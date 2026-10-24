"""Generate PIANO from melody.mid + beat_midi.txt, preserve original melody events."""
import argparse
import copy
import json
from pathlib import Path
import sys

import mido
import numpy as np
import torch

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.bonus.data import load_prompt,condition,vocabulary
from baseline.bonus.model import Config,AccompanimentTransformer
from baseline.music import PPB
from baseline.representations.generate_midi_like import sample
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import save_json


def allowed(tokens,vocab,sizes,tempos,max_events):
    last=tokens[-1];bars=tokens.count('Bar')
    group=lambda prefix:[t for t in vocab if t.startswith(prefix+'_')]
    if last=='<BOS>':return ['Bar']
    if last=='Bar':return [f'Beats_{sizes[bars-1]}']
    if last.startswith('Beats_'):return [tempos[bars-1]]
    if last.startswith('Pos_'):return group('Pitch')
    if last.startswith('Pitch_'):return group('Vel')
    if last.startswith('Vel_'):return group('Dur')
    start=len(tokens)-1-tokens[::-1].index('Bar');current=tokens[start:]
    ending=['Bar'] if bars<len(sizes) else (['<EOS>'] if any(t.startswith('Dur_') for t in tokens) else [])
    if sum(t.startswith('Pitch_') for t in current)>=max_events:return ending
    pos=next((int(t[4:]) for t in reversed(current) if t.startswith('Pos_')),0)
    return [f'Pos_{n}' for n in range(pos,sizes[bars-1]*PPB)]+ending


def note_grid(tokens,sizes):
    notes=[];bar=-1;pos=0;pitch=None;vel=81
    starts=np.cumsum([0]+[n*PPB for n in sizes])
    for token in tokens:
        if token=='Bar':bar+=1;pos=0;pitch=None
        elif token.startswith('Pos_'):pos=int(token[4:])
        elif token.startswith('Pitch_'):pitch=int(token[6:])
        elif token.startswith('Vel_'):vel=int(token[4:])
        elif token.startswith('Dur_') and pitch is not None:
            if not 0<=bar<len(sizes) or pos>=sizes[bar]*PPB:raise ValueError('Invalid generated bar position')
            onset=int(starts[bar])+pos;end=min(int(starts[-1]),onset+int(token[4:]))
            if end<=onset:raise ValueError('Nonpositive generated duration')
            notes.append((onset,end,pitch,vel));pitch=None
    if bar+1!=len(sizes) or not notes:raise ValueError('Expected complete nonempty accompaniment')
    return notes


def write_accompaniment(folder,tokens,output):
    melody,bars,times,starts,sizes=load_prompt(folder)
    notes=note_grid(tokens,sizes)
    original=mido.MidiFile(str(Path(folder)/'melody.mid'))
    if original.type==2:raise ValueError('Asynchronous MIDI unsupported')
    out=mido.MidiFile(type=1,ticks_per_beat=original.ticks_per_beat)
    out.tracks=copy.deepcopy(original.tracks)
    used={m.channel for track in original.tracks for m in track if not m.is_meta and hasattr(m,'channel')}
    channel=next(c for c in range(16) if c not in used and c!=9)
    piano=mido.MidiTrack([mido.MetaMessage('track_name',name='PIANO'),mido.Message('program_change',program=0,channel=channel)])
    def to_tick(grid):
        seconds=float(np.interp(grid/PPB,np.arange(len(times)),times))
        return int(round(melody.time_to_tick(seconds)))
    events=[]
    for onset,end,pitch,vel in notes:
        first,last=to_tick(onset),to_tick(end)
        if last<=first:last=first+1
        events.extend([(first,1,mido.Message('note_on',note=pitch,velocity=vel,channel=channel)),
                       (last,0,mido.Message('note_off',note=pitch,velocity=0,channel=channel))])
    previous=0
    for tick,_,msg in sorted(events,key=lambda e:(e[0],e[1])):
        piano.append(msg.copy(time=tick-previous));previous=tick
    final=max(previous,to_tick(sum(sizes)*PPB))
    piano.append(mido.MetaMessage('end_of_track',time=final-previous));out.tracks.append(piano)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);out.save(str(output))
    restored=mido.MidiFile(str(output))
    if original.tracks!=restored.tracks[:len(original.tracks)]:raise ValueError('Original melody MIDI events changed')
    return {'melody_events_preserved':True,'generated_notes':len(notes),'bounded_bars':len(sizes),
            'bar_beats':sizes,'end_seconds':float(times[starts[16]]),'piano_channel':channel,
            'condition_files':{name:sha256(Path(folder)/name) for name in ('melody.mid','beat_midi.txt')},
            'midi_sha256':sha256(output)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--prompts',type=Path,default=Path('POP909/prompts/bonus'))
    p.add_argument('--output',type=Path,default=Path('runs/bonus_accompaniment/generated'))
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--top-k',type=int,default=32)
    p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--max-events-per-bar',type=int,default=48)
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    import math
    if (not math.isfinite(a.temperature) or a.temperature<=0 or not 0<a.top_p<=1
            or a.top_k<0 or not 1<=a.limit<=10 or a.max_events_per_bar<1):p.error('Invalid generation settings')
    if list(a.output.glob('*.mid')) or (a.output/'sequences.json').exists():p.error('Existing generated data; choose a new output')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    saved=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    if saved.get('kind')!='bonus_encoder_decoder' or saved['vocab']!=vocabulary():p.error('Not a compatible Bonus checkpoint')
    model=AccompanimentTransformer(Config(**saved['config'])).to(a.device);model.load_state_dict(saved['model']);model.eval()
    vocab=saved['vocab'];inverse={i:t for t,i in vocab.items()}
    folders=sorted(p for p in a.prompts.iterdir() if p.is_dir())
    if len(folders)!=10:p.error('Expected exactly ten provided Bonus prompts')
    pieces=[];validation=[]
    args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    args['checkpoint_sha256']=sha256(a.checkpoint);args['checkpoint_step']=saved['step']
    save_json(a.output/'generation_args.json',args)
    with torch.inference_mode():
        for index,folder in enumerate(folders[:a.limit]):
            torch.manual_seed(a.seed+index)
            _,melody,times,starts,sizes=load_prompt(folder)
            encoded=[[vocab[t] for t in b] for b in melody]
            combined=['<BOS>']
            for first in range(0,16,4):
                count=min(4,16-first)
                # Never slide/truncate the generated target within a chunk.
                cap=min(a.max_events_per_bar,(model.config.target_context-4*count-4)//(4*count))
                if cap<1:raise ValueError('Target context too small even for a minimal complete chunk')
                source=torch.tensor([condition(encoded,first,count,vocab)],device=a.device)
                memory,padding=model.encode(source)
                tokens=['<BOS>'];local_sizes=sizes[first:first+count];tempos=[b[2] for b in melody[first:first+count]]
                while tokens[-1]!='<EOS>':
                    if len(tokens)>model.config.target_context:raise RuntimeError('Complete chunk exceeded context')
                    ids=torch.tensor([[vocab[t] for t in tokens]],device=a.device)
                    logits=model.decode(ids,memory,padding)[0,-1]
                    choices=allowed(tokens,vocab,local_sizes,tempos,cap)
                    token=inverse[sample(logits,[vocab[t] for t in choices],a.temperature,a.top_k,a.top_p)]
                    tokens.append(token)
                combined+=tokens[1:-1]
            combined+=['<EOS>']
            path=a.output/(folder.name+'.mid')
            record=write_accompaniment(folder,combined,path);record['song_name']=folder.name
            pieces.append({'song_name':folder.name,'piano_tokens':combined})
            validation.append(record)
            print(f'{folder.name}: 16 bars; {record["generated_notes"]} piano notes; original melody events preserved',flush=True)
    save_json(a.output/'validation.json',validation);save_json(a.output/'sequences.json',pieces)
    print(f'Generated {len(pieces)} conditioned pieces',flush=True)


if __name__=='__main__':main()
