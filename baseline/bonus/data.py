"""Separate MELODY/PIANO, using annotated bounded bars, never bonus GT as input."""
import argparse
import bisect
import json
from pathlib import Path
import sys

import numpy as np
import pretty_midi

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.music import beat_grid,PPB,MAX_DURATION,vocabulary as base_vocabulary
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import save_json


def vocabulary():
    result=base_vocabulary()
    for token in [f'Focus_{n}' for n in range(16)]+[f'TargetBars_{n}' for n in range(1,5)]:
        result[token]=len(result)
    return result


def named_track(midi,name):
    tracks=[i for i in midi.instruments if i.name.strip().upper()==name and not i.is_drum]
    if len(tracks)!=1:raise ValueError(f'Expected exactly one {name} track')
    return tracks[0]


def track_bars(instrument,times,starts,sizes):
    # A final downbeat marks the end, not another complete bar (same as TA GTBars).
    n=len(starts)-1
    if n<1:raise ValueError('No bounded annotated bars')
    events=[[] for _ in range(n)]
    end=float(times[starts[n]])
    ignored=0
    def coordinate(seconds):
        i=max(0,min(len(times)-2,bisect.bisect_right(times,seconds)-1))
        return i+(seconds-times[i])/(times[i+1]-times[i])
    for note in instrument.notes:
        if note.start>=end:
            ignored+=1;continue
        q=coordinate(note.start)
        b=max(0,bisect.bisect_right(starts,q)-1)
        if b>=n:raise ValueError('Onset outside bounded bars')
        pos=max(0,int((q-starts[b])*PPB))
        if pos>=sizes[b]*PPB:raise ValueError('Position outside bar')
        duration=max(1,min(MAX_DURATION,round((coordinate(min(note.end,end))-q)*PPB)))
        vel=min(125,max(1,note.velocity//4*4+1))
        events[b].append((pos,note.pitch,vel,duration))
    result=[]
    for b,notes in enumerate(events):
        start,size=starts[b],sizes[b]
        length=float(np.median(np.diff(times[start:start+size+1])))
        tempo=max(30,min(210,round((60/length)/5)*5))
        tokens=['Bar',f'Beats_{size}',f'Tempo_{tempo}']
        for pos,pitch,vel,dur in sorted(notes):
            tokens += [f'Pos_{pos}',f'Pitch_{pitch}',f'Vel_{vel}',f'Dur_{dur}']
        result.append(tokens)
    return result,ignored


def condition(melody_bars,focus,target_bars,vocab):
    if not 0<=focus<len(melody_bars)<=16 or not 1<=target_bars<=4 or focus+target_bars>len(melody_bars):
        raise ValueError('Invalid conditional window')
    return [vocab['<BOS>'],vocab[f'Focus_{focus}'],vocab[f'TargetBars_{target_bars}']]+[
        t for bar in melody_bars for t in bar]+[vocab['<EOS>']]


def training_pair(song,rng,vocab,target_context):
    n=len(song['melody'])
    first=rng.randint(0,max(0,n-16))
    melody=song['melody'][first:first+16]
    piano=song['piano'][first:first+16]
    focus=rng.randrange(len(melody))
    count=min(4,len(melody)-focus)
    while count>1 and sum(len(b) for b in piano[focus:focus+count])+1>target_context:
        count-=1
    target=[vocab['<BOS>']]+[t for b in piano[focus:focus+count] for t in b]+[vocab['<EOS>']]
    if len(target)-1>target_context:raise ValueError('Even one target bar exceeds context; increase target context')
    return condition(melody,focus,count,vocab),target


def load_prompt(folder):
    """Read melody.mid and beat_midi.txt ONLY. Piano/chord GT is evaluation-only."""
    folder=Path(folder)
    midi=pretty_midi.PrettyMIDI(str(folder/'melody.mid'))
    times,starts,sizes=beat_grid(folder/'beat_midi.txt')
    if len(starts)-1!=16:raise ValueError('Bonus prompt must contain exactly 16 bounded bars')
    bars,_=track_bars(named_track(midi,'MELODY'),times,starts,sizes)
    return midi,bars,times,starts,sizes[:16]


def prepare(dataset,output,expected=896):
    folders=sorted(p for p in Path(dataset).iterdir() if p.is_dir())
    if len(folders)!=expected:raise ValueError(f'Expected {expected} training songs, found {len(folders)}')
    output=Path(output)
    if (output/'pairs.json').exists():raise ValueError('Data exists; retain it or choose another output')
    vocab=vocabulary();songs=[];sources=[];notes={'MELODY':0,'PIANO':0};ignored={k:0 for k in notes}
    max_source=max_target_bar=0
    for folder in folders:
        path=folder/(folder.name+'.mid');beat=folder/'beat_midi.txt'
        midi=pretty_midi.PrettyMIDI(str(path));times,starts,sizes=beat_grid(beat)
        record={'song_name':folder.name,'beats_per_bar':sizes[:-1]}
        for name,key in [('MELODY','melody'),('PIANO','piano')]:
            bars,dropped=track_bars(named_track(midi,name),times,starts,sizes)
            record[key]=[[vocab[t] for t in bar] for bar in bars]
            notes[name]+=sum(sum(t.startswith('Pitch_') for t in bar) for bar in bars)
            ignored[name]+=dropped
        n=len(record['melody'])
        lengths=[len(b) for b in record['melody']]
        max_source=max(max_source,max(sum(lengths[i:i+16])+4 for i in range(max(1,n-15))))
        max_target_bar=max(max_target_bar,max(map(len,record['piano']))+1)
        songs.append(record);sources.append({'song_name':folder.name,'midi_sha256':sha256(path),'beats_sha256':sha256(beat)})
        if len(songs)%100==0:print(f'Track pairs {len(songs)}/{expected}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'tok2idx.json',vocab)
    save_json(output/'summary.json',{'songs':len(songs),'vocab_size':len(vocab),'quantization_steps_per_beat':PPB,
        'max_source_tokens_16_bars':max_source,'max_target_tokens_one_bar':max_target_bar,
        'notes_used':notes,'onsets_after_last_bounded_downbeat_excluded':ignored,
        'training_sources':sources,'condition':'16-bar melody, Focus_N + TargetBars_N; up to4bar piano targets',
        'target':'PIANO only; BRIDGE excluded; bonus prompt GT never read'})
    temp=output/'pairs.json.tmp';temp.write_text(json.dumps(songs,separators=(',',':')));temp.replace(output/'pairs.json')
    print(f'Prepared {len(songs)} paired songs; max16bar melody={max_source}; max1bar piano={max_target_bar}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('POP909/training'))
    p.add_argument('--output',type=Path,default=Path('data/bonus'))
    p.add_argument('--expected-songs',type=int,default=896)
    a=p.parse_args();prepare(a.dataset,a.output,a.expected_songs)
