"""Regenerate Basic A/B and depth samples with boundary-v2, without training.

Use a new output root. Completed groups resume only after source, checkpoint,
sampling, output hashes, and MIDI/token audits match the frozen identity.
"""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.audio.render_audio import sha256
from baseline.music import beat_grid, dump
import mido


def audit(folder, count, prompt=None, beats=None):
    pieces = json.loads((folder/'sequences.json').read_text())
    if len(pieces) != count:
        raise ValueError('Wrong piece count')
    expected = {s['song_name']+'.mid' for s in pieces}
    if {p.name for p in folder.glob('*.mid')} != expected:
        raise ValueError('Wrong MIDI file set')
    rows = []
    for s in pieces:
        bars, seen, pos, duplicates = [], set(), None, 0
        for t in s['tokens']:
            if t == 'Bar':
                bars.append([]); seen=set(); pos=None
            if bars: bars[-1].append(t)
            if t.startswith('Pos_'): pos=t
            if t.startswith('Pitch_'):
                key=(pos,t)
                if key in seen and (prompt is None or len(bars)>8): duplicates+=1
                seen.add(key)
        counts=[sum(t.startswith('Pitch_') for t in b) for b in bars]
        if len(bars)!=32 or s['tokens'][-1]!='<EOS>' or duplicates:
            raise ValueError('Invalid bars, ending, or duplicate generated onset/pitch')
        path=folder/(s['song_name']+'.mid')
        row={'song_name':s['song_name'],'midi_sha256':sha256(path),'bars':32,
             'notes_per_bar':counts,'generated_duplicates':duplicates,
             'last_bar_reaches_cap':counts[-1]>=128}
        if prompt is not None:
            original=mido.MidiFile(str(prompt)); times,starts,_=beat_grid(beats)
            boundary=float(times[starts[8]])
            expected_events=[];seconds=0.;tick=0;tempo=500000
            for msg in mido.merge_tracks(original.tracks):
                next_seconds=seconds+mido.tick2second(msg.time,original.ticks_per_beat,tempo)
                if next_seconds>=boundary:break
                tick+=msg.time;seconds=next_seconds
                if msg.type=='set_tempo':tempo=msg.tempo
                if msg.type!='end_of_track':expected_events.append((tick,msg.copy(time=0)))
            actual=[];tick=0
            for msg in mido.merge_tracks(mido.MidiFile(str(path)).tracks):
                tick+=msg.time
                if msg.type!='end_of_track':actual.append((tick,msg.copy(time=0)))
            if actual[:len(expected_events)]!=expected_events:
                raise ValueError('Original prompt events changed')
            row['prompt_events_preserved']=True
            row['prompt_event_count']=len(expected_events)
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/corrected_boundary_v2'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--stage',choices=['plan','generate','metrics','all'],default='plan')
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    files=['baseline/generate.py','baseline/model.py','baseline/music.py',
           'baseline/evaluation/evaluate.py','baseline/experiments/run_corrected_generation.py']
    jobs=[]
    for label,temp in [('t08',.8),('t10',1.),('t12',1.2)]:
        jobs.append({'path':f'basic_a/{label}','count':20,'temperature':temp,'checkpoint':'runs/baseline/best.pt'})
        for name in ('001','002','003'):
            folder=Path('POP909/prompts/basic-b')/name
            jobs.append({'path':f'basic_b/{name}/{label}','count':1,'temperature':temp,
                'checkpoint':'runs/baseline/best.pt','prompt':str(folder/f'{name}.mid'),
                'beats':str(folder/'beat_midi.txt')})
    jobs.append({'path':'depth/layers3','count':20,'temperature':1.,'checkpoint':'runs/depth_3_vs_6/train/last.pt'})
    for j in jobs:
        j['checkpoint_sha256']=sha256(j['checkpoint'])
        if 'prompt' in j:j['condition_sha256']={k:sha256(j[k]) for k in ('prompt','beats')}
    plan={'policy':'boundary-v2','seed':123,'bars':32,'top_k':32,'top_p':.95,
          'max_events_per_bar':128,'jobs':jobs,'source_sha256':{f:sha256(f) for f in files},
          'scope':'New inference only; historical outputs and model weights retained. MIDI-like results require separate protocol verification.'}
    planpath=root/'plan.json'
    if planpath.exists() and json.loads(planpath.read_text())!=plan:
        raise ValueError('Frozen plan differs; use a new output')
    dump(planpath,plan)
    if a.stage=='plan':print(planpath);return
    if a.stage in ('generate','all'):
        for j in jobs:
            folder=root/j['path'];marker=folder/'completed.json'
            if marker.exists():
                saved=json.loads(marker.read_text())
                if saved!={'job':j,'audit':audit(folder,j['count'],j.get('prompt'),j.get('beats'))}:
                    raise ValueError('Completed group changed')
                print('Verified existing group:',folder,flush=True);continue
            if folder.exists():raise ValueError(f'Partial output; inspect before resuming: {folder}')
            args=[sys.executable,'baseline/generate.py','--checkpoint',j['checkpoint'],'--output',str(folder),
                  '--count',str(j['count']),'--bars','32','--temperature',str(j['temperature']),
                  '--top-k','32','--top-p','.95','--seed','123','--generation-policy','boundary-v2','--device',a.device]
            if 'prompt' in j:args+=['--prompt',j['prompt'],'--prompt-beats',j['beats'],'--prompt-bars','8']
            subprocess.run(args,check=True)
            rows=audit(folder,j['count'],j.get('prompt'),j.get('beats'))
            dump(marker,{'job':j,'audit':rows})
            print('Audited:',folder,'last-bar cap count:',sum(r['last_bar_reaches_cap'] for r in rows),flush=True)
    if a.stage in ('metrics','all'):
        for label,folder in [('t08',root/'basic_a/t08'),('t10',root/'basic_a/t10'),
                             ('t12',root/'basic_a/t12'),('layers3',root/'depth/layers3')]:
            if not (folder/'completed.json').exists():raise ValueError('Generation audit missing')
            subprocess.run([sys.executable,'baseline/evaluation/evaluate.py','--data',str(folder),
                            '--output',str(root/'metrics'/f'{label}.csv')],check=True)
    print('Finished corrected MIDI and H1/H4/GS. WAV, SI, reports follow separately.',flush=True)


if __name__=='__main__':main()
