"""Matched-duration SI study: 60 generated samples and 20 sampled references.

Defaults only build/review a frozen plan. --execute renders and evaluates it.
No Git changes, model training or new music generation are performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys

import mido

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import save_json

GROUPS=('real','t08','t10','t12')


def build_plan(training,generated,duration,seed):
    if duration<=15:
        raise ValueError('Duration must exceed the long SI lower bound of 15 seconds')
    eligible=[]
    references=sorted(Path(training).glob('*/*.mid'))
    if len(references)!=896:
        raise ValueError(f'Expected 896 training songs, found {len(references)}')
    for n,path in enumerate(references,1):
        length=mido.MidiFile(str(path)).length
        if length>=duration:
            eligible.append((path,length))
        if n%100==0:
            print(f'Reference eligibility: {n}/{len(references)}',flush=True)
    if len(eligible)<20:
        raise ValueError('Fewer than 20 references meet the duration requirement')
    selected=sorted(random.Random(seed).sample(eligible,20),key=lambda item:item[0].stem)
    def describe(path,length=None):
        length=mido.MidiFile(str(path)).length if length is None else length
        if length<duration:
            raise ValueError(f'{path}: too short; do not silently drop generated samples')
        return {'song_name':path.stem,'source_path':str(path.resolve()),'source_sha256':sha256(path),
                'midi_duration_seconds':length}
    groups={'real':[describe(path,length) for path,length in selected]}
    for name in GROUPS[1:]:
        folder=Path(generated)/name
        expected=[folder/f'sample_{i:03d}.mid' for i in range(20)]
        if set(folder.glob('*.mid'))!=set(expected):
            raise ValueError(f'{folder}: expected exactly sample_000.mid through sample_019.mid')
        groups[name]=[describe(path) for path in expected]
    return {'schema_version':1,'study':'SI matched-duration sampled-reference comparison',
            'duration_seconds':duration,'start_seconds':0.,'bounds_seconds':[3.,8.,15.],
            'reference_seed':seed,'reference_population':896,'reference_eligible':len(eligible),
            'reference_selection':'Uniform sample without replacement from songs >= duration; Python random.Random',
            'groups':groups}


def write_summary(root,plan):
    """Merge only complete, identical-protocol group reports."""
    records=[]
    protocol=None
    for group in GROUPS:
        part=root/'groups'/group
        current=json.loads((part/'protocol.json').read_text())
        if protocol is None:
            protocol=current
        elif protocol!=current:
            raise ValueError('Group SI protocols differ; results cannot be merged')
        with (part/'per_piece.csv').open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        expected={item['song_name'] for item in plan['groups'][group]}
        if len(rows)!=20 or {r['song_name'] for r in rows}!=expected or any(r['group']!=group for r in rows):
            raise ValueError(f'{group}: SI results do not match all 20 planned songs')
        if any(float(r['duration_seconds'])!=plan['duration_seconds'] for r in rows):
            raise ValueError('Duration mismatch')
        if any(not math.isfinite(float(r[m])) for r in rows for m in ('SI_short','SI_mid','SI_long')):
            raise ValueError('Non-finite SI score')
        records.extend(rows)
    with (root/'per_piece.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary=[]
    for group in GROUPS:
        rows=[r for r in records if r['group']==group]
        result={'setting':group,'song_count':len(rows)}
        for metric in ('SI_short','SI_mid','SI_long'):
            values=[float(r[metric]) for r in rows]
            result[metric+'_mean']=statistics.mean(values)
            result[metric+'_std']=statistics.stdev(values)
        summary.append(result)
    for row in summary:
        for metric in ('SI_short','SI_mid','SI_long'):
            row[metric+'_abs_diff_from_reference']=abs(row[metric+'_mean']-summary[0][metric+'_mean'])
    with (root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines=['# SI:matched-duration batch comparison','',
           f'First{plan["duration_seconds"]:g}s; three generated groups of20and20fixed sampled real references.',
           f'896 real songs,{plan["reference_eligible"]}length-eligible; sampling seed={plan["reference_seed"]}.','',
           '| Setting | Songs | SI short mean +/- standard deviation | SI mid mean +/- standard deviation | SI long mean +/- standard deviation |',
           '|---|---:|---:|---:|---:|']
    for row in summary:
        values=[f'{row[m+"_mean"]:.4f} ± {row[m+"_std"]:.4f}' for m in ('SI_short','SI_mid','SI_long')]
        lines.append(f'| {row["setting"]} | {row["song_count"]} | '+' | '.join(values)+' |')
    lines+=['','## Mean differences from these20 real references','',
            '| Setting | short difference | mid difference | long difference |','|---|---:|---:|---:|']
    for row in summary[1:]:
        values=[f'{row[m+"_abs_diff_from_reference"]:.4f}' for m in ('SI_short','SI_mid','SI_long')]
        lines.append(f'| {row["setting"]} | '+' | '.join(values)+' |')
    lines+=['','## Interpretation and limits','',
            f'- short：3–8s;mid：8–15s;long：15–{plan["duration_seconds"]:g}s.',
            '- Higher is not universally better; proximity to this reference does not imply best listening quality.',
            '- Standard deviations describe variation across20 songs,not confidence intervals.',
            '- Real reference samples20 length-eligible songs; this is not the mean of all896 songs.',
            '- Fixed-duration openings can cover different bar counts and do not establish full-piece structure.',
            '- H4/GS uses full pieces and896 real songs; report separately from excerpt-based SI.',
            '- This study excludes Task B; SI is not originality/plagiarism detection.',
            '- Sources,selections,and hashes:study_plan.json; render records:audio/; protocols:groups/.','']
    (root/'summary.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,default=Path('POP909/training'))
    p.add_argument('--generated',type=Path,default=Path('runs/experiments/basic_a'))
    p.add_argument('--output',type=Path,default=Path('runs/metrics/si48_g0025'))
    p.add_argument('--duration',type=float,default=48.)
    p.add_argument('--gain',type=float,default=.025)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=4,choices=range(1,5))
    p.add_argument('--soundfont',type=Path)
    p.add_argument('--fluidsynth',default='fluidsynth')
    p.add_argument('--si-python',default=sys.executable,help='Python with requirements-si.txt')
    p.add_argument('--renderer-python',default=sys.executable,help='Python with mido for render_audio.py')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not math.isfinite(a.duration) or a.duration<=15 or not math.isfinite(a.gain) or not 0<a.gain<=5:
        p.error('Invalid duration or gain')
    root=a.output
    root.mkdir(parents=True,exist_ok=True)
    path=root/'study_plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['duration_seconds']!=a.duration or plan['reference_seed']!=a.seed:
            p.error('Existing plan differs; choose a new --output directory')
        # Validate stored sources, never silently select another song on resume.
        for items in plan['groups'].values():
            for item in items:
                if sha256(item['source_path'])!=item['source_sha256']:
                    raise ValueError('Input MIDI changed since the study plan was created')
    else:
        plan=build_plan(a.training,a.generated,a.duration,a.seed)
        save_json(path,plan)
    print(f'Frozen plan: {path}; '+', '.join(f'{g}={len(plan["groups"][g])}' for g in GROUPS),flush=True)
    if not a.execute:
        print('Plan only; add --execute and renderer settings to start audio/SI computation.')
        return
    if a.soundfont is None or not a.soundfont.is_file():
        p.error('--execute requires a valid --soundfont')
    here=Path(__file__).resolve().parents[1]
    # Each worker launches a separate CPU process; groups have independent output paths.
    def run_group(group):
        inputs=root/'inputs'/group
        inputs.mkdir(parents=True,exist_ok=True)
        expected=set()
        for item in plan['groups'][group]:
            target=inputs/(item['song_name']+'.mid')
            expected.add(target.name)
            if not target.exists() or sha256(target)!=item['source_sha256']:
                shutil.copyfile(item['source_path'],target)
        if {x.name for x in inputs.glob('*.mid')}!=expected:
            raise ValueError(f'Unexpected files in {inputs}')
        log=root/(group+'.log')
        with log.open('a',buffering=1) as stream:
            subprocess.run([a.renderer_python,str(here/'audio/render_audio.py'),'--input',str(inputs),
                            '--output',str(root/'audio'/group),'--soundfont',str(a.soundfont),
                            '--fluidsynth',a.fluidsynth,'--gain',str(a.gain)],stdout=stream,stderr=subprocess.STDOUT,check=True)
            subprocess.run([a.si_python,str(here/'evaluation/evaluate_si.py'),'--audio-root',str(root/'audio'),
                            '--groups',group,'--limit-per-group','20','--duration',str(a.duration),
                            '--scope','batch','--output',str(root/'groups'/group)],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f'Completed group: {group}',flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run_group,GROUPS))
    write_summary(root,plan)
    print(f'Finished 80 clips: {root}/summary.md',flush=True)


if __name__=='__main__':
    main()
