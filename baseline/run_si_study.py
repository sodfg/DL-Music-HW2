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
from render_audio import sha256
from evaluate_si import save_json

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
    lines=['# SI：匹配片段長度的批次比較','',
           f'每首分析前 {plan["duration_seconds"]:g} 秒。三組生成各 20 首；真實參考為固定抽樣的 20 首。',
           f'真實曲總數 896，符合長度條件 {plan["reference_eligible"]} 首；抽樣 seed={plan["reference_seed"]}。','',
           '| 設定 | 首數 | SI short 平均 ± 標準差 | SI mid 平均 ± 標準差 | SI long 平均 ± 標準差 |',
           '|---|---:|---:|---:|---:|']
    for row in summary:
        values=[f'{row[m+"_mean"]:.4f} ± {row[m+"_std"]:.4f}' for m in ('SI_short','SI_mid','SI_long')]
        lines.append(f'| {row["setting"]} | {row["song_count"]} | '+' | '.join(values)+' |')
    lines+=['','## 與這 20 首真實參考的平均差距','',
            '| 設定 | short 差距 | mid 差距 | long 差距 |','|---|---:|---:|---:|']
    for row in summary[1:]:
        values=[f'{row[m+"_abs_diff_from_reference"]:.4f}' for m in ('SI_short','SI_mid','SI_long')]
        lines.append(f'| {row["setting"]} | '+' | '.join(values)+' |')
    lines+=['','## 解讀與限制','',
            f'- short：3–8 秒；mid：8–15 秒；long：15–{plan["duration_seconds"]:g} 秒。',
            '- 分數不是越高越好；接近此參考組也不等於音樂最好聽。',
            '- 標準差是 20 首歌曲間的樣本標準差，不是信賴區間。',
            '- 真實組只有抽樣 20 首；不足長度的曲子未參與抽樣，不能稱為全 896 首平均。',
            '- 固定秒數代表各曲可能涵蓋不同小節數；只評估開頭片段，不能推論完整曲子結構。',
            '- H4/GS 原表是全曲及全部 896 首參考，與本次範圍不同，維持獨立表格。',
            '- 本研究不包含 Task B；SI 不是原創性／抄襲檢測。',
            '- 來源、選曲與輸入雜湊見 study_plan.json，渲染記錄見 audio/，算法協定見各 groups/。','']
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
    here=Path(__file__).resolve().parent
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
            subprocess.run([a.renderer_python,str(here/'render_audio.py'),'--input',str(inputs),
                            '--output',str(root/'audio'/group),'--soundfont',str(a.soundfont),
                            '--fluidsynth',a.fluidsynth,'--gain',str(a.gain)],stdout=stream,stderr=subprocess.STDOUT,check=True)
            subprocess.run([a.si_python,str(here/'evaluate_si.py'),'--audio-root',str(root/'audio'),
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
