"""Small REMI-like sampling pilot: fixed checkpoint/seeds, varying k then p.

Default: freeze/review plan. --execute creates 3x3 samples and listening report.
Reuse the existing three t10 baseline pieces; generate only six new pieces.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from baseline.audio.render_audio import sha256
from baseline.music import dump

SETTINGS = {'a_k32_p095': (32,.95), 'b_k64_p095': (64,.95), 'c_k64_p098': (64,.98)}


def validate(folder):
    songs=json.loads((folder/'sequences.json').read_text())
    names={f'sample_{i:03d}' for i in range(3)}
    if len(songs)!=3 or {s['song_name'] for s in songs}!=names:
        raise ValueError(f'{folder}: need exactly three complete samples')
    if {p.stem for p in folder.glob('*.mid')}!=names:
        raise ValueError('Missing or unexpected MIDI files')
    if any(s['tokens'][0]!='<BOS>' or s['tokens'][-1]!='<EOS>' or s['tokens'].count('Bar')!=32 for s in songs):
        raise ValueError('Malformed generated sequences')
    return songs


def pitch_ngrams(tokens,bars=4,n=4):
    """Simple pitch-stream overlap, NOT melody recognition or a quality score."""
    seen=0
    pitches=[]
    for token in tokens:
        if token=='Bar':seen+=1
        if seen>bars:break
        if token.startswith('Pitch_'):pitches.append(int(token[6:]))
    return {tuple(pitches[i:i+n]) for i in range(len(pitches)-n+1)}


def write_report(root):
    rows=[]
    for label in SETTINGS:
        songs=validate(root/label)
        for bars in (4,32):
            grams=[pitch_ngrams(s['tokens'],bars) for s in songs]
            overlaps=[len(a&b)/len(a|b) if a|b else None for a,b in itertools.combinations(grams,2)]
            if any(v is None for v in overlaps):raise ValueError('Not enough pitch events for overlap diagnostic')
            rows.append({'setting':label,'bars':bars,'song_count':3,'pair_count':3,
                         'mean_pitch_4gram_jaccard':statistics.mean(overlaps)})
    with (root/'overlap_diagnostic.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    lines=['# REMI-like 抽樣小比較（各 3 首，非正式 20 首評分）','',
           '相同六層 checkpoint、T=1.0、seed=123–125、32 小節；A 沿用既有 t10 的前三首。',
           'A→B 只改 top-k；B→C 只改 top-p。相同 seed 讓比較條件一致，不代表生成相同歌曲。','',
           '| 組別 | Temperature | Top-k | Top-p |','|---|---:|---:|---:|']
    for label,(k,p) in SETTINGS.items():lines.append(f'| {label} | 1.0 | {k} | {p} |')
    lines+=['','## 配對試聽','', '| Seed | A：原設定 | B：放寬 k | C：再放寬 p |', '|---|---|---|---|']
    for i in range(3):
        links=[f'[sample_{i:03d}]({label}/sample_{i:03d}.mid)' for label in SETTINGS]
        lines.append(f'| {123+i} | '+' | '.join(links)+' |')
    lines+=['','## 組內 pitch 4-gram 重疊診斷','',
            '把音符按 token 順序排列，計算每組三首間連續四個音高片段集合的 Jaccard 重疊率。',
            '包含和弦／伴奏排列，忽略節奏、力度及移調；不是旋律辨識、抄襲檢測或音樂品質指標。',
            '重疊率較低只表示這種表示下的共同片段較少，不代表更好聽。每組只有三首，結果僅供探索。','',
            '| 組別 | 前 4 小節重疊率 | 全 32 小節重疊率 |','|---|---:|---:|']
    for label in SETTINGS:
        values=[next(r['mean_pitch_4gram_jaccard'] for r in rows if r['setting']==label and r['bars']==b) for b in (4,32)]
        lines.append(f'| {label} | {values[0]:.1%} | {values[1]:.1%} |')
    lines+=['','## 聽感紀錄（待使用者填寫）','',
            '| 組別 | 前奏是否相似 | 曲內重複是否突兀 | 節奏／旋律是否自然 | 偏好及時間點 |',
            '|---|---|---|---|---|']
    for label in SETTINGS:lines.append(f'| {label} | 待聽 | 待聽 | 待聽 | |')
    lines+=['','- top-k 和 top-p 在合法事件集合上共同作用；改參數不保證每一步候選集合都會改變。',
            '- 增加隨機性可能提高變化，也可能使音樂更不連貫；不直接推薦重疊率最低的一組。',
            '- 尚未依此選正式繳交版本；確認偏好後才擴充到 20 首及正式評分。','']
    (root/'comparison.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/sampling_pilot'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    reference=Path('runs/experiments/basic_a/t10')
    checkpoint=Path('runs/baseline/best.pt')
    here=Path(__file__).resolve().parents[1]
    args=json.loads((reference/'generation_args.json').read_text())
    for key,value in {'count':20,'seed':123,'bars':32,'beats':4,'temperature':1.,'top_k':32,
                      'top_p':.95,'max_events_per_bar':128,'prompt':None}.items():
        if args[key]!=value:raise ValueError('Unexpected reference sampling settings')
    history=json.loads(Path('runs/depth_3_vs_6/plan.json').read_text())
    if sha256(checkpoint)!=history['reference_checkpoint_sha256'] or sha256(here/'generate.py')!=history['source_sha256']['generate.py']:
        raise ValueError('Reference model/generator changed; do not reuse historical samples')
    songs=json.loads((reference/'sequences.json').read_text())
    selected=[next(s for s in songs if s['song_name']==f'sample_{i:03d}') for i in range(3)]
    plan={'checkpoint_sha256':sha256(checkpoint),'generator_sha256':sha256(here/'generate.py'),
          'runner_sha256':sha256(__file__),'count_per_setting':3,'seed':123,'bars':32,'beats':4,
          'temperature':1.,'max_events_per_bar':128,'settings':SETTINGS,
          'reference_sequences_sha256':sha256(reference/'sequences.json'),
          'reference_midi_sha256':{s['song_name']:sha256(reference/(s['song_name']+'.mid')) for s in selected}}
    # Normalize tuple/list representation before comparing JSON on resume.
    plan=json.loads(json.dumps(plan))
    path=a.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Plan differs; choose a new output')
    dump(path,plan)
    print(f'Frozen plan: {path}; 3 settings x 3 seeds; reuse 3, generate 6',flush=True)
    if not a.execute:return
    import torch
    if a.device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA not available')
    for label,(k,top_p) in SETTINGS.items():
        folder=a.output/label
        marker=folder/'complete.json'
        if marker.exists():
            validate(folder)
            if json.loads(marker.read_text())!={p.name:sha256(p) for p in folder.glob('*.mid')}:
                raise ValueError('Completed MIDI files changed')
            continue
        if folder.exists() and any(folder.iterdir()):raise ValueError('Incomplete output exists; inspect before retrying')
        if label=='a_k32_p095':
            folder.mkdir(parents=True)
            for s in selected:shutil.copyfile(reference/(s['song_name']+'.mid'),folder/(s['song_name']+'.mid'))
            dump(folder/'sequences.json',selected)
            shutil.copyfile(reference/'tok2idx.json',folder/'tok2idx.json')
            dump(folder/'generation_args.json',{**args,'count':3,'output':str(folder),'reused_from':str(reference)})
        else:
            subprocess.run([sys.executable,str(here/'generate.py'),'--checkpoint',str(checkpoint),
                            '--output',str(folder),'--count','3','--bars','32','--seed','123',
                            '--temperature','1.0','--top-k',str(k),'--top-p',str(top_p),
                            '--device',a.device],check=True)
        validate(folder)
        dump(marker,{p.name:sha256(p) for p in folder.glob('*.mid')})
    write_report(a.output)
    print(f'Finished: {a.output}/comparison.md',flush=True)


if __name__=='__main__':main()
