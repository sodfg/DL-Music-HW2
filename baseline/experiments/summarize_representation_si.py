"""Merge protocol-identical SI: frozen real/Bar-Pos reference and MIDI-like.

The new group must first be rendered and evaluated using the commands in
REPRESENTATION_EXPERIMENT.md. This only reads verified results, no GPU needed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import verified_audio_record, save_json

METRICS=('SI_short','SI_mid','SI_long')


def summarize(root, reference):
    reference_plan=json.loads((reference/'study_plan.json').read_text())
    experiment=json.loads((root.parent/'plan.json').read_text())
    raw=json.loads((root.parent/'generated/event_sequences.json').read_text())
    names={f'sample_{i:03d}' for i in range(20)}
    if len(raw)!=20 or {s['song_name'] for s in raw}!=names:
        raise ValueError('MIDI-like group must contain all 20 generated pieces')
    expected_new={s['song_name']:sha256(root.parent/'generated'/(s['song_name']+'.mid')) for s in raw}
    groups=[('real',reference,'real'),('bar_pos',reference,'layers6'),('midi_like',root,'midi_like')]
    protocol=None
    records=[]
    provenance=[]
    for label,folder,group in groups:
        part=folder/'groups'/group
        current=json.loads((part/'protocol.json').read_text())
        if protocol is None: protocol=current
        if current!=protocol or current['duration_seconds']!=48 or current['start_seconds']!=0:
            raise ValueError('SI protocol differs; do not merge unlike studies')
        expected=(expected_new if label=='midi_like' else
                  {s['song_name']:s['source_sha256'] for s in reference_plan['groups'][group]})
        if len(expected)!=20: raise ValueError('Need 20 planned pieces per group')
        with (part/'per_piece.csv').open() as stream: rows=list(csv.DictReader(stream))
        if len(rows)!=20 or {r['song_name'] for r in rows}!=set(expected):
            raise ValueError('SI CSV differs from planned samples')
        for row in rows:
            wav=folder/'audio'/group/(row['song_name']+'.wav')
            audio=verified_audio_record(wav,folder/'audio')
            if (row['group']!=group or float(row['start_seconds'])!=0 or float(row['duration_seconds'])!=48
                    or audio['source_sha256']!=expected[row['song_name']]
                    or audio['settings']!=protocol['render_settings'] or audio['audio']['clipped_samples']
                    or row['wav_sha256']!=audio['wav_sha256'] or audio['midi_duration_seconds']<48
                    or any(not math.isfinite(float(row[m])) for m in METRICS)):
                raise ValueError(f'Unverified result/audio: {wav}')
            if label=='bar_pos' and sha256(Path('runs/experiments/basic_a/t10')/(row['song_name']+'.mid'))!=audio['source_sha256']:
                raise ValueError('Bar/Pos audio differs from the compared t10 MIDI')
            records.append({**row,'group':label})
        provenance.append({'group':label,'protocol_path':str(part/'protocol.json'),
                           'csv_sha256':sha256(part/'per_piece.csv'),'midi_sha256':expected})
    summary=[]
    for group in ('real','bar_pos','midi_like'):
        rows=[r for r in records if r['group']==group]
        result={'setting':group,'song_count':len(rows)}
        for m in METRICS:
            values=[float(r[m]) for r in rows]
            result[m+'_mean']=statistics.mean(values)
            result[m+'_std']=statistics.stdev(values)
            result[m+'_abs_diff_from_reference']=abs(result[m+'_mean']-(summary[0][m+'_mean'] if summary else result[m+'_mean']))
        summary.append(result)
    for name,rows in [('per_piece.csv',records),('comparison.csv',summary)]:
        with (root/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(root/'comparison_protocol.json',{'protocol':protocol,'groups':provenance,
        'reference_plan_sha256':sha256(reference/'study_plan.json'),
        'representation_plan_sha256':sha256(root.parent/'plan.json'),
        'reference_selection':reference_plan['reference_selection'],
        'reference_seed':reference_plan['reference_seed'],
        'reference_checkpoint_sha256':experiment['reference_checkpoint_sha256']})
    lines=['# 表示法比較：48 秒 SI','',
           '各 20 首，開頭 48 秒；真實組沿用 seed=42、從長度合格的 895 首中抽樣的 20 首。',
           '兩種表示法都使用六層模型、T=1.0、top-k=32、top-p=0.95、seed=123–142。','',
           '| 組別 | SI short 平均 ± 標準差 | SI mid 平均 ± 標準差 | SI long 平均 ± 標準差 |',
           '|---|---:|---:|---:|']
    labels={'real':'真實參考','bar_pos':'Bar/Pos','midi_like':'MIDI-like'}
    for r in summary:
        lines.append('| '+labels[r['setting']]+' | '+' | '.join(f'{r[m+"_mean"]:.4f} ± {r[m+"_std"]:.4f}' for m in METRICS)+' |')
    lines+=['','## 與真實平均的差距','','| 組別 | short | mid | long |','|---|---:|---:|---:|']
    for r in summary[1:]:
        lines.append('| '+labels[r['setting']]+' | '+' | '.join(f'{r[m+"_abs_diff_from_reference"]:.4f}' for m in METRICS)+' |')
    lines+=['','- short 3–8 秒，mid 8–15 秒，long 15–48 秒。',
            '- 逐項核對完整相同的算法、wrapper 雜湊、軟體版本、音色與增益協定，才沿用舊參考分數。',
            '- WAV 雜湊、MIDI 來源、無截波、片段長度及完整 20 首均已驗證。',
            '- 真實組只有抽樣 20 首。H4/GS 的 896 首全曲參考與這裡不同，維持分表。',
            '- SI 不是原創性或聽感評分，異常重複也可能得分；單訓練 seed 不推論普遍優劣。','']
    (root/'comparison.md').write_text('\n'.join(lines))
    print(f'Verified 60 pieces: {root}/comparison.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/representation_midi_like/si48'))
    p.add_argument('--reference',type=Path,default=Path('runs/depth_3_vs_6/si48'))
    a=p.parse_args();summarize(a.root,a.reference)
