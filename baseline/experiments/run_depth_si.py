"""Compare depth using matched 48-second SI clips; no training or generation.

Reuse verified reference/six-layer audio, render three-layer MIDI, then evaluate
all three groups with the same current SI wrapper. Default is plan only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import mido

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import save_json, verified_audio_record

METRICS = ('SI_short', 'SI_mid', 'SI_long')
GROUPS = ('real', 'layers6', 'layers3')


def freeze_plan(depth, study, output):
    original = json.loads((study/'study_plan.json').read_text())
    experiment = json.loads((depth/'plan.json').read_text())
    identity = json.loads((depth/'generation_identity.json').read_text())
    six_args = json.loads((Path(experiment['reference_samples'])/'generation_args.json').read_text())
    if identity['sampling'] != experiment['sampling'] or any(
            six_args[k] != v for k, v in experiment['sampling'].items()):
        raise ValueError('Three/six-layer sampling settings differ')
    if sha256(depth/'train/last.pt') != identity['checkpoint_sha256']:
        raise ValueError('Three-layer checkpoint differs from generation identity')
    if sha256(Path(experiment['reference_run'])/'best.pt') != experiment['reference_checkpoint_sha256']:
        raise ValueError('Six-layer checkpoint differs from depth plan')
    if original['duration_seconds'] != 48 or original['start_seconds'] != 0:
        raise ValueError('Expected the frozen first-48-second reference study')
    groups = {}
    settings = None
    for label, old_group in (('real', 'real'), ('layers6', 't10')):
        items = original['groups'][old_group]
        folder = study/'audio'/old_group
        if len(items) != 20 or {p.stem for p in folder.glob('*.wav')} != {x['song_name'] for x in items}:
            raise ValueError(f'{label}: expected exactly the 20 frozen reference-study WAVs')
        groups[label] = []
        for item in items:
            source = Path(item['source_path'])
            if sha256(source) != item['source_sha256']:
                raise ValueError(f'Study source changed: {source}')
            if label == 'layers6' and sha256(Path(experiment['reference_samples'])/source.name) != item['source_sha256']:
                raise ValueError('Six-layer MIDI differs from reference audio source')
            wav = folder/(item['song_name']+'.wav')
            record = verified_audio_record(wav, study/'audio')
            if record['source_sha256'] != item['source_sha256'] or record['audio']['clipped_samples']:
                raise ValueError('Audio source mismatch or clipping')
            if record['midi_duration_seconds'] < 48:
                raise ValueError('Not enough musical content')
            if settings is None:
                settings = record['settings']
            elif record['settings'] != settings:
                raise ValueError('Render settings differ')
            groups[label].append({**item, 'wav_sha256': record['wav_sha256']})
    files = sorted((depth/'generated').glob('*.mid'))
    if [p.name for p in files] != [f'sample_{i:03d}.mid' for i in range(20)]:
        raise ValueError('Expected exactly 20 three-layer samples')
    groups['layers3'] = []
    for path in files:
        seconds = mido.MidiFile(str(path)).length
        if seconds < 48:
            raise ValueError(f'Too short: {path}')
        groups['layers3'].append({'song_name': path.stem, 'source_path': str(path.resolve()),
                                 'source_sha256': sha256(path), 'midi_duration_seconds': seconds})
    plan = {'duration_seconds': 48., 'start_seconds': 0., 'bounds_seconds': [3., 8., 15.],
            'reference_seed': original['reference_seed'], 'reference_eligible': original['reference_eligible'],
            'reference_population': original['reference_population'],
            'reference_selection': original['reference_selection'],
            'depth_plan_sha256': sha256(depth/'plan.json'), 'generation_identity': identity,
            'reference_study_plan_sha256': sha256(study/'study_plan.json'),
            'render_settings': settings, 'groups': groups}
    path = output/'study_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Inputs changed; select a new output directory')
    save_json(path, plan)
    return plan


def write_report(root, plan):
    records, summary = [], []
    protocol = None
    for group in GROUPS:
        part = root/'groups'/group
        current = json.loads((part/'protocol.json').read_text())
        if protocol is None:
            protocol = current
        if (current != protocol or current['render_settings'] != plan['render_settings']
                or current['duration_seconds'] != 48 or current['start_seconds'] != 0
                or current['bounds_seconds'] != plan['bounds_seconds']):
            raise ValueError('SI protocols differ')
        with (part/'per_piece.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        expected = {x['song_name'] for x in plan['groups'][group]}
        if len(rows) != 20 or {r['song_name'] for r in rows} != expected:
            raise ValueError(f'{group}: results do not cover all 20 planned songs')
        for row in rows:
            wav = root/'audio'/group/(row['song_name']+'.wav')
            record = verified_audio_record(wav, root/'audio')
            source = next(x for x in plan['groups'][group] if x['song_name'] == row['song_name'])
            if (row['group'] != group or float(row['duration_seconds']) != 48 or float(row['start_seconds']) != 0
                    or record['settings'] != plan['render_settings'] or record['audio']['clipped_samples']
                    or record['source_sha256'] != source['source_sha256'] or row['wav_sha256'] != record['wav_sha256']
                    or any(not math.isfinite(float(row[m])) for m in METRICS)):
                raise ValueError('SI result/source/audio mismatch')
        records.extend(rows)
        result = {'setting': group, 'song_count': len(rows)}
        for metric in METRICS:
            values = [float(r[metric]) for r in rows]
            result[metric+'_mean'] = statistics.mean(values)
            result[metric+'_std'] = statistics.stdev(values)
            result[metric+'_abs_diff_from_reference'] = abs(result[metric+'_mean'] -
                (summary[0][metric+'_mean'] if summary else result[metric+'_mean']))
        summary.append(result)
    for filename, rows in (('per_piece.csv', records), ('comparison.csv', summary)):
        with (root/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 深度實驗：3 層與 6 層的 SI 比較', '',
             '每組 20 首，每首分析開頭 48 秒；生成設定均為 temperature=1.0、top-k=32、top-p=0.95、seed=123–142。',
             f'真實組沿用原 SI 研究 seed={plan["reference_seed"]} 的 20 首，從 {plan["reference_eligible"]} 首符合長度條件的曲目抽樣。', '',
             '| 組別 | 首數 | SI short 平均 ± 標準差 | SI mid 平均 ± 標準差 | SI long 平均 ± 標準差 |',
             '|---|---:|---:|---:|---:|']
    labels = {'real': '真實參考', 'layers6': '6 層 Transformer', 'layers3': '3 層 Transformer'}
    for row in summary:
        lines.append(f'| {labels[row["setting"]]} | 20 | ' + ' | '.join(
            f'{row[m+"_mean"]:.4f} ± {row[m+"_std"]:.4f}' for m in METRICS) + ' |')
    lines += ['', '## 與真實參考的平均差距', '',
              '| 模型 | short 差距 | mid 差距 | long 差距 |', '|---|---:|---:|---:|']
    for row in summary[1:]:
        lines.append(f'| {labels[row["setting"]]} | ' + ' | '.join(
            f'{row[m+"_abs_diff_from_reference"]:.4f}' for m in METRICS) + ' |')
    lines += ['', '- short：3–8 秒；mid：8–15 秒；long：15–48 秒。',
              '- 同一 SoundFont、FluidSynth 版本、gain=0.025；所有 WAV 驗證來源及雜湊，無截波。',
              '- 沿用真實組與六層的 WAV，但三組均由同一版 SI 程式重新計算；未訓練或生成新的 MIDI。',
              '- 標準差是歌曲間的樣本標準差；只有一個訓練 seed，不能推論普遍優劣。',
              '- SI 衡量重複結構，接近參考不等於較好聽，也不代表原創性。',
              '- 真實參考只有 20 首，且只量測前 48 秒；與全曲、896 首參考的 H4/GS 表維持分開。',
              '- 設定與來源見 study_plan.json、各 groups/*/protocol.json；逐首分數見 per_piece.csv。', '']
    (root/'comparison.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--depth', type=Path, default=Path('runs/depth_3_vs_6'))
    p.add_argument('--study', type=Path, default=Path('runs/metrics/si48_g0025'))
    p.add_argument('--output', type=Path, default=Path('runs/depth_3_vs_6/si48'))
    p.add_argument('--soundfont', type=Path)
    p.add_argument('--fluidsynth', default='fluidsynth')
    p.add_argument('--si-python', default=sys.executable)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    plan = freeze_plan(a.depth, a.study, a.output)
    print(f'Verified frozen plan: {a.output}/study_plan.json', flush=True)
    if not a.execute:
        return
    if a.soundfont is None or sha256(a.soundfont) != plan['render_settings']['soundfont_sha256']:
        p.error('Provide the same SoundFont as the reference study')
    here = Path(__file__).resolve().parents[1]
    audio = a.output/'audio'
    audio.mkdir(parents=True, exist_ok=True)
    for group, old in (('real', 'real'), ('layers6', 't10')):
        target = (a.study/'audio'/old).resolve()
        link = audio/group
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target:
            raise ValueError('Unexpected existing audio directory')
    subprocess.run([sys.executable, str(here/'audio/render_audio.py'), '--input', str(a.depth/'generated'),
                    '--output', str(audio/'layers3'), '--soundfont', str(a.soundfont),
                    '--fluidsynth', a.fluidsynth, '--gain', str(plan['render_settings']['gain'])], check=True)
    def evaluate(group):
        with (a.output/(group+'.log')).open('a', buffering=1) as log:
            subprocess.run([a.si_python, str(here/'evaluation/evaluate_si.py'), '--audio-root', str(audio),
                            '--groups', group, '--limit-per-group', '20', '--duration', '48', '--scope', 'batch',
                            '--output', str(a.output/'groups'/group)], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'SI complete: {group}', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(evaluate, GROUPS))
    write_report(a.output, plan)
    print(f'Finished: {a.output}/comparison.md', flush=True)


if __name__ == '__main__':
    main()
