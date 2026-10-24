"""Run the unmodified TA Bonus evaluator and require complete finite output."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts',type=Path,default=Path('POP909/prompts/bonus'))
    p.add_argument('--generated',type=Path,default=Path('runs/bonus_accompaniment/generated'))
    p.add_argument('--output',type=Path,default=Path('runs/bonus_accompaniment/metrics.csv'))
    p.add_argument('--expected-songs',type=int,default=10)
    a=p.parse_args()
    names={f'{i:03d}' for i in range(1,a.expected_songs+1)}
    if {p.stem for p in a.generated.glob('*.mid')}!=names:p.error('Generated set does not match all expected prompt IDs')
    audit=json.loads((a.generated/'validation.json').read_text())
    if len(audit)!=a.expected_songs or {r['song_name'] for r in audit}!=names:p.error('Missing generation audits')
    for row in audit:
        if not row['melody_events_preserved'] or row['bounded_bars']!=16 or sha256(a.generated/(row['song_name']+'.mid'))!=row['midi_sha256']:
            p.error('Unverified melody/track alignment')
        if any(sha256(a.prompts/row['song_name']/name)!=value for name,value in row['condition_files'].items()):
            p.error('Source melody/beat conditions changed after generation')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():p.error('Existing metric CSV; choose a new output')
    script=Path(__file__).resolve().parents[2]/'HW2_scripts/eval_bonus.py'
    subprocess.run([sys.executable,str(script),'--prompt_dir',str(a.prompts),
        '--gen_midi_dir',str(a.generated),'--output_csv',str(a.output)],check=True)
    with a.output.open() as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=a.expected_songs or {r['song_name'].zfill(3) for r in rows}!=names or any(int(r['n_bars'])!=16 for r in rows):
        p.error('TA evaluator skipped songs or used unexpected bar counts; do not report a partial mean')
    metrics=[k for k in rows[0] if k not in ('song_name','n_bars')]
    if any(not math.isfinite(float(r[m])) for r in rows for m in metrics):p.error('Nonfinite metric')
    total=sum(int(r['n_bars']) for r in rows)
    summary={m:{'pooled_mean':sum(float(r[m])*int(r['n_bars']) for r in rows)/total,
                'song_sample_std':statistics.stdev(float(r[m]) for r in rows) if len(rows)>1 else None} for m in metrics}
    save_json(a.output.with_suffix('.summary.json'),{'songs':len(rows),'bars':total,'metrics':summary,
         'ta_evaluator_sha256':sha256(script),'note':'CA-ceiling is the same chord estimator on GT piano, not a strict theoretical upper bound'})
    lines=['# Bonus: melody-conditioned accompaniment','',f'{len(rows)} songs x16 bars completed; original melody MIDI events preserved.','',
        '| Metric | Bar-weighted mean | Between-song sample standard deviation |','|---|---:|---:|']
    for m,v in summary.items():lines.append(f'| {m} | {v["pooled_mean"]:.4f} | '+(f'{v["song_sample_std"]:.4f}' if v['song_sample_std'] is not None else '—')+' |')
    lines+=['','- Inference reads only melody.mid/beat_midi.txt; GT piano/chord files are evaluator-only.',
        '- CA uses24 major/minor triads; unsupported/no-chord GT labels limit attainable scores.',
        '- CA-ceiling is a GT-estimator reference, not a guaranteed upper bound on generated CA.',
        '- OA compares per-bar pitch-class/velocity/duration/inter-onset distributions, not listening quality.',
        '- One training seed; training probe is not held-out validation. Listening and WAV checks are separate.','']
    a.output.with_suffix('.md').write_text('\n'.join(lines));print(f'Complete Bonus evaluation: {a.output.with_suffix(".md")}',flush=True)


if __name__=='__main__':main()
