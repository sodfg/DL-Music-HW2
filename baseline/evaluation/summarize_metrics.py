"""Summarize per-song evaluation CSVs without changing the original results.

Uses sample standard deviation (n-1). Non-finite or missing values are excluded
per metric and reported explicitly, rather than silently treated as zero.
"""
import argparse
import csv
import math
from pathlib import Path
import statistics

METRICS = {
    'H1': 'H1 (Pitch Entropy 1-Bar)',
    'H4': 'H4 (Pitch Entropy 4-Bar)',
    'GS': 'GS (Groove Similarity)',
}


def summarize_file(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'song_name', *METRICS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns: {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: no song rows')
    names = [r['song_name'].strip() for r in rows]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError(f'{path}: empty or duplicate song names')
    summary = {'setting': Path(path).stem, 'song_count': len(rows)}
    for short, column in METRICS.items():
        values = []
        for row in rows:
            raw = (row[column] or '').strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(f'{path}, {row["song_name"]}, {column}: invalid number {raw!r}') from error
            if math.isfinite(value):
                values.append(value)
        summary[short+'_valid_count'] = len(values)
        summary[short+'_missing_count'] = len(rows)-len(values)
        summary[short+'_mean'] = statistics.mean(values) if values else None
        summary[short+'_std'] = statistics.stdev(values) if len(values)>1 else None
    return summary


def build_summary(paths, reference='real'):
    summaries = [summarize_file(path) for path in paths]
    names = [r['setting'] for r in summaries]
    if len(set(names)) != len(names):
        raise ValueError('Input files must have distinct stems')
    try:
        real = next(r for r in summaries if r['setting'] == reference)
    except StopIteration as error:
        raise ValueError(f'Reference {reference!r} must be included in input files') from error
    for row in summaries:
        for metric in METRICS:
            value, baseline = row[metric+'_mean'], real[metric+'_mean']
            row[metric+'_abs_diff_from_reference'] = (
                abs(value-baseline) if value is not None and baseline is not None else None
            )
    return summaries


def readable_report(rows, reference='real'):
    def number(value):
        return f'{value:.4f}' if value is not None else 'Unavailable'
    labels = {'real': 'Real POP909', 't08': 'Temperature 0.8',
              't10': 'Temperature 1.0', 't12': 'Temperature 1.2'}
    def label(row):
        return labels.get(row['setting'], row['setting']).replace('|', r'\|')
    lines = [
        '# Music generation metric summary', '',
        'Aggregated from per-piece CSVs. H4 and GS are required; SI is reported separately.', '',
        '## Mean scores', '',
        'H4: pitch-class entropy over four bars. GS: similarity of bar-level onset patterns.',
        'Compare each metric against the real reference; higher is not universally better.', '',
        '| Data/setting | Songs | H4 mean | GS mean |',
        '|---|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f'| {label(row)} | {row["song_count"]} | '
                     f'{number(row["H4_mean"])} | {number(row["GS_mean"])} |')
    lines += ['', '## Distance from the real reference', '',
              f'Reference: `{reference}`. Smaller absolute mean differences indicate closer metric values.', '',
              '| Generation setting | H4 difference | GS difference |', '|---|---:|---:|']
    for row in rows:
        if row['setting'] != reference:
            lines.append(f'| {label(row)} | {number(row["H4_abs_diff_from_reference"])} | '
                         f'{number(row["GS_abs_diff_from_reference"])} |')
    candidates = [r for r in rows if r['setting'] != reference]
    for metric in ('H4', 'GS'):
        key = metric+'_abs_diff_from_reference'
        valid = [r for r in candidates if r[key] is not None]
        if valid:
            best = min(r[key] for r in valid)
            winners = '、'.join(label(r) for r in valid if abs(r[key]-best) < 1e-12)
            lines += ['', f'- **{metric} closest to the reference: {winners}** (difference {best:.4f}).']
    lines += ['', 'These are mean comparisons, not significance tests or overall music-quality rankings.',
              '', '## Between-song variation (standard deviation)', '',
              'Sample standard deviations (n-1) describe between-song variation, not confidence intervals or a lower-is-better objective.', '',
              '| Data/setting | H4 standard deviation | GS standard deviation |', '|---|---:|---:|']
    for row in rows:
        lines.append(f'| {label(row)} | {number(row["H4_std"])} | {number(row["GS_std"])} |')
    lines += ['', '## H1 and data completeness', '',
              '| Data/setting | H1 mean | H1 standard deviation | Valid counts H1 / H4 / GS |',
              '|---|---:|---:|---|']
    for row in rows:
        counts = ' / '.join(str(row[m+'_valid_count']) for m in METRICS)
        lines.append(f'| {label(row)} | {number(row["H1_mean"])} | {number(row["H1_std"])} | {counts} |')
    for row in rows:
        for metric in METRICS:
            missing = row[metric+'_missing_count']
            if missing:
                lines += ['', f'- Warning: {label(row)} excluded {missing} missing/nonfinite values for {metric}.']
    lines += ['', '## Comparison limits', '',
              '- GS uses the shared 64-position within-bar normalization.',
              '- Real songs vary in length and beat count; generated pieces have32 four-beat bars.',
              '- Metrics cannot establish originality or replace listening.',
              '- CSV retains full precision; displayed values use four decimal places.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('runs/metrics'))
    parser.add_argument('--files', nargs='+', default=['real.csv','t08.csv','t10.csv','t12.csv'])
    parser.add_argument('--reference', default='real', help='Reference CSV filename without .csv')
    parser.add_argument('--output', type=Path, help='Default: INPUT_DIR/summary.csv')
    args = parser.parse_args()
    paths = [args.input_dir/name for name in args.files]
    output = args.output or args.input_dir/'summary.csv'
    report = output.with_suffix('.md')
    if any(p.resolve() in [source.resolve() for source in paths] for p in (output, report)):
        parser.error('Output must not overwrite any input CSV')
    try:
        rows = build_summary(paths,args.reference)
    except (OSError,ValueError) as error:
        parser.exit(1,f'Error: {error}\n')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def number(value):
        return f'{value:.4f}' if value is not None else 'N/A'
    print('| Setting | Songs | H1 mean ± std | H4 mean ± std | GS mean ± std | ΔH4 | ΔGS |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for row in rows:
        parts = [f'{number(row[m+"_mean"])} ± {number(row[m+"_std"])}' for m in METRICS]
        print(f'| {row["setting"]} | {row["song_count"]} | '+ ' | '.join(parts)
              +f' | {number(row["H4_abs_diff_from_reference"])}'
              +f' | {number(row["GS_abs_diff_from_reference"])} |')
        for metric in METRICS:
            if row[metric+'_missing_count']:
                print(f'Warning: {row["setting"]} {metric}: '
                      f'{row[metric+"_missing_count"]} missing/non-finite values excluded.')
    report.write_text(readable_report(rows,args.reference),encoding='utf-8')
    print(f'\nSaved: {output}')
    print(f'Readable report: {report}')
    print('Std = variation across songs (sample std), not a confidence interval.')
    print(f'Δ = absolute difference of means from {args.reference}; no combined quality score.')


if __name__ == '__main__':
    main()
