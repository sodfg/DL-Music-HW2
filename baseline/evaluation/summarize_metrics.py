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
        return f'{value:.4f}' if value is not None else '未提供'
    labels = {'real': '真實 POP909', 't08': 'Temperature 0.8',
              't10': 'Temperature 1.0', 't12': 'Temperature 1.2'}
    def label(row):
        return labels.get(row['setting'], row['setting']).replace('|', r'\|')
    lines = [
        '# 音樂生成評估總表', '',
        '本頁由逐首評估 CSV 自動彙整。H4、GS 是必做指標；SI 尚未計算。', '',
        '## 先看平均分數', '',
        'H4：四小節內音高類別分布的熵。GS：小節間音符起始節奏的相似程度。',
        '分數不是越高越好；依作業要求，分別比較與真實資料有多接近。', '',
        '| 資料／設定 | 歌曲數 | H4 平均 | GS 平均 |',
        '|---|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f'| {label(row)} | {row["song_count"]} | '
                     f'{number(row["H4_mean"])} | {number(row["GS_mean"])} |')
    lines += ['', '## 與真實資料的距離', '',
              f'參考組：`{reference}`。下表是平均值的絕對差，越小表示該指標越接近參考組。', '',
              '| 生成設定 | H4 差距 | GS 差距 |', '|---|---:|---:|']
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
            lines += ['', f'- **{metric} 最接近參考組：{winners}**（差距 {best:.4f}）。']
    lines += ['', '這只是各指標平均值的比較，不是統計顯著性檢定，也不代表整體音樂品質排名。',
              '', '## 歌曲之間的差異（標準差）', '',
              '以下為樣本標準差（n−1）。越大表示同組歌曲在該指標上的差異越大；不是越小越好，也不是信賴區間。', '',
              '| 資料／設定 | H4 標準差 | GS 標準差 |', '|---|---:|---:|']
    for row in rows:
        lines.append(f'| {label(row)} | {number(row["H4_std"])} | {number(row["GS_std"])} |')
    lines += ['', '## 補充：H1 與資料完整性', '',
              '| 資料／設定 | H1 平均 | H1 標準差 | 有效筆數 H1 / H4 / GS |',
              '|---|---:|---:|---|']
    for row in rows:
        counts = ' / '.join(str(row[m+'_valid_count']) for m in METRICS)
        lines.append(f'| {label(row)} | {number(row["H1_mean"])} | {number(row["H1_std"])} | {counts} |')
    for row in rows:
        for metric in METRICS:
            missing = row[metric+'_missing_count']
            if missing:
                lines += ['', f'- 注意：{label(row)} 的 {metric} 排除了 {missing} 筆缺失或非有限數值。']
    lines += ['', '## 比較限制', '',
              '- GS 沿用本專案共同的 64 格小節相對位置正規化。',
              '- 真實資料是不同長度與拍點數的歌曲；生成資料固定 32 小節、每小節四拍。',
              '- 指標不能證明旋律原創性，也不能取代試聽。',
              '- CSV 保留完整精度；本頁顯示到小數點後四位。', '']
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
