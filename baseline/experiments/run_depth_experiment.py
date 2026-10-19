"""Controlled depth study. New three-layer run, existing six-layer reference.

Run --stage plan first. --stage all trains from scratch, generates 20 samples at
T=1.0, evaluates H1/H4/GS and writes a separate report. No package installation.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def train_command(python,script,reference,output,resume=False):
    result=[python,str(script),'--output',str(output)]
    for key in ('data','steps','batch_size','context','width','heads','lr','eval_every',
                'eval_batches','seed','validation_fraction'):
        result.extend(['--'+key.replace('_','-'),str(reference[key])])
    result.extend(['--layers','3','--device','cuda'])
    if resume:
        result.extend(['--resume',str(output/'last.pt')])
    return result


def verify_plan(existing,candidate,inspection=False):
    """Keep historical provenance intact after source layout/version changes."""
    if existing==candidate:
        return existing
    without_sources=lambda p:{k:v for k,v in p.items() if k!='source_sha256'}
    if inspection and without_sources(existing)==without_sources(candidate):
        print('Historical source hashes differ from current code; inspection/report only. '
              'Original plan retained. Use a NEW output for training/generation/evaluation.',flush=True)
        return existing
    raise ValueError('Existing plan differs; choose a new output directory')


def create_plan(reference_run,reference_samples,root,inspection=False):
    import torch
    from baseline.model import Config,MusicTransformer
    reference=json.loads((reference_run/'train_args.json').read_text())
    config=json.loads((reference_run/'config.json').read_text())
    generation=json.loads((reference_samples/'generation_args.json').read_text())
    if config['layers']!=6 or reference['layers']!=6 or reference['steps']!=20000 or reference['validation_fraction']!=0:
        raise ValueError('Expected existing six-layer, 20000-step, full-data reference')
    for k,v in {'count':20,'bars':32,'beats':4,'temperature':1.,'top_k':32,'top_p':.95,
                'seed':123,'max_events_per_bar':128,'prompt':None}.items():
        if generation[k]!=v:
            raise ValueError(f'Reference generation differs for {k}')
    checkpoint=Path(generation['checkpoint'])
    if checkpoint.resolve()!=(reference_run/'best.pt').resolve():
        raise ValueError('Reference samples must use the reference best.pt')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['step']!=20000 or saved['config']!=config:
        raise ValueError('Reference generation checkpoint must be exactly step 20000')
    data=Path(reference['data'])
    vocab=json.loads((data/'tok2idx.json').read_text())
    if saved['vocab']!=vocab or saved['split']!=json.loads((reference_run/'split.json').read_text()):
        raise ValueError('Reference vocabulary or split mismatch')
    if len(saved['split']['train'])!=896 or saved['split']['validation']:
        raise ValueError('Reference must train on all 896 songs')
    pieces=json.loads((reference_samples/'sequences.json').read_text())
    if len(pieces)!=20 or any(p['tokens'].count('Bar')!=32 for p in pieces):
        raise ValueError('Reference samples are incomplete')
    target=Config(**{**config,'layers':3})
    torch.manual_seed(reference['seed'])
    parameters=sum(p.numel() for p in MusicTransformer(target).parameters())
    plan={'question':'Effect of depth at fixed tokenization and training-token budget',
          'reference_run':str(reference_run),'reference_samples':str(reference_samples),
          'reference_args':reference,'reference_config':config,'target_config':asdict(target),
          'target_parameters':parameters,'reference_parameters':sum(t.numel() for k,t in saved['model'].items() if k!='head.weight'),
          'reference_checkpoint_step':saved['step'],'reference_checkpoint_sha256':digest(checkpoint),
          'data_sha256':digest(data/'sequences.json'),'vocab_sha256':digest(data/'tok2idx.json'),
          'source_sha256':{name:digest(Path(__file__).resolve().parents[1]/('evaluation/evaluate.py' if name=='evaluate.py' else name)) for name in ('train.py','model.py','music.py','generate.py','evaluate.py')},
          'sampling':{k:generation[k] for k in ('count','bars','beats','temperature','top_k','top_p','seed','max_events_per_bar')},
          'target_run':str(root/'train'),'target_samples':str(root/'generated'),
          'scope':'One training seed; training loss is not held-out generalization; SI/listening follow separately'}
    path=root/'plan.json'
    if path.exists():
        existing=json.loads(path.read_text())
        result=verify_plan(existing,plan,inspection)
        if result is existing:
            return existing
    dump(path,plan)
    return plan


def metric_summary(path,expected_count=20):
    with Path(path).open(newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=expected_count or len({r['song_name'] for r in rows})!=expected_count:
        raise ValueError(f'{path}: expected exactly {expected_count} distinct samples')
    result={}
    for short,name in [('H1','H1 (Pitch Entropy 1-Bar)'),('H4','H4 (Pitch Entropy 4-Bar)'),('GS','GS (Groove Similarity)')]:
        values=[float(r[name]) for r in rows]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'{path}: missing or non-finite metric')
        result[short+'_mean']=statistics.mean(values)
        result[short+'_std']=statistics.stdev(values)
    return result


def report(root,plan,reference_csv,real_csv):
    real=metric_summary(real_csv,896)
    rows=[{'model':'6 layers','parameters':plan['reference_parameters'],**metric_summary(reference_csv)},
          {'model':'3 layers','parameters':plan['target_parameters'],**metric_summary(root/'metrics.csv')}]
    for row in rows:
        for metric in ('H4','GS'):
            row[metric+'_abs_diff_from_real']=abs(row[metric+'_mean']-real[metric+'_mean'])
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines=['# 模型深度比較：3 層 vs 6 層','',
           '相同表示法、896 首、context 1024、batch 8、20000 步、seed 42；只改層數。',
           '生成固定 temperature 1.0、top-k 32、top-p 0.95、seed 123–142，每模型 20 首 32 小節。','',
           '| 模型 | 參數量 | H4 平均 ± 標準差 | GS 平均 ± 標準差 |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["model"]} | {row["parameters"]:,} | {row["H4_mean"]:.4f} ± {row["H4_std"]:.4f} | '
                     f'{row["GS_mean"]:.4f} ± {row["GS_std"]:.4f} |')
    lines+=['',f'原始 896 首全曲參考：H4={real["H4_mean"]:.4f}，GS={real["GS_mean"]:.4f}。',
            '', '| 模型 | H4 與真實平均差距 | GS 與真實平均差距 |', '|---|---:|---:|']
    for row in rows:
        lines.append(f'| {row["model"]} | {row["H4_abs_diff_from_real"]:.4f} | {row["GS_abs_diff_from_real"]:.4f} |')
    lines+=['','## 限制與下一步','',
            '- 固定步數與資料抽樣，近似固定訓練 token 預算；運算量不相同。',
            '- 只有一個訓練 seed，不宣稱統計顯著性。',
            '- 全部資料參與訓練，training_probe 不能當作獨立驗證成績。',
            '- 六層使用已確認為第 20000 步的 best.pt；三層固定使用第 20000 步 last.pt。',
            '- H4/GS 均使用相同 64 格正規化；與真實資料的差距及試聽仍須一起判斷。',
            '- SI 尚未加入本次深度比較；確認新曲長度後，才能選一致的分析範圍。',
            '- 三層程序 wall time 記於 execution.json，含初始化、讀檔與最後存檔；',
            '  六層舊記錄缺同一定義的成本量測，不能直接拿兩種計時宣稱加速比。',
            '- baseline 的模型與既有生成結果未被覆蓋。','']
    (root/'comparison.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','train','generate','evaluate','report','all'],default='plan')
    p.add_argument('--output',type=Path,default=Path('runs/depth_3_vs_6'))
    p.add_argument('--reference-run',type=Path,default=Path('runs/baseline'))
    p.add_argument('--reference-samples',type=Path,default=Path('runs/experiments/basic_a/t10'))
    p.add_argument('--reference-metrics',type=Path,default=Path('runs/metrics/t10.csv'))
    p.add_argument('--real-metrics',type=Path,default=Path('runs/metrics/real.csv'))
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    import torch
    root=a.output
    plan=create_plan(a.reference_run,a.reference_samples,root,inspection=a.stage in ('plan','report'))
    here=Path(__file__).resolve().parents[1]
    train=root/'train'
    samples=root/'generated'
    print(f'Plan verified: layers 6 -> 3; {plan["reference_parameters"]:,} -> {plan["target_parameters"]:,} parameters',flush=True)
    if a.stage=='plan':
        return
    if a.stage in ('train','all'):
        if not torch.cuda.is_available():
            p.error('CUDA required for this full experiment; use a separate tiny smoke run for CPU testing')
        if (train/'last.pt').exists():
            saved=torch.load(train/'last.pt',map_location='cpu',weights_only=False)
            if saved['step']==20000 and saved['config']==plan['target_config']:
                print('Training already complete; retaining checkpoint',flush=True)
            elif a.resume:
                subprocess.run(train_command(sys.executable,here/'train.py',plan['reference_args'],train,True),check=True)
            else:
                p.error('Partial run exists; inspect and use --resume or a new output folder')
        else:
            if any(train.glob('*.json')) or (train/'loss.jsonl').exists():
                p.error('Partial run metadata exists without checkpoint; use a new output folder')
            started=time.perf_counter()
            subprocess.run(train_command(sys.executable,here/'train.py',plan['reference_args'],train),check=True)
            dump(root/'execution.json',{'train_process_wall_seconds':time.perf_counter()-started,
                 'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
                 'gpu_name':torch.cuda.get_device_name(0),'python_version':platform.python_version(),
                 'device_visibility':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')})
    if a.stage in ('generate','all'):
        checkpoint=train/'last.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['step']!=20000 or saved['config']!=plan['target_config']:
            p.error('Three-layer final checkpoint is not ready')
        identity={'checkpoint_sha256':digest(checkpoint),'sampling':plan['sampling'],
                  'generator_sha256':plan['source_sha256']['generate.py']}
        marker=root/'generation_identity.json'
        if marker.exists() and json.loads(marker.read_text())!=identity:
            p.error('Generation identity mismatch; use a new output folder')
        if samples.exists() and not marker.exists():
            p.error('Unverified output folder exists; choose a new output folder')
        dump(marker,identity)
        done=samples/'sequences.json'
        if done.exists():
            pieces=json.loads(done.read_text())
            if len(pieces)!=20 or any(piece['tokens'].count('Bar')!=32 or not (samples/(piece['song_name']+'.mid')).is_file() for piece in pieces):
                p.error('Existing generated set incomplete or malformed')
            print('Generation already complete; retaining 20 samples',flush=True)
        else:
            command=[sys.executable,str(here/'generate.py'),'--checkpoint',str(checkpoint),'--output',str(samples)]
            for key,value in plan['sampling'].items():
                command.extend(['--'+key.replace('_','-'),str(value)])
            subprocess.run(command,check=True)
    if a.stage in ('evaluate','all'):
        subprocess.run([sys.executable,str(here/'evaluation/evaluate.py'),'--data',str(samples),
                        '--output',str(root/'metrics.csv')],check=True)
    if a.stage in ('report','all'):
        report(root,plan,a.reference_metrics,a.real_metrics)
        print(f'Finished: {root}/comparison.md',flush=True)


if __name__=='__main__':
    main()
