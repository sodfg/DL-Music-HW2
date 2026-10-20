"""Build a local submission draft. Never uploads or runs Git.

Only original prompt files, our code/models/results and licensed SoundFont are
included. A missing Task-A selection stays explicitly pending; no silent choice.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import verified_audio_record,save_json


def build(output,soundfont,task_a=None):
    import torch
    if output.exists():raise ValueError('Destination exists; choose a new output directory')
    if not soundfont.is_file():raise ValueError('SoundFont file missing')
    models={'baseline_6':Path('runs/baseline/best.pt'),
            'depth_3':Path('runs/depth_3_vs_6/train/last.pt'),
            'midi_like_6':Path('runs/representation_midi_like/train/last.pt')}
    # Preflight all required results before writing the draft.
    for path in models.values():
        if not path.is_file():raise ValueError(f'Model missing: {path}')
    pairs=[]
    for sid in ('001','002','003'):
        for setting in ('t08','t10','t12'):
            midi=Path('runs/experiments/basic_b')/sid/setting/(sid+'_000.mid')
            wav=Path('runs/listening/basic_b')/sid/setting/(sid+'_000.wav')
            record=verified_audio_record(wav,'runs/listening/basic_b')
            if record['source_sha256']!=sha256(midi) or record['audio']['clipped_samples']:
                raise ValueError('Task B audio unverified')
            pairs.append((Path('task_b')/sid/setting,midi,wav))
    if task_a:
        for i in range(20):
            name=f'sample_{i:03d}'
            midi=Path('runs/experiments/basic_a')/task_a/(name+'.mid')
            wav=Path('runs/metrics/si48_g0025/audio')/task_a/(name+'.wav')
            record=verified_audio_record(wav,'runs/metrics/si48_g0025/audio')
            if record['source_sha256']!=sha256(midi) or record['audio']['clipped_samples']:
                raise ValueError('Task A audio unverified')
            pairs.append((Path('task_a'),midi,wav))
    output.mkdir(parents=True)
    shutil.copytree('baseline',output/'code/baseline',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile('baseline/requirements.txt',output/'requirements.txt')
    exports={}
    for name,source in models.items():
        saved=torch.load(source,map_location='cpu',weights_only=False)
        if saved['step']!=20000:raise ValueError('Expected completed step-20000 model')
        target=output/'models'/(name+'.pt');target.parent.mkdir(exist_ok=True)
        minimal={k:saved[k] for k in ('config','vocab','model')}
        torch.save(minimal,target)
        restored=torch.load(target,map_location='cpu',weights_only=False)
        assert restored['config']==saved['config'] and restored['vocab']==saved['vocab']
        assert all(torch.equal(v,restored['model'][k]) for k,v in saved['model'].items())
        exports[name]={'source_sha256':sha256(source),'export_sha256':sha256(target),
                       'source_step':saved['step'],'contents':['config','vocab','model'],
                       'tensor_equality_verified':True}
    for dest,midi,wav in pairs:
        folder=output/dest;folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(midi,folder/midi.name);shutil.copyfile(wav,folder/wav.name)
    for sid in ('001','002','003'):
        folder=output/'prompts'/sid;folder.mkdir(parents=True)
        for name in (sid+'.mid','beat_midi.txt'):
            shutil.copyfile(Path('POP909/prompts/basic-b')/sid/name,folder/name)
    audio=output/'audio';audio.mkdir()
    shutil.copyfile(soundfont,audio/'GeneralUser-GS.sf2')
    shutil.copyfile(soundfont.parent/'GeneralUser-GS-LICENSE.txt',audio/'GeneralUser-GS-LICENSE.txt')
    tables=output/'experiment_tables';tables.mkdir()
    for name,path in {'task_a':'runs/metrics/summary.csv','sampling_si':'runs/metrics/si48_g0025/summary.csv',
                      'depth':'runs/depth_3_vs_6/comparison.csv','depth_si':'runs/depth_3_vs_6/si48/comparison.csv',
                      'representation':'runs/representation_midi_like/comparison.csv',
                      'representation_si':'runs/representation_midi_like/si48/comparison.csv'}.items():
        shutil.copyfile(path,tables/(name+'.csv'))
    save_json(output/'selections.json',{'task_a':task_a,'task_b':{s:None for s in ('001','002','003')},
        'cloud_link':None,'status':'DRAFT: final listening choices, report and cloud sharing pending'})
    save_json(output/'model_provenance.json',exports)
    (output/'task_a').mkdir(exist_ok=True)
    if not task_a:(output/'task_a/PENDING.md').write_text('尚未選定 Task A 設定。請試聽後選 t08/t10/t12，再打包20 MIDI與20 WAV。\n')
    (output/'README.md').write_text('''# DL in Music HW2 submission draft

這是本機繳交草稿，尚未上傳。選曲、正式報告與分享權限確認後才能當成定稿。
Task B 已包含九組 MIDI/WAV；每個 prompt 的最佳版本仍待填 selections.json。
Task A 若未選設定，task_a/PENDING.md 會保留待補，不會代替使用者選擇。

## Environment

Python 3.10+，PyTorch 2.5–2.6。原訓練環境：Python3.10.12、torch2.6.0+cu124、RTX A5000。
在這個資料夾開終端機，依自己的 CUDA 環境安裝 PyTorch，再安裝 requirements.txt。
既有共用環境若符合版本，可直接使用。音訊合成另外需要 FluidSynth 可執行檔。
包含的 GeneralUser GS SoundFont 授權見 audio/GeneralUser-GS-LICENSE.txt。

```bash
python3 -m pip install -r requirements.txt
```

## Basic Task A: generate new MIDI and WAV

```bash
python3 code/baseline/generate.py --checkpoint models/baseline_6.pt \\
  --output generated/new_a --count 20 --bars 32 \\
  --temperature 1.0 --top-k 32 --top-p 0.95 --seed 123
python3 code/baseline/audio/render_audio.py --input generated/new_a \\
  --output generated/new_a_wav --soundfont audio/GeneralUser-GS.sf2 \\
  --fluidsynth fluidsynth --gain 0.025
```

上例使用 t10，僅為可重現推論範例。正式提交設定以 selections.json 為準。
--output 可換成任何可寫入的新目錄。程式會依 CUDA 可用性選 GPU 或 CPU。

## Basic Task B: preserve prompt and generate continuation

```bash
python3 code/baseline/generate.py --checkpoint models/baseline_6.pt \\
  --prompt prompts/001/001.mid --prompt-beats prompts/001/beat_midi.txt \\
  --prompt-bars 8 --bars 32 --count 1 --temperature 1.0 --top-k 32 --top-p 0.95 \\
  --output generated/new_b_001
python3 code/baseline/audio/render_audio.py --input generated/new_b_001 \\
  --output generated/new_b_001_wav --soundfont audio/GeneralUser-GS.sf2 \\
  --fluidsynth fluidsynth --gain 0.025
```

002/003 使用對應 prompts 子資料夾。t08/t10/t12 分別設定 temperature=0.8/1.0/1.2。
提供的是原始提示檔，沒有訓練集或前處理資料。

## Optional models

models/depth_3.pt 可使用同一個 code/baseline/generate.py 推論。
models/midi_like_6.pt 使用下列獨立入口：

```bash
python3 code/baseline/representations/generate_midi_like.py \\
  --checkpoint models/midi_like_6.pt --output generated/new_midi_like \\
  --count 20 --bars 32 --temperature 1.0 --top-k 32 --top-p 0.95
```

模型匯出只保存推論需要的 model/config/vocab，移除了 optimizer 與隨機狀態。
所有權重張量已與原始 checkpoint 逐項比對一致。來源與雜湊見 model_provenance.json。
experiment_tables 提供已完成比較的摘要，重訓程式僅供原專案資料環境使用。
未實作 Bonus 伴奏任務，不宣稱已完成。
''')
    files={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()}
    save_json(output/'file_manifest.json',files)
    print(f'Local draft built: {output}; Task B=9 MIDI/9 WAV; Task A={task_a or "PENDING"}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/submission_draft'))
    p.add_argument('--soundfont',type=Path,required=True)
    p.add_argument('--task-a',choices=['t08','t10','t12'])
    a=p.parse_args();build(a.output,a.soundfont,a.task_a)
