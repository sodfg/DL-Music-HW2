"""Prepare a self-contained cloud-upload folder, never upload or run Git.

Stages: verified MIDI collection, louder submission-only WAVs, then package.
Original evaluation WAVs, generation records and model checkpoints stay intact.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import verified_audio_record
from baseline.music import dump


def collect():
    corrected=Path('runs/corrected_boundary_v2')
    selections=json.loads(Path('runs/submission_state/selections.json').read_text())
    if selections['task_a']!='t12':raise ValueError('Expected user-selected Task A t12')
    items=[]
    for i in range(20):
        name=f'sample_{i:03d}.mid'
        items.append(('task_a_t12/'+name,corrected/'basic_a/t12'/name))
    audits=json.loads(Path('runs/task_b_aligned/alignment_audit.json').read_text())
    for sid in ('001','002','003'):
        for setting in ('t08','t10','t12'):
            name=f'{sid}_000.mid';p=Path('runs/task_b_aligned')/sid/setting/name
            matching=[r for r in audits if r['output_sha256']==sha256(p)]
            if len(matching)!=1:raise ValueError('Aligned MIDI audit missing')
            audit=matching[0]
            if (audit['input_sha256']!=sha256(corrected/'basic_b'/sid/setting/name)
                    or audit['new_display_bars']!=32 or audit['max_event_time_error_seconds']>=.00005):
                raise ValueError('Task B alignment provenance differs')
            items.append((f'task_b/{sid}/{setting}/{name}',p))
    for i in range(1,11):
        name=f'{i:03d}.mid';p=Path('runs/bonus_aligned_v2')/name
        audit=next(r for r in json.loads(Path('runs/bonus_aligned_v2/alignment_audit.json').read_text()) if r['song_name']==f'{i:03d}')
        if audit['output_sha256']!=sha256(p) or audit['input_sha256']!=sha256(Path('runs/bonus_accompaniment/generated')/name) or audit['bounded_bars']!=16:
            raise ValueError('Bonus alignment provenance differs')
        items.append(('bonus/'+name,p))
    if len(items)!=39:raise ValueError('Expected 39 MIDI files')
    return selections,[{'relative_path':rel,'source_path':str(p),'source_sha256':sha256(p)} for rel,p in items]


def prepare_audio(a):
    selections,items=collect()
    plan={'task_a':selections['task_a'],'audio_gain':a.gain,'soundfont_sha256':sha256(a.soundfont),
          'instrument':'piano','sample_rate':44100,'items':items,
          'purpose':'Submission listening only. Evaluation WAVs remain at gain 0.025.'}
    path=a.work/'audio_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Audio plan differs; choose another --work folder')
    dump(path,plan)
    for item in items:
        target=a.work/'midi'/item['relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=item['source_sha256']:raise ValueError('Staged input changed')
        if not target.exists():shutil.copyfile(item['source_path'],target)
    subprocess.run([sys.executable,'baseline/audio/render_audio.py','--input',str(a.work/'midi'),
        '--output',str(a.work/'wav'),'--recursive','--soundfont',str(a.soundfont),
        '--fluidsynth',a.fluidsynth,'--gain',str(a.gain)],check=True)
    records=[]
    for item in items:
        wav=a.work/'wav'/Path(item['relative_path']).with_suffix('.wav')
        record=verified_audio_record(wav,a.work/'wav')
        if (record['source_sha256']!=item['source_sha256'] or record['audio']['clipped_samples']
                or record['settings']['gain']!=a.gain
                or record['settings']['soundfont_sha256']!=plan['soundfont_sha256']):
            raise ValueError('Invalid source or PCM clipping; lower the common gain in a new work directory')
        records.append({'relative_path':str(wav.relative_to(a.work/'wav')),
                        'midi_sha256':item['source_sha256'],'wav_sha256':record['wav_sha256'],
                        'audio':record['audio']})
    dump(a.work/'audio_completed.json',{'count':len(records),'gain':a.gain,
        'audio_plan_sha256':sha256(path),'maximum_peak':max(r['audio']['peak_amplitude'] for r in records),
        'files':records})
    print(f'Verified {len(records)} louder WAVs with zero clipping.',flush=True)


def package(a):
    import torch
    selections,items=collect();plan=json.loads((a.work/'audio_plan.json').read_text())
    done=json.loads((a.work/'audio_completed.json').read_text())
    if done['count']!=39 or done['audio_plan_sha256']!=sha256(a.work/'audio_plan.json') or plan['items']!=items:
        raise ValueError('Audio sources differ from completed preparation')
    if sha256(a.soundfont)!=plan['soundfont_sha256']:raise ValueError('SoundFont changed')
    if a.output.exists():raise ValueError('Upload destination exists; use a new destination')
    for suffix in ('.pdf','.pptx'):
        if not a.report.with_suffix(suffix).is_file():raise ValueError('Updated report is missing')
    models={'baseline_6':'runs/baseline/best.pt','depth_3':'runs/depth_3_vs_6/train/last.pt',
            'midi_like_6':'runs/representation_midi_like/train/last.pt',
            'bonus_accompaniment':'runs/bonus_accompaniment/train/last.pt'}
    for source in models.values():
        if not Path(source).is_file():raise ValueError('Inference model missing')
    a.output.mkdir(parents=True)
    for item in items:
        rel=Path(item['relative_path']);mid=Path(item['source_path']);wav=a.work/'wav'/rel.with_suffix('.wav')
        record=verified_audio_record(wav,a.work/'wav')
        if (record['source_sha256']!=sha256(mid) or record['audio']['clipped_samples']
                or record['settings']['gain']!=plan['audio_gain']
                or record['settings']['soundfont_sha256']!=plan['soundfont_sha256']):
            raise ValueError('Audio/source/settings mismatch')
        folder=a.output/'generation_results'/rel.parent;folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(mid,folder/rel.name);shutil.copyfile(wav,folder/wav.name)
    shutil.copytree('baseline',a.output/'code/baseline',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    supplied=a.output/'code/HW2_scripts';supplied.mkdir()
    for name in ('eval_basic.py','eval_bonus.py','tokenization.py','README.md'):
        p=Path('HW2_scripts')/name
        if p.is_file():shutil.copyfile(p,supplied/name)
    shutil.copyfile('baseline/requirements.txt',a.output/'requirements.txt')
    provenance={};folder=a.output/'models';folder.mkdir()
    for name,source in models.items():
        saved=torch.load(source,map_location='cpu',weights_only=False)
        if saved['step']!=20000:raise ValueError('Expected step20000 checkpoint')
        minimal={k:saved[k] for k in ('config','vocab','model')}
        if name=='bonus_accompaniment':minimal.update(kind=saved['kind'],step=saved['step'])
        target=folder/(name+'.pt');torch.save(minimal,target)
        reread=torch.load(target,map_location='cpu',weights_only=False)
        if not all(torch.equal(v,reread['model'][k]) for k,v in saved['model'].items()):raise ValueError('Model export changed weights')
        provenance[name]={'source_step':saved['step'],'source_sha256':sha256(source),
                          'export_sha256':sha256(target),'contents':list(minimal),'tensor_equality_verified':True}
    assets=a.output/'audio_assets';assets.mkdir()
    shutil.copyfile(a.soundfont,assets/'GeneralUser-GS.sf2')
    shutil.copyfile(a.soundfont.parent/'GeneralUser-GS-LICENSE.txt',assets/'GeneralUser-GS-LICENSE.txt')
    report=a.output/'report';report.mkdir()
    for suffix in ('.pdf','.pptx'):shutil.copyfile(a.report.with_suffix(suffix),report/('B12901157_report'+suffix))
    metrics=a.output/'metrics';metrics.mkdir()
    for name in ('summary.csv','summary.md'):shutil.copyfile(Path('runs/corrected_boundary_v2')/name,metrics/('basic_and_optional_'+name))
    for name in ('metrics.csv','metrics.md','metrics.summary.json'):shutil.copyfile(Path('runs/bonus_aligned_v2')/name,metrics/('bonus_'+name))
    checks=a.output/'verification';checks.mkdir()
    dump(checks/'audio_check.json',{'gain':done['gain'],'maximum_peak':done['maximum_peak'],'count':39,
         'soundfont_sha256':plan['soundfont_sha256'],'sample_rate':44100,'clipped_samples':0,
         'evaluation_gain':.025,'files':done['files']})
    public_audits=[{k:v for k,v in r.items() if k not in ('input','prompt_sha256','beat_sha256')} for r in json.loads(Path('runs/task_b_aligned/alignment_audit.json').read_text())]
    dump(checks/'task_b_alignment.json',public_audits)
    shutil.copyfile('runs/bonus_aligned_v2/alignment_audit.json',checks/'bonus_alignment.json')
    dump(checks/'model_provenance.json',provenance)
    selections.update(task_b_export='annotated-grid-9600ppq',submission_wav_gain=done['gain'],
                      status='DRAFT: Task B best-version labels, final listening and cloud link pending. All required audio files included.')
    dump(a.output/'selections.json',selections)
    (a.output/'BEST_VERSIONS.md').write_text('''# 最佳版本註記

Task A已選t12，完整20首全部提交。Bonus十首全部提交。
Task B九份MIDI與WAV全部提交，每個提示另標出最佳設定。

| 提示 | 最佳設定 |
|---|---|
| 001 | 待選：t08／t10／t12 |
| 002 | 待選：t08／t10／t12 |
| 003 | 待選：t08／t10／t12 |

同步更新selections.json與報告Task B頁，並在報告首頁與COOL留言填雲端連結。
''')
    (a.output/'README.md').write_text('''# HW2 雲端上傳整理包

目前為待補註記的草稿。音樂檔案已齊全，Task B每個提示的最佳版本與雲端連結仍待填入selections.json及報告。

## 內容

| 目錄 | 內容 |
|---|---|
| report | 報告PDF及可編輯PPTX。PDF另交NTU COOL |
| generation_results/task_a_t12 | 使用者選定T=1.2，20 MIDI＋20 WAV |
| generation_results/task_b | 001／002／003，各含t08、t10、t12，共9 MIDI＋9 WAV，全部上傳 |
| generation_results/bonus | 十個16小節MELODY＋PIANO作品，共10 MIDI＋10 WAV |
| models | 四個由本作業自行訓練的推論模型，無optimizer |
| code | 本作業程式與課程評估入口 |
| metrics | 新版Basic／Optional總表及Bonus分數 |
| audio_assets | GeneralUser GS SoundFont與授權，供重現合成音色 |
| verification | 檔案、音訊、MIDI刻度及模型權重核對記錄 |

共39 MIDI＋39 WAV。WAV統一提高音量，PCM16 stereo／44.1kHz，所有檔案無截波。
SI仍使用原gain0.025的評估音訊，提交WAV不覆寫或取代SI輸入。
Task B已對齊助教拍點，32個完整4/4小節，前8小節原演奏事件時間誤差小於0.05毫秒。
Bonus為16個標註小節，拍數隨標註變動，不強制為4/4。
最佳版本可先填入BEST_VERSIONS.md，再同步selections.json與報告。

不包含訓練資料、tokenized/preprocessed資料、快取、虛擬環境或其他人的神經網路模型。
課程提示也不包含於本包，請從課程提供資料另行取得。模型內的vocab是推論必要設定。

## 安裝與推論

Python3.10+。先依CUDA環境安裝適用PyTorch，再安裝requirements.txt。原訓練為torch2.6.0+cu124／RTX A5000。
已有符合版本的共用環境可直接使用。不需重新訓練。下列命令從本上傳包根目錄執行。
FluidSynth需另外安裝，或以--fluidsynth指定可執行檔位置。

```bash
python3 -m pip install -r requirements.txt
python3 code/baseline/generate.py --checkpoint models/baseline_6.pt --output generated/task_a --count 20 --bars 32 --temperature 1.2 --top-k 32 --top-p .95 --seed 123 --generation-policy boundary-v2
python3 code/baseline/audio/render_audio.py --input generated/task_a --output generated/task_a_wav --soundfont audio_assets/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain GAIN_VALUE
```

Task B（COURSE_BASIC_B替換為課程basic-b提示資料夾）：

```bash
python3 code/baseline/generate.py --checkpoint models/baseline_6.pt --prompt COURSE_BASIC_B/001/001.mid --prompt-beats COURSE_BASIC_B/001/beat_midi.txt --prompt-bars 8 --bars 32 --count 1 --temperature 1.2 --top-k 32 --top-p .95 --seed 123 --generation-policy boundary-v2 --output generated/basic_b/001/t12
python3 code/baseline/experiments/align_task_b.py --input-root generated/basic_b --prompts COURSE_BASIC_B --output generated/basic_b_aligned --prompt-id 001 --setting t12
python3 code/baseline/audio/render_audio.py --input generated/basic_b_aligned/001/t12 --output generated/basic_b_wav/001/t12 --soundfont audio_assets/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain GAIN_VALUE
```

002／003與其他temperature使用對應ID與設定。每個提示請標出最喜歡的一版，九組檔案全部保留。

Bonus（COURSE_BONUS替換為課程bonus提示資料夾，生成僅讀melody.mid與beat_midi.txt）：

```bash
python3 code/baseline/bonus/generate.py --checkpoint models/bonus_accompaniment.pt --prompts COURSE_BONUS --output generated/bonus
python3 code/baseline/experiments/align_bonus.py --input generated/bonus --prompts COURSE_BONUS --output generated/bonus_aligned
python3 code/baseline/audio/render_audio.py --input generated/bonus_aligned --output generated/bonus_wav --soundfont audio_assets/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain GAIN_VALUE
```

三層模型使用同一Basic入口，checkpoint換成models/depth_3.pt。
MIDI-like使用code/baseline/representations/generate_midi_like.py與models/midi_like_6.pt。
重新評估需要另取得課程資料與MusDr，參考code/baseline/README.md。
Bonus正式MIDI可使用助教入口重新評分：

```bash
python3 code/HW2_scripts/eval_bonus.py --prompt_dir COURSE_BONUS --gen_midi_dir generation_results/bonus --output_csv new_bonus_metrics.csv
```

本包在現有本機環境驗證推論，沒有宣稱已完成全新環境安裝測試。

## 上傳與繳交

1. 完成Task B最佳版本註記及最終聽感。
2. 整個資料夾上傳雲端，確認助教可以下載。
3. 將分享連結填入報告首頁與NTU COOL作業留言，並更新report內的PDF。
4. PDF交NTU COOL，程式、模型、音樂放雲端。
'''.replace('GAIN_VALUE',str(done['gain'])))
    update_manifest(a.output)
    print(f'Prepared cloud folder: {a.output}; 39 MIDI + 39 WAV, 4 models.',flush=True)


def update_manifest(root):
    files={p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*'))
           if p.is_file() and p.relative_to(root).as_posix()!='verification/file_manifest.json'}
    dump(root/'verification/file_manifest.json',files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['audio','package'],required=True)
    p.add_argument('--work',type=Path,default=Path('runs/upload_preparation_v2'))
    p.add_argument('--output',type=Path,default=Path('output/UPLOAD_HW2_B12901157'))
    p.add_argument('--report',type=Path,default=Path('output/report/B12901157_report_draft_v5'))
    p.add_argument('--soundfont',type=Path,required=True)
    p.add_argument('--fluidsynth',default='fluidsynth')
    p.add_argument('--gain',type=float,default=1.)
    a=p.parse_args()
    if a.stage=='audio':prepare_audio(a)
    else:package(a)


if __name__=='__main__':main()
