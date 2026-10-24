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


def build(output,soundfont,task_a=None,corrected_results=None,include_bonus=False):
    import torch
    if output.exists():raise ValueError('Destination exists; choose a new output directory')
    if not soundfont.is_file():raise ValueError('SoundFont file missing')
    models={'baseline_6':Path('runs/baseline/best.pt'),
            'depth_3':Path('runs/depth_3_vs_6/train/last.pt'),
            'midi_like_6':Path('runs/representation_midi_like/train/last.pt')}
    if include_bonus:models['bonus_accompaniment']=Path('runs/bonus_accompaniment/train/last.pt')
    if corrected_results:
        done=json.loads((corrected_results/'postprocess_completed.json').read_text())
        if done['wav_count']!=89 or done['summary_sha256']!=sha256(corrected_results/'summary.json'):
            raise ValueError('Corrected results incomplete or changed')
    # Preflight all required results before writing the draft.
    for path in models.values():
        if not path.is_file():raise ValueError(f'Model missing: {path}')
    pairs=[]
    for sid in ('001','002','003'):
        for setting in ('t08','t10','t12'):
            midi=(corrected_results/'basic_b' if corrected_results else Path('runs/experiments/basic_b'))/sid/setting/(sid+'_000.mid')
            audio_root=corrected_results/'audio/basic_b' if corrected_results else Path('runs/listening/basic_b')
            wav=audio_root/sid/setting/(sid+'_000.wav')
            record=verified_audio_record(wav,audio_root)
            if record['source_sha256']!=sha256(midi) or record['audio']['clipped_samples']:
                raise ValueError('Task B audio unverified')
            pairs.append((Path('task_b')/sid/setting,midi,wav))
    if task_a:
        for i in range(20):
            name=f'sample_{i:03d}'
            midi=(corrected_results/'basic_a' if corrected_results else Path('runs/experiments/basic_a'))/task_a/(name+'.mid')
            audio_root=corrected_results/'audio' if corrected_results else Path('runs/metrics/si48_g0025/audio')
            wav=audio_root/task_a/(name+'.wav')
            record=verified_audio_record(wav,audio_root)
            if record['source_sha256']!=sha256(midi) or record['audio']['clipped_samples']:
                raise ValueError('Task A audio unverified')
            pairs.append((Path('task_a'),midi,wav))
    if include_bonus:
        for sid in (f'{i:03d}' for i in range(1,11)):
            midi=Path('runs/bonus_accompaniment/generated')/(sid+'.mid')
            wav=Path('runs/bonus_accompaniment/audio')/(sid+'.wav')
            record=verified_audio_record(wav,'runs/bonus_accompaniment/audio')
            if record['source_sha256']!=sha256(midi) or record['audio']['clipped_samples']:
                raise ValueError('Bonus audio unverified')
            pairs.append((Path('bonus'),midi,wav))
    output.mkdir(parents=True)
    shutil.copytree('baseline',output/'code/baseline',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile('baseline/requirements.txt',output/'requirements.txt')
    exports={}
    for name,source in models.items():
        saved=torch.load(source,map_location='cpu',weights_only=False)
        if saved['step']!=20000:raise ValueError('Expected completed step-20000 model')
        target=output/'models'/(name+'.pt');target.parent.mkdir(exist_ok=True)
        minimal={k:saved[k] for k in ('config','vocab','model')}
        if name=='bonus_accompaniment':minimal.update(kind=saved['kind'],step=saved['step'])
        torch.save(minimal,target)
        restored=torch.load(target,map_location='cpu',weights_only=False)
        assert restored['config']==saved['config'] and restored['vocab']==saved['vocab']
        assert all(torch.equal(v,restored['model'][k]) for k,v in saved['model'].items())
        exports[name]={'source_sha256':sha256(source),'export_sha256':sha256(target),
                       'source_step':saved['step'],'contents':list(minimal),
                       'tensor_equality_verified':True}
    for dest,midi,wav in pairs:
        folder=output/dest;folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(midi,folder/midi.name);shutil.copyfile(wav,folder/wav.name)
    for sid in ('001','002','003'):
        folder=output/'prompts'/sid;folder.mkdir(parents=True)
        for name in (sid+'.mid','beat_midi.txt'):
            shutil.copyfile(Path('POP909/prompts/basic-b')/sid/name,folder/name)
    if include_bonus:
        for sid in (f'{i:03d}' for i in range(1,11)):
            folder=output/'prompts/bonus'/sid;folder.mkdir(parents=True)
            for name in ('melody.mid','beat_midi.txt'):
                shutil.copyfile(Path('POP909/prompts/bonus')/sid/name,folder/name)
    audio=output/'audio';audio.mkdir()
    shutil.copyfile(soundfont,audio/'GeneralUser-GS.sf2')
    shutil.copyfile(soundfont.parent/'GeneralUser-GS-LICENSE.txt',audio/'GeneralUser-GS-LICENSE.txt')
    tables=output/'experiment_tables';tables.mkdir()
    table_sources=({'corrected_summary':corrected_results/'summary.csv'} if corrected_results else {'task_a':'runs/metrics/summary.csv','sampling_si':'runs/metrics/si48_g0025/summary.csv',
                      'depth':'runs/depth_3_vs_6/comparison.csv','depth_si':'runs/depth_3_vs_6/si48/comparison.csv',
                      'representation':'runs/representation_midi_like/comparison.csv',
                      'representation_si':'runs/representation_midi_like/si48/comparison.csv'})
    for name,path in table_sources.items():
        shutil.copyfile(path,tables/(name+'.csv'))
    if corrected_results:
        for name in ('summary.md','summary.json','plan.json'):
            shutil.copyfile(corrected_results/name,tables/name)
    if include_bonus:
        for name in ('metrics.csv','metrics.md','metrics.summary.json'):
            shutil.copyfile(Path('runs/bonus_accompaniment')/name,tables/('bonus_'+name))
    save_json(output/'selections.json',{'task_a':task_a,'task_b':{s:None for s in ('001','002','003')},
        'cloud_link':None,'generation_policy':'boundary-v2' if corrected_results else 'legacy',
        'bonus_included':include_bonus,'status':'DRAFT: final listening choices, report and cloud sharing pending'})
    save_json(output/'model_provenance.json',exports)
    (output/'task_a').mkdir(exist_ok=True)
    if not task_a:(output/'task_a/PENDING.md').write_text('Task A not selected. Listen and choose t08/t10/t12 before packaging20 MIDI+20 WAV.\n')
    (output/'README.md').write_text('''# DL in Music HW2 submission draft

This is a local submission draft,not an uploaded final. Review selections,report,and sharing permissions first.
All9Task B MIDI/WAV pairs are included; record each preferred version in selections.json.
If Task A is unselected,task_a/PENDING.md remains; the script does not choose for the author.

## Environment

Python3.10+,PyTorch2.5–2.6. Original training:Python3.10.12,torch2.6.0+cu124,RTX A5000.
From this folder,install a PyTorch build matching your CUDA/CPU environment,then requirements.txt.
Reuse compatible shared environments. Audio rendering additionally requires FluidSynth.
Included GeneralUser GS SoundFont license:audio/GeneralUser-GS-LICENSE.txt.

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

The t10example demonstrates inference; formal selections are recorded in selections.json.
Use a fresh writable --output. GPU/CPU selection follows CUDA availability.

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

Use matching prompt folders for002/003; t08/t10/t12 set temperatures0.8/1.0/1.2.
Only the original prompts are provided here,not training or preprocessed data.

## Optional models

models/depth_3.pt uses the same code/baseline/generate.py inference entry point.
models/midi_like_6.pt uses this separate entry point:

```bash
python3 code/baseline/representations/generate_midi_like.py \\
  --checkpoint models/midi_like_6.pt --output generated/new_midi_like \\
  --count 20 --bars 32 --temperature 1.0 --top-k 32 --top-p 0.95
```

Model export retains inference model/config/vocab only,without optimizer or RNG state.
All weight tensors match the source checkpoint. See model_provenance.json for sources/hashes.
experiment_tables summarizes completed comparisons; training scripts need the original project data environment.
This draft does not include Bonus accompaniment.
''')
    readme=output/'README.md'
    contents=readme.read_text()
    if corrected_results:
        contents=contents.replace('--seed 123\n','--seed 123 --generation-policy boundary-v2\n')
        contents=contents.replace('--output generated/new_b_001\n','--output generated/new_b_001 --generation-policy boundary-v2\n')
        contents+='\nFormal Bar/Pos uses boundary-v2,also for the three-layer model. Historical generated scores are excluded.\n'
    if include_bonus:
        contents=contents.replace('This draft does not include Bonus accompaniment.','Bonus includes10melody-plus-accompaniment MIDI,10 WAV,a separate model,and metrics.')
        contents+='''
## Bonus inference

```bash
python3 code/baseline/bonus/generate.py --checkpoint models/bonus_accompaniment.pt --prompts prompts/bonus --output generated/new_bonus
python3 code/baseline/audio/render_audio.py --input generated/new_bonus --output generated/new_bonus_wav --soundfont audio/GeneralUser-GS.sf2 --fluidsynth fluidsynth --gain 0.025
```

Bonus prompts include only melody.mid/beat_midi.txt,not GT accompaniment/chord answers.
'''
    readme.write_text(contents)
    files={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()}
    save_json(output/'file_manifest.json',files)
    print(f'Local draft built: {output}; Task B=9 MIDI/9 WAV; Task A={task_a or "PENDING"}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/submission_draft'))
    p.add_argument('--soundfont',type=Path,required=True)
    p.add_argument('--task-a',choices=['t08','t10','t12'])
    p.add_argument('--corrected-results',type=Path,help='Verified boundary-v2 results, excludes historical generated tables')
    p.add_argument('--include-bonus',action='store_true')
    a=p.parse_args();build(a.output,a.soundfont,a.task_a,a.corrected_results,a.include_bonus)
