"""Separate Bonus pipeline: paired seq2seq training, ten accompaniments, TA CA/OA.

Default plan only. Does not use GT piano/chords as generation conditions.
WAV rendering follows on the local audio environment after downloading results.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.bonus.data import vocabulary
from baseline.bonus.model import Config,AccompanimentTransformer
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import save_json


def create_plan(data,root,steps,batch_size):
    import torch
    summary=json.loads((data/'summary.json').read_text())
    if summary['songs']!=896 or json.loads((data/'tok2idx.json').read_text())!=vocabulary():
        raise ValueError('Full Bonus experiment requires all896 song pairs and the correct vocabulary')
    config=Config(len(vocabulary()),max(256,math.ceil(summary['max_source_tokens_16_bars']/64)*64))
    torch.manual_seed(42);model=AccompanimentTransformer(config)
    here=Path(__file__).resolve().parents[1]
    names=['bonus/data.py','bonus/model.py','bonus/train.py','bonus/generate.py','bonus/evaluate.py',
           'representations/generate_midi_like.py','experiments/run_bonus.py','music.py']
    plan={'task':'Bonus melody-conditioned piano accompaniment','architecture':'encoder-decoder Transformer from random initialization',
          'config':asdict(config),'parameters':sum(p.numel() for p in model.parameters()),
          'training':{'steps':steps,'batch_size':batch_size,'lr':.0003,'seed':42,'all_songs':896,'evaluation':'training_probe'},
          'window':'full up-to16bar melody; Focus_N and TargetBars_N select up-to4bar piano targets',
          'generation':{'temperature':1.,'top_k':32,'top_p':.95,'seed':123,'max_events_per_bar':48,'prompt_bars':16},
          'prompt_inputs_sha256':{f'{n:03d}/{name}':sha256(here.parent/'POP909/prompts/bonus'/f'{n:03d}'/name) for n in range(1,11) for name in ('melody.mid','beat_midi.txt')},
          'summary_sha256':sha256(data/'summary.json'),
          'data':str(data),'pairs_sha256':sha256(data/'pairs.json'),'vocab_sha256':sha256(data/'tok2idx.json'),
          'source_sha256':{name:sha256(here/name) for name in names},
          'ta_evaluator_sha256':sha256(here.parent/'HW2_scripts/eval_bonus.py'),
          'generation_inputs':['melody.mid','beat_midi.txt'],'evaluation_only':['piano.mid','chord_midi.txt']}
    path=root/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Plan/source changed; choose a new --output')
    save_json(path,plan);return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','train','generate','evaluate','all'],default='plan')
    p.add_argument('--data',type=Path,default=Path('data/bonus'))
    p.add_argument('--output',type=Path,default=Path('runs/bonus_accompaniment'))
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.batch_size)<1:p.error('Counts must be positive')
    import torch
    plan=create_plan(a.data,a.output,a.steps,a.batch_size)
    print(f'Bonus plan: {plan["parameters"]:,} parameters, all896 songs, {a.steps} steps',flush=True)
    if a.stage=='plan':return
    here=Path(__file__).resolve().parents[1]
    training=a.output/'train';checkpoint=training/'last.pt';generated=a.output/'generated'
    if a.stage in ('train','all'):
        if not torch.cuda.is_available():p.error('CUDA required for full run; use separate tiny CPU smoke')
        complete=False
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['config']!=plan['config'] or saved['identity']['pairs_sha256']!=plan['pairs_sha256']:p.error('Checkpoint identity differs')
            complete=saved['step']==a.steps
            if not complete and not a.resume:p.error('Partial checkpoint; inspect and use --resume')
        if not complete:
            command=[sys.executable,str(here/'bonus/train.py'),'--data',str(a.data),'--output',str(training),
                     '--steps',str(a.steps),'--batch-size',str(a.batch_size),'--device','cuda']
            if checkpoint.exists():command+=['--resume',str(checkpoint)]
            started=time.perf_counter();subprocess.run(command,check=True)
            save_json(a.output/'execution.json',{'train_process_wall_seconds':time.perf_counter()-started,
               'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
               'python':platform.python_version(),'device_visibility':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')})
    if a.stage in ('generate','all'):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['step']!=a.steps or saved['config']!=plan['config'] or saved['identity']['pairs_sha256']!=plan['pairs_sha256']:
            p.error('Final Bonus checkpoint not ready')
        if len(saved.get('seen_songs',[]))!=896:p.error('Training has not sampled all896 songs')
        marker=a.output/'generation_identity.json'
        identity={'checkpoint_sha256':sha256(checkpoint),'source_sha256':plan['source_sha256'],'sampling':plan['generation']}
        if marker.exists() and json.loads(marker.read_text())!=identity:p.error('Different generation identity')
        if generated.exists() and not marker.exists():p.error('Unverified generated folder')
        save_json(marker,identity)
        if (generated/'sequences.json').exists():
            records=json.loads((generated/'validation.json').read_text())
            if len(records)!=10 or {r['song_name'] for r in records}!={f'{n:03d}' for n in range(1,11)}:p.error('Incomplete Bonus set')
            for r in records:
                if sha256(generated/(r['song_name']+'.mid'))!=r['midi_sha256'] or not r['melody_events_preserved']:p.error('Generated MIDI changed')
            print('Retaining completed ten prompt accompaniments',flush=True)
        else:
            subprocess.run([sys.executable,str(here/'bonus/generate.py'),'--checkpoint',str(checkpoint),'--output',str(generated)],check=True)
    if a.stage in ('evaluate','all'):
        metric=a.output/'metrics.csv'
        if metric.exists():p.error('Metrics already exist; inspect retained report rather than silently overwrite')
        subprocess.run([sys.executable,str(here/'bonus/evaluate.py'),'--generated',str(generated),'--output',str(metric)],check=True)
        print(f'Finished Bonus MIDI and objective metrics: {metric.with_suffix(".md")}; ten WAVs still need rendering',flush=True)


if __name__=='__main__':main()
