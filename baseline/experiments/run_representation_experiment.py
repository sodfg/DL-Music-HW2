"""Six-layer baseline vs beat-grid MIDI-like at fixed training-token budget.

Plan by default. --stage all trains a NEW model, generates 20 Task-A samples,
evaluates H4/GS via the existing common adapter, and creates a comparison report.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.audio.render_audio import sha256
from baseline.experiments.run_depth_experiment import dump, metric_summary
from baseline.model import Config, MusicTransformer
from baseline.representations.midi_like import vocabulary


def make_plan(data, root):
    import torch
    reference_run = Path('runs/baseline')
    reference_samples = Path('runs/experiments/basic_a/t10')
    args = json.loads((reference_run/'train_args.json').read_text())
    config = json.loads((reference_run/'config.json').read_text())
    generation = json.loads((reference_samples/'generation_args.json').read_text())
    expected = {'count':20, 'bars':32, 'beats':4, 'temperature':1., 'top_k':32, 'top_p':.95,
                'seed':123, 'max_events_per_bar':128}
    if (args['layers'] != 6 or args['steps'] != 20000 or args['validation_fraction'] != 0
            or config['layers'] != 6 or generation['prompt'] is not None
            or any(generation[k] != v for k,v in expected.items())):
        raise ValueError('Expected original full-data six-layer reference and t10 sampling')
    checkpoint = reference_run/'best.pt'
    if Path(generation['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('Reference generation checkpoint differs')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['step'] != 20000 or saved['config'] != config or len(saved['split']['train']) != 896 or saved['split']['validation']:
        raise ValueError('Reference must be the full-data step-20000 checkpoint')
    summary = json.loads((data/'summary.json').read_text())
    original = Path(args['data'])
    if (summary['songs'] != 896 or summary['source_sequences_sha256'] != sha256(original/'sequences.json')
            or summary['source_vocab_sha256'] != sha256(original/'tok2idx.json')
            or json.loads((data/'tok2idx.json').read_text()) != vocabulary()):
        raise ValueError('MIDI-like data must come from exactly the baseline quantized data')
    songs = json.loads((data/'sequences.json').read_text())
    if len(songs) != 896 or {s['song_name'] for s in songs} != set(saved['split']['train']):
        raise ValueError('Different training-song population')
    target = Config(**{**config, 'vocab_size':len(vocabulary())})
    torch.manual_seed(args['seed'])
    target_parameters = sum(p.numel() for p in MusicTransformer(target).parameters())
    here = Path(__file__).resolve().parents[1]
    sources = ['train.py', 'model.py', 'music.py', 'evaluation/evaluate.py',
               'representations/midi_like.py', 'representations/generate_midi_like.py',
               'experiments/run_representation_experiment.py']
    plan = {'question':'Representation comparison at fixed layers, grid quantization and training-token budget',
            'reference_args':args, 'reference_config':config, 'target_config':asdict(target),
            'reference_checkpoint_sha256':sha256(checkpoint),
            'reference_parameters':sum(v.numel() for k,v in saved['model'].items() if k!='head.weight'),
            'target_parameters':target_parameters, 'sampling':expected,
            'data':str(data), 'data_sha256':sha256(data/'sequences.json'),
            'vocab_sha256':sha256(data/'tok2idx.json'), 'metadata_sha256':sha256(data/'bar_metadata.json'),
            'preparation':summary, 'source_sha256':{name:sha256(here/name) for name in sources}}
    path = root/'plan.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != plan:
            raise ValueError('Plan/source/data changed; use a new --output directory')
    else:
        dump(path,plan)
    return plan


def report(root, plan):
    real = metric_summary('runs/metrics/real.csv',896)
    rows = [dict(representation='Bar/Pos baseline',parameters=plan['reference_parameters'],
                 **metric_summary('runs/metrics/t10.csv')),
            dict(representation='MIDI-like',parameters=plan['target_parameters'],
                 **metric_summary(root/'metrics.csv'))]
    for row in rows:
        for metric in ('H4','GS'):
            row[metric+'_abs_diff_from_real'] = abs(row[metric+'_mean']-real[metric+'_mean'])
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines=['# Token representation:Bar/Pos versus MIDI-like','',
           '896 songs,same quantized notes/tempos,6 layers,width512,context1024,batch8,20000 steps,seed42.',
           '20 pieces of32 four-beat bars per group,T1,k32,p0.95,seeds123–142.','',
           '| Representation | Parameters | H4 mean ± standard deviation | GS mean ± standard deviation |','|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["representation"]} | {r["parameters"]:,} | {r["H4_mean"]:.4f} ± {r["H4_std"]:.4f} | '
                     f'{r["GS_mean"]:.4f} ± {r["GS_std"]:.4f} |')
    lines += ['',f'896-full-song reference:H4={real["H4_mean"]:.4f},GS={real["GS_mean"]:.4f}.','',
              '| Representation | H4 meandifference | GS meandifference |','|---|---:|---:|']
    for r in rows:
        lines.append(f'| {r["representation"]} | {r["H4_abs_diff_from_real"]:.4f} | {r["GS_abs_diff_from_real"]:.4f} |')
    prep=plan['preparation']
    lines += ['', '## Comparison scope and limitations', '',
        f'- Bar/Pos:{prep["totals"]["baseline_tokens"]:,}tokens; MIDI-like:{prep["totals"]["midi_like_tokens"]:,}tokens.',
        f'- Dataset-average estimated beats per1024 tokens: {prep["baseline_approx_beats_per_1024_tokens"]:.2f} / '
        f'{prep["midi_like_approx_beats_per_1024_tokens"]:.2f} beats; individual crops can differ.',
        '- Same architecture settings; vocabularies change embedding and total parameter counts.',
        '- Fixed training-token budget,not equal notes,epochs,music seconds,or FLOPs.',
        '- MIDI-like TimeShift uses1/16beat,not milliseconds. No learned Bar/Pos/Beats tokens.',
        '- Per-bar Tempo events remain; an external4-beat clock enforces32 bars,so bar-structure cues remain.',
        '- Both enforce valid sequences,at least1note and at most128 onsets per bar,but legal candidate sets differ.',
        '- MIDI-like caps notes at256 steps and retains overlapping same-pitch MIDI events.',
        '- sequences.json is the H4/GS onset view; actual events are in event_sequences.json. Render the MIDI for audio.',
        '- Task A representation comparison only; does not replace Task B or claim standard MidiTok compatibility.',
        '- Cross-entropy across vocabularies is not a music-quality ranking. One seed; listening is still required.',
        '- SI is evaluated separately with matching48-second excerpts.','']
    (root/'comparison.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','train','generate','evaluate','report','all'],default='plan')
    p.add_argument('--data',type=Path,default=Path('data/midi_like'))
    p.add_argument('--output',type=Path,default=Path('runs/representation_midi_like'))
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    import torch
    plan=make_plan(a.data,a.output)
    print(f'Plan verified: 6 layers; MIDI-like vocabulary={len(vocabulary())}; parameters={plan["target_parameters"]:,}',flush=True)
    if a.stage=='plan':return
    here=Path(__file__).resolve().parents[1]
    train=a.output/'train'
    samples=a.output/'generated'
    if a.stage in ('train','all'):
        if not torch.cuda.is_available():p.error('CUDA required for the full experiment')
        last=train/'last.pt'
        complete=False
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False)
            if saved['config']!=plan['target_config'] or saved['vocab']!=vocabulary():
                p.error('Existing checkpoint differs')
            complete=saved['step']==20000
            if not complete and not a.resume:p.error('Partial run exists; inspect and use --resume')
        elif train.exists() and any(train.iterdir()):
            p.error('Partial training output without checkpoint; choose new output')
        if not complete:
            command=[sys.executable,str(here/'train.py'),'--data',str(a.data),'--output',str(train)]
            for key in ('steps','batch_size','context','width','layers','heads','lr','eval_every',
                        'eval_batches','seed','validation_fraction'):
                command.extend(['--'+key.replace('_','-'),str(plan['reference_args'][key])])
            command.extend(['--device','cuda'])
            if last.exists():command.extend(['--resume',str(last)])
            started=time.perf_counter()
            subprocess.run(command,check=True)
            dump(a.output/'execution.json',{'train_process_wall_seconds':time.perf_counter()-started,
                 'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
                 'gpu':torch.cuda.get_device_name(0),'python':platform.python_version(),
                 'device_visibility':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')})
        else:print('Training already complete; retaining checkpoint',flush=True)
    if a.stage in ('generate','all'):
        checkpoint=train/'last.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['step']!=20000 or saved['config']!=plan['target_config'] or saved['vocab']!=vocabulary():
            p.error('Final MIDI-like checkpoint not ready')
        identity={'checkpoint_sha256':sha256(checkpoint),'sampling':plan['sampling'],
                  'generator_sha256':plan['source_sha256']['representations/generate_midi_like.py']}
        marker=a.output/'generation_identity.json'
        if marker.exists() and json.loads(marker.read_text())!=identity:p.error('Generation identity changed')
        if samples.exists() and not marker.exists():p.error('Unverified sample directory')
        dump(marker,identity)
        done=samples/'event_sequences.json'
        if done.exists():
            from baseline.representations.midi_like import read_events,metric_tokens
            pieces=json.loads(done.read_text())
            views=json.loads((samples/'sequences.json').read_text())
            names={f'sample_{i:03d}' for i in range(20)}
            if len(pieces)!=20 or {x['song_name'] for x in pieces}!=names:p.error('Incomplete samples')
            if views!=[{'song_name':x['song_name'],'tokens':metric_tokens(x['tokens'],[4]*32)} for x in pieces]:
                p.error('Metric view differs from generated event sequence')
            for piece in pieces:
                if read_events(piece['tokens'])[1]!=32*64 or not (samples/(piece['song_name']+'.mid')).exists():
                    p.error('Incomplete generated MIDI')
            print('Generation already complete; retaining 20 samples',flush=True)
        else:
            command=[sys.executable,str(here/'representations/generate_midi_like.py'),
                     '--checkpoint',str(checkpoint),'--output',str(samples)]
            for key,value in plan['sampling'].items():command.extend(['--'+key.replace('_','-'),str(value)])
            subprocess.run(command,check=True)
    if a.stage in ('evaluate','all'):
        subprocess.run([sys.executable,str(here/'evaluation/evaluate.py'),'--data',str(samples),
                        '--output',str(a.output/'metrics.csv')],check=True)
    if a.stage in ('report','all'):
        report(a.output,plan)
        print(f'Finished: {a.output}/comparison.md',flush=True)


if __name__=='__main__':main()
