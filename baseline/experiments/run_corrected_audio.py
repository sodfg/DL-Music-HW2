"""Download corrected groups as they finish, render WAVs, recompute SI, summarize.

No training or Git changes. --watch-remote observes an already authorized EDA
batch. Historical reference/MIDI-like scores are reused only with verified inputs
and matching SI protocols. Historical Bar/Pos generated scores are never reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.audio.render_audio import sha256
from baseline.evaluation.evaluate_si import verified_audio_record
from baseline.experiments.run_corrected_generation import audit
from baseline.music import dump

GROUPS={'t08':'basic_a/t08','t10':'basic_a/t10','t12':'basic_a/t12','layers3':'depth/layers3'}


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def summarize(root):
    plan=json.loads((root/'plan.json').read_text())
    refs={g:root/'si/groups'/g for g in GROUPS}
    refs['real']=Path('runs/depth_3_vs_6/si48/groups/real')
    refs['midi_like']=Path('runs/representation_midi_like/si48/groups/midi_like')
    protocol=None;si={};provenance=[]
    for group,folder in refs.items():
        current=json.loads((folder/'protocol.json').read_text())
        if protocol is None:protocol=current
        if current!=protocol:raise ValueError(f'SI protocol mismatch: {group}')
        rows=read_csv(folder/'per_piece.csv')
        if len(rows)!=20 or len({r['song_name'] for r in rows})!=20:raise ValueError('Need 20 unique SI pieces')
        for r in rows:
            if float(r['duration_seconds'])!=48 or float(r['start_seconds'])!=0:raise ValueError('Wrong SI crop')
            if group in GROUPS:
                wav=root/'audio'/group/(r['song_name']+'.wav');audio_root=root/'audio'
                midi=root/GROUPS[group]/(r['song_name']+'.mid')
            elif group=='real':
                audio_root=Path('runs/depth_3_vs_6/si48/audio');wav=audio_root/'real'/(r['song_name']+'.wav')
                frozen=json.loads(Path('runs/depth_3_vs_6/si48/study_plan.json').read_text())
                item=next(x for x in frozen['groups']['real'] if x['song_name']==r['song_name']);midi=Path(item['source_path'])
                if sha256(midi)!=item['source_sha256']:raise ValueError('Reference input changed')
            else:
                audio_root=Path('runs/representation_midi_like/si48/audio');wav=audio_root/'midi_like'/(r['song_name']+'.wav')
                midi=Path('runs/representation_midi_like/generated')/(r['song_name']+'.mid')
            record=verified_audio_record(wav,audio_root)
            if record['source_sha256']!=sha256(midi) or record['wav_sha256']!=r['wav_sha256'] or record['audio']['clipped_samples'] or record['settings']!=protocol['render_settings']:
                raise ValueError(f'Unverified audio: {wav}')
        si[group]={}
        for m in ('SI_short','SI_mid','SI_long'):
            values=[float(r[m]) for r in rows]
            if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite SI')
            si[group][m+'_mean']=statistics.mean(values);si[group][m+'_std']=statistics.stdev(values)
        provenance.append({'group':group,'csv':str(folder/'per_piece.csv'),'csv_sha256':sha256(folder/'per_piece.csv'),'reused':group in ('real','midi_like')})
    # Reuse unchanged real metrics only after confirming the exact training token data.
    historical=json.loads(Path('runs/depth_3_vs_6/plan.json').read_text())
    if sha256('data/baseline/sequences.json')!=historical['data_sha256']:raise ValueError('Real data changed')
    identity=json.loads(Path('runs/representation_midi_like/generation_identity.json').read_text())
    if sha256('baseline/representations/generate_midi_like.py')!=identity['generator_sha256']:raise ValueError('MIDI-like generator changed')
    if sha256('runs/representation_midi_like/train/last.pt')!=identity['checkpoint_sha256']:raise ValueError('MIDI-like checkpoint changed')
    metrics={g:root/'metrics'/f'{g}.csv' for g in GROUPS}
    metrics.update(real=Path('runs/metrics/real.csv'),midi_like=Path('runs/representation_midi_like/metrics.csv'))
    summary=[]
    for group,file in metrics.items():
        rows=read_csv(file);expected=896 if group=='real' else 20
        if len(rows)!=expected:raise ValueError('Wrong metric count')
        row={'setting':group,'song_count':expected,'policy':'boundary-v2' if group in GROUPS else 'unchanged reference' if group=='real' else 'MIDI-like stateful generator'}
        for short,col in [('H1','H1 (Pitch Entropy 1-Bar)'),('H4','H4 (Pitch Entropy 4-Bar)'),('GS','GS (Groove Similarity)')]:
            values=[float(r[col]) for r in rows]
            if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite metric')
            row[short+'_mean']=statistics.mean(values);row[short+'_std']=statistics.stdev(values)
        row.update(si[group]);summary.append(row)
        provenance.append({'group':group,'metric_csv':str(file),'sha256':sha256(file),'reused':group in ('real','midi_like')})
    real=next(r for r in summary if r['setting']=='real')
    for row in summary:
        for m in ('H1','H4','GS','SI_short','SI_mid','SI_long'):
            row[m+'_abs_diff_from_reference']=abs(row[m+'_mean']-real[m+'_mean'])
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    dump(root/'summary.json',{'groups':summary,'provenance':provenance,'si_protocol':protocol,'plan_sha256':sha256(root/'plan.json')})
    lines=['# Formal results after generation-rule correction','','All Bar/Pos groups use boundary-v2 and existing checkpoints. Historical generated scores are excluded.','',
           '| Group | Songs | H4 mean ± standard deviation | GS mean ± standard deviation |','|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["setting"]} | {r["song_count"]} | {r["H4_mean"]:.4f} ± {r["H4_std"]:.4f} | {r["GS_mean"]:.4f} ± {r["GS_std"]:.4f} |')
    lines+=['','## SI:20 pieces per group,first48 seconds','','| Group | short | mid | long |','|---|---:|---:|---:|']
    for r in summary:lines.append('| '+r['setting']+' | '+' | '.join(f'{r[m+"_mean"]:.4f} ± {r[m+"_std"]:.4f}' for m in ('SI_short','SI_mid','SI_long'))+' |')
    lines+=['','- Real H4/GS:896 full songs; SI:20seed42 sampled first48-second excerpts. Scopes differ.',
            '- 3/6 layers:T1,k32,p0.95,seeds123–142,same boundary-v2,and existing checkpoints.',
            '- MIDI-like reuses20verified pieces and matching-protocol scores. Clock/NoteOn/Off constraints differ,so representation comparisons include inference rules rather than isolated causal effects.',
            '- Between-song sample standard deviation. Do not rank losses across vocabularies.',
            '- Task B retains8prompt bars and continues24 bars. Current selections/links are recorded separately in submission_state.','']
    (root/'summary.md').write_text('\n'.join(lines))
    print('Updated summaries:',root/'summary.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/corrected_boundary_v2'))
    p.add_argument('--watch-remote',action='store_true')
    p.add_argument('--host',default='eda_union_sodfg')
    p.add_argument('--remote-root',default='DL_music_hw2/runs/corrected_boundary_v2')
    p.add_argument('--soundfont',type=Path,required=True)
    p.add_argument('--fluidsynth',required=True)
    p.add_argument('--si-python',default='.venv-si/bin/python')
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    def fetch(relative=''):
        dest=root/relative;dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a',f'{a.host}:{a.remote_root}/{relative}/',str(dest)+'/'],check=True)
    def work(group):
        folder=root/GROUPS[group];rows=audit(folder,20)
        dump(folder/'local_validation.json',rows)
        log=root/f'postprocess_{group}.log'
        with log.open('a',buffering=1) as stream:
            subprocess.run([sys.executable,'baseline/audio/render_audio.py','--input',str(folder),'--output',str(root/'audio'/group),
                '--soundfont',str(a.soundfont),'--fluidsynth',a.fluidsynth,'--gain','.025'],stdout=stream,stderr=subprocess.STDOUT,check=True)
            subprocess.run([a.si_python,'baseline/evaluation/evaluate_si.py','--audio-root',str(root/'audio'),'--groups',group,
                '--limit-per-group','20','--duration','48','--scope','batch','--output',str(root/'si/groups'/group)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('Audio and SI finished:',group,flush=True)
    futures={};status=None
    with ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            if a.watch_remote:
                probe="import pathlib,json; p=pathlib.Path.home()/"+repr(a.remote_root)+"; print(json.dumps({'done':[str(x.parent.relative_to(p)) for x in p.rglob('completed.json')], 'metrics':len(list((p/'metrics').glob('*.csv')))}))"
                remote=json.loads(subprocess.check_output(['ssh',a.host,'python3 -c '+shlex.quote(probe)],text=True))
                if remote!=status:print('Remote progress:',remote,flush=True);status=remote
                ready=remote['done'];finished=len(ready)==13 and remote['metrics']==4
            else:
                ready=[path for path in GROUPS.values() if (root/path/'completed.json').exists()];finished=True
            for group,path in GROUPS.items():
                if path in ready and group not in futures:
                    if a.watch_remote:fetch(path)
                    futures[group]=pool.submit(work,group)
            for f in futures.values():
                if f.done():f.result()
            if finished:break
            time.sleep(30)
        if a.watch_remote:fetch()
        if set(futures)!=set(GROUPS):raise ValueError('Generation incomplete')
        for f in futures.values():f.result()
    # Validate all nine continuations before rendering the selected basic-b tree.
    for name in ('001','002','003'):
        prompt=Path('POP909/prompts/basic-b')/name
        for setting in ('t08','t10','t12'):
            folder=root/'basic_b'/name/setting
            audit(folder,1,prompt/f'{name}.mid',prompt/'beat_midi.txt')
    subprocess.run([sys.executable,'baseline/audio/render_audio.py','--input',str(root/'basic_b'),'--output',str(root/'audio/basic_b'),
        '--recursive','--soundfont',str(a.soundfont),'--fluidsynth',a.fluidsynth,'--gain','.025'],check=True)
    summarize(root)
    dump(root/'postprocess_completed.json',{'summary_sha256':sha256(root/'summary.json'),'wav_count':len(list((root/'audio').rglob('*.wav'))),
        'status':'MIDI, WAV and metrics complete. Report refresh and listening selection follow.'})
    print('Finished 89 corrected WAV files and all new metric summaries.',flush=True)


if __name__=='__main__':main()
