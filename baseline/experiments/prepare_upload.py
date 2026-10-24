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
                      status=('Selections complete; final review and cloud link/upload pending.' if all(selections['task_b'].values()) else 'DRAFT: Task B best-version labels and cloud link pending.'))
    dump(a.output/'selections.json',selections)
    lines=['# Preferred versions','','Task A: t12. All20 pairs are submitted. Bonus: all10 pairs.','',
           '| Task B prompt | Preferred setting |','|---|---|']
    for sid,setting in selections['task_b'].items():
        lines.append(f'| {sid} | {setting or "Pending"} |')
    lines += ['', 'All nine Task B MIDI/WAV pairs remain included.',
              'See selections.json for sharing links. Both URLs belong on the report cover.']
    (a.output/'BEST_VERSIONS.md').write_text('\n'.join(lines)+'\n')
    template=Path(__file__).resolve().parents[1]/'docs/UPLOAD_README.md'
    (a.output/'README.md').write_text(template.read_text().replace('GAIN_VALUE',str(done['gain'])))
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
    p.add_argument('--report',type=Path,default=Path('output/report/B12901157_report_draft_v7'))
    p.add_argument('--soundfont',type=Path,required=True)
    p.add_argument('--fluidsynth',default='fluidsynth')
    p.add_argument('--gain',type=float,default=1.)
    a=p.parse_args()
    if a.stage=='audio':prepare_audio(a)
    else:package(a)


if __name__=='__main__':main()
