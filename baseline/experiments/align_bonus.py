"""Align 16-bar Bonus exports to annotations without changing performed events.

Each annotated beat is one display quarter note. Meter numerators follow the
annotated beat count (including 2/4 and 6/4 display conventions). The original
score's meter denominator is not inferred from these beat annotations.
"""
import argparse
import bisect
import csv
import math
import statistics
from pathlib import Path
import sys
import mido

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from baseline.audio.render_audio import sha256
from baseline.experiments.align_task_b import timed_tracks
from baseline.music import beat_grid, dump

IGNORE={'set_tempo','time_signature','end_of_track'}


def align_bonus(input_path,beat_path,output_path,bars=16,ppq=19200):
    input_path,beat_path,output_path=map(Path,(input_path,beat_path,output_path))
    if output_path.exists():raise ValueError('Output exists; use a new folder')
    midi=mido.MidiFile(str(input_path));tracks,_=timed_tracks(midi)
    if midi.type==2:raise ValueError('Asynchronous MIDI not supported')
    times,starts,sizes=beat_grid(beat_path)
    if len(starts)<=bars or abs(float(times[0]))>1e-9 or starts[0]!=0:
        raise ValueError('Expected zero-based complete annotated bars')
    beat_count=starts[bars];knots=list(map(float,times[:beat_count+1]))
    performance=[(s,m) for track in tracks for _,s,m in track if m.type not in IGNORE]
    # Retain tiny original note/control overhang caused by the source tick grid.
    # Extend only the final beat's end, not the sounding event times.
    annotation_end=knots[-1];knots[-1]=max(knots[-1],max(s for s,_ in performance))
    if knots[-1]-annotation_end>.005:raise ValueError('Unexpected source overhang beyond 5ms')
    def tick_for(seconds):
        i=max(0,min(beat_count-1,bisect.bisect_right(knots,seconds)-1))
        return round((i+(seconds-knots[i])/(knots[i+1]-knots[i]))*ppq)
    out=mido.MidiFile(type=1,ticks_per_beat=ppq);conductor=mido.MidiTrack();out.tracks.append(conductor)
    metadata=[];previous_size=None
    for start,size in zip(starts[:bars],sizes[:bars]):
        if size!=previous_size:
            metadata.append((start*ppq,mido.MetaMessage('time_signature',numerator=size,denominator=4)))
            previous_size=size
    for i in range(beat_count):
        metadata.append((i*ppq,mido.MetaMessage('set_tempo',tempo=round((knots[i+1]-knots[i])*1e6))))
    end=beat_count*ppq;previous=0
    for tick,msg in sorted(metadata,key=lambda x:x[0]):
        conductor.append(msg.copy(time=tick-previous));previous=tick
    conductor.append(mido.MetaMessage('end_of_track',time=end-previous))
    for events in tracks:
        track=mido.MidiTrack();out.tracks.append(track);previous=0
        for _,seconds,msg in events:
            if msg.type in IGNORE:continue
            tick=tick_for(seconds)
            if not previous<=tick<=end:raise ValueError('Event outside annotated range')
            track.append(msg.copy(time=tick-previous));previous=tick
        track.append(mido.MetaMessage('end_of_track',time=end-previous))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    staged=output_path.with_suffix('.mid.tmp');out.save(str(staged))
    new_tracks,_=timed_tracks(mido.MidiFile(str(staged)));error=0.;count=0
    for old,new in zip(tracks,new_tracks[1:]):
        old=[(s,m) for _,s,m in old if m.type not in IGNORE]
        new=[(s,m) for _,s,m in new if m.type not in IGNORE]
        if len(old)!=len(new):raise ValueError('Event count changed')
        for (s1,m1),(s2,m2) in zip(old,new):
            if m1!=m2:raise ValueError('Performance event changed')
            error=max(error,abs(s1-s2));count+=1
    if error>=.00005:raise ValueError('Performance timing differs by >=0.05ms')
    staged.replace(output_path)
    return {'song_name':input_path.stem,'input_sha256':sha256(input_path),'output_sha256':sha256(output_path),
        'beat_sha256':sha256(beat_path),'bounded_bars':bars,'beats_per_bar':sizes[:bars],
        'display_meter_convention':'Each annotated beat is a quarter; numerator follows annotation, original denominator not inferred',
        'end_tick':end,'ticks_per_beat':ppq,'max_event_time_error_seconds':error,
        'final_beat_extension_seconds':knots[-1]-annotation_end,'performance_events_verified':count,
        'duration_difference_seconds':out.length-midi.length}


def summarize_metrics(folder):
    """Summarize untouched TA CSV after aligned-MIDI evaluation."""
    folder=Path(folder)
    with (folder/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=10 or {r['song_name'].zfill(3) for r in rows}!={f'{i:03d}' for i in range(1,11)} or any(int(r['n_bars'])!=16 for r in rows):
        raise ValueError('Expected ten unique 16-bar Bonus results')
    metrics={}
    for k in rows[0]:
        if k in ('song_name','n_bars'):continue
        values=[float(r[k]) for r in rows]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite metric')
        metrics[k]={'pooled_mean':statistics.mean(values),'song_sample_std':statistics.stdev(values)}
    note='Aligned MIDI reevaluated with untouched TA evaluator. All ten songs have 16 bars. Tiny tick retiming can change duration/IOI histogram bins. CA-ceiling is a GT-estimator reference, not a theoretical upper bound.'
    dump(folder/'metrics.summary.json',{'songs':10,'bars':160,'metrics':metrics,'ta_evaluator_sha256':sha256(Path('HW2_scripts/eval_bonus.py')),'note':note})
    lines=['# Bonus：正式上傳 MIDI 的重新評分','', '10 首、160 小節。使用助教原始 eval_bonus.py。','', '| 指標 | 平均 | 歌曲間標準差 |','|---|---:|---:|']
    lines += [f"| {k} | {v['pooled_mean']:.4f} | {v['song_sample_std']:.4f} |" for k,v in metrics.items()]
    lines += ['', '匯出依標註拍點調整 MIDI 刻度，保留演奏事件，時間誤差小於 0.05 毫秒。微小誤差可能改變音長／IOI 的直方圖分箱，因此本表重新評估全部十首。', 'CA-ceiling 是同一估計器在真實鋼琴上的參考值，不是嚴格理論上限。']
    (folder/'metrics.md').write_text('\n'.join(lines)+'\n')
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('runs/bonus_accompaniment/generated'))
    p.add_argument('--prompts',type=Path,default=Path('POP909/prompts/bonus'))
    p.add_argument('--output',type=Path,default=Path('runs/bonus_aligned_v2'))
    p.add_argument('--summarize-metrics',action='store_true',help='Summarize an existing TA metrics.csv in --output, without rewriting MIDI')
    a=p.parse_args();rows=[]
    if a.summarize_metrics:
        summarize_metrics(a.output);return
    for i in range(1,11):
        sid=f'{i:03d}';row=align_bonus(a.input/(sid+'.mid'),a.prompts/sid/'beat_midi.txt',a.output/(sid+'.mid'))
        rows.append(row);print(sid,'16 annotated/display bars; max event delta ms=',row['max_event_time_error_seconds']*1000,flush=True)
    dump(a.output/'alignment_audit.json',rows)


if __name__=='__main__':main()
