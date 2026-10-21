"""Align Task B MIDI tick grids to annotated beats, retaining performed events.

Creates separate visualization/export copies. Does not sample tokens, train a
model, overwrite input files, or alter historical generation audits/metrics.
"""
import argparse
import bisect
import json
from pathlib import Path
import sys

import mido

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.audio.render_audio import sha256
from baseline.music import beat_grid, dump


def timed_tracks(midi):
    """Return per-track absolute ticks and seconds under the global tempo map."""
    tick, seconds, tempo = 0, 0., 500000
    tempo_ticks, tempo_seconds, tempos = [0], [0.], [tempo]
    for msg in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
        tick += msg.time
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            if tick == tempo_ticks[-1]:
                tempo_seconds[-1], tempos[-1] = seconds, tempo
            else:
                tempo_ticks.append(tick); tempo_seconds.append(seconds); tempos.append(tempo)
    def to_seconds(tick):
        i = bisect.bisect_right(tempo_ticks, tick)-1
        return tempo_seconds[i]+mido.tick2second(tick-tempo_ticks[i], midi.ticks_per_beat, tempos[i])
    result=[]
    for track in midi.tracks:
        tick=0; events=[]
        for msg in track:
            tick+=msg.time
            events.append((tick,to_seconds(tick),msg.copy(time=0)))
        result.append(events)
    return result, to_seconds


def prompt_cutoff(source, boundary):
    # Match preserve_prompt's rounding convention, including varying source tempo.
    tick, seconds, tempo = 0, 0., 500000
    for msg in mido.merge_tracks(source.tracks):
        next_seconds=seconds+mido.tick2second(msg.time,source.ticks_per_beat,tempo)
        if next_seconds>=boundary:
            return tick+round(mido.second2tick(boundary-seconds,source.ticks_per_beat,tempo))
        tick+=msg.time;seconds=next_seconds
        if msg.type=='set_tempo':tempo=msg.tempo
    return tick+round(mido.second2tick(boundary-seconds,source.ticks_per_beat,tempo))


def align_midi(input_path, prompt_path, beat_path, output_path, prompt_bars=8,
               total_bars=32, ppq=9600, tolerance_seconds=.0002):
    input_path,prompt_path,beat_path,output_path=map(Path,(input_path,prompt_path,beat_path,output_path))
    if output_path.exists():raise ValueError(f'Refusing to overwrite {output_path}')
    if not 0<prompt_bars<total_bars or not 1<=ppq<=32767:raise ValueError('Invalid dimensions')
    original=mido.MidiFile(str(input_path));prompt=mido.MidiFile(str(prompt_path))
    if original.type==2 or original.ticks_per_beat!=prompt.ticks_per_beat:
        raise ValueError('Expected synchronous MIDI with original prompt tick resolution')
    times,starts,sizes=beat_grid(beat_path)
    if len(starts)<=prompt_bars or any(n!=4 for n in sizes[:prompt_bars]):
        raise ValueError('This Task B export supports annotated four-beat bars only')
    beat_count=starts[prompt_bars]
    if beat_count!=4*prompt_bars or abs(float(times[0]))>1e-9:
        raise ValueError('Pickup/nonzero first beat requires a separate meter policy')
    seam=prompt_cutoff(prompt,float(times[beat_count]))
    tracks,to_seconds=timed_tracks(original);seam_seconds=to_seconds(seam)
    knots=list(map(float,times[:beat_count+1]));knots[-1]=seam_seconds
    if any(b<=a for a,b in zip(knots,knots[1:])):raise ValueError('Invalid beat map')
    old_end=max(tick for t in tracks for tick,_,_ in t)
    expected_end=seam+4*(total_bars-prompt_bars)*original.ticks_per_beat
    if old_end!=expected_end:raise ValueError('Source timeline does not contain the expected complete tail')
    def new_tick(tick,seconds):
        if tick>=seam:
            return beat_count*ppq+round((tick-seam)*ppq/original.ticks_per_beat)
        i=max(0,min(beat_count-1,bisect.bisect_right(knots,seconds)-1))
        return round((i+(seconds-knots[i])/(knots[i+1]-knots[i]))*ppq)
    out=mido.MidiFile(type=1,ticks_per_beat=ppq)
    conductor=mido.MidiTrack();out.tracks.append(conductor)
    metadata=[(0,mido.MetaMessage('time_signature',numerator=4,denominator=4))]
    for i in range(beat_count):
        metadata.append((i*ppq,mido.MetaMessage('set_tempo',tempo=round((knots[i+1]-knots[i])*1e6))))
    for track in tracks:
        for tick,seconds,msg in track:
            if msg.type=='set_tempo' and tick>=seam:
                metadata.append((new_tick(tick,seconds),msg))
    if not any(tick==beat_count*ppq and msg.type=='set_tempo' for tick,msg in metadata):
        raise ValueError('Missing tail tempo at the prompt boundary')
    previous=0
    for tick,msg in sorted(metadata,key=lambda item:item[0]):
        conductor.append(msg.copy(time=tick-previous));previous=tick
    final_tick=total_bars*4*ppq
    conductor.append(mido.MetaMessage('end_of_track',time=final_tick-previous))
    for events in tracks:
        track=mido.MidiTrack();out.tracks.append(track);previous=0
        for tick,seconds,msg in events:
            if msg.type in ('set_tempo','time_signature','end_of_track'):continue
            dest=new_tick(tick,seconds)
            if dest<previous:raise ValueError('Event order changed')
            track.append(msg.copy(time=dest-previous));previous=dest
        track.append(mido.MetaMessage('end_of_track',time=final_tick-previous))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    out.save(str(output_path))
    reread=mido.MidiFile(str(output_path));new_tracks,new_seconds=timed_tracks(reread)
    max_error=0.;count=0
    for old,new in zip(tracks,new_tracks[1:]):
        old=[(s,m) for _,s,m in old if m.type not in ('set_tempo','time_signature','end_of_track')]
        new=[(s,m) for _,s,m in new if m.type not in ('set_tempo','time_signature','end_of_track')]
        if len(old)!=len(new):raise ValueError('Performance event count changed')
        for (s1,m1),(s2,m2) in zip(old,new):
            if m1!=m2:raise ValueError('Performance event contents changed')
            max_error=max(max_error,abs(s1-s2));count+=1
    end_error=abs(new_seconds(final_tick)-to_seconds(old_end))
    if max(max_error,end_error)>tolerance_seconds:
        raise ValueError(f'Timing tolerance exceeded: {max_error}, end={end_error}')
    return {'input':str(input_path.resolve()),'input_sha256':sha256(input_path),
            'prompt_sha256':sha256(prompt_path),'beat_sha256':sha256(beat_path),
            'output_sha256':sha256(output_path),'prompt_bars':prompt_bars,'total_bars':total_bars,
            'display_meter':'4/4','ticks_per_beat':ppq,'end_tick':final_tick,
            'prompt_end_tick':beat_count*ppq,'prompt_end_seconds':seam_seconds,
            'annotation_boundary_seconds':float(times[beat_count]),
            'old_display_bars':old_end/original.ticks_per_beat/4,
            'new_display_bars':final_tick/ppq/4,'performance_events_verified':count,
            'max_event_time_error_seconds':max_error,'end_time_error_seconds':end_error,
            'tolerance_seconds':tolerance_seconds,'note':'Export grid only; event bytes except timing, tempo and meter preserved. Not a new generation.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=Path('runs/corrected_boundary_v2/basic_b'))
    p.add_argument('--output',type=Path,default=Path('runs/task_b_aligned'))
    p.add_argument('--prompts',type=Path,default=Path('POP909/prompts/basic-b'))
    p.add_argument('--prompt-id',choices=['001','002','003'])
    p.add_argument('--setting',choices=['t08','t10','t12'])
    a=p.parse_args();rows=[]
    for name in ([a.prompt_id] if a.prompt_id else ['001','002','003']):
        prompt=a.prompts/name
        for setting in ([a.setting] if a.setting else ['t08','t10','t12']):
            filename=f'{name}_000.mid'
            row=align_midi(a.input_root/name/setting/filename,prompt/f'{name}.mid',prompt/'beat_midi.txt',a.output/name/setting/filename)
            rows.append(row)
            print(f'{name}/{setting}: {row["old_display_bars"]:.3f} -> 32 display bars; max timing error={row["max_event_time_error_seconds"]*1000:.4f} ms',flush=True)
    dump(a.output/'alignment_audit.json',rows)


if __name__=='__main__':main()
