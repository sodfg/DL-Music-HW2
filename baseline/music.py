"""Beat-annotation-based representation, independent of incorrect MIDI meters.
Adapted conceptually from the course HW2_scripts/tokenization.py.
Beats_N means annotated beats per bar, not a claim about meter denominator.
"""
import bisect
import json
from pathlib import Path
import numpy as np
import pretty_midi
import mido

PPB = 16
MAX_BEATS = 32
MAX_DURATION = 256

def vocabulary():
    tokens = ['<PAD>', '<BOS>', '<EOS>', 'Bar']
    for prefix, values in [('Beats', range(1, MAX_BEATS + 1)),
                           ('Tempo', range(30, 211, 5)),
                           ('Pos', range(MAX_BEATS * PPB)),
                           ('Pitch', range(128)), ('Vel', range(1, 128, 4)),
                           ('Dur', range(1, MAX_DURATION + 1))]:
        tokens.extend(f'{prefix}_{n}' for n in values)
    return {t: i for i, t in enumerate(tokens)}

def beat_grid(path):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) < 2 or any(len(r) < 3 for r in rows):
        raise ValueError(f'Invalid beat annotations: {path}')
    times = np.array([float(r[0]) for r in rows])
    orders = [float(r[2]) for r in rows]
    if not np.all(np.diff(times) > 0):
        raise ValueError('Beat times must be strictly increasing')
    downs = [i for i, o in enumerate(orders) if o == 1]
    if not downs:
        raise ValueError('No annotated downbeats')
    # Retain pickup as a separate partial bar, matching the TA convention.
    starts = ([0] if downs[0] > 0 else []) + downs
    end_index = len(times)
    sizes = [b-a for a,b in zip(starts, starts[1:] + [end_index])]
    if max(sizes) > MAX_BEATS:
        raise ValueError(f'Bar exceeds {MAX_BEATS} annotated beats')
    extended = np.append(times, times[-1] + times[-1] - times[-2])
    return extended, starts, sizes

def encode(midi_path, beat_path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    times, starts, sizes = beat_grid(beat_path)
    events = [[] for _ in starts]
    def coordinate(seconds):
        idx = max(0, min(len(times)-2, bisect.bisect_right(times, seconds)-1))
        return idx + (seconds-times[idx])/(times[idx+1]-times[idx])
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            q = coordinate(note.start)
            b = max(0, min(len(starts)-1, bisect.bisect_right(starts, q)-1))
            pos = int(max(0, (q-starts[b])*PPB))
            if pos >= sizes[b]*PPB:
                raise ValueError(f'Note outside annotated last bar: {note.start}')
            dur = max(1, min(MAX_DURATION, round((coordinate(note.end)-q)*PPB)))
            vel = min(125, max(1, note.velocity//4*4+1))
            events[b].append((pos, note.pitch, vel, dur))
    result = ['<BOS>']
    for b, (start, size) in enumerate(zip(starts, sizes)):
        beat_length = float(np.median(np.diff(times[start:start+size+1])))
        tempo = max(30, min(210, round((60/beat_length)/5)*5))
        result.extend(['Bar', f'Beats_{size}', f'Tempo_{tempo}'])
        for pos, pitch, vel, dur in sorted(events[b]):
            result.extend([f'Pos_{pos}', f'Pitch_{pitch}', f'Vel_{vel}', f'Dur_{dur}'])
    result.append('<EOS>')
    return result

def decode(tokens, output, default_beats=4, default_tempo=120):
    """Write MIDI in ticks so tempo changes and bar boundaries remain explicit."""
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    events = [(0, 0, mido.Message('program_change', program=0, time=0))]
    start, beats, tempo, bar = 0, default_beats, default_tempo, -1
    pos, pitch, vel = 0, None, 81
    boundary_ticks = [0]
    for token in tokens:
        if token == 'Bar':
            if bar >= 0:
                start += beats * 480
                boundary_ticks.append(start)
            bar += 1
            pos, pitch = 0, None
        elif token.startswith('Beats_'):
            beats = int(token.split('_')[1])
        elif token.startswith('Tempo_'):
            tempo = int(token.split('_')[1])
            events.append((start, 0, mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo))))
        elif token.startswith('Pos_'):
            pos = int(token.split('_')[1])
        elif token.startswith('Pitch_'):
            pitch = int(token.split('_')[1])
        elif token.startswith('Vel_'):
            vel = int(token.split('_')[1])
        elif token.startswith('Dur_') and pitch is not None:
            onset = start + pos * 480//PPB
            end = onset + int(token.split('_')[1])*480//PPB
            events.extend([(onset, 2, mido.Message('note_on', note=pitch, velocity=vel)),
                           (end, 1, mido.Message('note_off', note=pitch, velocity=0))])
            pitch = None
    total = start + beats*480 if bar >= 0 else 0
    # Crop final sustained notes at the requested piece boundary.
    events = [(min(t, total), order, msg) for t, order, msg in events]
    previous = 0
    for tick, _, message in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(message.copy(time=tick-previous))
        previous = tick
    track.append(mido.MetaMessage('end_of_track', time=max(0, total-previous)))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(output))
    return boundary_ticks + [total]

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))

def preserve_prompt(source_path, generated_path, boundary_seconds, generated_boundary_tick):
    """Keep original MIDI events before boundary; append tail in musical ticks.

    Includes source tempo/control events. Notes crossing the boundary are ended
    there to avoid a dangling note-on. Tail tempo events begin at that boundary.
    """
    source = mido.MidiFile(str(source_path))
    generated = mido.MidiFile(str(generated_path))
    events, active = [], set()
    seconds, tick, tempo = 0., 0, 500000
    cutoff = None
    for msg in mido.merge_tracks(source.tracks):
        next_seconds = seconds + mido.tick2second(msg.time, source.ticks_per_beat, tempo)
        if next_seconds >= boundary_seconds:
            cutoff = tick + round(mido.second2tick(boundary_seconds-seconds, source.ticks_per_beat, tempo))
            break
        seconds, tick = next_seconds, tick+msg.time
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        if msg.type != 'end_of_track':
            events.append((tick, 0, msg.copy(time=0)))
        if msg.type == 'note_on' and msg.velocity > 0:
            active.add((msg.channel,msg.note))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            active.discard((msg.channel,msg.note))
    if cutoff is None:
        cutoff = tick + round(mido.second2tick(boundary_seconds-seconds,source.ticks_per_beat,tempo))
    for channel,note in sorted(active):
        events.append((cutoff,1,mido.Message('note_off',channel=channel,note=note,velocity=0)))
    # End sustain from the prompt and select piano for the generated channel.
    events.append((cutoff,1,mido.Message('control_change',control=64,value=0,channel=0)))
    events.append((cutoff,1,mido.Message('program_change',program=0,channel=0)))
    tick = 0
    tail_active = set()
    final_tick = cutoff
    for msg in mido.merge_tracks(generated.tracks):
        tick += msg.time
        if tick < generated_boundary_tick:
            continue
        dest = cutoff + round((tick-generated_boundary_tick)*source.ticks_per_beat/generated.ticks_per_beat)
        final_tick = max(final_tick,dest)
        if msg.type == 'end_of_track':
            continue
        # Ignore note-offs from quantized prefix notes crossing the seam.
        if msg.type == 'note_on' and msg.velocity > 0:
            tail_active.add((msg.channel,msg.note))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel,msg.note)
            if key not in tail_active:
                continue
            tail_active.discard(key)
        events.append((dest,2,msg.copy(time=0)))
    out = mido.MidiFile(ticks_per_beat=source.ticks_per_beat)
    track = mido.MidiTrack()
    out.tracks.append(track)
    previous = 0
    for tick,_,msg in sorted(events,key=lambda e:(e[0],e[1])):
        track.append(msg.copy(time=tick-previous))
        previous=tick
    track.append(mido.MetaMessage('end_of_track',time=max(0,final_tick-previous)))
    out.save(str(generated_path))
