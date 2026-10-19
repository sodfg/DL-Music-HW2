"""Beat-grid MIDI-like: NoteOn/NoteOff, Velocity, TimeShift, and Tempo.

Use the SAME quantized notes/tempi as the existing annotation-based encoder.
No Bar/Pos/Beats tokens are fed to the model. Original bar lengths are sidecar
metadata for roundtrip and metrics only. TimeShift units are 1/16 annotated beat,
not milliseconds. Same-pitch overlaps retain every MIDI on/off event.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import sys

import mido

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.music import PPB, MAX_DURATION, dump, vocabulary as bar_vocabulary

MAX_SHIFT = 256


def vocabulary():
    tokens = ['<PAD>', '<BOS>', '<EOS>']
    for prefix, values in [('Tempo', range(30, 211, 5)), ('Vel', range(1, 128, 4)),
                           ('NoteOn', range(128)), ('NoteOff', range(128)),
                           ('TimeShift', range(1, MAX_SHIFT+1))]:
        tokens.extend(f'{prefix}_{v}' for v in values)
    return {t: i for i, t in enumerate(tokens)}


def canonical_events(tokens):
    """Return baseline-equivalent absolute-grid events, including clipped offs."""
    events, bars = [], []
    start = beats = 0
    pos = 0
    pitch = None
    vel = 81
    for token in tokens:
        if token == 'Bar':
            start += beats*PPB
            beats = 0
        elif token.startswith('Beats_'):
            beats = int(token[6:])
            bars.append(beats)
        elif token.startswith('Tempo_'):
            events.append((start, 'tempo', int(token[6:]), 0))
        elif token.startswith('Pos_'):
            pos = int(token[4:])
        elif token.startswith('Pitch_'):
            pitch = int(token[6:])
        elif token.startswith('Vel_'):
            vel = int(token[4:])
        elif token.startswith('Dur_') and pitch is not None:
            onset = start+pos
            events.extend([(onset, 'on', pitch, vel), (onset+int(token[4:]), 'off', pitch, 0)])
            pitch = None
    total = start+beats*PPB
    priority = {'tempo': 0, 'off': 1, 'on': 2}
    # Stable ordering matches baseline.music.decode, including simultaneous notes.
    events = sorted(((min(t, total), k, p, v) for t,k,p,v in events), key=lambda e: (e[0], priority[e[1]]))
    if not bars or total <= 0:
        raise ValueError('No annotated bars')
    return events, bars, total


def from_canonical(tokens):
    events, bars, total = canonical_events(tokens)
    result, clock = ['<BOS>'], 0
    for tick, kind, value, velocity in events:
        while clock < tick:
            step = min(MAX_SHIFT, tick-clock)
            result.append(f'TimeShift_{step}')
            clock += step
        if kind == 'tempo':
            result.append(f'Tempo_{value}')
        elif kind == 'on':
            result.extend([f'Vel_{velocity}', f'NoteOn_{value}'])
        else:
            result.append(f'NoteOff_{value}')
    while clock < total:
        step = min(MAX_SHIFT, total-clock)
        result.append(f'TimeShift_{step}')
        clock += step
    return result+['<EOS>'], bars


def read_events(tokens):
    if not tokens or tokens[0] != '<BOS>' or tokens[-1] != '<EOS>':
        raise ValueError('Missing BOS/EOS')
    clock, velocity, pending = 0, 81, False
    active = Counter()
    events = []
    vocab = vocabulary()
    for token in tokens[1:-1]:
        if token not in vocab or token.startswith('<'):
            raise ValueError(f'Invalid event token: {token}')
        prefix, raw = token.split('_')
        value = int(raw)
        if pending and prefix != 'NoteOn':
            raise ValueError('Velocity must be followed by NoteOn')
        if prefix == 'TimeShift':
            clock += value
        elif prefix == 'Tempo':
            events.append((clock, 'tempo', value, 0))
        elif prefix == 'Vel':
            velocity, pending = value, True
        elif prefix == 'NoteOn':
            if not pending:
                raise ValueError('NoteOn requires an explicit velocity')
            active[value] += 1
            pending = False
            events.append((clock, 'on', value, velocity))
        elif prefix == 'NoteOff':
            if active[value] <= 0:
                raise ValueError('NoteOff without an active note')
            active[value] -= 1
            events.append((clock, 'off', value, 0))
    if pending or any(active.values()):
        raise ValueError('Unfinished note events')
    return events, clock


def decode(tokens, output):
    events, total = read_events(tokens)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack([mido.Message('program_change', program=0)])
    midi.tracks.append(track)
    previous = 0
    for grid, kind, value, velocity in events:
        tick = grid*(480//PPB)
        if kind == 'tempo':
            msg = mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(value))
        else:
            msg = mido.Message('note_on' if kind == 'on' else 'note_off', note=value, velocity=velocity)
        track.append(msg.copy(time=tick-previous))
        previous = tick
    track.append(mido.MetaMessage('end_of_track', time=total*(480//PPB)-previous))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(output))


def metric_tokens(tokens, beats_per_bar):
    """Export an onset-preserving baseline view solely for H1/H4/GS evaluation.

    These metrics use note-on pitches and positions, not durations. FIFO pairs
    overlapping equal pitches only for this view; rendered audio uses raw events.
    Never re-tokenize MIDI via unreliable embedded meter/tempo estimation.
    """
    events, total = read_events(tokens)
    if total != sum(beats_per_bar)*PPB:
        raise ValueError('Timeline/bar metadata mismatch')
    starts, now = [], 0
    for beats in beats_per_bar:
        starts.append(now)
        now += beats*PPB
    import bisect
    notes, tempos = [[] for _ in starts], {}
    active = defaultdict(deque)
    for tick, kind, value, velocity in events:
        if kind == 'tempo':
            if tick not in starts:
                raise ValueError('Expected baseline-matched tempo at bar boundaries')
            tempos[tick] = value
        elif kind == 'on':
            b = bisect.bisect_right(starts, tick)-1
            if b < 0 or tick >= total:
                raise ValueError('Note onset outside piece')
            note = [tick-starts[b], value, velocity, 1]
            notes[b].append(note)
            active[value].append((tick, note))
        else:
            onset, note = active[value].popleft()
            note[3] = max(1, min(MAX_DURATION, tick-onset))
    output = ['<BOS>']
    for start, beats, bar in zip(starts, beats_per_bar, notes):
        output.extend(['Bar', f'Beats_{beats}', f'Tempo_{tempos.get(start,120)}'])
        for pos, pitch, vel, duration in sorted(bar):
            output.extend([f'Pos_{pos}', f'Pitch_{pitch}', f'Vel_{vel}', f'Dur_{duration}'])
    return output+['<EOS>']


def prepare(source, output, expected=896):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Choose a separate data output directory')
    if (output/'sequences.json').exists():
        raise ValueError('Prepared data already exists; retain it or choose a new output')
    songs = json.loads((source/'sequences.json').read_text())
    if len(songs) != expected or len({s['song_name'] for s in songs}) != expected:
        raise ValueError('Wrong number of distinct source songs')
    vocab, converted, metadata = vocabulary(), [], []
    totals = Counter()
    from baseline.audio.render_audio import sha256
    for song in songs:
        tokens, bars = from_canonical(song['tokens'])
        original, _, total = canonical_events(song['tokens'])
        events, restored_total = read_events(tokens)
        if original != events or total != restored_total:
            raise ValueError(f'Event roundtrip failed: {song["song_name"]}')
        if set(tokens)-vocab.keys():
            raise ValueError('Unknown tokens')
        # Verify the metric view preserves every per-bar onset/pitch/velocity.
        def onsets(seq):
            return [t for t in seq if not t.startswith('Dur_')]
        if onsets(metric_tokens(tokens, bars)) != onsets(song['tokens']):
            raise ValueError(f'Onset metric view changed: {song["song_name"]}')
        converted.append({'song_name': song['song_name'], 'tokens': tokens})
        metadata.append({'song_name': song['song_name'], 'beats_per_bar': bars,
                         'baseline_tokens': len(song['tokens']), 'midi_like_tokens': len(tokens),
                         'grid_steps': total})
        totals.update({'baseline_tokens': len(song['tokens']), 'midi_like_tokens': len(tokens),
                       'beats': sum(bars), 'notes': sum(e[1]=='on' for e in events), 'bars': len(bars)})
        if len(converted)%100 == 0:
            print(f'Event and metric roundtrip: {len(converted)}/{expected}', flush=True)
    dump(output/'tok2idx.json', vocab)
    dump(output/'bar_metadata.json', metadata)
    dump(output/'summary.json', {'representation': 'beat-grid MIDI-like', 'songs': len(songs),
        'vocab_size': len(vocab), 'source_sequences_sha256': sha256(source/'sequences.json'),
        'source_vocab_sha256': sha256(source/'tok2idx.json'), 'grid_steps_per_beat': PPB,
        'time_shift_max': MAX_SHIFT, 'totals': dict(totals),
        'baseline_approx_beats_per_1024_tokens': 1024*totals['beats']/totals['baseline_tokens'],
        'midi_like_approx_beats_per_1024_tokens': 1024*totals['beats']/totals['midi_like_tokens'],
        'roundtrip': 'All quantized MIDI note/tempo events and piece durations match; all metric onsets match'})
    decode(converted[0]['tokens'], output/'roundtrip.mid')
    # Completion marker last, so incomplete preparation cannot look complete.
    dump(output/'sequences.json', converted)
    print(f'Prepared {len(songs)} songs, vocabulary={len(vocab)}; all event roundtrips passed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/baseline'))
    p.add_argument('--output', type=Path, default=Path('data/midi_like'))
    p.add_argument('--expected-songs', type=int, default=896)
    a = p.parse_args()
    prepare(a.source, a.output, a.expected_songs)
