"""Musical timing/overlap preservation and exact-length grammar checks."""
from collections import Counter
from pathlib import Path
import random
import tempfile
import unittest

import mido

from baseline.music import decode as baseline_decode
from baseline.representations.midi_like import (from_canonical, read_events, canonical_events,
                                                metric_tokens, decode, vocabulary)
from baseline.representations.generate_midi_like import EventState


class MidiLikeTests(unittest.TestCase):
    def source(self):
        return ['<BOS>', 'Bar', 'Beats_12', 'Tempo_90',
                'Pos_0', 'Pitch_60', 'Vel_81', 'Dur_100',
                'Pos_16', 'Pitch_60', 'Vel_41', 'Dur_4',
                'Pos_160', 'Pitch_67', 'Vel_125', 'Dur_48',
                'Bar', 'Beats_4', 'Tempo_120',
                'Pos_0', 'Pitch_64', 'Vel_81', 'Dur_256', '<EOS>']

    def test_midi_event_roundtrip_preserves_overlaps_tempo_and_crop(self):
        source = self.source()
        tokens, bars = from_canonical(source)
        self.assertEqual(bars, [12,4])
        self.assertFalse(any(t.startswith(('Pos_', 'Beats_')) or t=='Bar' for t in tokens))
        self.assertFalse(set(tokens)-vocabulary().keys())
        events, total = read_events(tokens)
        expected, _, duration = canonical_events(source)
        self.assertEqual((events,total), (expected,duration))
        def timeline(path):
            clock, result = 0, []
            for msg in mido.merge_tracks(mido.MidiFile(str(path)).tracks):
                clock += msg.time
                result.append((clock,msg.copy(time=0)))
            return result
        with tempfile.TemporaryDirectory() as temp:
            old, new = Path(temp)/'old.mid', Path(temp)/'new.mid'
            baseline_decode(source,old)
            decode(tokens,new)
            self.assertEqual(timeline(old),timeline(new))
        view=metric_tokens(tokens,bars)
        without_durations=lambda seq:[t for t in seq if not t.startswith('Dur_')]
        self.assertEqual(without_durations(source),without_durations(view))

    def test_invalid_note_events_and_wrong_bar_metadata_fail(self):
        for tokens in (['<BOS>','NoteOff_60','<EOS>'],
                       ['<BOS>','Vel_81','NoteOn_60','<EOS>'],
                       ['<BOS>','Vel_81','TimeShift_1','<EOS>']):
            with self.assertRaises(ValueError):
                read_events(tokens)
        tokens,_=from_canonical(self.source())
        with self.assertRaises(ValueError):
            metric_tokens(tokens,[4,4])

    def test_random_grammar_runs_terminate_and_preserve_every_bar(self):
        for seed in range(12):
            rng=random.Random(seed)
            state=EventState(bars=5,beats=4,max_events=3)
            tokens=['<BOS>']
            for _ in range(2000):
                token=rng.choice(state.choices())
                state.advance(token)
                tokens.append(token)
                if token=='<EOS>':break
            self.assertEqual(tokens[-1],'<EOS>')
            events,total=read_events(tokens)
            self.assertEqual(total,5*64)
            onsets=Counter(t//64 for t,k,_,_ in events if k=='on')
            self.assertEqual(set(onsets),set(range(5)))
            self.assertTrue(all(1<=n<=3 for n in onsets.values()))
            self.assertEqual(metric_tokens(tokens,[4]*5).count('Bar'),5)
            with self.assertRaises(ValueError):
                state.advance('TimeShift_1')


if __name__ == '__main__':
    unittest.main()
