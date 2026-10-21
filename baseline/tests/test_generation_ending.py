"""Regression checks for last-bar pileup and simultaneous duplicate notes."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate import allowed
from baseline.music import vocabulary


class EndingTests(unittest.TestCase):
    def setUp(self):
        self.vocab = vocabulary()
        self.tokens = ['<BOS>', 'Bar', 'Beats_4', 'Tempo_120',
                       'Pos_63', 'Pitch_96', 'Vel_65', 'Dur_8']

    def test_final_boundary_is_available_before_cap(self):
        choices = allowed(self.tokens, self.vocab, 1, 4, 128, 'boundary-v2')
        self.assertIn('Bar', choices)
        self.assertIn('<EOS>', choices)
        self.assertNotIn('Bar', allowed(self.tokens, self.vocab, 1, 4, 128))

    def test_same_onset_pitch_is_blocked_but_chords_allowed(self):
        choices = allowed(self.tokens+['Pos_63'], self.vocab, 1, 4, 128, 'boundary-v2')
        self.assertNotIn('Pitch_96', choices)
        self.assertIn('Pitch_60', choices)

    def test_repeated_pitch_at_later_time_remains_allowed(self):
        tokens = [t.replace('Pos_63', 'Pos_0') for t in self.tokens]
        choices = allowed(tokens+['Pos_16'], self.vocab, 1, 4, 128, 'boundary-v2')
        self.assertIn('Pitch_96', choices)

    def test_saturated_position_cannot_create_empty_pitch_choices(self):
        tokens = self.tokens[:4]
        for pitch in range(128):
            tokens += ['Pos_63', f'Pitch_{pitch}', 'Vel_65', 'Dur_8']
        choices = allowed(tokens, self.vocab, 1, 4, 256, 'boundary-v2')
        self.assertEqual(choices, ['Bar', '<EOS>'])
