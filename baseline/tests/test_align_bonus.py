"""Annotation meters may vary; original sounding events stay in seconds."""
from pathlib import Path
import tempfile
import unittest
import mido
from baseline.experiments.align_bonus import align_bonus


class AlignBonusTests(unittest.TestCase):
    def fixture(self,root,overhang=.0005):
        midi=mido.MidiFile(ticks_per_beat=10000);track=mido.MidiTrack();midi.tracks.append(track)
        track.extend([mido.MetaMessage('track_name',name='MELODY'),mido.MetaMessage('set_tempo',tempo=500000),
            mido.MetaMessage('time_signature',numerator=4,denominator=4),
            mido.Message('note_on',note=60,velocity=90),
            mido.Message('note_off',note=60,time=round((6+overhang)*20000)),mido.MetaMessage('end_of_track')])
        source=root/'source.mid';midi.save(source)
        beat=root/'beat.txt';beat.write_text('\n'.join(f'{i*.5:.6f} 1 {int(i in (0,6,8,12))}' for i in range(13)))
        return source,beat

    def test_variable_meters_preserve_timing_and_small_overhang(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,beat=self.fixture(root);out=root/'aligned.mid'
            row=align_bonus(source,beat,out,bars=3)
            self.assertEqual(row['beats_per_bar'],[6,2,4])
            self.assertEqual(row['bounded_bars'],3)
            self.assertLess(row['max_event_time_error_seconds'],.00005)
            meters=[m.numerator for t in mido.MidiFile(out).tracks for m in t if m.type=='time_signature']
            self.assertEqual(meters,[6,2,4])
            self.assertAlmostEqual(row['final_beat_extension_seconds'],.0005)

    def test_excess_overhang_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,beat=self.fixture(root,.01);out=root/'bad.mid'
            with self.assertRaises(ValueError):align_bonus(source,beat,out,bars=3)
            self.assertFalse(out.exists())
