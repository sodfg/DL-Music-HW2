"""Mixed prompt/tail clocks must align without changing performed events."""
from pathlib import Path
import tempfile
import unittest
import mido
from baseline.experiments.align_task_b import align_midi


class AlignTaskBTests(unittest.TestCase):
    def fixture(self,root):
        prompt=root/'prompt.mid';mixed=root/'mixed.mid';beat=root/'beat_midi.txt'
        source=mido.MidiFile(ticks_per_beat=240);t=mido.MidiTrack();source.tracks.append(t)
        t.extend([mido.MetaMessage('set_tempo',tempo=500000),mido.MetaMessage('time_signature',numerator=4,denominator=4),
                  mido.Message('program_change',program=0),mido.Message('control_change',control=7,value=90),
                  mido.Message('note_on',note=60,velocity=83,time=121),
                  mido.Message('note_off',note=60,time=122),mido.MetaMessage('end_of_track',time=2637)])
        source.save(prompt)
        song=mido.MidiFile(ticks_per_beat=240);t=mido.MidiTrack();song.tracks.append(t)
        t.extend([m.copy() for m in source.tracks[0][:-1]])
        t.extend([mido.MetaMessage('set_tempo',tempo=750000,time=2637),
                  mido.Message('note_on',note=65,velocity=92,time=0),mido.Message('note_off',note=65,time=211),
                  mido.MetaMessage('end_of_track',time=1709)])
        song.save(mixed)
        beat.write_text('\n'.join(f'{i*.75:.6f} 1 {1 if i%4==0 else 0}' for i in range(9)))
        return mixed,prompt,beat

    def test_mixed_clock_aligns_to_four_bars_and_preserves_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mixed,prompt,beat=self.fixture(root)
            r=align_midi(mixed,prompt,beat,root/'aligned.mid',prompt_bars=2,total_bars=4)
            self.assertEqual(r['old_display_bars'],5)
            self.assertEqual(r['new_display_bars'],4)
            self.assertLess(r['max_event_time_error_seconds'],.0002)
            self.assertAlmostEqual(mido.MidiFile(root/'aligned.mid').length,12.,places=5)
            self.assertEqual(r['performance_events_verified'],6)

    def test_refuses_to_overwrite_and_rejects_wrong_tail_length(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);mixed,prompt,beat=self.fixture(root);out=root/'existing.mid';out.write_bytes(b'keep')
            with self.assertRaises(ValueError):align_midi(mixed,prompt,beat,out,prompt_bars=2,total_bars=4)
            self.assertEqual(out.read_bytes(),b'keep')
            with self.assertRaises(ValueError):align_midi(mixed,prompt,beat,root/'bad.mid',prompt_bars=2,total_bars=5)
