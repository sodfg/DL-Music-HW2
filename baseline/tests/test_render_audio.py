from array import array
from pathlib import Path
import tempfile
import unittest
import wave

import mido

from baseline.audio.render_audio import collect_files, inspect_wav, piano_copy, sha256


class RenderTests(unittest.TestCase):
    def test_piano_copy_preserves_note_ticks_and_source(self):
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d)/'source.mid', Path(d)/'piano.mid'
            midi = mido.MidiFile(type=0)
            midi.tracks.append(mido.MidiTrack([
                mido.Message('program_change', program=40, time=0),
                mido.Message('control_change', control=0, value=5, time=20),
                mido.Message('note_on', note=60, velocity=90, time=10),
                mido.Message('program_change', program=41, time=3),
                mido.Message('note_off', note=60, time=47),
            ]))
            midi.save(str(source))
            original_hash = sha256(source)
            piano_copy(source, output)
            self.assertEqual(original_hash, sha256(source))
            def note_ticks(path):
                tick, notes = 0, []
                for msg in mido.merge_tracks(mido.MidiFile(str(path)).tracks):
                    tick += msg.time
                    if msg.type in ('note_on','note_off'):
                        notes.append((tick,msg.type,msg.note,msg.velocity))
                return notes
            self.assertEqual(note_ticks(source),note_ticks(output))
            self.assertTrue(all(msg.program==0 for msg in mido.MidiFile(str(output)).tracks[0]
                                if msg.type=='program_change'))

    def test_wav_levels_and_silence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'audio.wav'
            def write(samples):
                with wave.open(str(p),'wb') as stream:
                    stream.setnchannels(2)
                    stream.setsampwidth(2)
                    stream.setframerate(8000)
                    stream.writeframes(array('h',samples).tobytes())
            write([0,32767,-32768,1000])
            stats=inspect_wav(p)
            self.assertEqual(stats['clipped_samples'],2)
            self.assertEqual(stats['duration_seconds'],2/8000)
            self.assertEqual(stats['peak_amplitude'],1.)
            write([0,0,0,0])
            with self.assertRaises(ValueError):
                inspect_wav(p)

    def test_recursive_paths_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'t08').mkdir()
            (root/'t08'/'sample.mid').touch()
            (root/'top.mid').touch()
            self.assertEqual(len(collect_files(root)[1]),1)
            self.assertEqual(len(collect_files(root,True)[1]),2)
            (root/'top.midi').touch()
            with self.assertRaises(ValueError):
                collect_files(root)


if __name__ == '__main__':
    unittest.main()
