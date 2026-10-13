"""Run with: python -m unittest discover -s baseline -p 'test_*.py'."""
import tempfile
from pathlib import Path
import unittest
import torch
import pretty_midi
import mido
from model import Config, MusicTransformer
from music import decode, encode, preserve_prompt

class PipelineTests(unittest.TestCase):
    def test_attention_cannot_see_future(self):
        torch.manual_seed(1)
        model = MusicTransformer(Config(32, context=16, width=16, layers=2, heads=4, dropout=0.)).eval()
        a = torch.tensor([[1,2,3,4,5]])
        b = torch.tensor([[1,2,3,9,8]])
        with torch.no_grad():
            torch.testing.assert_close(model(a)[:,:3],model(b)[:,:3],atol=1e-6,rtol=1e-5)

    def test_long_bar_and_tempo_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'song.mid'
            tokens=['<BOS>','Bar','Beats_12','Tempo_120','Pos_184','Pitch_60','Vel_81','Dur_4',
                    'Bar','Beats_4','Tempo_60','Pos_0','Pitch_64','Vel_77','Dur_16','<EOS>']
            ticks = decode(tokens,out)
            midi = pretty_midi.PrettyMIDI(str(out))
            notes = midi.instruments[0].notes
            self.assertEqual([n.pitch for n in notes],[60,64])
            self.assertAlmostEqual(notes[0].start,5.75,places=5)
            self.assertAlmostEqual(notes[1].start,6.,places=5)
            self.assertAlmostEqual(notes[1].end,7.,places=5)
            self.assertEqual(ticks,[0,5760,7680])

    def test_prompt_events_preserved_and_tail_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            source=d/'source.mid'
            generated=d/'generated.mid'
            original=pretty_midi.PrettyMIDI(initial_tempo=90)
            inst=pretty_midi.Instrument(0)
            inst.notes=[pretty_midi.Note(99,60,.123,1.234),pretty_midi.Note(70,65,2.,4.)]
            original.instruments=[inst]
            original.write(str(source))
            ticks=decode(['Bar','Beats_4','Tempo_120','Pos_0','Pitch_50','Vel_81','Dur_128',
                          'Bar','Beats_4','Tempo_60','Pos_0','Pitch_72','Vel_81','Dur_16'],generated)
            preserve_prompt(source,generated,3.,ticks[1])
            result=pretty_midi.PrettyMIDI(str(generated))
            notes=sorted([n for i in result.instruments for n in i.notes],key=lambda n:n.start)
            reread=pretty_midi.PrettyMIDI(str(source)).instruments[0].notes
            self.assertAlmostEqual(notes[0].start,reread[0].start,places=5)
            self.assertAlmostEqual(notes[0].end,reread[0].end,places=5)
            self.assertEqual(notes[0].velocity,99)
            self.assertAlmostEqual(notes[1].end,3.,places=3)
            self.assertEqual(notes[2].pitch,72)
            self.assertAlmostEqual(notes[2].start,3.,places=3)
            self.assertAlmostEqual(notes[2].end,4.,places=3)
            self.assertAlmostEqual(result.get_end_time(),4.,places=3)

if __name__ == '__main__':
    unittest.main()
