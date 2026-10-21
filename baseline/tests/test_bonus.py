"""Meaningful conditional attention, no-leakage and melody/time alignment tests."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import mido
import torch

from baseline.bonus.data import vocabulary,condition,load_prompt,training_pair
from baseline.bonus.model import Config,AccompanimentTransformer
from baseline.bonus.generate import write_accompaniment,allowed


class BonusTests(unittest.TestCase):
    def test_decoder_cannot_see_future_but_uses_melody(self):
        torch.manual_seed(9)
        model=AccompanimentTransformer(Config(40,32,32,24,1,1,4,0.)).eval()
        src=torch.tensor([[1,8,9,2]])
        a=torch.tensor([[1,10,11,12,2]]);b=torch.tensor([[1,10,11,19,20]])
        with torch.no_grad():
            x=model(src,a);y=model(src,b);z=model(torch.tensor([[1,25,26,2]]),a)
        torch.testing.assert_close(x[:,:3],y[:,:3],rtol=1e-5,atol=1e-6)
        self.assertGreater(float((x-z).abs().max()),1e-5)
        with torch.no_grad():
            padded=model(torch.tensor([[1,8,9,2,0,0]]),a)
        torch.testing.assert_close(x,padded,rtol=1e-4,atol=1e-6)

    def test_complete_target_bars_not_truncated(self):
        vocab=vocabulary();bar=[vocab[t] for t in ['Bar','Beats_4','Tempo_120','Pos_0','Pitch_60','Vel_81','Dur_8']]
        song={'melody':[bar]*16,'piano':[bar]*16}
        source,target=training_pair(song,random.Random(3),vocab,18)
        self.assertEqual(len(source),16*7+4)
        self.assertEqual(target[-1],vocab['<EOS>'])
        self.assertLessEqual(len(target)-1,18)
        self.assertEqual((len(target)-2)%len(bar),0)
        with self.assertRaises(ValueError):training_pair(song,random.Random(3),vocab,2)
        with self.assertRaises(ValueError):condition([bar]*16,15,4,vocab)

    def prompt(self,folder):
        starts=[0]+[6+4*i for i in range(16)]
        rows=[]
        for beat in range(starts[-1]+1):
            bar=max(i for i,start in enumerate(starts) if start<=beat)
            rows.append(f'{beat*.5} 1 {beat-starts[bar]+1}')
        (folder/'beat_midi.txt').write_text('\n'.join(rows)+'\n')
        midi=mido.MidiFile(ticks_per_beat=480)
        track=mido.MidiTrack([mido.MetaMessage('track_name',name='MELODY'),
            mido.MetaMessage('set_tempo',tempo=500000),mido.Message('program_change',program=0),
            mido.Message('note_on',note=60,velocity=89,time=0),
            mido.Message('note_off',note=60,velocity=0,time=960),
            mido.MetaMessage('end_of_track',time=starts[-1]*480-960)])
        midi.tracks.append(track);midi.save(str(folder/'melody.mid'))
        # Invalid GT files detect any accidental attempt to use them for conditioning.
        (folder/'piano.mid').write_bytes(b'GT MUST NOT BE READ')
        (folder/'chord_midi.txt').write_text('GT MUST NOT BE READ')

    def test_writer_preserves_melody_and_six_beat_first_bar_without_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.prompt(folder)
            _,bars,_,_,sizes=load_prompt(folder)
            self.assertEqual(sizes,[6]+[4]*15)
            tokens=['<BOS>']
            for n in sizes:tokens+=['Bar',f'Beats_{n}','Tempo_120','Pos_0','Pitch_48','Vel_81','Dur_8']
            tokens+=['<EOS>'];output=folder/'out.mid'
            record=write_accompaniment(folder,tokens,output)
            original=mido.MidiFile(str(folder/'melody.mid'));generated=mido.MidiFile(str(output))
            self.assertEqual(generated.tracks[:len(original.tracks)],original.tracks)
            self.assertEqual(record['bounded_bars'],16)
            self.assertEqual(record['generated_notes'],16)
            self.assertAlmostEqual(record['end_seconds'],33.)
            self.assertEqual(set(record['condition_files']),{'melody.mid','beat_midi.txt'})

    def test_grammar_copies_condition_meter_and_tempo(self):
        vocab=vocabulary()
        self.assertEqual(allowed(['<BOS>','Bar'],vocab,[6],['Tempo_90'],8),['Beats_6'])
        self.assertEqual(allowed(['<BOS>','Bar','Beats_6'],vocab,[6],['Tempo_90'],8),['Tempo_90'])
        tokens=['<BOS>','Bar','Beats_6','Tempo_90','Pos_80','Pitch_60','Vel_81','Dur_8']
        choices=allowed(tokens,vocab,[6],['Tempo_90'],8)
        self.assertIn('Pos_95',choices);self.assertNotIn('Pos_79',choices);self.assertIn('<EOS>',choices)


if __name__=='__main__':unittest.main()
