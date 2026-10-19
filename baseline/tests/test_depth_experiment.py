import tempfile
from pathlib import Path
import csv
import unittest
from baseline.experiments.run_depth_experiment import train_command,metric_summary,verify_plan

class DepthTests(unittest.TestCase):
    def test_controlled_training_command(self):
        reference={'data':'data/baseline','steps':20000,'batch_size':8,'context':1024,
                   'width':512,'heads':8,'lr':.0003,'eval_every':500,'eval_batches':20,
                   'seed':42,'validation_fraction':0.,'layers':6}
        args=train_command('python3',Path('train.py'),reference,Path('runs/depth/train'))
        options=dict(zip(args[2::2],args[3::2]))
        self.assertEqual(options['--layers'],'3')
        self.assertEqual(options['--steps'],'20000')
        self.assertEqual(options['--validation-fraction'],'0.0')
        self.assertEqual(options['--output'],'runs/depth/train')
        self.assertNotIn('--resume',options)

    def test_historical_plan_is_preserved_for_inspection_only(self):
        original={'data_sha256':'same-data','source_sha256':{'evaluate.py':'old-code'}}
        candidate={'data_sha256':'same-data','source_sha256':{'evaluate.py':'new-code'}}
        self.assertIs(verify_plan(original,candidate,inspection=True),original)
        self.assertEqual(original['source_sha256']['evaluate.py'],'old-code')
        with self.assertRaises(ValueError):
            verify_plan(original,candidate,inspection=False)
        with self.assertRaises(ValueError):
            verify_plan(original,{**candidate,'data_sha256':'different-data'},inspection=True)

    def test_metrics_require_complete_finite_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'metric.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f)
                w.writerow(['song_name','H1 (Pitch Entropy 1-Bar)','H4 (Pitch Entropy 4-Bar)','GS (Groove Similarity)'])
                for i in range(20):
                    w.writerow([str(i),2,2.5,.9])
            self.assertEqual(metric_summary(p)['H4_mean'],2.5)
            text=p.read_text().replace('2.5','nan',1)
            p.write_text(text)
            with self.assertRaises(ValueError):
                metric_summary(p)

if __name__=='__main__':
    unittest.main()
