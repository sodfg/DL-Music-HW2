import csv
import json
from pathlib import Path
import tempfile
import unittest

from run_si_study import GROUPS,write_summary


class SIStudyTests(unittest.TestCase):
    def fixture(self,root):
        plan={'duration_seconds':48.,'reference_eligible':895,'reference_seed':42,'groups':{}}
        for group in GROUPS:
            path=root/'groups'/group
            path.mkdir(parents=True)
            (path/'protocol.json').write_text(json.dumps({'duration_seconds':48,'render_settings':{'gain':.025}}))
            names=[f'song_{i:03d}' for i in range(20)]
            plan['groups'][group]=[{'song_name':name} for name in names]
            rows=[{'group':group,'song_name':name,'duration_seconds':48,
                   'SI_short':.2 if group=='real' else .3,'SI_mid':.1,'SI_long':0} for name in names]
            with (path/'per_piece.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return plan

    def test_complete_groups_and_mean_differences(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plan=self.fixture(root)
            write_summary(root,plan)
            with (root/'summary.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([int(r['song_count']) for r in rows],[20]*4)
            self.assertAlmostEqual(float(rows[1]['SI_short_abs_diff_from_reference']),.1)
            self.assertIn('抽樣', (root/'summary.md').read_text())

    def test_mismatched_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plan=self.fixture(root)
            (root/'groups'/'t12'/'protocol.json').write_text(json.dumps({'duration_seconds':32}))
            with self.assertRaises(ValueError):
                write_summary(root,plan)
            self.assertFalse((root/'summary.csv').exists())


if __name__=='__main__':
    unittest.main()
