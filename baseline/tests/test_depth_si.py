"""Verify complete, protocol-matched depth SI reports and reject bad inputs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline.experiments.run_depth_si import GROUPS, METRICS, write_report


class DepthSITests(unittest.TestCase):
    def fixture(self, root):
        settings = {'gain': .025}
        protocol = {'render_settings': settings, 'start_seconds': 0.,
                    'duration_seconds': 48., 'bounds_seconds': [3., 8., 15.]}
        plan = {'render_settings': settings, 'bounds_seconds': [3., 8., 15.],
                'reference_seed': 42, 'reference_eligible': 895, 'groups': {}}
        for n, group in enumerate(GROUPS):
            folder = root/'groups'/group
            folder.mkdir(parents=True)
            (folder/'protocol.json').write_text(json.dumps(protocol))
            rows = []
            plan['groups'][group] = []
            for i in range(20):
                name = f'sample_{i:03d}'
                plan['groups'][group].append({'song_name': name, 'source_sha256': 'midi'})
                rows.append({'group': group, 'song_name': name, 'duration_seconds': 48,
                             'start_seconds': 0, 'wav_sha256': 'wav',
                             **{m: .1 + .01*n for m in METRICS}})
            with (folder/'per_piece.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return plan, {'settings': settings, 'audio': {'clipped_samples': 0},
                      'source_sha256': 'midi', 'wav_sha256': 'wav'}

    def test_complete_report_and_reference_distances(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, audio = self.fixture(root)
            with patch('baseline.experiments.run_depth_si.verified_audio_record', return_value=audio):
                write_report(root, plan)
            with (root/'comparison.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['song_count'] for r in rows], ['20']*3)
            self.assertAlmostEqual(float(rows[2]['SI_long_abs_diff_from_reference']), .02)
            self.assertIn('48 秒', (root/'comparison.md').read_text())

    def test_mismatched_protocol_and_audio_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, audio = self.fixture(root)
            audio['source_sha256'] = 'changed'
            with patch('baseline.experiments.run_depth_si.verified_audio_record', return_value=audio):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    write_report(root, plan)
            plan, audio = self.fixture_existing(root)
            path = root/'groups/layers3/protocol.json'
            protocol = json.loads(path.read_text())
            protocol['duration_seconds'] = 32
            path.write_text(json.dumps(protocol))
            with patch('baseline.experiments.run_depth_si.verified_audio_record', return_value=audio):
                with self.assertRaisesRegex(ValueError, 'protocols differ'):
                    write_report(root, plan)

    def fixture_existing(self, root):
        import shutil
        shutil.rmtree(root/'groups')
        return self.fixture(root)


if __name__ == '__main__':
    unittest.main()
