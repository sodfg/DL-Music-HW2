"""Inference must reject checkpoints that request arbitrary Python execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


class UnsafeCheckpoint:
    def __reduce__(self):
        return print, ('UNSAFE_CHECKPOINT_EXECUTED',)


class CheckpointSafetyTests(unittest.TestCase):
    def test_inference_entrypoints_reject_executable_pickle(self):
        root = Path(__file__).resolve().parents[2]
        scripts = ('baseline/generate.py', 'baseline/bonus/generate.py',
                   'baseline/representations/generate_midi_like.py')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder / 'untrusted.pt'
            torch.save(UnsafeCheckpoint(), checkpoint)
            for index, script in enumerate(scripts):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, str(root / script), '--checkpoint', str(checkpoint),
                         '--output', str(folder / str(index)), '--device', 'cpu'],
                        cwd=root, capture_output=True, text=True, timeout=45)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('UNSAFE_CHECKPOINT_EXECUTED', result.stdout)
                    self.assertIn('Weights only load failed', result.stderr)
