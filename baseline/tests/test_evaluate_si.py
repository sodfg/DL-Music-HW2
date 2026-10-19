import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from baseline.evaluation.evaluate_si import si_values, verified_audio_record, file_hash


class SITests(unittest.TestCase):
    def test_official_duration_row_indexing(self):
        fitness=np.zeros((65,65))
        # 2 Hz, length in frames = row + 1.
        fitness[5,0]=.3       # 3 seconds; short lower boundary
        fitness[15,0]=.6     # 8 seconds; shared short/mid boundary
        fitness[29,0]=.8     # 15 seconds; shared mid/long boundary
        fitness[63,0]=.9     # 32 seconds
        fitness[64,0]=1.     # 32.5 seconds: excluded by actual audio duration
        result=si_values(fitness,2,32)
        self.assertEqual(result,{'SI_short':.6,'SI_mid':.8,'SI_long':.9})

    def test_insufficient_duration_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            si_values(np.zeros((30,30)),2,15)
        fitness=np.zeros((65,65))
        fitness[1,1]=np.nan
        with self.assertRaises(ValueError):
            si_values(fitness,2,32)

    def test_manifest_hash_verification(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'t08').mkdir()
            path=root/'t08'/'sample.wav'
            path.write_bytes(b'fixture')
            record={'wav_sha256':file_hash(path),'settings':{'gain':.1}}
            # Recursive renderer places one manifest at the root.
            (root/'render_manifest.json').write_text(json.dumps({'files':{'t08/sample.wav':record}}))
            self.assertEqual(verified_audio_record(path,root),record)
            path.write_bytes(b'modified')
            with self.assertRaises(ValueError):
                verified_audio_record(path,root)


if __name__=='__main__':
    unittest.main()
