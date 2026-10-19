import csv
from pathlib import Path
import tempfile
import unittest
from baseline.evaluation.summarize_metrics import METRICS, summarize_file, build_summary


class SummaryTests(unittest.TestCase):
    def write_csv(self, folder, name, rows):
        path=Path(folder)/name
        with path.open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['song_name',*METRICS.values()])
            writer.writerows(rows)
        return path

    def test_mean_sample_std_and_reference_difference(self):
        with tempfile.TemporaryDirectory() as d:
            real=self.write_csv(d,'real.csv',[['a',1,2,.5],['b',3,4,.7]])
            generated=self.write_csv(d,'t08.csv',[['c',2,2,.4],['d',4,2,.4]])
            rows=build_summary([real,generated])
            self.assertEqual(rows[0]['H4_mean'],3.)
            self.assertAlmostEqual(rows[0]['H4_std'],2**.5)
            self.assertEqual(rows[1]['H4_abs_diff_from_reference'],1.)
            self.assertAlmostEqual(rows[1]['GS_abs_diff_from_reference'],.2)

    def test_missing_values_have_per_metric_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.write_csv(d,'x.csv',[['a','nan',2,.5],['b','',4,'inf']])
            row=summarize_file(path)
            self.assertEqual(row['song_count'],2)
            self.assertEqual(row['H1_valid_count'],0)
            self.assertEqual(row['H1_missing_count'],2)
            self.assertIsNone(row['H1_mean'])
            self.assertEqual(row['GS_valid_count'],1)
            self.assertIsNone(row['GS_std'])
            self.assertEqual(row['H4_mean'],3.)

    def test_duplicate_songs_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=self.write_csv(d,'x.csv',[['a',1,2,.5],['a',1,2,.5]])
            with self.assertRaises(ValueError):
                summarize_file(path)


if __name__ == '__main__':
    unittest.main()
