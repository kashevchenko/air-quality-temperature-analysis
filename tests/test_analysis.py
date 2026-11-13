"""Invariant checks against the supplied dataset and generated outputs."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import ROOT, POLLUTANTS, load_and_clean, correlations, evaluate_models, regressions


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "data/air_quality_dataset.csv"
        cls.raw, cls.clean, cls.audit = load_and_clean(cls.path)

    def test_cleaning_and_integrity(self):
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.audit['input_sha256'])
        self.assertTrue(np.isfinite(self.clean).all().all())
        self.assertFalse(self.clean[POLLUTANTS].lt(0).any().any())
        self.assertFalse(self.clean.duplicated().any())
        self.assertEqual(len(self.raw), len(self.clean) + self.audit['rows_excluded_missing_or_nonfinite'] + self.audit['rows_excluded_negative_after_missing'] + self.audit['duplicate_analysis_rows_removed'])

    def test_missing_schema_and_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.csv'
            with self.assertRaises(FileNotFoundError): load_and_clean(path)
            pd.DataFrame({'wrong': [1]}).to_csv(path, index=False)
            with self.assertRaises(ValueError): load_and_clean(path)

    def test_correlations(self):
        corr = correlations(self.clean)
        for row in corr.itertuples():
            self.assertAlmostEqual(row.pearson_r, pearsonr(self.clean.Temperature, self.clean[row.pollutant]).statistic)
            self.assertGreaterEqual(row.pearson_q + 1e-15, row.pearson_p)
        self.assertTrue(corr[['pearson_q','spearman_q']].ge(0).all().all())
        self.assertTrue(corr[['pearson_q','spearman_q']].le(1).all().all())

    def test_split_and_polynomial_coefficients(self):
        scores, models, split, coefs = evaluate_models(self.clean)
        again, _, split2, _ = evaluate_models(self.clean)
        pd.testing.assert_frame_equal(scores, again)
        pd.testing.assert_frame_equal(split, split2)
        self.assertEqual(split.source_row_0based.nunique(), len(self.clean))
        train = split.loc[split.partition.eq('train'), 'source_row_0based']
        test = split.loc[split.partition.eq('test'), 'source_row_0based']
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(scores), 27)
        x = self.clean[['Temperature']]
        for row in coefs.itertuples():
            expected = row.beta0 + row.beta1*x.Temperature + row.beta2*x.Temperature**2
            np.testing.assert_allclose(models[row.pollutant, row.model].predict(x), expected, atol=1e-9)
        for pollutant in POLLUTANTS:
            self.assertAlmostEqual(models[pollutant, 'Mean baseline'].predict(x.iloc[:1])[0], self.clean.loc[train, pollutant].mean())

    def test_robust_regression(self):
        table = regressions(self.clean)
        self.assertEqual(len(table), 18)
        self.assertTrue((table.ci_low <= table.coefficient).all())
        self.assertTrue((table.ci_high >= table.coefficient).all())

    def test_artifacts(self):
        for name in ['correlation_matrix', 'temperature_correlations', 'regression_analysis', 'polynomial_comparison', 'model_comparison', 'distributions']:
            with Image.open(ROOT / 'results' / f'{name}.png') as im:
                self.assertGreaterEqual(im.info['dpi'][0], 299.9)
                im.verify()
        for path in (ROOT/'results').glob('*.csv'):
            self.assertGreater(len(pd.read_csv(path)), 0)


if __name__ == '__main__':
    unittest.main()
