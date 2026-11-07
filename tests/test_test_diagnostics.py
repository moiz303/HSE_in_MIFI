import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.test_diagnostics import reference_from_endpoints, evaluate_predictions, run


class LocalTestDiagnostic(unittest.TestCase):
    def test_ratio_coverage_and_id_alignment_without_guessing_missing_targets(self):
        test = pd.DataFrame({"index": [0, 1, 2, 3, 4], "CC50_uM": [20, 30, np.nan, ">100", 10],
            "IC50_uM_VACV": [2, np.nan, 2, 2, 0]})
        reference = reference_from_endpoints(test)
        prediction = pd.DataFrame({"index": [4, 3, 2, 1, 0], "SI VACV": [100, 100, 100, 100, 8]})
        report = evaluate_predictions(reference, prediction)
        self.assertEqual(report["n_evaluated"], 1)
        self.assertEqual(report["coverage"], .2)
        self.assertEqual(report["mse_ratio_subset"], 4)
        self.assertIsNone(report["full_test_ratio_mse"])
        self.assertIsNone(report["official_kaggle_score"])
        self.assertEqual(reference.status.tolist(), ["scorable", "missing_IC50", "missing_CC50",
            "censored_or_approximate_endpoint", "nonpositive_endpoint"])

    def test_missing_duplicate_or_bad_predictions_fail_instead_of_changing_coverage(self):
        reference = reference_from_endpoints(pd.DataFrame({"index": [0, 1], "CC50_uM": [10, 20],
            "IC50_uM_VACV": [1, np.nan]}))
        for ids, values in [([0], [10]), ([0, 0], [10, 10]), ([0, 1], [10, np.nan]),
                            ([0, 1], [10, -1]), ([0, 2], [10, 10])]:
            with self.assertRaises(ValueError):
                evaluate_predictions(reference, pd.DataFrame({"index": ids, "SI VACV": values}))

    def test_aggregate_report_only_and_input_files_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test = root / "test.csv"
            prediction = root / "submission.csv"
            pd.DataFrame({"index": [0, 1], "CC50_uM": [10, 20], "IC50_uM_VACV": [1, 2]}).to_csv(test, index=False)
            pd.DataFrame({"index": [1, 0], "SI VACV": [12, 8]}).to_csv(prediction, index=False)
            originals = [test.read_bytes(), prediction.read_bytes()]
            report = run(test, [prediction], root / "check")
            self.assertEqual(report["results"][0]["mse_ratio_subset"], 4)
            self.assertEqual(originals, [test.read_bytes(), prediction.read_bytes()])
            written = json.loads((root / "check.json").read_text())
            self.assertNotIn("ratio_target", written)
            self.assertEqual(set(p.name for p in root.iterdir()), {"test.csv", "submission.csv", "check.json", "check.csv"})
