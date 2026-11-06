"""Проверки сборки Kaggle CSV на маленьких фреймах, без модели и сети."""

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src import pipeline
from train_model import predict


class SubmissionContract(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame({"index": [10, 11], "SMILES": ["CCO", "CC"]})
        self.valid = pd.DataFrame({"index": [10, 11], "SI VACV": [1.5, 2.0]})

    def test_submission_csv_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reports" / "submission.csv"
            pipeline.save_submission(self.test, [1.5, 2.0], "index", "SI VACV", path)
            pd.testing.assert_frame_equal(pd.read_csv(path), self.valid)
            saved = path.read_bytes()
            with self.assertRaises(SystemExit):
                pipeline.save_submission(self.test, [1.5, 2.0], "index", "index", path)
            self.assertEqual(path.read_bytes(), saved)
            invalid = ([1.5], [None, 2.0], ["bad", 2.0], [float("inf"), 2.0])
            for predictions in invalid:
                with self.subTest(predictions=predictions), self.assertRaises(SystemExit):
                    pipeline.save_submission(self.test, predictions, "index", "SI VACV", path)
                self.assertEqual(path.read_bytes(), saved)

    def test_missing_precondition_names_next_command(self):
        path = pipeline.ARTIFACTS / "missing-test-fixture.csv"
        with self.assertRaisesRegex(SystemExit, "сначала запустите `python -m src.pipeline train`"):
            pipeline.require_file(path, "сначала запустите `python -m src.pipeline train`")

    def test_missing_test_file_points_to_kaggle_download_step(self):
        config = pipeline.load_config()
        config["submission"]["test_csv"] = "data/raw/missing-test-fixture.csv"
        with self.assertRaisesRegex(SystemExit, "python -m src.pipeline kaggle"):
            pipeline.step_submission(config)

    def test_training_script_missing_test_file_is_actionable(self):
        path = pipeline.ROOT / "data/raw/missing-test-fixture.csv"
        with self.assertRaisesRegex(SystemExit, "python -m src.pipeline kaggle"):
            predict(path, "direct", "index", "SMILES", "SI VACV", path)

    def test_missing_config_has_an_actionable_error(self):
        path = pipeline.ROOT / "configs" / "missing-test-fixture.json"
        with self.assertRaisesRegex(SystemExit, "нет конфигурации"):
            pipeline.load_config(path)


if __name__ == "__main__":
    unittest.main()
