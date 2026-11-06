"""Проверки сборки Kaggle CSV на маленьких фреймах, без модели и сети."""

import unittest

import pandas as pd

from src import pipeline
from train_model import predict


class SubmissionContract(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame({"index": [10, 11], "SMILES": ["CCO", "CC"]})
        self.valid = pd.DataFrame({"index": [10, 11], "SI VACV": [1.5, 2.0]})

    def test_valid_submission_passes(self):
        pipeline.validate_submission(self.test, self.valid, "index", "SI VACV")

    def test_required_columns_are_enforced(self):
        with self.assertRaisesRegex(SystemExit, "нужны колонки"):
            pipeline.validate_submission(
                self.test, self.valid.rename(columns={"SI VACV": "SI"}), "index", "SI VACV"
            )

    def test_row_count_must_match_test(self):
        with self.assertRaisesRegex(SystemExit, "нужно 2 строк"):
            pipeline.validate_submission(self.test, self.valid.iloc[:1], "index", "SI VACV")

    def test_empty_nonnumeric_and_infinite_predictions_are_rejected(self):
        for value in (None, "bad", float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(SystemExit, "пустые или нечисловые"):
                invalid = pd.DataFrame({
                    "index": [10, 11], "SI VACV": pd.Series([value, 2.0], dtype=object),
                })
                pipeline.validate_submission(self.test, invalid, "index", "SI VACV")

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
