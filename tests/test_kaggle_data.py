"""Тесты запрета на арифметику и загрузки данных соревнования.

Организаторы запретили выводить целевую переменную из колонок тестового файла.
Эти тесты закрепляют запрет в коде: если кто-то снова начнёт считать SI из
CC50_uM и IC50_uM_VACV на пути сабмита, тест покраснеет.
"""

import unittest

import pandas as pd

from src.data import kaggle_data as kd


class ForbiddenFeatures(unittest.TestCase):
    def test_cc50_in_features_is_rejected(self):
        frame = pd.DataFrame({"SMILES": ["CCO"], "CC50_uM": [10.0]})
        with self.assertRaises(ValueError) as caught:
            kd.assert_no_forbidden_features(frame)
        self.assertIn("CC50_uM", str(caught.exception))

    def test_ic50_in_features_is_rejected(self):
        frame = pd.DataFrame({"SMILES": ["CCO"], "IC50_uM_VACV": [1.0]})
        with self.assertRaises(ValueError):
            kd.assert_no_forbidden_features(frame)

    def test_smiles_only_frame_passes(self):
        frame = pd.DataFrame({"index": [0], "SMILES": ["CCO"], "strain": ["WR"]})
        kd.assert_no_forbidden_features(frame)

    def test_the_arithmetic_helpers_are_gone(self):
        # Прямая проверка, что путь не вернулся под старыми именами.
        for name in ("selectivity_from_columns", "verify_rule",
                     "INACTIVE_MARKER", "NO_CC50_VALUE"):
            self.assertFalse(
                hasattr(kd, name),
                f"{name} восстановлен, а арифметика по тесту запрещена организаторами",
            )


class TrainingFrame(unittest.TestCase):
    def test_columns_are_renamed_to_the_contract(self):
        raw = pd.DataFrame({
            "SMILES": ["CCO"], "CC50_uM": [100.0], "IC50_uM_VACV": [4.0],
            "SI VACV": [25.0], "Culture": ["Vero"], "Title": ["x"], "strain": ["WR"],
        })

        class Stub(dict):
            pass

        # Подменяем чтение файла: сети и данных соревнования в тестах нет.
        original = kd.load
        kd.load = lambda name, directory=kd.KAGGLE_DIR: raw.copy()
        try:
            frame = kd.training_frame()
        finally:
            kd.load = original

        for column in ("smiles", "CC50", "IC50", "SI", "cell_line", "source", "virus"):
            self.assertIn(column, frame.columns)
        self.assertEqual(frame.iloc[0]["source"], "kaggle_train")
        self.assertEqual(frame.iloc[0]["virus"], "vaccinia virus")
        self.assertAlmostEqual(frame.iloc[0]["SI"], 25.0)


if __name__ == "__main__":
    unittest.main()
