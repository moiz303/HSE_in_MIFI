"""Тесты правила вычисления SI из колонок соревнования.

Сети нет, данные соревнования в репозиторий не коммитятся — поэтому фикстуры
собраны руками и повторяют три случая, которые встречаются в train.
"""

import unittest

import numpy as np
import pandas as pd

from src.data import kaggle_data as kd


def frame(rows):
    return pd.DataFrame(rows, columns=[kd.CC50_COLUMN, kd.IC50_COLUMN]).astype(float)


class Rule(unittest.TestCase):
    def test_both_values_divide(self):
        result = kd.selectivity_from_columns(frame([[220.0, 64.0], [15.0, 2.18]]))
        self.assertAlmostEqual(result.iloc[0], 220.0 / 64.0)
        self.assertAlmostEqual(result.iloc[1], 15.0 / 2.18)

    def test_missing_ic50_is_the_inactive_marker(self):
        # В train все 81 строка с пропущенным IC50 имеют SI ровно 10000.
        result = kd.selectivity_from_columns(frame([[26.2, np.nan]]))
        self.assertAlmostEqual(result.iloc[0], kd.INACTIVE_MARKER)

    def test_missing_cc50_is_zero(self):
        # В train обе такие строки имеют SI ровно 0.0, хотя IC50 задан.
        result = kd.selectivity_from_columns(frame([[np.nan, 100.0]]))
        self.assertAlmostEqual(result.iloc[0], kd.NO_CC50_VALUE)

    def test_missing_cc50_wins_over_missing_ic50(self):
        result = kd.selectivity_from_columns(frame([[np.nan, np.nan]]))
        self.assertAlmostEqual(result.iloc[0], kd.NO_CC50_VALUE)

    def test_equal_values_give_one(self):
        result = kd.selectivity_from_columns(frame([[50.0, 50.0]]))
        self.assertAlmostEqual(result.iloc[0], 1.0)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            kd.selectivity_from_columns(pd.DataFrame({kd.CC50_COLUMN: [1.0]}))

    def test_non_numeric_text_is_treated_as_missing(self):
        table = pd.DataFrame({kd.CC50_COLUMN: ["нет"], kd.IC50_COLUMN: ["10"]})
        result = kd.selectivity_from_columns(table)
        self.assertAlmostEqual(result.iloc[0], kd.NO_CC50_VALUE)


class VerifyRule(unittest.TestCase):
    def test_summary_reports_a_perfect_match(self):
        train = frame([[220.0, 64.0], [26.2, np.nan], [np.nan, 100.0]])
        train[kd.TARGET_COLUMN] = [220.0 / 64.0, kd.INACTIVE_MARKER, kd.NO_CC50_VALUE]
        summary = kd.verify_rule(train)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["within_tolerance"], 3)
        self.assertLess(summary["mse"], 1e-12)

    def test_summary_catches_a_wrong_convention(self):
        # Если бы маркер неактивности был другим, проверка обязана это увидеть.
        train = frame([[26.2, np.nan]])
        train[kd.TARGET_COLUMN] = [1.0]
        summary = kd.verify_rule(train)
        self.assertEqual(summary["within_tolerance"], 0)
        self.assertGreater(summary["mse"], 1.0)

    def test_rounding_in_the_source_file_stays_within_tolerance(self):
        # SI в файле соревнования дан с тремя знаками: 3.4375 записано как 3.438.
        train = frame([[220.0, 64.0]])
        train[kd.TARGET_COLUMN] = [3.438]
        summary = kd.verify_rule(train)
        self.assertEqual(summary["within_tolerance"], 1)


if __name__ == "__main__":
    unittest.main()
