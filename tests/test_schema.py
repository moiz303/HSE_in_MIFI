"""Тесты контракта данных. Стадия 0, владелец A."""

import unittest

import pandas as pd

from src.data import schema


class ParseMeasurement(unittest.TestCase):
    def test_plain_number(self):
        self.assertEqual(schema.parse_measurement("12.5"), (12.5, ""))

    def test_censored_values_keep_the_qualifier(self):
        self.assertEqual(schema.parse_measurement(">100"), (100.0, ">"))
        self.assertEqual(schema.parse_measurement("<0.1"), (0.1, "<"))
        self.assertEqual(schema.parse_measurement(" >= 50 "), (50.0, ">="))

    def test_garbage_is_not_silently_zero(self):
        self.assertEqual(schema.parse_measurement("нет данных"), (None, ""))
        self.assertEqual(schema.parse_measurement(None), (None, ""))


class UnitConversion(unittest.TestCase):
    def test_molar_scale(self):
        self.assertAlmostEqual(schema.to_micromolar(1, "nM"), 0.001)
        self.assertAlmostEqual(schema.to_micromolar(1, "mM"), 1000.0)
        self.assertAlmostEqual(schema.to_micromolar(2.5, "uM"), 2.5)

    def test_mass_units_need_molecular_weight(self):
        with self.assertRaises(schema.UnitConversionError):
            schema.to_micromolar(10, "ug/mL")
        # 10 µg/mL при M = 250 г/моль → 40 µM
        self.assertAlmostEqual(schema.to_micromolar(10, "ug/mL", 250.0), 40.0)

    def test_unknown_unit_fails_loudly(self):
        with self.assertRaises(schema.UnitConversionError):
            schema.to_micromolar(1, "parrots")


class ValidateTable(unittest.TestCase):
    def frame(self, **overrides):
        row = {"smiles": "CCO", "IC50": 1.0, "CC50": 100.0, "SI": 100.0}
        row.update(overrides)
        return pd.DataFrame([row])

    def test_good_row_passes(self):
        accepted, rejected = schema.validate_table(self.frame())
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)

    def test_missing_required_column_raises(self):
        with self.assertRaises(ValueError):
            schema.validate_table(pd.DataFrame([{"smiles": "CCO"}]))

    def test_inconsistent_si_is_rejected(self):
        accepted, rejected = schema.validate_table(self.frame(SI=3.0))
        self.assertEqual(len(accepted), 0)
        self.assertIn("SI", rejected.iloc[0]["reason"])

    def test_missing_or_false_label_flags_do_not_bypass_validation(self):
        for flag in (float("nan"), pd.NA, None, False, "False", "false", "0"):
            with self.subTest(flag=str(flag)):
                accepted, rejected = schema.validate_table(self.frame(SI=3.0, target_is_label=flag))
                self.assertEqual(len(accepted), 0)
                self.assertEqual(len(rejected), 1)

    def test_actual_label_keeps_missing_ic50(self):
        for flag in (True, "True", "1"):
            with self.subTest(flag=flag):
                accepted, rejected = schema.validate_table(
                    self.frame(IC50=float("nan"), SI=10000.0, target_is_label=flag))
                self.assertEqual(len(accepted), 1)
                self.assertEqual(len(rejected), 0)

    def test_current_zero_label_is_kept_but_negative_label_is_rejected(self):
        accepted, rejected = schema.validate_table(self.frame(IC50=float('nan'),SI=0.,target_is_label=True))
        self.assertEqual(len(accepted),1)
        self.assertEqual(len(rejected),0)
        accepted, rejected = schema.validate_table(self.frame(IC50=float('nan'),SI=-1.,target_is_label=True))
        self.assertEqual(len(accepted),0)
        self.assertEqual(len(rejected),1)

    def test_non_numeric_is_rejected_with_reason(self):
        accepted, rejected = schema.validate_table(self.frame(IC50="нет"))
        self.assertEqual(len(accepted), 0)
        self.assertTrue(rejected.iloc[0]["reason"])

    def test_unit_error_scale_is_rejected(self):
        # 1e9 µM — это 1000 моль/л, такого измерения не существует.
        accepted, rejected = schema.validate_table(self.frame(IC50=1e9, CC50=1e11, SI=100.0))
        self.assertEqual(len(accepted), 0)

    def test_source_frame_is_not_mutated(self):
        frame = self.frame()
        schema.validate_table(frame)
        self.assertNotIn("reason", frame.columns)


if __name__ == "__main__":
    unittest.main()
