"""Тесты стандартизации и агрегации. Стадия 1, владелец A."""

import unittest

import pandas as pd

from src.data import standardize


class Standardize(unittest.TestCase):
    def test_salt_is_stripped_to_the_largest_fragment(self):
        self.assertEqual(
            standardize.standardize_smiles("CC(=O)Oc1ccccc1C(=O)O.[Na+]"),
            standardize.standardize_smiles("CC(=O)Oc1ccccc1C(=O)O"),
        )

    def test_same_molecule_written_differently_collapses(self):
        self.assertEqual(
            standardize.standardize_smiles("c1ccccc1O"),
            standardize.standardize_smiles("Oc1ccccc1"),
        )

    def test_invalid_smiles_returns_none(self):
        self.assertIsNone(standardize.standardize_smiles("this is not a molecule"))
        self.assertIsNone(standardize.standardize_smiles("C(C"))

    def test_molecular_weight_is_available_for_unit_conversion(self):
        self.assertAlmostEqual(standardize.molecular_weight("CCO"), 46.07, places=1)


class Aggregate(unittest.TestCase):
    def test_zero_si_is_not_silently_converted_to_missing_or_discarded(self):
        frame = pd.DataFrame([
            {'standard_smiles':'C','IC50':float('nan'),'CC50':10.,'SI':0.},
            {'standard_smiles':'CC','IC50':1.,'CC50':10.,'SI':0.},
            {'standard_smiles':'CC','IC50':1.,'CC50':10.,'SI':10.}])
        out = standardize.aggregate_duplicates(frame).set_index('standard_smiles')
        self.assertEqual(out.loc['C','SI'],0.)
        self.assertEqual(out.loc['CC','SI'],5.)
        self.assertIn('0',out.loc['CC','SI_values'])

    def test_conflicting_measurements_collapse_to_log_median(self):
        frame = pd.DataFrame([
            {"standard_smiles": "CCO", "IC50": 1.0, "CC50": 100.0, "SI": 100.0, "doi": "10.1/a"},
            {"standard_smiles": "CCO", "IC50": 100.0, "CC50": 100.0, "SI": 1.0, "doi": "10.1/b"},
            {"standard_smiles": "CCO", "IC50": 10.0, "CC50": 100.0, "SI": 10.0, "doi": "10.1/a"},
        ])
        result = standardize.aggregate_duplicates(frame)
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertAlmostEqual(row["IC50"], 10.0)
        self.assertEqual(row["n_measurements"], 3)
        # Разброс на два порядка должен быть виден в отчёте, а не потеряться.
        self.assertAlmostEqual(row["IC50_spread_log10"], 2.0)
        self.assertEqual(row["doi_all"], "10.1/a;10.1/b")

    def test_distinct_structures_are_kept_apart(self):
        frame = pd.DataFrame([
            {"standard_smiles": "CCO", "IC50": 1.0, "CC50": 10.0, "SI": 10.0},
            {"standard_smiles": "CCC", "IC50": 2.0, "CC50": 20.0, "SI": 10.0},
        ])
        self.assertEqual(len(standardize.aggregate_duplicates(frame)), 2)


if __name__ == "__main__":
    unittest.main()
