import unittest

import numpy as np
import pandas as pd

from src.data.vacv_supplement import eligible_supplement
from src.tanimoto_compare_new_dataset import exact_cc50_rows


class TanimotoSupplementTests(unittest.TestCase):
    def test_fold_filter_blocks_held_molecule_and_scaffold(self):
        supplement = pd.DataFrame({
            "standard_smiles": ["c1ccccc1", "Cc1ccccc1", "c1ccncc1"],
            "endpoint": ["CC50"] * 3,
            "CC50": [10.0, 20.0, 30.0],
            "CC50_qualifier": [""] * 3,
        })
        molecule = eligible_supplement(supplement, ["c1ccccc1"], "molecule")
        self.assertNotIn("c1ccccc1", molecule.standard_smiles.tolist())
        self.assertIn("Cc1ccccc1", molecule.standard_smiles.tolist())
        scaffold = eligible_supplement(supplement, ["c1ccccc1"], "scaffold")
        self.assertNotIn("Cc1ccccc1", scaffold.standard_smiles.tolist())
        self.assertIn("c1ccncc1", scaffold.standard_smiles.tolist())

    def test_only_exact_cc50_is_added_and_ic50_stays_missing(self):
        supplement = pd.DataFrame({
            "standard_smiles": ["c1ccccc1", "Cc1ccccc1", "c1ccncc1"],
            "endpoint": ["CC50", "CC50", "EC50"],
            "CC50": [10.0, 20.0, np.nan],
            "CC50_qualifier": ["", ">", ""],
            "EC50": [np.nan, np.nan, 1.0],
            "source": ["literature_vacv_2023"] * 3,
        })
        added = exact_cc50_rows(supplement)
        self.assertEqual(len(added), 1)
        self.assertEqual(added.iloc[0].CC50, 10.0)
        self.assertTrue(added.IC50.isna().all())
        self.assertTrue(added.EC50.isna().all())
        self.assertEqual(added.iloc[0].source, "literature_vacv_2023")


if __name__ == "__main__":
    unittest.main()
