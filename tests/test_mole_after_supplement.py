import unittest

import numpy as np
import pandas as pd

from src.mole_after_supplement import append_eligible_supplement


class NewDatasetLeakageTests(unittest.TestCase):
    def test_appends_exact_cc50_only_after_scaffold_guard(self):
        training = pd.DataFrame({"standard_smiles": ["CCO"], "IC50": [3.0], "CC50": [30.0],
                                 "IC50_qualifier": [""], "CC50_qualifier": [""]})
        supplement = pd.DataFrame({
            "standard_smiles": ["CCO", "c1ccccc1", "CCO"],
            "endpoint": ["CC50", "CC50", "EC50"],
            "relation": ["=", "=", "="],
            "value": [8.0, 9.0, 2.0],
            "doi": ["doi"] * 3,
        })
        vectors = {smiles: np.ones(1000) for smiles in supplement.standard_smiles.unique()}

        augmented, exact, counts = append_eligible_supplement(
            training, supplement, ["c1ccccc1"], "scaffold", vectors)

        self.assertEqual(len(augmented), 2)
        self.assertEqual(len(exact), 1)
        self.assertEqual(counts["exact_cc50"], 1)
        self.assertEqual(counts["structures"], 1)
        self.assertEqual(augmented.iloc[-1].CC50, 8.0)
        self.assertTrue(pd.isna(augmented.iloc[-1].IC50))
        self.assertEqual(augmented.iloc[-1].source, "literature_vacv_2023")


if __name__ == "__main__":
    unittest.main()
