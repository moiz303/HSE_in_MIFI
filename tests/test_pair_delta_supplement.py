import unittest

import numpy as np
import pandas as pd

from src.data.standardize import standardize_smiles
from src.pair_delta_supplement import append_eligible
from src.splits import murcko_scaffold


def row(smiles, endpoint, relation, value):
    return {"standard_smiles": standardize_smiles(smiles), "endpoint": endpoint,
        "relation": relation, "IC50": np.nan, "IC50_qualifier": "", "CC50":
        value if endpoint == "CC50" else np.nan,
        "CC50_qualifier": relation.replace("=", "") if endpoint == "CC50" else "",
        "EC50": value if endpoint == "EC50" else np.nan}


class PairDeltaSupplementTests(unittest.TestCase):
    def setUp(self):
        self.held = standardize_smiles("c1ccccc1")
        self.original = pd.DataFrame({"standard_smiles": [self.held], "IC50": [2.],
            "IC50_qualifier": [""], "CC50": [20.], "CC50_qualifier": [""]})
        self.extra = pd.DataFrame([
            row("c1ccccc1", "CC50", "=", 30.),
            row("Cc1ccccc1", "CC50", "=", 40.),
            row("C1CCCCC1", "CC50", "=", 50.),
            row("C1CCCC1", "CC50", ">", 60.),
            row("CCO", "EC50", "=", 70.),
        ])
        self.matrix = {smiles: np.full(3, index, dtype=float)
                       for index, smiles in enumerate([*self.original.standard_smiles,
                                                       *self.extra.standard_smiles])}

    def test_only_exact_cc50_is_appended_and_never_as_ic50(self):
        train, _, ic50, cc50, weights, extra = append_eligible(
            self.original, [1.], self.extra, {self.held}, "molecule", self.matrix)
        expected = {standardize_smiles("Cc1ccccc1"), standardize_smiles("C1CCCCC1")}
        self.assertEqual(set(extra.standard_smiles), expected)
        self.assertTrue(np.isnan(ic50[1:]).all())
        self.assertTrue(np.isfinite(cc50[1:]).all())
        np.testing.assert_allclose(weights, [1., .5, .5])
        self.assertEqual(len(train), 3)

    def test_scaffold_protocol_removes_all_supplement_rows_on_held_scaffold(self):
        _, _, ic50, cc50, _, extra = append_eligible(
            self.original, [1.], self.extra, {self.held}, "scaffold", self.matrix)
        self.assertNotIn(murcko_scaffold(self.held), set(extra.standard_smiles.map(murcko_scaffold)))
        self.assertEqual(len(extra), 1)
        self.assertTrue(np.isnan(ic50[1:]).all())
        np.testing.assert_allclose(cc50[1:], [50.])


if __name__ == "__main__":
    unittest.main()
