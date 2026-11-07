import unittest
from pathlib import Path
import numpy as np
from src.data.vacv_supplement import load_supplement, eligible_supplement
from src.splits import murcko_scaffold

class SupplementContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_supplement(Path(__file__).resolve().parents[1])

    def test_endpoints_are_not_relabelled_or_cartesian_expanded(self):
        self.assertEqual(len(self.rows), 126)
        self.assertEqual(self.rows.record_id.nunique(), 126)
        self.assertTrue(self.rows.IC50.isna().all())
        self.assertTrue(self.rows.SI.isna().all())
        self.assertEqual(self.rows.endpoint.eq('EC50').sum(), 84)
        self.assertEqual(self.rows.CC50.notna().sum(), 42)
        exact = self.rows.CC50.notna() & self.rows.CC50_qualifier.eq('')
        self.assertEqual(exact.sum(), 17)
        self.assertEqual(self.rows.loc[exact, 'standard_smiles'].nunique(), 11)

    def test_held_molecules_and_scaffolds_are_excluded(self):
        held = self.rows.standard_smiles.iloc[0]
        molecule = eligible_supplement(self.rows, [held], 'molecule')
        self.assertNotIn(held, set(molecule.standard_smiles))
        scaffold = eligible_supplement(self.rows, [held], 'scaffold')
        self.assertNotIn(murcko_scaffold(held), set(scaffold.standard_smiles.map(murcko_scaffold)))

if __name__ == '__main__':
    unittest.main()
