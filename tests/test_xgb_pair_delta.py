import unittest

import numpy as np

from src.models.xgb_pair_delta import (assert_reference_partition, build_training_pairs,
                                       pair_features, tanimoto)
from src.splits import murcko_scaffold


class PairDelta(unittest.TestCase):
    def test_pair_features_encode_ordered_difference_and_similarity(self):
        q = np.array([[1., 4., 6.]])
        r = np.array([[2., 1., 3.]])
        np.testing.assert_array_equal(pair_features(q, r, [0.75]), [[1, 4, 6, 2, 1, 3, -1, 3, 3, .75]])
        np.testing.assert_allclose(tanimoto([[1, 1, 0]], [[1, 0, 1]]), [1/3])

    def test_training_pairs_are_directed_log_deltas_with_train_only_indices(self):
        x = np.zeros((3, 2061), dtype=np.float32)
        x[:, 0] = 1
        x[:, 2048] = [0., 1., 2.]
        pair_x, delta, pair_w = build_training_pairs(x, [1., 10., 100.], [1., .5, 1.], 2, .9)
        self.assertEqual(pair_x.shape, (6, 3 * x.shape[1] + 1))
        np.testing.assert_allclose(np.sort(delta), [-2., -1., -1., 1., 1., 2.])
        np.testing.assert_allclose(np.sort(pair_w), np.sort([np.sqrt(.5)]*4 + [1., 1.]))

    def test_no_eligible_reference_yields_no_pairs_and_enables_fallback(self):
        x = np.zeros((3, 2061), dtype=np.float32)
        x[np.arange(3), np.arange(3)] = 1
        pairs, delta, weights = build_training_pairs(x, [1., 10., 100.], np.ones(3), 2, 1.)
        self.assertEqual(pairs.shape, (0, 3 * x.shape[1] + 1))
        self.assertEqual(len(delta), 0)
        self.assertEqual(len(weights), 0)

    def test_reference_bank_rejects_held_structures_and_scaffolds(self):
        assert_reference_partition(["c1ccccc1"], ["C1CCCCC1"], murcko_scaffold, "molecule")
        with self.assertRaisesRegex(AssertionError, "structure"):
            assert_reference_partition(["c1ccccc1"], ["c1ccccc1"], murcko_scaffold, "molecule")
        with self.assertRaisesRegex(AssertionError, "scaffold"):
            assert_reference_partition(["c1ccccc1"], ["Cc1ccccc1"], murcko_scaffold, "scaffold")


if __name__ == "__main__":
    unittest.main()
