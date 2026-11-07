import unittest

import numpy as np

from src.models.tanimoto_krr import TanimotoKRR, tanimoto_kernel


class TanimotoKRRTests(unittest.TestCase):
    def test_tanimoto_kernel_matches_set_similarity(self):
        fingerprints = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
        kernel = tanimoto_kernel(fingerprints)
        np.testing.assert_allclose(kernel, [[1, 1 / 3, 0], [1 / 3, 1, 0], [0, 0, 0]])
        np.testing.assert_allclose(kernel, kernel.T)

    def test_independent_masks_and_log_concentration_ratio(self):
        fingerprints = np.eye(4, dtype=np.uint8)
        model = TanimotoKRR({"alpha": 0.01}).fit(
            fingerprints,
            np.array([1.0, 10.0, np.nan, 100.0]),
            np.array([10.0, np.nan, 100.0, 1000.0]),
        )
        self.assertEqual(model.n_train["IC50"], 3)
        self.assertEqual(model.n_train["CC50"], 3)
        self.assertEqual(model.n_train["IC50_skipped"], 1)
        predicted = model.predict_log_concentrations(fingerprints[:1])
        np.testing.assert_allclose(model.predict_log_si(fingerprints[:1]),
                                   predicted["CC50"] - predicted["IC50"])
        self.assertTrue(np.isfinite(model.predict_si(fingerprints[:1])).all())

    def test_rejects_non_binary_fingerprints_and_bad_alpha(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            tanimoto_kernel([[0, 0.5]], [[1, 0]])
        with self.assertRaisesRegex(ValueError, "alpha"):
            TanimotoKRR({"alpha": 0})


if __name__ == "__main__":
    unittest.main()
