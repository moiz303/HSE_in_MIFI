import unittest

import numpy as np
import pandas as pd

from src.models.mole_ridge import RidgeTwoHead


class MolERidgeTests(unittest.TestCase):
    def test_fits_two_independent_exact_concentration_heads(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(40, 5))
        ic50 = np.power(10, 1.0 + features[:, 0] * 0.2)
        cc50 = np.power(10, 1.5 + features[:, 1] * 0.3)
        frame = pd.DataFrame({"IC50": ic50, "CC50": cc50,
                              "IC50_qualifier": [""] * 39 + [">"],
                              "CC50_qualifier": [""] * 38 + ["<", ""]})

        model = RidgeTwoHead(alpha=1.0).fit(features, frame)
        prediction = model.predict_si(features)

        self.assertEqual(model.n_train, {"IC50": 39, "CC50": 39})
        self.assertEqual(prediction.shape, (40,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_rejects_wrong_feature_width(self):
        features = np.arange(40, dtype=float).reshape(10, 4)
        frame = pd.DataFrame({"IC50": np.arange(1, 11), "CC50": np.arange(2, 12),
                              "IC50_qualifier": [""] * 10, "CC50_qualifier": [""] * 10})
        model = RidgeTwoHead().fit(features, frame)
        with self.assertRaises(ValueError):
            model.predict_si(np.zeros((2, 5)))


if __name__ == "__main__":
    unittest.main()
