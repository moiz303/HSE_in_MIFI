"""Тесты линейной поправки масштаба. Сети нет, данные синтетические.

Проверяем механику, а не то, что поправка помогает: помогает она или нет —
вопрос к опыту на реальных данных (reports/experiment_calibration.json).
"""

import unittest

import numpy as np

import experiment_calibration as calibration


class FitCoefficients(unittest.TestCase):
    def test_affine_recovers_an_exact_linear_shift(self):
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        true = 7.0 + 3.0 * predicted
        a, b = calibration.fit_affine(predicted, true)
        self.assertAlmostEqual(a, 7.0, places=6)
        self.assertAlmostEqual(b, 3.0, places=6)

    def test_scale_has_no_shift_by_construction(self):
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        a, b = calibration.fit_scale(predicted, 3.0 * predicted)
        self.assertEqual(a, 0.0)
        self.assertAlmostEqual(b, 3.0, places=6)

    def test_scale_survives_all_zero_predictions(self):
        # Вырожденный случай: делить не на что. Поправка должна стать тождеством.
        a, b = calibration.fit_scale(np.zeros(5), np.arange(5.0))
        self.assertEqual((a, b), (0.0, 1.0))

    def test_affine_minimises_mse_better_than_identity(self):
        rng = np.random.default_rng(3)
        predicted = rng.uniform(0, 100, 200)
        true = 5.0 + 1.6 * predicted + rng.normal(0, 5, 200)
        a, b = calibration.fit_affine(predicted, true)
        fitted = calibration.mse(calibration.apply_calibration(predicted, a, b), true)
        self.assertLess(fitted, calibration.mse(predicted, true))


class Applying(unittest.TestCase):
    def test_negative_results_are_clipped_to_zero(self):
        # SI — отношение концентраций, отрицательным быть не может.
        result = calibration.apply_calibration(np.array([0.0, 1.0, 10.0]), -5.0, 1.0)
        self.assertTrue((result >= 0).all())
        self.assertEqual(result[0], 0.0)

    def test_large_predictions_are_not_capped_back(self):
        # Смысл гипотезы — поднять крупные прогнозы; верхней обрезки тут нет.
        result = calibration.apply_calibration(np.array([20000.0]), 0.0, 2.0)
        self.assertAlmostEqual(float(result[0]), 40000.0)

    def test_identity_coefficients_change_nothing(self):
        predicted = np.array([0.5, 7.0, 1234.0])
        np.testing.assert_allclose(
            calibration.apply_calibration(predicted, 0.0, 1.0), predicted
        )


class OutOfFold(unittest.TestCase):
    """Главное свойство схемы: прогноз для молекулы делает модель, её не видевшая."""

    def test_every_row_gets_a_prediction_from_a_model_that_did_not_see_it(self):
        smiles = [
            "CCO", "CCN", "CCC", "c1ccccc1", "c1ccccc1O", "c1ccccc1N",
            "C1CCCCC1", "C1CCCCC1O", "C1CCCCC1N", "c1ccncc1", "c1ccncc1C",
            "c1ccncc1O", "CC(=O)O", "CC(=O)N", "CC(=O)OC", "c1ccc2ccccc2c1",
            "c1ccc2ccccc2c1O", "c1ccc2ccccc2c1N", "CCCCO", "CCCCN",
        ]
        rng = np.random.default_rng(5)
        features = rng.random((len(smiles), 4))
        si = 10 ** rng.uniform(0, 3, len(smiles))
        index = np.arange(len(smiles))

        out_of_fold = calibration.oof_predictions(
            features, si, smiles, index,
            {"transform": "raw", "loss": "squared_error"},
            folds=4, seed=1,
        )
        self.assertEqual(len(out_of_fold), len(smiles))
        self.assertTrue(np.isfinite(out_of_fold).all())
        self.assertTrue((out_of_fold >= 0).all())


if __name__ == "__main__":
    unittest.main()
