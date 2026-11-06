"""Тесты перебираемой модели. Сети нет, данные синтетические и крошечные."""

import unittest

import numpy as np

import sweep
from src.models.flexible import FlexibleSI


def data(n: int = 40, seed: int = 11):
    rng = np.random.default_rng(seed)
    features = rng.random((n, 6))
    si = 10 ** rng.uniform(-1, 4, n)
    return features, si


class Transforms(unittest.TestCase):
    def test_every_transform_round_trips(self):
        features, si = data()
        for transform in ("raw", "log", "sqrt"):
            model = FlexibleSI(transform=transform).fit(features, si)
            predicted = model.predict_si(features)
            self.assertTrue(np.isfinite(predicted).all(), transform)
            self.assertTrue((predicted >= 0).all(), transform)

    def test_unknown_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            FlexibleSI(transform="кубический")

    def test_unknown_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            FlexibleSI(loss="как-нибудь")

    def test_positive_only_loss_requires_raw_scale(self):
        # gamma и poisson определены для положительной цели: в логах цель
        # становится отрицательной, и обучение было бы бессмысленным.
        for loss in ("gamma", "poisson"):
            with self.assertRaises(ValueError, msg=loss):
                FlexibleSI(transform="log", loss=loss)
            FlexibleSI(transform="raw", loss=loss)


class Clipping(unittest.TestCase):
    def test_prediction_respects_the_cap(self):
        features, si = data()
        model = FlexibleSI(clip_si=1000.0).fit(features, si)
        self.assertLessEqual(model.predict_si(features).max(), 1000.0)

    def test_no_cap_by_default(self):
        self.assertIsNone(FlexibleSI().clip_si)

    def test_old_pickle_without_clip_attribute(self):
        # Модели коммитятся; объект из прежней версии кода не должен ломать
        # предсказание.
        features, si = data()
        model = FlexibleSI(clip_si=500.0).fit(features, si)
        del model.clip_si
        self.assertTrue(np.isfinite(model.predict_si(features)).all())


class Weights(unittest.TestCase):
    def test_weighting_changes_the_fit(self):
        features, si = data()
        plain = FlexibleSI(weight_power=0.0).fit(features, si).predict_si(features)
        weighted = FlexibleSI(weight_power=1.0).fit(features, si).predict_si(features)
        self.assertFalse(np.allclose(plain, weighted))

    def test_weights_are_finite_for_large_targets(self):
        features, si = data()
        si[0] = 2.2e5
        model = FlexibleSI(weight_power=1.0, clip_si=36066.7).fit(features, si)
        self.assertTrue(np.isfinite(model.predict_si(features)).all())


class Quantile(unittest.TestCase):
    def test_higher_quantile_predicts_higher(self):
        features, si = data()
        low = FlexibleSI(loss="quantile", quantile=0.3).fit(features, si).predict_si(features)
        high = FlexibleSI(loss="quantile", quantile=0.9).fit(features, si).predict_si(features)
        self.assertGreater(high.mean(), low.mean())


class Grid(unittest.TestCase):
    def test_base_grid_has_no_duplicates(self):
        grid = sweep.candidates("base")
        self.assertEqual(len(grid), len({tuple(sorted(item.items())) for item in grid}))

    def test_full_grid_includes_the_base_one(self):
        base, full = sweep.candidates("base"), sweep.candidates("full")
        for item in base:
            self.assertIn(item, full)

    def test_every_combination_in_the_grid_trains(self):
        features, si = data(n=30)
        for settings in sweep.candidates("full"):
            model = FlexibleSI(**settings).fit(features, si)
            predicted = model.predict_si(features)
            self.assertTrue(np.isfinite(predicted).all(), settings)
            self.assertTrue(model.describe())


if __name__ == "__main__":
    unittest.main()
