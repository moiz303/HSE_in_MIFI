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


class FeaturePresets(unittest.TestCase):
    SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "не молекула",
              "CN1CCC[C@H]1c1cccnc1"]

    def test_every_preset_builds_a_clean_matrix(self):
        from src.features import fingerprints

        for name, preset in fingerprints.PRESETS.items():
            matrix = fingerprints.featurize(self.SMILES, **preset)
            self.assertEqual(matrix.shape[0], len(self.SMILES), name)
            self.assertFalse(np.isnan(matrix).any(), name)
            self.assertTrue(np.isfinite(matrix).all(), name)

    def test_presets_differ_in_width(self):
        from src.features import fingerprints

        widths = {
            name: fingerprints.featurize(self.SMILES, **preset).shape[1]
            for name, preset in fingerprints.PRESETS.items()
        }
        self.assertGreater(len(set(widths.values())), 3)

    def test_old_config_calls_still_work(self):
        # configs/baseline.json передаёт use_morgan и use_descriptors.
        from src.features import fingerprints

        matrix = fingerprints.featurize(
            self.SMILES, use_morgan=True, use_descriptors=True
        )
        self.assertEqual(matrix.shape[1], fingerprints.MORGAN_BITS + 13)

    def test_invalid_smiles_gives_a_zero_row_not_a_crash(self):
        from src.features import fingerprints

        matrix = fingerprints.featurize(["не молекула"], maccs=True)
        self.assertTrue(np.isfinite(matrix).all())


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


class Forests(unittest.TestCase):
    """Лес как baseline: классика QSAR на фингерпринтах, которой у нас не было."""

    def test_forest_learners_train_and_predict(self):
        features, si = data(n=30)
        for learner in ("rf", "extra"):
            model = FlexibleSI(learner=learner, clip_si=20000.0).fit(features, si)
            predicted = model.predict_si(features)
            self.assertTrue(np.isfinite(predicted).all(), learner)
            self.assertLessEqual(predicted.max(), 20000.0, learner)

    def test_losses_the_forest_cannot_express_are_rejected(self):
        # Лес предсказывает среднее по листу: квантиль и гамма ему недоступны,
        # у него настраивается только критерий расщепления.
        for loss in ("quantile", "gamma"):
            with self.assertRaises(ValueError, msg=loss):
                FlexibleSI(learner="rf", loss=loss)

    def test_forest_criteria_that_exist_are_accepted(self):
        for loss in ("squared_error", "absolute_error", "poisson"):
            FlexibleSI(learner="rf", loss=loss)

    def test_forest_is_in_the_base_grid(self):
        learners = {item.get("learner", "hgb") for item in sweep.candidates("base")}
        self.assertIn("rf", learners)
        self.assertIn("extra", learners)
