"""Smoke-прогон: признаки, обе стратегии, метрики. Стадии 3–4, владелец B.

Синтетические данные, никакой сети. Задача теста — поймать сломанный пайплайн
за секунды, а не измерить качество.
"""

import unittest

import numpy as np

from src import evaluate, splits
from src.features import fingerprints
from src.models import baseline

SMILES = [
    "CCO", "CCCC", "c1ccccc1", "C1CCCCC1", "c1ccncc1", "O=C(O)CCl",
    "O=C(Nc1ccccc1)c1ccccc1", "Cc1ccc2ncccc2c1", "CC(=O)Oc1ccccc1C(=O)O",
    "CN1CCC[C@H]1c1cccnc1", "c1ccc2[nH]ccc2c1", "OCC1OC(O)C(O)C(O)C1O",
]


class Features(unittest.TestCase):
    def test_shape_is_stable(self):
        matrix = fingerprints.featurize(SMILES)
        expected = fingerprints.MORGAN_BITS + len(fingerprints.DESCRIPTOR_NAMES)
        self.assertEqual(matrix.shape, (len(SMILES), expected))

    def test_invalid_smiles_does_not_crash_featurization(self):
        matrix = fingerprints.featurize(["CCO", "not a molecule"])
        self.assertEqual(matrix.shape[0], 2)
        self.assertFalse(np.isnan(matrix).any())

    def test_identical_molecules_give_identical_rows(self):
        matrix = fingerprints.featurize(["c1ccccc1", "c1ccccc1"])
        np.testing.assert_array_equal(matrix[0], matrix[1])


class TrainingLoop(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.features = fingerprints.featurize(SMILES)
        self.ic50 = 10 ** rng.uniform(-1, 2, len(SMILES))
        self.cc50 = self.ic50 * 10 ** rng.uniform(0.5, 2.5, len(SMILES))
        self.si = self.cc50 / self.ic50

    def test_direct_strategy_runs_end_to_end(self):
        model = baseline.build("direct").fit(self.features, self.si)
        predicted = model.predict_log_si(self.features)
        self.assertEqual(predicted.shape, (len(SMILES),))
        self.assertTrue(np.isfinite(predicted).all())

    def test_two_head_strategy_runs_end_to_end(self):
        model = baseline.build("two_head").fit(self.features, self.ic50, self.cc50)
        predicted = model.predict_log_si(self.features)
        self.assertTrue(np.isfinite(predicted).all())

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            baseline.build("magic")

    def test_two_head_trains_when_some_ic50_are_missing(self):
        # Так размечены строки организаторов: IC50 не определён, цель задана
        # меткой SI = 10000. Раньше обучение падало на ValueError: y contains NaN.
        ic50 = self.ic50.copy()
        ic50[:3] = np.nan
        model = baseline.build("two_head").fit(self.features, ic50, self.cc50)
        self.assertEqual(model.n_train["IC50"], len(SMILES) - 3)
        self.assertEqual(model.n_train["IC50_skipped"], 3)
        self.assertEqual(model.n_train["CC50"], len(SMILES))
        self.assertTrue(np.isfinite(model.predict_log_si(self.features)).all())

    def test_each_head_uses_its_own_rows(self):
        # Непарные данные: у части строк есть только IC50, у части только CC50.
        ic50, cc50 = self.ic50.copy(), self.cc50.copy()
        ic50[:2] = np.nan
        cc50[-4:] = np.nan
        model = baseline.build("two_head").fit(self.features, ic50, cc50)
        self.assertEqual(model.n_train["IC50"], len(SMILES) - 2)
        self.assertEqual(model.n_train["CC50"], len(SMILES) - 4)

    def test_direct_skips_rows_without_a_target(self):
        si = self.si.copy()
        si[:2] = np.nan
        model = baseline.build("direct").fit(self.features, si)
        self.assertEqual(model.n_train["SI"], len(SMILES) - 2)

    def test_non_positive_values_are_skipped_not_clipped(self):
        ic50 = self.ic50.copy()
        ic50[0] = 0.0
        ic50[1] = -5.0
        model = baseline.build("two_head").fit(self.features, ic50, self.cc50)
        self.assertEqual(model.n_train["IC50_skipped"], 2)

    def test_clipping_caps_the_prediction(self):
        # Метрика соревнования — MSE по сырому SI, и её определяет хвост:
        # прогноз выше максимума из данных организаторов стоит сотни миллионов.
        cap = 1000.0
        model = baseline.build("direct", clip_si=cap).fit(self.features, self.si)
        predicted = baseline.from_log(model.predict_log_si(self.features))
        self.assertLessEqual(predicted.max(), cap * 1.000001)

    def test_clipping_also_caps_two_head(self):
        cap = 100.0
        model = baseline.build("two_head", clip_si=cap).fit(
            self.features, self.ic50, self.cc50
        )
        predicted = baseline.from_log(model.predict_log_si(self.features))
        self.assertLessEqual(predicted.max(), cap * 1.000001)

    def test_direct_clips_the_training_target_too(self):
        # Три молекулы ChEMBL с SI до 2.2e5 лежат вне диапазона соревнования и
        # без обрезки перетягивают обучение на себя.
        si = self.si.copy()
        si[0] = 2.2e5
        capped = baseline.build("direct", clip_si=1000.0).fit(self.features, si)
        free = baseline.build("direct").fit(self.features, si)
        self.assertLess(
            baseline.from_log(capped.predict_log_si(self.features)).max(),
            baseline.from_log(free.predict_log_si(self.features)).max(),
        )

    def test_no_clipping_by_default(self):
        model = baseline.build("direct")
        self.assertIsNone(model.clip_si)

    def test_sentinel_blends_by_probability(self):
        # Метка стоит на молекулах с общим мотивом, поэтому классификатор
        # должен их отделить, а прогноз на них — уйти к 10000.
        rng = np.random.default_rng(3)
        features = np.vstack([
            rng.normal(0, 0.1, (20, 6)),      # активные
            rng.normal(5, 0.1, (20, 6)),      # неактивные, явно другой кластер
        ])
        si = np.concatenate([10 ** rng.uniform(0, 2, 20), np.full(20, 10000.0)])
        is_label = np.concatenate([np.zeros(20, bool), np.ones(20, bool)])

        model = baseline.build("sentinel").fit(features, si, is_label)
        predicted = model.predict_si(features)
        self.assertLess(predicted[:20].mean(), 1000.0)
        self.assertGreater(predicted[20:].mean(), 5000.0)
        self.assertEqual(model.n_train["метки"], 20)
        self.assertEqual(model.n_train["регрессия"], 20)

    def test_sentinel_regression_never_sees_the_label_rows(self):
        # Иначе регрессия учится предсказывать разметку, а не концентрацию.
        features = np.random.default_rng(4).random((30, 5))
        si = np.concatenate([np.full(10, 10000.0), 10 ** np.linspace(0, 2, 20)])
        is_label = np.concatenate([np.ones(10, bool), np.zeros(20, bool)])
        model = baseline.build("sentinel").fit(features, si, is_label)
        self.assertEqual(model.n_train["регрессия"], 20)
        self.assertEqual(model.regressor.n_train["SI"], 20)

    def test_sentinel_survives_a_single_class(self):
        features = np.random.default_rng(5).random((12, 4))
        si = 10 ** np.linspace(0, 2, 12)
        model = baseline.build("sentinel").fit(features, si, np.zeros(12, bool))
        predicted = model.predict_si(features)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertLess(predicted.max(), 10000.0)

    def test_sentinel_probability_is_a_probability(self):
        rng = np.random.default_rng(6)
        features = np.vstack([rng.normal(0, 0.1, (15, 4)), rng.normal(4, 0.1, (15, 4))])
        si = np.concatenate([np.full(15, 3.0), np.full(15, 10000.0)])
        is_label = np.concatenate([np.zeros(15, bool), np.ones(15, bool)])
        model = baseline.build("sentinel").fit(features, si, is_label)
        probability = model.predict_proba_label(features)
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_a_head_with_no_usable_rows_fails_clearly(self):
        with self.assertRaises(ValueError) as caught:
            baseline.build("two_head").fit(
                self.features, np.full(len(SMILES), np.nan), self.cc50
            )
        self.assertIn("IC50", str(caught.exception))

    def test_log_roundtrip(self):
        values = np.array([0.1, 1.0, 1000.0])
        np.testing.assert_allclose(baseline.from_log(baseline.to_log(values)), values, rtol=1e-9)


class Metrics(unittest.TestCase):
    def test_perfect_prediction_scores_zero_error(self):
        true = np.array([0.0, 1.0, 2.0, 3.0])
        result = evaluate.metrics(true, true)
        self.assertAlmostEqual(result["rmse_log10"], 0.0)
        self.assertAlmostEqual(result["r2_log10"], 1.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            evaluate.metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_bootstrap_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(1)
        true = rng.normal(size=120)
        pred = true + rng.normal(scale=0.3, size=120)
        point = evaluate.metrics(true, pred)["rmse_log10"]
        low, high = evaluate.bootstrap_ci(true, pred, n_resamples=200)
        self.assertLess(low, point)
        self.assertGreater(high, point)

    def test_applicability_similarity_is_one_for_a_known_molecule(self):
        train = fingerprints.morgan_matrix(["c1ccccc1", "CCO"])
        query = fingerprints.morgan_matrix(["CCO"])
        similarity = evaluate.applicability_warning(train, query)
        self.assertAlmostEqual(similarity[0], 1.0, places=6)


class SplitsWiring(unittest.TestCase):
    def test_manifest_roundtrip(self):
        import tempfile
        from pathlib import Path

        split = splits.scaffold_split(SMILES)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            splits.save_manifest(path, split, splits.DEFAULT_SEED, len(SMILES))
            self.assertEqual(splits.load_manifest(path), split)


if __name__ == "__main__":
    unittest.main()
