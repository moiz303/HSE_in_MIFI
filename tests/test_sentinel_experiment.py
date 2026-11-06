"""Тесты метрик опыта про метки. Сети нет, данные синтетические.

Проверяется арифметика метрик, а не то, что классификатор помогает: помогает
или нет — вопрос к опыту на реальных данных.
"""

import unittest

import numpy as np

import experiment_sentinel as sentinel


class Auc(unittest.TestCase):
    LABELS = np.array([0, 0, 0, 1, 1, 1], dtype=bool)

    def test_perfect_ranking(self):
        self.assertAlmostEqual(
            sentinel.auc(self.LABELS, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])), 1.0
        )

    def test_reversed_ranking(self):
        self.assertAlmostEqual(
            sentinel.auc(self.LABELS, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])), 0.0
        )

    def test_constant_score_is_a_coin_flip(self):
        # Связки обязаны усредняться, иначе AUC смещается на одинаковых
        # вероятностях — а это частый случай у леса.
        self.assertAlmostEqual(sentinel.auc(self.LABELS, np.full(6, 0.5)), 0.5)

    def test_single_class_gives_nan(self):
        self.assertTrue(np.isnan(sentinel.auc(np.ones(4, dtype=bool), np.arange(4.0))))


class Calibration(unittest.TestCase):
    def test_honest_probability_beats_overconfident_one(self):
        rng = np.random.default_rng(0)
        probability = rng.uniform(0, 1, 4000)
        labels = (rng.uniform(size=4000) < probability).astype(float)
        overconfident = np.clip((probability - 0.5) * 3 + 0.5, 0, 1)
        self.assertLess(
            sentinel.expected_calibration_error(labels, probability),
            sentinel.expected_calibration_error(labels, overconfident),
        )

    def test_perfect_calibration_is_near_zero(self):
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertLess(sentinel.expected_calibration_error(labels, np.full(4, 0.5)), 1e-9)

    def test_probability_of_one_on_all_positives(self):
        labels = np.ones(10)
        self.assertLess(sentinel.expected_calibration_error(labels, np.ones(10)), 1e-9)


class Classifiers(unittest.TestCase):
    def test_every_family_builds_and_gives_probabilities(self):
        rng = np.random.default_rng(1)
        features = rng.random((40, 6))
        target = (features[:, 0] > 0.5).astype(int)
        for kind in ("hgb", "rf", "extra", "logreg"):
            classifier = sentinel.build_classifier(kind)
            classifier.fit(features, target)
            probability = classifier.predict_proba(features)[:, 1]
            self.assertTrue(((probability >= 0) & (probability <= 1)).all(), kind)

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):
            sentinel.build_classifier("нейросеть")
