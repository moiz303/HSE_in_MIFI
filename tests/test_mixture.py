"""Тесты вариантов смеси для строк-меток. Сети нет, данные синтетические."""

import unittest

import numpy as np

import experiment_mixture as sentinel


def out(base=500.0, plain=400.0, probability=0.8):
    return {
        "база": np.array([base]),
        "регрессия_неметки": np.array([plain]),
        "p_все": np.array([probability]),
        "p_в_распределении": np.array([probability]),
    }


class Variants(unittest.TestCase):
    def test_base_is_passed_through_untouched(self):
        built = sentinel.build_variants(out())
        self.assertEqual(built["база"][0], 500.0)
        self.assertEqual(built["регрессия_неметки"][0], 400.0)

    def test_full_weight_mixture_is_the_textbook_formula(self):
        # сж=1: p * 10000 + (1 - p) * второе слагаемое.
        built = sentinel.build_variants(out(probability=0.8))
        self.assertAlmostEqual(built["смесь_все_сж1"][0], 0.8 * 10000 + 0.2 * 400.0)
        self.assertAlmostEqual(built["сбазой_все_сж1"][0], 0.8 * 10000 + 0.2 * 500.0)

    def test_shrink_scales_the_label_weight_not_the_label(self):
        # Сжатие умножает вероятность, а не значение метки: при сж=0.5 и p=0.8
        # вес метки 0.4, остаток 0.6.
        built = sentinel.build_variants(out(probability=0.8))
        self.assertAlmostEqual(built["сбазой_все_сж0.5"][0], 0.4 * 10000 + 0.6 * 500.0)

    def test_zero_probability_leaves_the_second_term_alone(self):
        built = sentinel.build_variants(out(probability=0.0))
        for name, expected in (("сбазой_все_сж1", 500.0), ("смесь_все_сж1", 400.0)):
            self.assertAlmostEqual(built[name][0], expected, msg=name)

    def test_base_blend_never_drops_below_the_base_when_label_is_higher(self):
        # Метка 10000 выше прогноза, поэтому подмешивание может только поднять.
        built = sentinel.build_variants(out(base=500.0, probability=0.3))
        for name, value in built.items():
            if name.startswith("сбазой"):
                self.assertGreaterEqual(value[0], 500.0, name)

    def test_every_shrink_level_produces_both_families(self):
        built = sentinel.build_variants(out())
        for shrink in sentinel.SHRINK:
            if shrink == 0.0:
                continue
            for family in ("смесь", "сбазой"):
                for subset in ("все", "в_распределении"):
                    self.assertIn(f"{family}_{subset}_сж{shrink:g}", built)


class SentinelProbability(unittest.TestCase):
    def test_single_class_training_part_gives_a_constant(self):
        # Вырожденный fold: меток в обучающей части нет, классификатор не
        # обучить. Вероятность должна стать постоянной, а не упасть.
        features = np.random.default_rng(2).random((10, 3))
        label = np.zeros(10, dtype=int)
        probability = sentinel.sentinel_probability(
            features, label, np.arange(6), np.arange(6, 10)
        )
        self.assertEqual(len(probability), 4)
        self.assertTrue(np.allclose(probability, 0.0))

    def test_empty_training_part_gives_zero(self):
        features = np.random.default_rng(2).random((4, 3))
        label = np.array([0, 1, 0, 1])
        probability = sentinel.sentinel_probability(
            features, label, np.array([], dtype=int), np.arange(4)
        )
        self.assertTrue(np.allclose(probability, 0.0))


if __name__ == "__main__":
    unittest.main()
