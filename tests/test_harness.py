"""Тесты общего каркаса опытов: на нём теперь держатся все проверки гипотез."""

import unittest

import numpy as np
import pandas as pd

from src.experiments import harness

CORES = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1"]


def toy(n: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.random((n, 5))
    si = np.clip(10 ** (1 + 2 * features[:, 0]), 1, None)
    smiles = [CORES[i % len(CORES)] + "C" * (i % 3) for i in range(n)]
    frame = pd.DataFrame({"SI": si, "standard_smiles": smiles})
    return frame, features, si, smiles, np.ones(n, dtype=bool)


class Folds(unittest.TestCase):
    def test_folds_cover_every_row_once(self):
        _, _, _, smiles, _ = toy()
        folds = harness.fold_indices(smiles, 4, 1)
        assigned = np.concatenate(folds)
        self.assertEqual(sorted(assigned), list(range(len(smiles))))

    def test_fold_sets_differ_between_repeats(self):
        _, _, _, smiles, _ = toy()
        sets = harness.fold_sets(smiles, 4, 2)
        self.assertEqual(len(sets), 2)
        self.assertFalse(np.array_equal(sets[0][0], sets[1][0]))


class Oof(unittest.TestCase):
    def test_empty_fold_is_skipped(self):
        _, features, si, _, _ = toy(n=60)
        prediction = harness.oof_predictions(
            harness.model_variant(harness.BASE_SETTINGS, features, si),
            60, [np.arange(60), np.array([], dtype=int)],
        )
        self.assertEqual(len(prediction), 60)

    def test_predictions_land_only_in_held_rows(self):
        _, features, si, smiles, _ = toy()
        folds = harness.fold_indices(smiles, 4, 1)
        prediction = harness.oof_predictions(
            harness.model_variant(harness.BASE_SETTINGS, features, si),
            len(si), folds[:1],
        )
        outside = np.setdiff1d(np.arange(len(si)), folds[0])
        self.assertTrue(np.isnan(prediction[outside]).all())


class Compare(unittest.TestCase):
    def test_base_is_marked_and_verdicts_are_filled(self):
        frame, features, si, smiles, inside = toy()
        folds = harness.fold_sets(smiles, 4, 2)
        table = harness.compare({
            "база": harness.model_variant(harness.BASE_SETTINGS, features, si),
            "лог": harness.model_variant(
                {**harness.BASE_SETTINGS, "transform": "log"}, features, si
            ),
        }, si, inside, folds)
        self.assertEqual(int(table["это_база"].sum()), 1)
        self.assertEqual(str(table.loc[table["это_база"], "вариант"].iloc[0]), "база")
        self.assertTrue(set(table["вердикт"]) <= {"база", "лучше", "хуже", "в пределах шума"})

    def test_unknown_base_is_rejected(self):
        frame, features, si, smiles, inside = toy(n=80)
        with self.assertRaises(ValueError):
            harness.compare(
                {"а": harness.model_variant(harness.BASE_SETTINGS, features, si)},
                si, inside, harness.fold_sets(smiles, 3, 1), base="нет такого",
            )

    def test_identical_variants_are_within_noise(self):
        # Один и тот же вариант под двумя именами обязан дать «в пределах шума»:
        # иначе сравнение systematically смещено.
        frame, features, si, smiles, inside = toy()
        folds = harness.fold_sets(smiles, 4, 2)
        table = harness.compare({
            "один": harness.model_variant(harness.BASE_SETTINGS, features, si),
            "тот же": harness.model_variant(harness.BASE_SETTINGS, features, si),
        }, si, inside, folds)
        other = table[~table["это_база"]].iloc[0]
        self.assertEqual(other["вердикт"], "в пределах шума")
        self.assertAlmostEqual(float(other["разница_с_базой"]), 0.0, places=6)
