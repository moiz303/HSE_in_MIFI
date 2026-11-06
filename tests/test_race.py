"""Тесты гонки кандидатов: отсев и остановка по метрике, а не по времени."""

import unittest

import numpy as np
import pandas as pd

import sweep

CORES = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "C1CCNCC1", "c1cscn1"]


def toy(n: int = 180, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.random((n, 5))
    si = np.clip(10 ** (1 + 2 * features[:, 0]) + rng.normal(0, 5, n), 1, None)
    smiles = [CORES[i % len(CORES)] + "C" * (i % 4) for i in range(n)]
    frame = pd.DataFrame({"SI": si, "standard_smiles": smiles})
    return frame, features, smiles, np.ones(n, dtype=bool)


class PairedInterval(unittest.TestCase):
    def test_interval_brackets_the_mean(self):
        differences = np.random.default_rng(0).normal(5.0, 1.0, 500)
        low, high = sweep.paired_interval(differences)
        self.assertLess(low, differences.mean())
        self.assertGreater(high, differences.mean())

    def test_clear_difference_excludes_zero(self):
        low, high = sweep.paired_interval(np.full(200, 3.0) + 0.01)
        self.assertGreater(low, 0.0)

    def test_no_difference_includes_zero(self):
        differences = np.random.default_rng(1).normal(0.0, 1.0, 500)
        low, high = sweep.paired_interval(differences)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)


class EmptyFolds(unittest.TestCase):
    def test_empty_fold_does_not_crash(self):
        # Если почти все молекулы делят каркас, групповое разбиение складывает
        # их в одну часть, и остальные фолды оказываются пустыми.
        frame, features, _, _ = toy(n=60)
        si = frame["SI"].to_numpy(float)
        prediction = sweep.oof_predictions(
            {"transform": "raw", "loss": "squared_error"},
            features, si,
            [np.arange(60), np.array([], dtype=int)],
        )
        self.assertEqual(len(prediction), 60)


class Race(unittest.TestCase):
    GRID = [
        {"transform": "raw", "loss": "squared_error", "clip_si": 20000.0},
        {"transform": "log", "loss": "squared_error", "clip_si": 20000.0},
        {"transform": "raw", "loss": "squared_error", "weight_power": 1.0,
         "clip_si": 20000.0},
    ]

    def test_race_stops_on_the_metric_not_the_clock(self):
        frame, features, smiles, inside = toy()
        _, verdict = sweep.race(
            self.GRID, frame, features, inside, smiles,
            folds=4, min_repeats=2, max_repeats=8, deadline=float("inf"),
        )
        # Бюджет не ограничивали, значит остановка должна быть содержательной.
        self.assertNotIn("бюджет", verdict["остановка"])
        self.assertLess(len(verdict["история"]), 8)

    def test_a_clearly_worse_candidate_is_eliminated_early(self):
        frame, features, smiles, inside = toy()
        results, _ = sweep.race(
            self.GRID, frame, features, inside, smiles,
            folds=4, min_repeats=2, max_repeats=6, deadline=float("inf"),
        )
        repeats = {row["описание"]: row["повторов"] for row in results}
        # Кто-то обязан выбыть раньше остальных, иначе отсев не работает.
        self.assertLess(min(repeats.values()), max(repeats.values()))

    def test_single_candidate_race_finishes(self):
        frame, features, smiles, inside = toy(n=120)
        results, verdict = sweep.race(
            self.GRID[:1], frame, features, inside, smiles,
            folds=3, min_repeats=2, max_repeats=4, deadline=float("inf"),
        )
        self.assertEqual(len(results), 1)
        self.assertIn("лидер", verdict)

    def test_deadline_is_still_respected_when_given(self):
        frame, features, smiles, inside = toy()
        _, verdict = sweep.race(
            self.GRID, frame, features, inside, smiles,
            folds=4, min_repeats=2, max_repeats=8, deadline=0.0,
        )
        self.assertIn("бюджет", verdict["остановка"])
