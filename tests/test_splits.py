"""Тест на утечки между частями. Стадия 2, владелец B.

Самая дорогая ошибка в задаче: при утечке метрика выглядит отлично, а на
закрытом тесте Kaggle результат рассыпается. Поэтому тест стоит в CI.
"""

import unittest

from src import splits

# Два химических ряда с общими каркасами плюс несколько одиночных молекул.
BENZAMIDES = [
    "O=C(Nc1ccccc1)c1ccccc1",
    "O=C(Nc1ccc(C)cc1)c1ccccc1",
    "O=C(Nc1ccc(Cl)cc1)c1ccccc1",
    "O=C(Nc1ccc(F)cc1)c1ccccc1",
    "O=C(Nc1ccc(OC)cc1)c1ccccc1",
]
QUINOLINES = [
    "c1ccc2ncccc2c1",
    "Cc1ccc2ncccc2c1",
    "Clc1ccc2ncccc2c1",
    "COc1ccc2ncccc2c1",
]
SINGLES = ["CCO", "CCCC", "c1ccccc1", "C1CCCCC1", "c1ccncc1", "O=C(O)CCl"]
DATASET = BENZAMIDES + QUINOLINES + SINGLES


class ScaffoldSplit(unittest.TestCase):
    def setUp(self):
        self.split = splits.scaffold_split(DATASET, seed=splits.DEFAULT_SEED)

    def test_every_molecule_lands_exactly_once(self):
        assigned = [index for indices in self.split.values() for index in indices]
        self.assertEqual(sorted(assigned), list(range(len(DATASET))))

    def test_no_molecule_or_scaffold_crosses_the_border(self):
        splits.assert_no_leaks(DATASET, self.split)

    def test_a_chemical_series_stays_in_one_part(self):
        where = {}
        for name, indices in self.split.items():
            for index in indices:
                where[DATASET[index]] = name
        self.assertEqual(len({where[s] for s in BENZAMIDES}), 1)
        self.assertEqual(len({where[s] for s in QUINOLINES}), 1)

    def test_same_seed_gives_the_same_split(self):
        again = splits.scaffold_split(DATASET, seed=splits.DEFAULT_SEED)
        self.assertEqual(self.split, again)

    def test_leak_detector_actually_fires(self):
        broken = {"train": [0, 1], "validation": [1], "holdout": [2]}
        with self.assertRaises(AssertionError):
            splits.assert_no_leaks(DATASET, broken)

    def test_fractions_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            splits.scaffold_split(DATASET, fractions={"train": 0.5, "validation": 0.2})


if __name__ == "__main__":
    unittest.main()
