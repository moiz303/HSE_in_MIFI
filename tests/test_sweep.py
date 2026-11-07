"""Regression tests for model-race reporting."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import sweep


class RaceReporting(unittest.TestCase):
    def test_race_reports_std_of_per_repeat_mse(self):
        frame = pd.DataFrame({"SI": [1.0, 1.0, 1.0]})
        grid = [{"learner": "hgb"}, {"learner": "xgb"}]
        predictions = [
            np.ones(3),
            np.ones(3),
            np.full(3, 2.0),
            np.full(3, 3.0),
        ]

        with (
            patch.object(sweep, "fold_indices", return_value=[np.arange(3)]),
            patch.object(sweep, "oof_predictions", side_effect=predictions),
            patch.object(sweep, "paired_interval", return_value=(-1.0, 1.0)),
            patch.object(sweep, "log"),
        ):
            results, _ = sweep.race(
                grid=grid,
                frame=frame,
                features=np.zeros((3, 1)),
                in_distribution=np.ones(3, dtype=bool),
                smiles=["CC", "CCC", "CCCC"],
                folds=2,
                min_repeats=1,
                max_repeats=2,
                deadline=float("inf"),
            )

        by_learner = {result["learner"]: result for result in results}
        self.assertAlmostEqual(
            by_learner["hgb"]["mse_в_распределении_std"], 0.5
        )
        self.assertAlmostEqual(
            by_learner["xgb"]["mse_в_распределении_std"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
