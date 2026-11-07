# MolE Ridge after curated VACV supplement

This is a controlled data addition to the frozen-MolE/Ridge experiment in [mole_ridge_experiment.md](mole_ridge_experiment.md). It compares the same model and the same original OOF structures, targets, 3×3 molecule/scaffold folds, and per-fold Ridge alphas. Alphas were frozen from the earlier nested train-only selection to isolate the effect of adding measurements. No alpha was selected from the after-MSE.

`load_supplement` first applies the repository-global holdout scaffold guard. The result has 126 observations on 19 structures: 84 EC50 and 42 CC50 observations. Of the CC50 observations, 17 exact values on 11 structures pass the model's exact-target mask. Only those 17 values were appended to training rows. Each added row has `IC50=NaN`; the 84 EC50 rows were never converted or used, and the 25 censored CC50 values were not treated as exact. Raw endpoint labels and assay provenance remain in `data/raw/vacv_2023_reporter_observations.csv`.

Within each outer fold, supplemental observations are filtered against the held structures (molecule protocol) or held scaffolds (scaffold protocol) before they enter outer training. The original 267 evaluation rows are then scored unchanged. Across folds, 13–17 exact CC50 observations were eligible per outer training set. The committed development split and holdout were not rebuilt or scored.

| Features | Molecule before → after MSE | Change | Scaffold before → after MSE | Change |
|---|---:|---:|---:|---:|
| MolE | 43.298M → 33.382M | −22.90% | 10,860.99M → 138,724.70M | +1177.27% |
| Morgan | 29.677M → 29.682M | +0.014% | 29.680M → 29.681M | +0.002% |
| MolE + Morgan | 29.581M → 29.568M | −0.043% | 29.697M → 29.700M | +0.009% |

The small fusion differences are effectively zero and point in opposite directions. The MolE-only molecule improvement is not robust: its scaffold score worsened sharply. This supplement run did not establish a Kaggle gain.

Fixed validation was also kept separate: its 67 original rows were evaluated using train-only models and the same alpha selected in the earlier train-only run. Fusion raw SI MSE moved from `51,836,265` to `51,842,315` (+0.012%). The development refits saved under `artifacts/mole/*_new_dataset.joblib` include train and validation plus the 17 globally safe exact CC50 observations, so these refits have no independent validation score. Holdout and test endpoints were never used.

Full counts, fold manifests, per-repeat scores, and split-specific supplement counts are in `reports/mole_ridge_new_dataset_comparison.json`, `reports/mole_ridge_new_dataset_oof.csv`, `reports/mole_ridge_new_dataset_splits.json`, and `reports/mole_ridge_new_dataset_comparison.csv`.
