# Architecture 4: shared D-MPNN with a direct SI head

Run the nested OOF experiment and final development refit with:

```bash
PYTHONPATH=/Users/f37/HSE_arch_env/lib/python3.12/site-packages:$PWD python3 -m src.dmpnn_workflow
```

The checked-in settings are in `configs/dmpnn_arch4.json`. The loader validates the committed split manifest; the fit receives only train and validation frames. It scores the original Kaggle training targets from `kaggle_reference.csv`; duplicate structures are count-weighted in the direct-SI loss. IC50 and CC50 use independent masks, exact log-MSE, and one-sided losses for censored bounds. Unknown or conflicting qualifiers are skipped. Inner scaffold folds choose auxiliary-loss weight and stopping epoch; outer OOF folds are not used for selection. The final artifact is refit on the 1007 development structures. Holdout endpoints and test endpoints are not used.

## Results

The seed 20261106 scaffold OOF scored 267 original Kaggle rows:

| Targets | Weighted XGBoost baseline | D-MPNN | Change |
|---|---:|---:|---:|
| All rows | 29,657,226.52 | 23,912,715.09 | -19.37% |
| Service-label rows | 99,643,915.00 | 49,691,463.00 | -50.13% |
| Measured rows | 6,676,522.73 | 15,448,051.60 | +131.38% |

The overall reduction is driven by lower error on the service-label subgroup; error on measured SI more than doubles. This is an OOF development comparison, not a Kaggle score, and it does not establish a model promotion. The existing default and baseline remain unchanged.

Artifacts: `reports/dmpnn_arch4_training.json`, `reports/dmpnn_arch4_oof.csv`, and `artifacts/dmpnn_arch4.pt`. The checkpoint records its training settings and runtime package versions.
