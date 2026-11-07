# Direct CatBoost SI VACV baseline
This isolated workflow trains one seeded CPU `CatBoostRegressor` directly on
`SI VACV`. Its only model inputs are the existing Morgan 2048-bit fingerprint
and 13 RDKit descriptors generated from `SMILES`. `CC50_uM` and
`IC50_uM_VACV` are not used for feature construction, filtering, validation, or
prediction. Zero-valued SI labels are retained.
From the repository root, run:
```bash
python -m baseline_catboost.run
```
The defaults load:
- `attached_assets/train_v2.csv` (320 rows)
- `attached_assets/test.csv` (597 rows)
- `attached_assets/sample_submit.csv` (597 rows)
The runner checks the required headers, SMILES parseability, target values,
expected row counts, and test/template ID order. It creates `outputs/` with:
- `submission.csv` — exactly `index,SI VACV`, in sample-template order, with
  model predictions replacing the template target values
- `catboost_si_vacv.cbm` — the final model refit on every labeled training row
- `validation.json` — deterministic Murcko scaffold split details and raw-SI
  validation MSE
The validation MSE is a local scaffold-validation result, not the official
competition score. To use alternate inputs, pass `--train`, `--test`, and
`--sample` paths and set `--output-dir`. The supplied row counts are checked by
default; use `--no-expected-row-counts` for other dataset sizes. Test and
template IDs must always match exactly in row order.
Run focused tests with:
```bash
python -m unittest tests.test_baseline_catboost -v
```