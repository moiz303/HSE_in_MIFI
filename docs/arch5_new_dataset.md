# arch5: optional VACV reporter CC50 supplement

Reproduce the isolated comparison with:

```sh
python -m src.arch5_compare --config configs/xgb_arch5_new_dataset.json
```

The opt-in source is Chiem et al., DOI [10.1128/spectrum.04745-22](https://doi.org/10.1128/spectrum.04745-22). The raw files preserve 138 individual endpoint observations and a separate 184-row CC50/EC50 context table. The context table is an audit artifact, not a training target. After removing reserved holdout scaffolds, the supplement contributes 17 exact CC50 observations across 11 structures. EC50 is retained in its own column and never used as IC50.

The outer 3-fold scaffold comparison filters existing public data to development-training scaffolds, then appends only supplement observations outside the held scaffold and the globally reserved holdout scaffolds. The nested public OOF fit filters the passed public pool by each inner held scaffold, so eligible supplement structures outside the original development scaffold whitelist remain available without crossing the fold boundary. The final refit appends globally holdout-safe exact CC50 observations after the development-scaffold public whitelist. Kaggle targets remain their original per-row targets; no median target aggregation or SI-pair training is used.

| OOF run | Raw SI MSE | Scored predictions |
|---|---:|---:|
| Existing public source hierarchy | 31,529,999.321 | 801 |
| With exact supplement CC50 | 32,442,107.646 | 801 |

The optional supplement increased pooled 3-repeat scaffold OOF MSE by 912,108.325 (2.89%). Its Kaggle-only concentration-head control was 29,619,524.359. This experiment did not improve the source hierarchy and does not change any default or model promotion setting.

The new model, report, and row-level OOF predictions use the `_new_dataset` suffix. The saved model records `holdout_used=false`, `supplement=true`, the training config, the public head target counts, and the Ridge residual heads. A reload smoke check produced finite predictions for three structures. No test concentrations or Kaggle submissions were used.

Artifacts: `reports/arch5_source_hierarchy_new_dataset.json`, `reports/arch5_source_hierarchy_new_dataset_oof.csv`, and `artifacts/arch5_source_hierarchy_new_dataset.joblib`. The run and test logs are preserved alongside them.
