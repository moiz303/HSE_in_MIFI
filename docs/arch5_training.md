# Architecture 5 training

Reproduce: `python -m src.arch5_compare --config configs/xgb_arch5.json` in scientific runtime including XGBoost. 3-fold scaffold development CV x3 seeds (20261106–20261108), nested scaffold public predictions for residual fitting. Original Kaggle reference is filtered to development structures; holdout/test targets excluded. Final public/residual models saved in `artifacts/arch5_source_hierarchy.joblib` with inference formula and config.

Raw original-target SI MSE: 29,579,318.71 vs Kaggle-only control 29,619,524.36. Weighted XGBoost from architecture 1 on verified identical original rows/folds/repeats: 29,682,392.14 (hierarchy gain ~0.35%, exploratory). These are OOF scores, not Kaggle public scores. Baseline default unchanged.

Real-data runs caught two runner issues: filtering the full Kaggle reference to development, and retaining unrelated public scaffolds inside nested OOF (only held scaffolds should be excluded). The latter is covered by a new regression test. Five architecture-specific tests passed; saved artifact reload/inference passed. Huge apparent loss in an intermediate buggy run was corrected before final results.
