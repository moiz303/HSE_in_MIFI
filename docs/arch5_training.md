# Architecture 5 training

Reproduce: `python -m src.arch5_compare --config configs/xgb_arch5.json` in scientific runtime including XGBoost. 3-fold scaffold development CV, seed 20261106, nested scaffold public predictions for residual fitting. Original Kaggle reference is filtered to development structures; holdout/test targets excluded. Final public/residual models saved in `artifacts/arch5_source_hierarchy.joblib` with inference formula and config.

See `reports/arch5_source_hierarchy.json` for raw SI MSE on original targets and Kaggle-only control. This is one-repeat CV, not a Kaggle public score. Baseline default unchanged. Four architecture-specific tests passed.
