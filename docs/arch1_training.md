# Architecture 1 training

Runtime: `/Users/f37/HSE_arch_env/bin/python` (Python 3.12).

Reproduce: `python -m src.tanimoto_compare`. Fixed alpha=1.0, no tuning on outer folds. 1007 development structures, 267 original Kaggle rows, 3 folds x 3 repeats in molecule and scaffold protocols (seeds 20261106–20261108). Holdout/test not used. Final artifact: `artifacts/tanimoto_krr_final.joblib`, fitted on development only.

Raw SI MSE: molecule 29,740,969.16 vs weighted XGB 29,659,649.91; scaffold 29,746,809.16 vs 29,682,392.14. No confirmed improvement, baseline remains unchanged. These are OOF scores, not Kaggle public scores. Fixed-split report is supplementary and uses historical aggregated SI; use original-target OOF report for comparison.

Real execution revealed pandas read-only target arrays; extraction now copies before masking, covered by a regression test. Four architecture-specific tests passed.
