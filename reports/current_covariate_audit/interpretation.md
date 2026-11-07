# Train/test covariate-shift diagnostic

Only development/test SMILES and frozen author embeddings enter the source
classifier. SI and official concentration fields never enter its fitting.
Five-fold source OOF groups identical molecules across both sources.

Source ROC AUC is 0.53765. Log-loss is 0.73013, worse than the constant source
prior's 0.61832. Estimated density weights reduce effective sample size from
267 to 110.4. This classifier does not support useful importance weighting.

Weighted OOF numbers in weighted_comparison.csv are diagnostic estimates under
the strong assumption that P(SI|SMILES) stays unchanged. They are neither official
Kaggle Scores nor calibrated predictions of them. The failed GP remains top-ranked
by these weights despite its official 18.56M failure; this approach does not solve
the validation disagreement. No model was promoted, retrained with these weights,
or submitted as a result of this audit.

22 test structures also occur in development, each with one development label.
No test concentration or target was read. Exact structural matches do not prove
the SI of another assay, so this overlap alone is not evidence for a new submission.
