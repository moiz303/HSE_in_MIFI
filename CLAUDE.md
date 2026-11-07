# Project instructions

Read README.md. Default is configs/baseline.json, using recovered concentration XGBoost heads. Kaggle CV-hybrid score 6944427.34748 was worse than previous 1975037.74735. Hybrid promotion is disabled unless model.promote_candidate=true explicitly. Never claim a validation gain implies a leaderboard gain. Research configs remain available; holdout stays excluded.

User authorized final refit on train+validation (option 1), available explicitly via configs/xgb_final.json with fixed baseline parameters and separate artifacts. Do not report validation as independent after this refit. Pipeline inference defaults to the preserved verified baseline unless another config is explicitly supplied. No Kaggle submission without separate user instruction.

Current training default is option 2, configs/xgb_ensemble.json: five concentration pairs with seeds 20261106..20261110 on the SAME train+validation rows as option 1. Arithmetic mean of member SI predictions, no classifier. Preserve option 1 artifacts; do not claim ensemble improvement before evaluation. Holdout remains excluded.

Never train on this Windows machine. Real-data model fitting, sweeps and comparisons run on the teammate's Mac through `./run_mac.sh`. Local data preparation, static checks and unittest toy-model tests are allowed.

The previous HGB, sentinel, mixture, calibration and sweep implementations were removed. Do not restore their defaults or load historical model artifacts. Historical findings are summarized in `docs/previous_experiments.md`.

Keep raw data and source provenance. Concentrations are ?M; model log10(IC50) and log10(CC50), then subtract to obtain log10(SI). Mask missing/nonpositive/nonfinite and censored targets independently for each head. Never derive missing concentrations from an SI label.

Preserve the committed scaffold split. Fit on train only; report validation without claiming superiority from a single split. Holdout is reserved for final release. Do not derive predictions or assay flags from test concentrations.
