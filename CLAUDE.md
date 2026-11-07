# Project instructions

Read README.md. Default is configs/xgb_hybrid.json: XGBoost concentration heads plus an experimental label classifier, selected against the unchanged baseline on fixed validation. CatBoost remains an explicit comparison config.

Never train on this Windows machine. Real-data model fitting, sweeps and comparisons run on the teammate's Mac through `./run_mac.sh`. Local data preparation, static checks and unittest toy-model tests are allowed.

The previous HGB, sentinel, mixture, calibration and sweep implementations were removed. Do not restore their defaults or load historical model artifacts. Historical findings are summarized in `docs/previous_experiments.md`.

Keep raw data and source provenance. Concentrations are ?M; model log10(IC50) and log10(CC50), then subtract to obtain log10(SI). Mask missing/nonpositive/nonfinite and censored targets independently for each head. Never derive missing concentrations from an SI label.

Preserve the committed scaffold split. Fit on train only; report validation without claiming superiority from a single split. Holdout is reserved for final release. Do not derive predictions or assay flags from test concentrations.
