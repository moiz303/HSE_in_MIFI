# Project instructions

Latest requested experiment/training default: configs/xgb_tail.json. Compare bounded high-SI emphasis (IC50 cap2/cap4, both heads cap2) against fixed reliability weights on identical development molecule/scaffold OOF splits, 3 folds x 3 repeats. Factors use only exact paired concentrations and explicitly measured labels in each fit fold, never test targets; sqrt((CC50/IC50)/100), clipped to [1,cap]. Require >=2% molecule MSE gain, wins in >=2/3 repeats, and <=2% deterioration on scaffold, measured and SI<=100 subsets. Fall back to reliability weights if checks fail. Preserve earlier models and the verified inference default; no automatic Kaggle submission. AGENTS.md is preserved as requested by the user; this paragraph supersedes its earlier training-default descriptions.

Windows tail experiment completed 2026-11-07: all candidates failed the predeclared checks; selected weighted. Best molecule improvement only 0.205% (ic4), with SI<=100 scaffold MSE 10.56x baseline. Saved tail submission exactly matches the previous Windows analogue alpha=0 output. See reports/tail_comparison.json and docs/kaggle_progress.md. Do not describe this as a successful improvement or a new official score.

Read README.md. Default is configs/baseline.json, using recovered concentration XGBoost heads. Kaggle CV-hybrid score 6944427.34748 was worse than previous 1975037.74735. Hybrid promotion is disabled unless model.promote_candidate=true explicitly. Never claim a validation gain implies a leaderboard gain. Research configs remain available; holdout stays excluded.

User authorized final refit on train+validation (option 1), available explicitly via configs/xgb_final.json with fixed baseline parameters and separate artifacts. Do not report validation as independent after this refit. Pipeline inference defaults to the preserved verified baseline unless another config is explicitly supplied. No Kaggle submission without separate user instruction.

Current training default is option 3, configs/xgb_weighted.json: fixed reliability-weighted concentration heads. The Mac run first compares options 1/2/3 using original Kaggle OOF targets in molecule and scaffold protocols (3 folds, 3 repeats), then fits option 3 on train+validation. See docs/local_scoring.md. Preserve previous artifacts. No automatic promotion or Kaggle submissions. Holdout remains excluded.

Latest authorized experiment/default: configs/xgb_analog.json. Weighted XGBoost plus independent training-only IC50/CC50 neighbor banks; Morgan/Tanimoto, k=5, minimum similarity=0.5. Alpha 0/.1/.25/.5 selected only via original training OOF targets with scaffold/measured-target safeguards. Holdout and test endpoints never enter the banks or alpha selection. Preserve weighted and prior artifacts. Do not confuse this molecular-analogue experiment with the earlier five-seed option 2.

The user has explicitly lifted the earlier no-training restriction: real-data training and comparisons are now authorized on this Windows laptop. Use ../env/Scripts/python.exe for local runs; ./run_mac.sh remains the Mac-specific launcher. Preserve holdout exclusions and logs. Kaggle submissions still require a separate explicit request.

The previous HGB, sentinel, mixture, calibration and sweep implementations were removed. Do not restore their defaults or load historical model artifacts. Historical findings are summarized in `docs/previous_experiments.md`.

Keep raw data and source provenance. Concentrations are ?M; model log10(IC50) and log10(CC50), then subtract to obtain log10(SI). Mask missing/nonpositive/nonfinite and censored targets independently for each head. Never derive missing concentrations from an SI label.

Preserve the committed scaffold split. Fit on train only; report validation without claiming superiority from a single split. Holdout is reserved for final release. Do not derive predictions or assay flags from test concentrations.

User now explicitly authorizes post-inference LOCAL error diagnostics using test CC50/IC50. Use src/test_diagnostics.py on saved predictions only; do not import it into training or inference. No guessing missing endpoints/labels, no automated selection from this diagnostic, no exact Kaggle-score claim. Outputs are gitignored aggregates; derived per-row test targets are not exported.
