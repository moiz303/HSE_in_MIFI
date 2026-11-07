# Current project instructions

Read preserved AGENTS.md; the current user-approved protocol below supersedes
its obsolete XGBoost/Mac-default descriptions. Historical instructions are in
legacy/CLAUDE_v1.md, not current defaults.

## Competition protocol

Competition: molecular-activity-prediction-hack, target SI VACV, raw MSE.
The live data-description API forbids CC50_uM and IC50_uM_VACV for model building
and prediction. User confirmed no organizer exception: STRICT interpretation.
Never use these competition columns as features, labels, auxiliary targets or
filters. Current training reads only SMILES + SI VACV; inference index + SMILES.
Independent public measurements may produce public SI or separate auxiliary
IC50/EC50/CC50 tasks. Never substitute official competition concentrations.
Current package: src/current/, config configs/current.json, root train_model.py.
Legacy concentration models are historical and ineligible for the current strict
workflow; their old public scores are not scores of new SI models.

## Data and evaluation

Authoritative train_v2.csv SHA256:
d9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8.
81 legacy targets10000 became0; all other official train columns/order unchanged.
Data/current contains versioned cohorts, public dataset, rejection journal,
source hashes and old-role-compatible splits. Keep old processed tables immutable;
only their structures/roles, not targets, are read for the current split.
267 original competition development rows (66 zeros); 53 excluded/reserved rows
never fitted or evaluated, including2 unmapped legacy-rejected structures.
Public:109 exact IC50/CC50 pairs,87 structures; fit80 pairs/63 new structures.
Same document+nonempty cell line, no censoring/EC50/TC50/GI50, DOI retained.
Public competition overlap and reserved scaffolds excluded. No source averaging.

Initial comparison: fourteen models,7 SI methods ×2 source regimes, same18 OOF folds
(molecule/scaffold,3 folds ×3 repeats), fixed seeds. Raw-SI first block and log1p
controls retained; log controls motivated by TRAIN OOF loss/tail, not test labels.
OOF selection on all current original SI including zeros, before each block's
separate endpoint diagnostics. Initial OOF/current best OFFICIAL candidate:
chemeleon_domain_log1p__public_plus_kaggle; mean OOF5061848.47814.
Best separate endpoint subset MSE1968835.32348 is catboost_log1p mixed; it does
NOT change selection. Original testing_main CatBoost control4187780.78852 is
exactly reproduced. Best verified current Kaggle Score1586333.78789/ref58103006.
Later lowest raw OOF4973199.155 mixed GP FAILED Kaggle18558089.19131;
never automatically promote it. Solo hurdle1latent Kaggle1949484.90648 also lost.
All configurations/source regimes/strata/official scores: reports/current/all_models.csv/md.
One development observation accounts for about96% of some models' raw MSE.
Strata are diagnostics; they do not replace official MSE. Goal top10 OR
Score<800000 is not yet achieved.
Exploratory structure-cluster bootstrap for selected-vs-zero includes no gain;
multiple selection on reused OOF is not independent validation.

## Actions and reproduction

Training on this Windows laptop explicitly authorized. Use ../env_arch/Scripts/python.exe,
Python3.12, requirements-current.txt; Mac launcher uses .venv-current.
No automatic Kaggle submissions from training scripts. User explicitly authorized
best-hypothesis uploads in the followup goal: top10 OR Score<800000, four attempts
at goal start. Live attempt/ref/Score registry: reports/current/kaggle_goal_submissions.json.
Check it and current team submissions before any new upload; never duplicate a submission.
Initial four authorized attempts are used (20/20 team submissions on2026-11-07).
User expects reset2026-11-08 03:00Europe/Moscow /00:00UTC. Verify live remaining
quota after reset and upload only selected, reproducible hypotheses.
One selected biological candidate is queued for2026-11-08 00:00:05UTC:
reports/current/queued_submission_plan.json. Its separate process has one upload
call, bounded network/score waits, a one-shot lock, and live quota/hash/dedup checks.
If user pauses/cancels the goal before upload, create the cancel_file named in
reports/current/queued_submission_plan.json
and stop the confirmed live queue process. Do not relaunch merely because a state
file is stale: inspect the tool session/process and remote submission history.
Queue state/log/lock live under logs/; API refs/scores are appended to the tracked
reports/current/kaggle_goal_submissions.json. Never duplicate an ambiguous upload.
No messages to organizers/others without explicit authorization.
Holdout reserved. Test endpoint diagnostics allowed only separately after saved
predictions via src.test_diagnostics; never fit, tune, infer labels or promote
models from them. Diagnostic raw reports remain ignored/local.
Current candidate pointer: reports/current/best_candidate.json. Prior verified
public1614079.37878 remains historical only, not current protocol eligibility.

Tests: python -m unittest discover -s tests -t . -v.
Train: python train_model.py --stage all. Summaries: python -m src.current.results.
Fresh SMILES-only artifact verification: python -m src.current.verify.
Public collector: python -m src.current.collect (immutable new snapshot).
Current source audit: docs/current_protocol.md, reports/current/competition_audit.json.
Independent audit: reports/current/protocol_audit.json/md. GP log1p means are
Gaussian retransformation heuristics, not exact expectations of clipped SI.
TabPFN fresh-process reproduction failed; do not upload/promote until repaired.
User allows up to3agents, with distinct file ownership and no agent Kaggle uploads.
Keep reporting completed milestones and failures; avoid silent unbounded sweeps.
New source changes require new prepared version; never silently reuse checkpoints.
Preserve AGENTS.md unchanged. Commit/push completed verified work to origin/main;
no force push, no discarding unrelated changes. Keep legacy module paths required
to open historical artifacts. Prior sentinel experiment cancelled; do not resume.
