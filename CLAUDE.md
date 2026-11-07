# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Never train on this machine

**Do not run model training here.** No `python -m src.pipeline train`, no `fit()` on real data, no sweeps, no multi-variant comparison runs. This laptop is not the machine for that workload.

Instead: export the data and hand the teammate a runnable script. They train on their Mac.

```bash
python -m src.pipeline export      # writes the two data files below — safe to run here
python train_model.py --help       # the script the TEAMMATE runs on the Mac
```

- `data/processed/train.csv` — обучающая выборка
- `data/processed/validation.csv` — проверочная выборка
- `data/processed/splits.json` — манифест разбиения; holdout живёт только здесь и в выборки не экспортируется

Safe to run locally: data download, schema validation, standardisation, the export step, and the `unittest` suite (seconds, fits only toy models in `test_smoke`). If a conclusion needs training, state what to run and on which machine, then stop.

## Status

Scaffold is in place; the science is not. Working: `src/data/schema.py` (data contract, unit conversion, censored values), `src/data/standardize.py` (RDKit standardisation, log-median aggregation), `src/splits.py` (Murcko scaffold split + leak assertion), `src/features/fingerprints.py`, `src/models/baseline.py` (`direct` and `two_head` strategies), `src/evaluate.py`, `src/pipeline.py` (CLI). Deliberately stubbed with `NotImplementedError`: the database downloads in `src/data/download.py` (owner A), plus Chemprop and ChemBERTa branches (owner C) which have no module yet.

## Commands

```bash
python -m unittest discover -s tests -t . -v   # 36 tests, all passing, no network
python -m src.pipeline {data|splits|train|evaluate|submission|all} [--config configs/baseline.json]
python -m unittest tests.test_splits -v        # one test module
python -m unittest tests.test_splits.ScaffoldSplit.test_same_seed_gives_the_same_split
```

Run from the repo root — `src` is imported as a top-level package, so `python -I` breaks it (`-I` drops the cwd from `sys.path`). `Makefile` wraps the same commands, but `make` is usually absent on the team's Windows machines. There is no pytest, no PyYAML and no make in the venv: tests are stdlib `unittest`, configs are JSON. Keep it that way unless the whole team installs otherwise.

## Task context (Nuclear IT Hack, MIFI/ИИКС chemoinformatics lab — one night)

Goal: from a molecule's SMILES predict its **selectivity index** against vaccinia virus (VACV) — a regression target, `SI = CC50 / IC50`. Scored on a private Kaggle competition; the leaderboard result must be **reproducible from this repo** (clone → run pipeline → produce the submission `.csv` → same order of metric), otherwise the team is skipped at the pitch stage.

Three deliverables, all required:
1. Model training + evaluation code.
2. A **data-collection script** that pulls molecules from open databases and emits `smiles, IC50, CC50, SI, doi` (DOI where available), with the source URLs/queries documented and a validation step.
3. The **assembled dataset** itself, committed in that format.

The provided starter dataset is small, so dataset expansion from public sources (ChEMBL is already installed, see below) is as important as modelling.

Terminology discipline (also stated in `README.md`): the target is **vaccinia virus**, used as the surrogate/vaccine strain — not variola. Always record virus, strain, cell line and assay protocol alongside a measurement, and describe predictions as a research-stage computational ranking, never as evidence of efficacy or safety.

## Environment

The virtualenv is **outside** the repo, in the parent directory, and is deliberately gitignored:

```bash
# from the repo root
source ../env/Scripts/activate      # Git Bash
../env/Scripts/Activate.ps1         # PowerShell
../env/Scripts/python.exe -V        # Python 3.14.7
```

Already present: `rdkit`, `scikit-learn`, `catboost`, `pandas`, `scipy`, `numpy`, `chembl_webresource_client`, `requests-cache`, `joblib`, `matplotlib`, `tensorflow`/`keras`, `jupyter`.

Not present: `torch`, `chemprop`, `transformers`. README stages 5–6 (Chemprop D-MPNN, ChemBERTa-2) therefore need an install first, and on **Python 3.14** wheels for those may not exist yet — budget for a separate 3.11/3.12 env for those branches rather than assuming `pip install` will work in this one.

Use `requests-cache` for anything that hits an external API so re-runs of the collection script are cheap and the pipeline stays reproducible offline.

## Working agreement (from README.md)

Development is **staged and gated**, stages 0–9: data contract → load/clean → splits → baseline features → baseline evaluation → Chemprop → ChemBERTa-2 → fusion → calibration & applicability domain → independent holdout + release. The gates are the point, not decoration:

- One stage = one isolated change set = its own branch/PR, touching only that stage's files, with its own tests and acceptance criteria. Do not start the next stage before the current gate passes.
- A failed gate means fixing the current task. Never report a stage as successful on a failed gate, and never widen a stage's file footprint to make it pass.
- Splits are **scaffold-grouped** (Murcko) with a pinned seed and a committed split manifest. No identical standardised molecule or scaffold may appear in two splits.
- Fit every transformation (scaling, feature selection, imputation) on train only.
- The holdout is touched **once**, at stage 9. It is not used for hyperparameters, thresholds, model selection or ensemble weights.
- Fusion/ensembling is fitted on **out-of-fold** predictions only, and is kept only if it beats the best single baseline on validation.
- An extra branch (Chemprop, ChemBERTa) is merged into the final model only if it demonstrably improves validation on the same split — otherwise it stays out.
- The raw input table is never overwritten; cleaning writes derived files and a log with a per-row exclusion reason.

## Modelling notes specific to this target

- `SI` is a ratio of two concentrations: it spans orders of magnitude and is right-skewed. Model it in log space (and likewise IC50/CC50); report the metric Kaggle scores, but keep log-space diagnostics.
- Measurements are frequently **censored** (`>100 µM`, `<0.1`). The data contract must represent censoring explicitly rather than silently casting to a float, and the cleaning step must decide — in writing — whether censored rows train, validate or are dropped.
- Units are mixed across sources (µM, nM, µg/mL). Normalise to one unit at parse time; µg/mL → molar needs the molecular weight, so it is a conversion with a failure mode, not a multiplication.
- The same compound often appears in several papers with conflicting values. Aggregate per standardised structure (median in log space) and keep the per-source values plus DOIs for audit.
- Predicting IC50 and CC50 separately and dividing can beat predicting SI directly, or lose to it — it is an experiment to run on the fixed split, not an assumption.
- Standardise structures before anything else (RDKit: largest fragment / salt strip, neutralise, canonical tautomer if used consistently) and deduplicate on the standardised canonical SMILES, not the raw string.

## Reproducibility requirements

Because the Kaggle score must be reproducible by a third party:
- Pin seeds and record them; pin dependency versions in a committed requirements file.
- The submission must be produced by a single documented entry point, end to end from the committed data.
- Save model artifacts, the split manifest, metrics and checksums for the released version.

## Other agent configs

An OpenAI Codex config exists at `~/.codex/config.toml`. If you want its user-level items (MCP servers, slash commands, subagents, skills, instructions) brought into Claude Code, reply `/import` to scan and list what is importable, then `/import --yes=<digest>` with the digest from the scan output to apply it. If `/import` is not available on this surface, run `claude import` from a terminal instead.
