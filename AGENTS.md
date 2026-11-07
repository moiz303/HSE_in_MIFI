# Repository Guidelines

## Project Structure & Module Organization

This Python project predicts vaccinia virus (VACV) selectivity index, `SI = CC50 / IC50`, from SMILES. Read `CLAUDE.md` before changing pipeline behavior.

- `src/data/`: collection, schema validation, unit conversion, and structure standardization.
- `src/features/`, `src/models/`: molecular features and separate CatBoost (default) or XGBoost regressors for IC50 and CC50.
- `src/xgb_workflow.py`: fixed-split verification, training, evaluation, and SMILES-only submission. `docs/previous_experiments.md` records lessons from removed experiments.
- `src/pipeline.py`: CLI; `configs/`: JSON run configurations.
- `tests/`: unit and smoke tests. `data/`: raw, interim, and processed datasets plus source documentation. `reports/` and `artifacts/`: metrics, submissions, and models.

## Build, Test, and Development Commands

Run commands from the repository root, using the existing parent-directory environment (`../env/Scripts/Activate.ps1` in PowerShell).

- `python -m pip install -r requirements.txt`: install pinned dependencies.
- `python -m src.pipeline --help`: inspect available pipeline steps.
- `python -m src.pipeline export`: prepare train/validation files for the training machine.
- `python -m unittest discover -s tests -t . -v`: run the offline suite, matching CI.
- `python -m unittest tests.test_splits -v`: run one test module.

**Never train models on this machine.** Do not run pipeline `train`/`all`, sweeps, or real-data experiment comparisons. Hand runnable scripts and exports to the teammate training on Mac. Toy-model smoke tests are allowed.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_CASE` constants. Follow existing type hints and explanatory docstrings. Keep configurations in JSON. No formatter or linter is configured; avoid unrelated formatting changes or introducing pytest/PyYAML without team agreement.

## Testing & Scientific Integrity

Use standard-library `unittest`, with `tests/test_*.py` modules and `test_*` methods. CI uses Python 3.12; no numeric coverage threshold is configured. Cover changed behavior, especially censoring, units, standardization, and leakage prevention.

Preserve seeded Murcko-scaffold splits; fit transformations on train only. Reserve holdout for the final release evaluation. Fit ensembles on out-of-fold predictions. Never overwrite raw data; retain rejection reasons and source provenance. Cache external API requests.

## Commit & Pull Request Guidelines

History mixes informal messages with `feat(scope):`, `fix(scope):`, and `refactor:`; prefer descriptive messages in the latter style. Keep each branch/PR focused on one stage. Describe behavior changes, relevant issues, validation commands/results, and reproducibility impacts. Model changes require comparison on identical splits before promotion; failed gates must be fixed before advancing.

User workflow: after each completed, verified edit, commit and push to `origin/main`. Never force-push or discard unrelated changes.
