"""Train and run a direct CatBoost regressor for the SI VACV endpoint."""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from rdkit import Chem
from src import splits
from src.features.fingerprints import featurize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "attached_assets/train_v2.csv"
DEFAULT_TEST = ROOT / "attached_assets/test.csv"
DEFAULT_TEMPLATE = ROOT / "attached_assets/sample_submit.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ID_COLUMN = "index"
SMILES_COLUMN = "SMILES"
TARGET_COLUMN = "SI VACV"
FORBIDDEN_ENDPOINT_COLUMNS = ("CC50_uM", "IC50_uM_VACV")
EXPECTED_TRAIN_ROWS = 320
EXPECTED_TEST_ROWS = 597
SEED = 20261106
VALIDATION_FRACTIONS = {"train": 0.8, "validation": 0.2}
MODEL_PARAMETERS = {
    "iterations": 600,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5.0,
    "random_seed": SEED,
    "thread_count": 1,
    "loss_function": "RMSE",
    "task_type": "CPU",
    "verbose": False,
    "allow_writing_files": False,
}


@dataclass
class TrainingResult:
    model: Any
    submission: pd.DataFrame
    summary: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} CSV is missing required column(s): {', '.join(missing)}."
        )


def _validate_smiles(values: pd.Series, label: str) -> list[str]:
    smiles: list[str] = []
    invalid_rows: list[int] = []
    for row_number, value in enumerate(values, start=2):
        if pd.isna(value) or not str(value).strip():
            invalid_rows.append(row_number)
            continue
        value = str(value).strip()
        try:
            molecule = Chem.MolFromSmiles(value)
        except Exception:
            molecule = None
        if molecule is None:
            invalid_rows.append(row_number)
        else:
            smiles.append(value)
    if invalid_rows:
        row_list = ", ".join(map(str, invalid_rows[:10]))
        more = " (additional rows omitted)" if len(invalid_rows) > 10 else ""
        raise ValueError(
            f"{label} CSV has missing or unparseable SMILES at CSV row(s) "
            f"{row_list}{more}; correct those structures and retry."
        )
    return smiles


def _parse_targets(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any():
        rows = ", ".join(str(index + 2) for index in np.flatnonzero(invalid.to_numpy())[:10])
        raise ValueError(
            f"Training column {TARGET_COLUMN!r} contains non-numeric value(s) "
            f"at CSV row(s) {rows}."
        )
    present = numeric.notna().to_numpy()
    values_array = numeric.to_numpy(dtype=float, na_value=np.nan)
    nonfinite = present & ~np.isfinite(values_array)
    if nonfinite.any():
        rows = ", ".join(str(index + 2) for index in np.flatnonzero(nonfinite)[:10])
        raise ValueError(
            f"Training column {TARGET_COLUMN!r} must contain finite values; "
            f"found a non-finite value at CSV row(s) {rows}."
        )
    negative = present & (values_array < 0)
    if negative.any():
        rows = ", ".join(str(index + 2) for index in np.flatnonzero(negative)[:10])
        raise ValueError(
            f"Training column {TARGET_COLUMN!r} cannot be negative; "
            f"found a negative value at CSV row(s) {rows}."
        )
    if not present.any():
        raise ValueError(
            f"Training CSV has no labeled rows in {TARGET_COLUMN!r}; "
            "at least one finite SI VACV target is required."
        )
    return pd.Series(values_array, index=values.index, name=TARGET_COLUMN)


def _validate_ids(frame: pd.DataFrame, label: str) -> list[str]:
    ids = frame[ID_COLUMN].astype("string")
    missing = ids.isna() | ids.str.strip().eq("")
    if missing.any():
        rows = ", ".join(str(index + 2) for index in np.flatnonzero(missing.to_numpy())[:10])
        raise ValueError(f"{label} CSV has a missing {ID_COLUMN!r} at CSV row(s) {rows}.")
    normalized = ids.str.strip()
    duplicates = normalized.duplicated(keep=False)
    if duplicates.any():
        examples = ", ".join(normalized[duplicates].drop_duplicates().head(5).tolist())
        raise ValueError(f"{label} CSV contains duplicate {ID_COLUMN!r} values: {examples}.")
    return normalized.tolist()


def validate_tables(
        train: pd.DataFrame,
        test: pd.DataFrame,
        template: pd.DataFrame,
        *,
        expected_train_rows: int | None = None,
        expected_test_rows: int | None = None,
) -> pd.Series:
    """Validate required inputs and return numeric SI labels.
    Missing SI labels are permitted and excluded from fitting. Zero labels are
    retained. No other training or test columns are inspected for row selection.
    """
    _require_columns(train, (SMILES_COLUMN, TARGET_COLUMN), "Training")
    _require_columns(test, (ID_COLUMN, SMILES_COLUMN), "Test")
    _require_columns(template, (ID_COLUMN, TARGET_COLUMN), "Sample-submission")
    if set(template.columns) != {ID_COLUMN, TARGET_COLUMN} or len(template.columns) != 2:
        raise ValueError(
            f"Sample-submission CSV must have exactly these columns: "
            f"{ID_COLUMN}, {TARGET_COLUMN}."
        )
    for frame, label, expected in (
            (train, "Training", expected_train_rows),
            (test, "Test", expected_test_rows),
    ):
        if frame.empty:
            raise ValueError(f"{label} CSV has no data rows.")
        if expected is not None and len(frame) != expected:
            raise ValueError(
                f"{label} CSV has {len(frame)} data rows; expected {expected}. "
                "Check that the supplied file is complete or set the appropriate "
                "expected-row option."
            )
    if len(test) != len(template):
        raise ValueError(
            f"Test CSV has {len(test)} rows but sample-submission CSV has "
            f"{len(template)} rows; they must have the same row count."
        )
    _validate_smiles(train[SMILES_COLUMN], "Training")
    _validate_smiles(test[SMILES_COLUMN], "Test")
    test_ids = _validate_ids(test, "Test")
    template_ids = _validate_ids(template, "Sample-submission")
    if test_ids != template_ids:
        mismatch = next(
            index
            for index, (test_id, template_id) in enumerate(
                zip(test_ids, template_ids), start=2
            )
            if test_id != template_id
        )
        raise ValueError(
            f"Test and sample-submission IDs do not match in row order "
            f"(first mismatch at CSV row {mismatch})."
        )
    return _parse_targets(train[TARGET_COLUMN])


def features_from_frame(frame: pd.DataFrame) -> np.ndarray:
    """Generate the existing Morgan-2048 plus 13 RDKit descriptor features."""
    _require_columns(frame, (SMILES_COLUMN,), "Feature input")
    smiles = _validate_smiles(frame[SMILES_COLUMN], "Feature input")
    return featurize(smiles)


def predict_frame(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Predict from SMILES only; endpoint and metadata columns are ignored."""
    predictions = np.asarray(model.predict(features_from_frame(frame)), dtype=float).reshape(-1)
    if len(predictions) != len(frame):
        raise ValueError(
            f"Model returned {len(predictions)} predictions for {len(frame)} test rows."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced a non-finite test prediction.")
    return predictions


def build_submission(template: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Copy the template, replace its target values, and keep its row order."""
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(predictions) != len(template):
        raise ValueError(
            f"Received {len(predictions)} predictions for {len(template)} template rows."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Cannot write non-finite submission predictions.")
    submission = template.copy(deep=True)
    submission[TARGET_COLUMN] = predictions
    return submission.loc[:, [ID_COLUMN, TARGET_COLUMN]]


def build_model() -> Any:
    """Create the single seeded CPU CatBoost regressor used for both fits."""
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**MODEL_PARAMETERS)


def train_direct(
        train: pd.DataFrame,
        test: pd.DataFrame,
        template: pd.DataFrame,
        *,
        expected_train_rows: int | None = None,
        expected_test_rows: int | None = None,
) -> TrainingResult:
    """Run fixed scaffold validation, then refit the same model on all labels."""
    targets = validate_tables(
        train,
        test,
        template,
        expected_train_rows=expected_train_rows,
        expected_test_rows=expected_test_rows,
    )
    eligible = targets.notna().to_numpy()
    labeled = train.loc[eligible].reset_index(drop=True)
    y_all = targets.loc[eligible].to_numpy(dtype=float)
    smiles = labeled[SMILES_COLUMN].astype(str).tolist()
    x_all = featurize(smiles)
    split = splits.scaffold_split(
        smiles,
        fractions=VALIDATION_FRACTIONS,
        seed=SEED,
    )
    splits.assert_no_leaks(smiles, split)
    fit_indices = split["train"]
    validation_indices = split["validation"]
    if not fit_indices or not validation_indices:
        raise ValueError(
            "The fixed Murcko scaffold split produced an empty training or "
            "validation partition; cannot report a validation MSE."
        )
    model = build_model()
    model.fit(x_all[fit_indices], y_all[fit_indices])
    validation_predictions = np.asarray(
        model.predict(x_all[validation_indices]), dtype=float
    ).reshape(-1)
    if len(validation_predictions) != len(validation_indices):
        raise ValueError("CatBoost returned an unexpected number of validation predictions.")
    if not np.isfinite(validation_predictions).all():
        raise ValueError("CatBoost produced a non-finite validation prediction.")
    validation_mse = float(
        np.mean((y_all[validation_indices] - validation_predictions) ** 2)
    )
    # Refit this same estimator on every labeled row, including zero targets.
    model.fit(x_all, y_all)
    test_predictions = predict_frame(model, test)
    submission = build_submission(template, test_predictions)
    summary = {
        "validation": {
            "metric": "mean_squared_error",
            "target": TARGET_COLUMN,
            "scale": "raw SI",
            "mse": validation_mse,
            "method": "Murcko scaffold grouped split",
            "seed": SEED,
            "train_rows": len(fit_indices),
            "validation_rows": len(validation_indices),
        },
        "final_fit": {
            "labeled_training_rows": len(y_all),
            "zero_targets_retained": int(np.count_nonzero(y_all == 0)),
            "feature_count": int(x_all.shape[1]),
            "feature_source": "SMILES: Morgan fingerprint plus RDKit descriptors",
            "model_parameters": MODEL_PARAMETERS,
        },
        "interpretation": (
            "This is a local scaffold-validation metric, not the official "
            "competition score."
        ),
    }
    return TrainingResult(model=model, submission=submission, summary=summary)


def read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={ID_COLUMN: "string"})
    except FileNotFoundError as exc:
        raise ValueError(f"{label} CSV was not found at {path}.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Could not parse {label} CSV at {path}: {exc}") from exc


def write_outputs(result: TrainingResult, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    model_path = output_dir / "catboost_si_vacv.cbm"
    summary_path = output_dir / "validation.json"
    result.submission.to_csv(submission_path, index=False)
    result.model.save_model(str(model_path))
    summary_path.write_text(
        json.dumps(result.summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return submission_path, model_path, summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one direct CatBoost model for SI VACV using only SMILES-derived "
            "molecular features."
        )
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="training CSV")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="test CSV")
    parser.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="sample-submission CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for submission, model, and validation summary",
    )
    parser.add_argument(
        "--expected-train-rows",
        type=int,
        default=EXPECTED_TRAIN_ROWS,
        help="expected number of training rows (default: 320)",
    )
    parser.add_argument(
        "--expected-test-rows",
        type=int,
        default=EXPECTED_TEST_ROWS,
        help="expected number of test and template rows (default: 597)",
    )
    parser.add_argument(
        "--no-expected-row-counts",
        action="store_true",
        help="allow nonstandard row counts while still checking test/template alignment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    expected_train = None if args.no_expected_row_counts else args.expected_train_rows
    expected_test = None if args.no_expected_row_counts else args.expected_test_rows
    try:
        train = read_csv(args.train, "Training")
        test = read_csv(args.test, "Test")
        template = read_csv(args.sample, "Sample-submission")
        result = train_direct(
            train,
            test,
            template,
            expected_train_rows=expected_train,
            expected_test_rows=expected_test,
        )
        submission_path, model_path, summary_path = write_outputs(
            result, args.output_dir
        )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    validation = result.summary["validation"]
    print(
        f"Local scaffold-validation raw SI MSE: {validation['mse']:.6f} "
        f"({validation['train_rows']} train, "
        f"{validation['validation_rows']} validation rows)."
    )
    print(f"Submission: {submission_path}")
    print(f"Model: {model_path}")
    print(f"Validation summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
