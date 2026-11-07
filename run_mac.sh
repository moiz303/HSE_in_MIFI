#!/usr/bin/env bash
# Mac only: tests -> two concentration heads -> fixed validation -> optional submission.
# ./run_mac.sh [configs/xgb_ensemble.json]
set -euo pipefail
STARTED=$SECONDS
trap 'echo "[ERROR] run failed after $((SECONDS-STARTED))s; inspect the log above" >&2' ERR
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$(uname -s)" == "Darwin" ]] || { echo "Training is allowed only on Mac" >&2; exit 1; }
[[ $# -le 1 ]] || { echo "Usage: ./run_mac.sh [config.json]" >&2; exit 1; }
CONFIG="${1:-configs/xgb_ensemble.json}"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
echo "[1/6] Environment: $(uname -sm); config=$CONFIG"
PY=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
[[ -n "$PY" ]] || { echo "Install Python: brew install python@3.12" >&2; exit 1; }
[[ -d .venv ]] || "$PY" -m venv .venv
source .venv/bin/activate
python --version
echo "[2/6] Installing pinned dependencies"
python -m pip install -r requirements.txt
python -c 'import catboost, xgboost, rdkit' || { echo "Check dependencies; XGBoost on Mac may require: brew install libomp" >&2; exit 1; }
mkdir -p reports artifacts
MODEL_PREFIX="$(python -c 'import sys; from src.pipeline import load_config; from src.xgb_workflow import report_prefix; print(report_prefix(load_config(sys.argv[1])))' "$CONFIG")"
LOG_FILE="reports/${MODEL_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOG_FILE"
# pipefail makes test/training failures fatal even when output is piped to tee.
{
  echo "[3/6] Tests; elapsed=$((SECONDS-STARTED))s"
  python -m unittest discover -s tests -t . -v
  echo "[4/6] Training and reporting (final mode skips validation scoring); elapsed=$((SECONDS-STARTED))s"
  python -c 'import json,sys; from src.pipeline import load_config; print(json.dumps(load_config(sys.argv[1]), indent=2))' "$CONFIG"
  python -u train_model.py --config "$CONFIG"
  echo "[5/6] Building submission from selected model; elapsed=$((SECONDS-STARTED))s"
  TEST_CSV="$(python -c 'import sys; from src.pipeline import load_config; print(load_config(sys.argv[1])["submission"]["test_csv"])' "$CONFIG")"
  if [[ -f "$TEST_CSV" ]]; then
    python -u -m src.pipeline submission --config "$CONFIG"
  else
    echo "No test CSV: training/evaluation complete; submission skipped."
    echo "On the machine with test data: python -m src.pipeline submission --config $CONFIG"
  fi
} 2>&1 | tee "$LOG_FILE"
echo "[6/6] Completed in $((SECONDS-STARTED))s"
echo "Done. See the model_file in $CONFIG, reports/${MODEL_PREFIX}_*.json and $LOG_FILE"
