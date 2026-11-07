#!/usr/bin/env bash
# Current competition workflow; only SMILES and train_v2 SI enter fitting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$(uname -s)" == "Darwin" ]] || { echo "Use python train_model.py on Windows/Linux" >&2; exit 1; }
[[ $# -le 1 ]] || { echo "Usage: ./run_mac.sh [configs/current.json]" >&2; exit 1; }
CONFIG="${1:-configs/current.json}"
command -v python3.12 >/dev/null || { echo "Install Python 3.12: brew install python@3.12" >&2; exit 1; }
[[ -d .venv-current ]] || python3.12 -m venv .venv-current
source .venv-current/bin/activate
mkdir -p logs reports/current artifacts/current
LOG="logs/current_mac_$(date +%Y%m%d_%H%M%S).log"
STARTED=$SECONDS
trap 'echo "[ERROR] stopped after $((SECONDS-STARTED))s; inspect log" >&2' ERR
{
  echo "[1/5] Environment: $(uname -sm); config=$CONFIG; log=$LOG"
  python --version
  python -m pip install -r requirements-current.txt
  echo "[2/5] Tests; elapsed=$((SECONDS-STARTED))s"
  python -m unittest tests.test_current tests.test_kaggle_data -v
  for FILE in train_v2.csv test.csv sample_submit.csv; do
    if [[ ! -f "data/raw/kaggle/$FILE" ]]; then
      echo "[data] Download $FILE; Kaggle authentication required"
      kaggle competitions download molecular-activity-prediction-hack -f "$FILE" -p data/raw/kaggle
    fi
  done
  echo "[3/5] Versioned dataset; no holdout or forbidden endpoint features"
  python -u train_model.py --config "$CONFIG" --stage prepare
  echo "[4/5] Same 18 OOF folds, both source regimes, final fits; elapsed=$((SECONDS-STARTED))s"
  python -u train_model.py --config "$CONFIG" --stage train
  echo "[5/5] Build comparison from saved OOF; elapsed=$((SECONDS-STARTED))s"
  python -m src.current.results
  echo "[done] reports/current/comparison.md; OOF-selected candidate: reports/current/best_candidate.json"
  echo "Kaggle is not submitted automatically. Endpoint diagnostics are a separate optional command."
} 2>&1 | tee "$LOG"
