#!/usr/bin/env bash
# Обучение модели SI на маке. Один файл: окружение, проверки, обучение, итог.
#
#   ./run_mac.sh                 обе стратегии на фиксированном разбиении
#   ./run_mac.sh 5               5 scaffold-разбиений, сравнение стратегий
#   ./run_mac.sh 5 direct        5 разбиений, только стратегия direct
#
# Данные уже в репозитории: сеть нужна только для pip. Holdout не трогается.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REPEATS="${1:-0}"
STRATEGY="${2:-both}"
VENV=".venv"
LOG_DIR="reports"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

# Цвета только если вывод идёт в терминал: в файле логов escape-коды мешают.
if [[ -t 1 ]]; then
  B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[36m'; RED=$'\033[31m'
else
  B=''; DIM=''; R=''; GREEN=''; YELLOW=''; BLUE=''; RED=''
fi

STEP=0
TOTAL=5
START_TS=$(date +%s)

elapsed() {
  local seconds=$(( $(date +%s) - START_TS ))
  printf '%dм %02dс' $(( seconds / 60 )) $(( seconds % 60 ))
}

step() {
  STEP=$(( STEP + 1 ))
  printf '\n%s[%d/%d]%s %s%s%s %s(%s)%s\n' \
    "$BLUE" "$STEP" "$TOTAL" "$R" "$B" "$1" "$R" "$DIM" "$(elapsed)" "$R"
}

ok()   { printf '  %s✓%s %s\n' "$GREEN" "$R" "$1"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$R" "$1"; }
die()  { printf '\n%s✗ %s%s\n' "$RED" "$1" "$R" >&2; exit 1; }

printf '%s╭─────────────────────────────────────────────────╮%s\n' "$B" "$R"
printf '%s│  VACV · индекс селективности · обучение модели   │%s\n' "$B" "$R"
printf '%s╰─────────────────────────────────────────────────╯%s\n' "$B" "$R"
printf '%sмашина:%s %s · %s\n' "$DIM" "$R" "$(uname -s)" "$(uname -m)"
if [[ "$REPEATS" -gt 0 ]]; then
  printf '%sрежим:%s  %s разбиений, стратегия: %s\n' "$DIM" "$R" "$REPEATS" "$STRATEGY"
else
  printf '%sрежим:%s  фиксированное разбиение, стратегия: %s\n' "$DIM" "$R" "$STRATEGY"
fi
printf '%sлог:%s    %s\n' "$DIM" "$R" "$LOG_FILE"

# ---------------------------------------------------------------- 1. Python
step "Проверяю Python"
PY=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
[[ -n "$PY" ]] || die "не нашёл python3. Поставь: brew install python@3.12"

PY_VERSION="$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "$PY $PY_VERSION"
# На 3.13+ колёс rdkit может не быть. Предупреждаем до установки, а не после.
case "$PY_VERSION" in
  3.10|3.11|3.12) ;;
  *) warn "rdkit надёжно ставится на 3.10-3.12; на $PY_VERSION установка может упасть" ;;
esac

# ------------------------------------------------------------ 2. Окружение
step "Готовлю окружение"
if [[ ! -d "$VENV" ]]; then
  $PY -m venv "$VENV"
  ok "создал $VENV"
else
  ok "$VENV уже есть"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if python -c 'import rdkit, sklearn, pandas, scipy' 2>/dev/null; then
  ok "зависимости на месте, pip не нужен"
else
  printf '  %sставлю зависимости, это долго только в первый раз…%s\n' "$DIM" "$R"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt || die "pip не смог установить requirements.txt"
  ok "зависимости установлены"
fi

# ----------------------------------------------------------------- 3. Данные
step "Проверяю данные"
for file in data/processed/train.csv data/processed/validation.csv; do
  [[ -f "$file" ]] || die "нет $file. На машине со данными: python -m src.pipeline export"
  rows=$(( $(wc -l < "$file") - 1 ))
  ok "$(basename "$file"): $rows строк"
done
if [[ -f data/processed/splits.json ]]; then
  ok "манифест разбиения на месте (holdout только в нём, в обучение не идёт)"
fi

# ------------------------------------------------------------------ 4. Тесты
step "Гоняю тесты"
if python -m unittest discover -s tests -t . >/dev/null 2>&1; then
  ok "все тесты зелёные"
else
  warn "тесты красные — обучение запускаю, но результату верить рано"
  printf '  %sподробности: python -m unittest discover -s tests -t . -v%s\n' "$DIM" "$R"
fi

# --------------------------------------------------------------- 5. Обучение
step "Обучаю"
mkdir -p "$LOG_DIR"
ARGS=(--strategy "$STRATEGY")
[[ "$REPEATS" -gt 0 ]] && ARGS+=(--repeats "$REPEATS")

# python -u, чтобы строки шли в терминал сразу, а не после завершения.
# tee пишет и в лог: утром никто не вспомнит числа из прокрученного терминала.
set +e
python -u train_model.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
[[ $STATUS -eq 0 ]] || die "обучение упало (код $STATUS), см. $LOG_FILE"

# ------------------------------------------------------------------- Итог
printf '\n%s╭─────────────────────────────────────────────────╮%s\n' "$GREEN" "$R"
printf '%s│  Готово за %-36s │%s\n' "$GREEN" "$(elapsed)" "$R"
printf '%s╰─────────────────────────────────────────────────╯%s\n' "$GREEN" "$R"
printf '  метрики:  %s\n' "reports/train_metrics.json"
printf '  модели:   %s\n' "artifacts/model_*.joblib"
printf '  лог:      %s\n' "$LOG_FILE"

if [[ "$REPEATS" -eq 0 ]]; then
  printf '\n%sОдно разбиение из 140 молекул: разница между стратегиями внутри шума.%s\n' "$DIM" "$R"
  printf '%sЧтобы закрыть вопрос direct против two_head: ./run_mac.sh 5%s\n' "$DIM" "$R"
fi
printf '\n%sКоммить метрики и лог в репозиторий — они идут в финальный отчёт.%s\n' "$DIM" "$R"
