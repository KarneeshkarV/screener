#!/usr/bin/env bash
# India PIT mid / small / mid+small matrix. Resumable. Shard by strategy.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
source .env
set +a

export SCREENER_PRICE_PROVIDER=yfinance
export SCREENER_AGENT=0
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

WORKERS="${WORKERS:-4}"
NICE=(nice -n 15)
if command -v ionice >/dev/null 2>&1; then
  NICE=(ionice -c3 "${NICE[@]}")
fi

mapfile -t STRATEGIES < <(
  uv run python - <<'PY'
from scripts.run_pit_midsmall_study import STRATEGIES
for s in STRATEGIES:
    print(s.name)
PY
)

study() { "${NICE[@]}" uv run python scripts/run_pit_midsmall_study.py "$@"; }

PRICE=()
FUND=()
mapfile -t PRICE < <(
  uv run python - <<'PY'
from scripts.run_pit_midsmall_study import STRATEGIES
for s in STRATEGIES:
    if not s.fund:
        print(s.name)
PY
)
mapfile -t FUND < <(
  uv run python - <<'PY'
from scripts.run_pit_midsmall_study import STRATEGIES
for s in STRATEGIES:
    if s.fund:
        print(s.name)
PY
)

echo "=== warm price cache 5y midsmall ($(date -Is)) ==="
study -u midsmall -y 5 -s momentum_12_1

echo "=== price-only workers=${WORKERS} n=${#PRICE[@]} ($(date -Is)) ==="
printf '%s\n' "${PRICE[@]}" \
  | xargs -P "${WORKERS}" -I {} "${NICE[@]}" \
      uv run python scripts/run_pit_midsmall_study.py --price-only -s {}

echo "=== fund workers=${WORKERS} n=${#FUND[@]} ($(date -Is)) ==="
printf '%s\n' "${FUND[@]}" \
  | xargs -P "${WORKERS}" -I {} "${NICE[@]}" \
      uv run python scripts/run_pit_midsmall_study.py --fund-only -s {}

study --summary-only
echo "=== done ($(date -Is)) ==="
