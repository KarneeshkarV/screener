#!/usr/bin/env bash
# Run the whole momentum study in dependency order, then publish the site.
#
#     ./scripts/run_momentum_study_all.sh
#     WORKERS=4 ./scripts/run_momentum_study_all.sh
#
# Every phase is resumable - a run whose JSON already exists is skipped - so an
# interrupted job picks up where it stopped and re-running this script is cheap.
# Phases are ordered by value per hour, because the tail of the sweep is the
# part most likely to be cut short:
#
#   1. baselines across every window, both markets, so the headline table is
#      complete first;
#   2. holding period crossed with the regime overlay on the ten-year window,
#      which is the pair that interacts - a filter that blocks entries changes
#      how often slots turn over;
#   3. construction levers one at a time against the baseline;
#   4. the same cross on the five-year window, which is the real robustness
#      check: a lever that only helps on one window has not been shown to help.
#
# The site is rebuilt after each phase, so a browser open on the report picks up
# partial results without waiting for the whole job.
set -uo pipefail

cd "$(dirname "$0")/.."

# This runs for hours on a machine someone else is using, so it is deliberately
# a background citizen.
#
# Sizing, measured rather than guessed: the heaviest cell (India, ten years,
# ~920 tickers) peaks at 875 MB and takes 100 seconds. Six workers is therefore
# ~5.3 GB against the 11 GB free here, and the numeric libraries are capped at
# two threads each so peak demand lands near ten of sixteen cores instead of
# saturating them. nice/ionice mean any interactive process preempts the whole
# job regardless.
WORKERS="${WORKERS:-6}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

NICE=(nice -n 15)
if command -v ionice >/dev/null 2>&1; then
  NICE=(ionice -c3 "${NICE[@]}")
fi

STRATEGIES=(
  momentum_12_1 momentum_6_6 momentum_12_1_trend momentum_12_1_riskadj
  momentum_12_1_volmanaged momentum_12_1_dynamic momentum_12_1_defensive
  dual_momentum_gem dual_momentum_market dual_momentum_paa dual_momentum_daa
  tsmom_12 tsmom_blend
  faber_sma10 absolute_momentum industry_trend_breakout
)

study() { "${NICE[@]}" uv run python scripts/run_momentum_study.py "$@"; }
publish() { "${NICE[@]}" uv run python scripts/build_momentum_site.py; }

# Warm the on-disk price cache one market at a time before fanning out. Two
# workers that both miss on the same ticker would otherwise fetch and write the
# same parquet file concurrently.
warm() {
  echo "=== warming price cache ($(date -Is)) ==="
  for market in india us; do
    study -m "${market}" -y 10 -s momentum_12_1
  done
}

# Shard by strategy: each worker owns a disjoint set of run keys, so no two
# processes can write the same output file.
parallel_phase() {
  local name=$1
  shift
  echo "=== phase: ${name}  workers=${WORKERS}  ($(date -Is)) ==="
  printf '%s\n' "${STRATEGIES[@]}" \
    | xargs -P "${WORKERS}" -I {} "${NICE[@]}" \
        uv run python scripts/run_momentum_study.py -s {} "$@"
  publish
}

warm
parallel_phase "baselines"
parallel_phase "hold x regime, 10y" -y 10 --regime-sweep --hold-sweep
parallel_phase "construction levers, 10y" -y 10 --lever-sweep
parallel_phase "hold x regime, 5y" -y 5 --regime-sweep --hold-sweep
parallel_phase "construction levers, 5y" -y 5 --lever-sweep

echo "=== done ($(date -Is)) ==="
