#!/usr/bin/env bash
# Size x sizing sweep, 5yr, both markets, all 7 strategies.
#   top    in {5,10,20,40}
#   sizing in {equal_slot, fixed_fraction, inverse_vol, atr_risk, fixed_risk}
# atr_risk/fixed_risk get --stop-loss 0.08 (fixed_risk requires one).
# Race-safe + resumable: one result file per job; concurrent workers never
# write the same file; a job with an existing valid result is skipped.
set -u
cd "${SCREENER_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)}"
SP="${ANALYSIS_DIR:-$PWD/profiling/_analysis}"; mkdir -p "$SP"
RAW=$SP/sweep_nohold_raw; RES=$SP/sweep_nohold_res
mkdir -p "$RAW" "$RES"
CONCURRENCY=4

STRATS="mark_minervini mq_us1 mq_us2 mq_us3 mq_in1 mq_in2 mq_in3"
TOPS="5 10 20 40"
SIZINGS="equal_slot fixed_fraction inverse_vol atr_risk fixed_risk"
needs_fund() { case "$1" in mq_us1|mq_us2|mq_in1|mq_in2|mq_in3) return 0;; *) return 1;; esac; }
extract() { grep -m1 -E "│ $2 " "$1" | sed -E 's/.*│[^│]*│ *([^ │]+) *│.*/\1/'; }

run_one() {
  local MKT="$1" STRAT="$2" TOP="$3" SIZING="$4"
  local ID="${MKT}_${STRAT}_t${TOP}_${SIZING}"
  local RF="$RES/${ID}.csv"
  # skip if already completed: 5th field (total_return) is a real value.
  # Only empty/ERROR results get re-run on relaunch; NO_TRADES is final.
  if [ -s "$RF" ]; then
    local v; v=$(cut -d, -f5 "$RF")
    case "$v" in ""|ERROR) : ;; *) return 0;; esac
  fi
  local LOG="$RAW/${ID}.log"
  local UNIV="" FUND="" STOP=""
  [ "$MKT" = "india" ] && UNIV="--universe dynamic --dynamic-base nifty500 --universe-size 500 --universe-rebalance monthly"
  needs_fund "$STRAT" && FUND="--fundamentals-provider fmp"
  # No time cap (--hold 100000): exit = sma50 crossunder OR 8% stop-loss.
  # Stop applied to ALL sizings now, so downside is capped without the hold cap.
  STOP="--stop-loss 0.08"
  timeout 900 uv run screener backtest-rolling -m "$MKT" --years 5 --hold 100000 \
    --top "$TOP" --strategy "$STRAT" --sizing "$SIZING" $STOP $UNIV $FUND > "$LOG" 2>&1
  local tr cg sh dd hr al pf td br
  if grep -q "Total Return" "$LOG"; then
    tr=$(extract "$LOG" "Total Return"); cg=$(extract "$LOG" "CAGR")
    sh=$(extract "$LOG" "Sharpe"); dd=$(extract "$LOG" "Max Drawdown")
    hr=$(extract "$LOG" "Hit Rate"); al=$(extract "$LOG" "Alpha \(ann\.\)")
    pf=$(extract "$LOG" "Profit Factor"); td=$(extract "$LOG" "Trades")
    br=$(extract "$LOG" "Benchmark Return")
  elif grep -qi "No trades" "$LOG"; then tr="NO_TRADES"; else tr="ERROR"; fi
  echo "$MKT,$STRAT,$TOP,$SIZING,$tr,$cg,$sh,$dd,$hr,$al,$pf,$td,$br" > "$RF"
}

total=0; done=0
for MKT in us india; do for STRAT in $STRATS; do for TOP in $TOPS; do for SIZING in $SIZINGS; do
  total=$((total+1))
  # throttle to CONCURRENCY background workers
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n 2>/dev/null || sleep 1; done
  run_one "$MKT" "$STRAT" "$TOP" "$SIZING" &
done; done; done; done
wait
echo ">>> ALL LAUNCHED total=$total  results=$(ls "$RES" | wc -l)  $(date +%H:%M:%S)"
