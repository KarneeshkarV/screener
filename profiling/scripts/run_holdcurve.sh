#!/usr/bin/env bash
# Hold-time curve, 5yr, both markets, all 7 strategies, equal_slot only.
#   hold in {5,10,20,30,40,60,100}   (the no-cap inf point is reused from sweep_nohold_res)
#   sizing = equal_slot  (sizing proven Sharpe-neutral; collapsed to isolate hold-time)
#   stop 0.08 on every run so the only thing varying is the hold cap.
# Race-safe + resumable: one result file per job; a job with a valid result is skipped.
set -u
cd "${SCREENER_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)}"
SP="${ANALYSIS_DIR:-$PWD/profiling/_analysis}"; mkdir -p "$SP"
RAW=$SP/holdcurve_raw; RES=$SP/holdcurve_res
mkdir -p "$RAW" "$RES"
CONCURRENCY=4

STRATS="mark_minervini mq_us1 mq_us2 mq_us3 mq_in1 mq_in2 mq_in3"
TOPS="5 10 20 40"
HOLDS="5 10 20 30 40 60 100"
SIZING="equal_slot"
needs_fund() { case "$1" in mq_us1|mq_us2|mq_in1|mq_in2|mq_in3) return 0;; *) return 1;; esac; }
extract() { grep -m1 -E "│ $2 " "$1" | sed -E 's/.*│[^│]*│ *([^ │]+) *│.*/\1/'; }

run_one() {
  local MKT="$1" STRAT="$2" TOP="$3" HOLD="$4"
  local ID="${MKT}_${STRAT}_t${TOP}_h${HOLD}"
  local RF="$RES/${ID}.csv"
  if [ -s "$RF" ]; then
    local v; v=$(cut -d, -f6 "$RF")
    case "$v" in ""|ERROR) : ;; *) return 0;; esac
  fi
  local LOG="$RAW/${ID}.log"
  local UNIV="" FUND=""
  [ "$MKT" = "india" ] && UNIV="--universe dynamic --dynamic-base nifty500 --universe-size 500 --universe-rebalance monthly"
  needs_fund "$STRAT" && FUND="--fundamentals-provider fmp"
  timeout 900 uv run screener backtest-rolling -m "$MKT" --years 5 --hold "$HOLD" \
    --top "$TOP" --strategy "$STRAT" --sizing "$SIZING" --stop-loss 0.08 \
    $UNIV $FUND > "$LOG" 2>&1
  local tr cg sh dd hr al pf td br
  if grep -q "Total Return" "$LOG"; then
    tr=$(extract "$LOG" "Total Return"); cg=$(extract "$LOG" "CAGR")
    sh=$(extract "$LOG" "Sharpe"); dd=$(extract "$LOG" "Max Drawdown")
    hr=$(extract "$LOG" "Hit Rate"); al=$(extract "$LOG" "Alpha \(ann\.\)")
    pf=$(extract "$LOG" "Profit Factor"); td=$(extract "$LOG" "Trades")
    br=$(extract "$LOG" "Benchmark Return")
  elif grep -qi "No trades" "$LOG"; then tr="NO_TRADES"; else tr="ERROR"; fi
  # cols: market,strategy,top,hold,sizing,total_return,cagr,sharpe,max_dd,hit_rate,alpha,profit_factor,trades,bench_return
  echo "$MKT,$STRAT,$TOP,$HOLD,$SIZING,$tr,$cg,$sh,$dd,$hr,$al,$pf,$td,$br" > "$RF"
}

total=0
for MKT in us india; do for STRAT in $STRATS; do for TOP in $TOPS; do for HOLD in $HOLDS; do
  total=$((total+1))
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n 2>/dev/null || sleep 1; done
  run_one "$MKT" "$STRAT" "$TOP" "$HOLD" &
done; done; done; done
wait
echo ">>> ALL DONE total=$total  results=$(ls "$RES" | wc -l)  $(date +%H:%M:%S)"
