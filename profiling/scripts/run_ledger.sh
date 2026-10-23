#!/usr/bin/env bash
# Trade-ledger sweep: same 448 configs as the hold-time curve, but with --csv so we
# capture the per-trade ledger and can measure how often the SAME stock gets re-traded.
#   holds {5,10,20,30,40,60,100} + inf (inf => --hold 100000, matches the no-cap point)
#   equal_slot, stop 0.08, both markets, top{5,10,20,40}, 7 strategies = 448 runs.
# Race-safe + resumable. Per combo we save the raw ledger and a one-line repeat-metric.
set -u
cd "${SCREENER_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)}"
SP="${ANALYSIS_DIR:-$PWD/profiling/_analysis}"; mkdir -p "$SP"
RAW=$SP/ledger_raw; RES=$SP/ledger_res
mkdir -p "$RAW" "$RES"
CONCURRENCY=4
STRATS="mark_minervini mq_us1 mq_us2 mq_us3 mq_in1 mq_in2 mq_in3"
TOPS="5 10 20 40"
HOLDS="5 10 20 30 40 60 100 inf"
needs_fund() { case "$1" in mq_us1|mq_us2|mq_in1|mq_in2|mq_in3) return 0;; *) return 1;; esac; }

run_one() {
  local MKT="$1" STRAT="$2" TOP="$3" HOLD="$4"
  local ID="${MKT}_${STRAT}_t${TOP}_h${HOLD}"
  local LG="$RAW/${ID}.csv" RF="$RES/${ID}.csv"
  # resumable: skip if the metric file already has a valid (non-ERROR) row
  if [ -s "$RF" ]; then
    local v; v=$(cut -d, -f5 "$RF")
    case "$v" in ""|ERROR) : ;; *) return 0;; esac
  fi
  local HOLDARG="$HOLD"; [ "$HOLD" = "inf" ] && HOLDARG="100000"
  local UNIV="" FUND=""
  [ "$MKT" = "india" ] && UNIV="--universe dynamic --dynamic-base nifty500 --universe-size 500 --universe-rebalance monthly"
  needs_fund "$STRAT" && FUND="--fundamentals-provider fmp"
  timeout 900 uv run screener backtest-rolling -m "$MKT" --years 5 --hold "$HOLDARG" \
    --top "$TOP" --strategy "$STRAT" --sizing equal_slot --stop-loss 0.08 --csv \
    $UNIV $FUND > "$LG" 2>"$RAW/${ID}.err"
  # metric via python: total trades, unique tickers, repeat ratio, %multi, max count on one name, top3
  python3 - "$LG" "$MKT" "$STRAT" "$TOP" "$HOLD" > "$RF" <<'PY'
import sys, csv
from collections import Counter
lg,mkt,strat,top,hold = sys.argv[1:6]
try:
    with open(lg, newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows or "ticker" not in rows[0]:
        raise ValueError("no ledger")
    c=Counter(r["ticker"] for r in rows if r.get("ticker"))
    tot=sum(c.values()); uniq=len(c)
    ratio = tot/uniq if uniq else 0.0
    multi = sum(1 for k,v in c.items() if v>1)          # names traded 2+ times
    pct_multi = 100.0*sum(v for k,v in c.items() if v>1)/tot if tot else 0.0
    mx = max(c.values()) if c else 0
    top3="|".join(f"{k}:{v}" for k,v in c.most_common(3))
    print(f"{mkt},{strat},{top},{hold},{tot},{uniq},{ratio:.3f},{multi},{pct_multi:.1f},{mx},{top3}")
except Exception:
    print(f"{mkt},{strat},{top},{hold},ERROR,,,,,,")
PY
}

total=0
for MKT in us india; do for STRAT in $STRATS; do for TOP in $TOPS; do for HOLD in $HOLDS; do
  total=$((total+1))
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n 2>/dev/null || sleep 1; done
  run_one "$MKT" "$STRAT" "$TOP" "$HOLD" &
done; done; done; done
wait
echo ">>> LEDGER SWEEP DONE total=$total results=$(ls "$RES" | wc -l) $(date +%H:%M:%S)"
