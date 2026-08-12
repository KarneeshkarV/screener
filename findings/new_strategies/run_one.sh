#!/usr/bin/env bash
# Usage: run_one.sh <market> <strategy> <years> <hold> <fund> <extra-args-file> <outdir>
set -u
MARKET="$1"; STRATEGY="$2"; YEARS="$3"; HOLD="$4"; FUND="$5"; EXTRA_FILE="$6"; OUTDIR="$7"
cd /home/karneeshkar/screener
EXTRA=()
EXTRA_FILE_ABS="$OUTDIR/$EXTRA_FILE"
if [ -f "$EXTRA_FILE_ABS" ]; then read -r -a EXTRA < "$EXTRA_FILE_ABS"; fi

ARGS=(backtest-rolling -m "$MARKET" --top 10 --hold "$HOLD" --years "$YEARS" --strategy "$STRATEGY" --initial-capital 100000)
if [ "$MARKET" = "india" ]; then
  ARGS+=(--universe nifty500 --cost-model india --slippage-bps 10)
else
  ARGS+=(--universe sp500 --cost-model flat --commission-bps 1 --slippage-bps 5)
fi
if [ "$FUND" = "fmp" ]; then ARGS+=(--fundamentals-provider fmp); fi
ARGS+=("${EXTRA[@]}")

if [ "$EXTRA_FILE" = "-" ]; then
  TAG="${MARKET}__${STRATEGY}__${YEARS}y__h${HOLD}"
else
  TAG="${MARKET}__${STRATEGY}__${YEARS}y__h${HOLD}__${EXTRA_FILE%.txt}"
fi
LOG="$OUTDIR/${TAG}.log"
timeout 2700 .venv/bin/screener --log-level ERROR --agent-detail summary "${ARGS[@]}" > "$LOG" 2>&1
echo "$MARKET $STRATEGY $YEARS hold=$HOLD exit=$? -> $(basename "$LOG")"
