#!/usr/bin/env bash
# Intraday options snapshot recorder.
#
# Snapshots each market's watchlist option chains into the first-class contract
# store at ~/.screener/contracts/{market}/{date}/{underlying}.parquet. US uses
# delayed CBOE quotes (yfinance fallback); India uses the NSE live chain API.
#
# `options record --every 15m` runs a session-bounded loop: it no-ops outside
# session hours and exits once the session closes, so ONE session-open cron per
# market covers the whole day. Passes are idempotent (dedupe on
# contract+snapshot_ts), so rerunning after a failure is safe. Free intraday
# option history barely exists — every session this does not run is history that
# can never be recovered, so keep the cron running each session.
#
# Because each market's --every loop blocks until that session closes, run one
# cron entry per market at (or just before) its open. Times are UTC; adjust for
# DST on the US line.
#
# Cron example (also documented in README.md):
#   # India: session 09:15–15:30 IST ≈ 03:45–10:00 UTC
#   40 3 * * 1-5 MARKETS=india /root/screneer_main/screener/scripts/daily_options_record.sh >> "$HOME/.screener/options-logs/cron.log" 2>&1
#   # US: session 09:30–16:00 ET ≈ 13:30–20:00 UTC (EST; 12:30–19:00 on EDT)
#   25 13 * * 1-5 MARKETS=us    /root/screneer_main/screener/scripts/daily_options_record.sh >> "$HOME/.screener/options-logs/cron.log" 2>&1
#
# Alternatively drive a plain 15-minute cron with `options record --once` (set
# EVERY= empty and ONCE=1 below) scheduled across each session's UTC window.
set -uo pipefail

REPO_DIR="${SCREENER_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
MARKETS="${MARKETS:-us india}"
EVERY="${EVERY:-15m}"
ONCE="${ONCE:-0}"
LOG_DIR="${LOG_DIR:-$HOME/.screener/options-logs}"
KEEP_DAYS="${KEEP_DAYS:-30}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
# cron runs with a minimal PATH; make sure uv is reachable.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

stamp=$(date -u +%F)
echo "[$stamp] options record | markets: $MARKETS | every: ${EVERY:-<once>} once: $ONCE"

if [ "$ONCE" = "1" ]; then
  mode_args=(--once)
else
  mode_args=(--every "$EVERY")
fi

for market in $MARKETS; do
  log="$LOG_DIR/options-record-$market-$stamp.log"
  if uv run screener --log-level ERROR options record -m "$market" "${mode_args[@]}" >>"$log" 2>&1; then
    echo "[$stamp] record ok      $market: $(tail -n 1 "$log")"
  else
    echo "[$stamp] record FAILED  $market (see $log)"
  fi
done

# Storage watch: surface (non-fatally) when a store exceeds its size budget.
# Set SCREENER_BARS_BUDGET_MB / SCREENER_CONTRACTS_BUDGET_MB to enable.
if uv run screener cache storage-watch 2>&1 | sed "s/^/[$stamp] storage: /"; then :; else
  echo "[$stamp] storage: OVER BUDGET — see line(s) above"
fi

find "$LOG_DIR" -type f -mtime "+$KEEP_DAYS" -delete
echo "[$stamp] done"
