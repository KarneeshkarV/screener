#!/usr/bin/env bash
# Daily 1m bar recorder.
#
# Appends the trailing window of 1m bars for each market's active universe
# (default: the market's default universe — sp500 / nifty50) into the
# interval-partitioned bar store at ~/.screener/bars/{market}/1m/. Run once
# per day after the close, the archive grows past yfinance's ~30-day free 1m
# history cap; 5m/15m/30m/1h backtest requests are then served by resampling
# the stored 1m series locally instead of separate per-interval downloads.
#
# Runs are idempotent (overlapping bars dedupe on write), so rerunning after a
# failure is safe. A gap longer than the provider cap (~30d) leaves a hole the
# recorder cannot backfill — keep the cron daily.
#
# Cron example (also documented in README.md):
#   30 1 * * 2-6 /root/screneer_main/screener/scripts/daily_bars_record.sh >> "$HOME/.screener/bars-logs/cron.log" 2>&1
set -uo pipefail

REPO_DIR="${SCREENER_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
MARKETS="${MARKETS:-us india}"
DAYS="${DAYS:-2}"
LOG_DIR="${LOG_DIR:-$HOME/.screener/bars-logs}"
KEEP_DAYS="${KEEP_DAYS:-30}"

mkdir -p "$LOG_DIR"
# One run at a time: a hung run must not overlap the next day's cron.
exec 9>"$LOG_DIR/.daily_bars_record.lock"
if ! flock -n 9; then
  echo "another daily_bars_record run is active; exiting"
  exit 0
fi
cd "$REPO_DIR" || { echo "cannot cd to $REPO_DIR" >&2; exit 1; }
# cron runs with a minimal PATH; make sure uv is reachable.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

stamp=$(date -u +%F)
echo "[$stamp] markets: $MARKETS | days: $DAYS"

failed=0
read -ra markets <<< "$MARKETS"
for market in "${markets[@]}"; do
  log="$LOG_DIR/bars-record-$market-$stamp.log"
  if uv run screener --log-level ERROR bars record -m "$market" --days "$DAYS" >"$log" 2>&1; then
    echo "[$stamp] record ok      $market: $(tail -n 1 "$log")"
  else
    echo "[$stamp] record FAILED  $market (see $log)"
    failed=1
  fi
done

# Storage watch: surface (non-fatally) when a store exceeds its size budget.
# Set SCREENER_BARS_BUDGET_MB / SCREENER_CONTRACTS_BUDGET_MB to enable.
if uv run screener cache storage-watch 2>&1 | sed "s/^/[$stamp] storage: /"; then :; else
  echo "[$stamp] storage: OVER BUDGET — see line(s) above"
fi

find "$LOG_DIR" -type f -name 'bars-record-*.log' -mtime "+$KEEP_DAYS" -delete
echo "[$stamp] done"
exit "$failed"
