#!/usr/bin/env bash
# Daily screen → backtest replay bridge.
#
# Phase 1: run `screener screen` for EVERY registered criterion (all plugins in
#   screener/criteria/plugins/) on each market in $MARKETS, persisting each run
#   to ~/.screener/history.db.
# Phase 2: for each market:criteria pair, replay the most recent persisted run
#   that is at least $REPLAY_AGE_DAYS old via `backtest-historical --from-run`,
#   writing the metrics log and HTML tear-sheet to $LOG_DIR.
# Phase 3: back up the local history.db to Turso via `screener history-backup`
#   (reads TURSO_* creds from the repo's .env). Non-fatal.
#
# Pairs without an old-enough run are skipped (logged, non-fatal), so the first
# $REPLAY_AGE_DAYS days after install only accumulate history.
#
# Cron example (also documented in README.md):
#   30 11 * * 1-5 /root/screneer_main/screener/scripts/daily_screen_replay.sh >> "$HOME/.screener/replay-logs/cron.log" 2>&1
set -uo pipefail

REPO_DIR="${SCREENER_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
MARKETS="${MARKETS:-us india}"
REPLAY_AGE_DAYS="${REPLAY_AGE_DAYS:-7}"
HOLD="${HOLD:-5}"
TOP_N="${TOP_N:-50}"
LOG_DIR="${LOG_DIR:-$HOME/.screener/replay-logs}"
KEEP_DAYS="${KEEP_DAYS:-30}"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"
# cron runs with a minimal PATH; make sure uv is reachable.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

stamp=$(date -u +%F)
criteria="${CRITERIA:-$(uv run python -c "import screener.cli, screener.criteria as c; print(' '.join(sorted(c.CRITERIA)))")}"
echo "[$stamp] markets: $MARKETS | criteria: $criteria"

for market in $MARKETS; do
  for crit in $criteria; do
    log="$LOG_DIR/screen-$market-$crit-$stamp.log"
    if uv run screener --log-level ERROR screen -m "$market" -c "$crit" -n "$TOP_N" >"$log" 2>&1; then
      echo "[$stamp] screen ok      $market:$crit"
    else
      echo "[$stamp] screen FAILED  $market:$crit (see $log)"
    fi
  done
done

for market in $MARKETS; do
  for crit in $criteria; do
    out="$LOG_DIR/replay-$market-$crit-$stamp"
    if uv run screener --log-level ERROR backtest-historical \
        --from-run "$market:$crit" --run-age-days "$REPLAY_AGE_DAYS" \
        --hold "$HOLD" --report "$out.html" >"$out.log" 2>&1; then
      echo "[$stamp] replay ok      $market:$crit → $out.log"
    else
      echo "[$stamp] replay skipped $market:$crit: $(tail -n 1 "$out.log")"
      rm -f "$out.html"
    fi
  done
done

backup_log="$LOG_DIR/history-backup-$stamp.log"
if uv run screener --log-level ERROR history-backup >"$backup_log" 2>&1; then
  echo "[$stamp] backup ok      history.db → Turso ($(tail -n 1 "$backup_log"))"
else
  echo "[$stamp] backup FAILED  history.db → Turso (see $backup_log)"
fi

find "$LOG_DIR" -type f -mtime "+$KEEP_DAYS" -delete
echo "[$stamp] done"
