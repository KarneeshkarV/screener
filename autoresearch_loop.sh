#!/usr/bin/env bash
# Karpathy-style autoresearch loop for trading strategies.
#
# Each iteration:
#   1. Launch `claude -p` (full tool access) to mutate autoresearch_strategies.py.
#      If the sandbox is unchanged after the first call, retry once with a
#      sharper directive before giving up on the iteration's mutation.
#   2. Run autoresearch.py on in-sample + out-of-sample windows (OHLCV is
#      parquet-cached, so only iter 1 pays the fetch cost).
#   3. Append all N strategy results for this iteration to journal.jsonl.
#   4. Snapshot the mutation with a git commit if the sandbox actually changed.
#
# Usage:
#   ./autoresearch_loop.sh                   # 20 iter, US, 150 tickers, 3y, 20 slots
#   ./autoresearch_loop.sh 20 us 500 3 20    # positional: ITER MARKET LIMIT YEARS SLOTS
#   ITER=20 MARKET=us LIMIT=500 YEARS=3 SLOTS=20 ./autoresearch_loop.sh
#
# Requires: claude (Claude Code CLI) on PATH, uv, git.
set -uo pipefail

ITER="${1:-${ITER:-20}}"
MARKET="${2:-${MARKET:-us}}"
LIMIT="${3:-${LIMIT:-150}}"
YEARS="${4:-${YEARS:-3}}"
SLOTS="${5:-${SLOTS:-20}}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

JOURNAL="$ROOT/journal.jsonl"
RUNLOG="$ROOT/autoresearch_runs.log"
SANDBOX="$ROOT/autoresearch_strategies.py"

command -v claude >/dev/null 2>&1 || { echo "claude CLI not found on PATH." >&2; exit 1; }
command -v uv     >/dev/null 2>&1 || { echo "uv not found on PATH." >&2; exit 1; }
command -v git    >/dev/null 2>&1 || { echo "git not found on PATH." >&2; exit 1; }

[[ -d .git ]] || { echo "run this from the repo root" >&2; exit 1; }

mkdir -p .autoresearch/ohlcv
touch "$JOURNAL" "$RUNLOG"

echo "=== autoresearch: $ITER iterations | market=$MARKET limit=$LIMIT years=${YEARS}y slots=$SLOTS ==="
echo "journal: $JOURNAL"
echo "runlog:  $RUNLOG"
echo

START_TS="$(date -Iseconds)"
echo "--- run starting $START_TS ---" >> "$RUNLOG"

# Ensure a clean pre-mutation baseline so `git diff` is meaningful.
git add -N "$SANDBOX" 2>/dev/null || true

invoke_claude() {
  # Single Claude invocation. Accepts the prompt on stdin and appends
  # stdout+stderr to the runlog, then prints the full agent output on stdout
  # so the caller can grep HYPOTHESIS.
  local prompt="$1"
  claude -p \
    --permission-mode=acceptEdits \
    --allowedTools=Edit,Write,Read,Bash,WebSearch,WebFetch,Grep,Glob \
    "$prompt" 2>&1 | tee -a "$RUNLOG"
}

for i in $(seq 1 "$ITER"); do
  echo
  echo "================ iteration $i / $ITER ================"
  ITER_TS="$(date -Iseconds)"
  echo "[$ITER_TS] iter $i begin" >> "$RUNLOG"

  # Recent journal context for the agent (last 40 lines).
  JOURNAL_TAIL="$(tail -n 40 "$JOURNAL" 2>/dev/null || true)"

  read -r -d '' PROMPT_MAIN <<PROMPT_EOF || true
You are iteration ${i}/${ITER} of an autoresearch loop searching for profitable
long-only trading strategies. The evaluator runs IS + OOS on every strategy
registered in NEW_STRATEGIES and has already been run for previous iterations
(see journal tail below).

HARD REQUIREMENTS — failure on any of these wastes an iteration:
  A. You MUST edit autoresearch_strategies.py using the Edit or Write tool.
     Adding a new strat_<name> function and registering it in NEW_STRATEGIES.
     Verify the diff with: Bash: git diff -- autoresearch_strategies.py
  B. You MUST print exactly one line to stdout in the form:
        HYPOTHESIS: <<=120 char one-liner describing the idea>>
     Print this AFTER the edit succeeds. No other lines may start with
     "HYPOTHESIS:".
  C. Do NOT edit any other file. Forbidden: run_pinescript_strategies.py,
     engine.py, portfolio.py, slippage.py, metrics.py, autoresearch.py,
     autoresearch_loop.sh.
  D. Do NOT run the backtest yourself. The driver runs autoresearch.py.

STRATEGY CONSTRAINTS:
  - Long-only, bar-close decisions, no lookahead (use .shift() / prev-bar
    arrays for entry/exit conditions that reference prior values).
  - df columns: date, open, high, low, close, volume, adj_close.
  - Use _walk(entries, exits, close, df["date"].values) to produce trades.
  - Import helpers from run_pinescript_strategies: _ema, _sma, _rma, _stdev,
    _rsi, _atr, _supertrend_dir, _walk, Trade.
  - Pick an idea that's DIFFERENT from what's in the journal tail. You may
    use WebSearch / WebFetch to pull ideas from quant literature.

JOURNAL TAIL (most recent first):
---
${JOURNAL_TAIL}
---

Begin. Edit the sandbox, then print HYPOTHESIS: <text>.
PROMPT_EOF

  # Record sandbox checksum BEFORE claude runs so we can detect a real edit.
  PRE_SHA="$(git hash-object "$SANDBOX" 2>/dev/null || echo none)"

  AGENT_OUT="$(invoke_claude "$PROMPT_MAIN")"

  POST_SHA="$(git hash-object "$SANDBOX" 2>/dev/null || echo none)"
  HYPOTHESIS="$(echo "$AGENT_OUT" | grep -m1 '^HYPOTHESIS:' | sed 's/^HYPOTHESIS: *//' | head -c 240)"

  # Retry once if the sandbox wasn't touched OR no HYPOTHESIS line was emitted.
  if [[ "$PRE_SHA" == "$POST_SHA" || -z "$HYPOTHESIS" ]]; then
    echo "[iter $i] first pass produced no mutation / no HYPOTHESIS — retrying once" | tee -a "$RUNLOG"
    read -r -d '' PROMPT_RETRY <<PROMPT_EOF || true
Iteration ${i} retry. On the previous attempt you did not satisfy the hard
requirements. Execute them NOW with no further reasoning:

  1. Use Edit (or Write) to modify autoresearch_strategies.py: add a new
     strat_<name>(df) function AND register it in NEW_STRATEGIES.
  2. Run: Bash: git diff --stat -- autoresearch_strategies.py
     to confirm the file changed.
  3. Print exactly one line starting with 'HYPOTHESIS: ' describing the
     idea in <=120 chars.

No other output, no other edits. The sandbox is at $SANDBOX.
PROMPT_EOF
    AGENT_OUT_RETRY="$(invoke_claude "$PROMPT_RETRY")"
    POST_SHA="$(git hash-object "$SANDBOX" 2>/dev/null || echo none)"
    HYPOTHESIS_RETRY="$(echo "$AGENT_OUT_RETRY" | grep -m1 '^HYPOTHESIS:' | sed 's/^HYPOTHESIS: *//' | head -c 240)"
    [[ -n "$HYPOTHESIS_RETRY" ]] && HYPOTHESIS="$HYPOTHESIS_RETRY"
  fi

  SANDBOX_CHANGED="no"
  [[ "$PRE_SHA" != "$POST_SHA" ]] && SANDBOX_CHANGED="yes"
  [[ -z "$HYPOTHESIS" ]] && HYPOTHESIS="(no hypothesis; sandbox_changed=$SANDBOX_CHANGED)"

  echo "[iter $i] sandbox_changed=$SANDBOX_CHANGED  hypothesis: $HYPOTHESIS"

  # Always evaluate — even no-op iterations produce 9 baseline journal entries,
  # which is useful for tracking drift from prior runs.
  echo "[iter $i] evaluating..."
  uv run python autoresearch.py evaluate \
    --market "$MARKET" \
    --years "$YEARS" \
    --limit "$LIMIT" \
    --slots "$SLOTS" \
    --iteration "$i" \
    --hypothesis "$HYPOTHESIS" \
    --journal "$JOURNAL" 2>&1 | tee -a "$RUNLOG"
  EVAL_RC="${PIPESTATUS[0]}"

  if [[ "$EVAL_RC" -ne 0 ]]; then
    echo "[iter $i] evaluator failed (rc=$EVAL_RC) — reverting sandbox" | tee -a "$RUNLOG"
    git checkout -- "$SANDBOX" 2>>"$RUNLOG" || true
    continue
  fi

  # Commit only if the sandbox actually changed.
  if [[ "$SANDBOX_CHANGED" == "yes" ]]; then
    git add "$SANDBOX"
    git -c user.name="autoresearch" -c user.email="autoresearch@local" \
        commit -m "autoresearch iter ${i}: ${HYPOTHESIS}" >>"$RUNLOG" 2>&1 || true
  fi

  echo "[iter $i] done"
done

echo
echo "=== autoresearch complete: $ITER iterations ==="
echo "total journal lines (all runs):"
grep -c "" "$JOURNAL" || true
echo
echo "top 10 by OOS score across the entire journal:"
uv run python - <<'PY'
import json
from pathlib import Path
rows = []
for line in Path("journal.jsonl").read_text().splitlines():
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
rows.sort(key=lambda r: r.get("score", 0) or 0, reverse=True)
for r in rows[:10]:
    oos = r.get("oos") or {}
    print(f"  iter={r.get('iteration'):>2}  {r.get('strategy',''):<30} "
          f"score={r.get('score',0):+.3f}  "
          f"dsr={(oos.get('dsr') or 0):.2f}  "
          f"ret={(oos.get('total_return') or 0):+.1%}  "
          f"mdd={(oos.get('max_drawdown') or 0):+.1%}  "
          f"sortino={(oos.get('sortino') or 0):+.2f}")
PY
