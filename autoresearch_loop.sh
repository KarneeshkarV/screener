#!/usr/bin/env bash
# Karpathy-style autoresearch loop for trading strategies.
#
# Each iteration:
#   1. Build a leaderboard + tried-strategies list from journal.jsonl so the
#      agent has actual benchmark context (top scores, ideas already covered).
#   2. Launch `claude -p` (full tool access) to mutate autoresearch_strategies.py.
#      If the sandbox is unchanged after the first call, retry once with a
#      sharper directive before giving up on the iteration's mutation.
#   3. Compute the strategies that have NOT yet been evaluated for today's OOS
#      window (per market) and run autoresearch.py with --only on just those.
#      Stock + previously-added strategies stay cached in the journal across
#      iterations of the same day, so each iter only pays for the new strat.
#   4. Snapshot the mutation with a git commit if the sandbox actually changed.
#
# Usage:
#   ./autoresearch_loop.sh                   # 20 iter, US, 150 tickers, 3y, 20 slots
#   ./autoresearch_loop.sh 20 us 500 3 20    # positional: ITER MARKET LIMIT YEARS SLOTS
#   ITER=20 MARKET=us LIMIT=500 YEARS=3 SLOTS=20 ./autoresearch_loop.sh
#   MARKET=both ./autoresearch_loop.sh       # evaluate both us and india per iter
#
# Iteration numbering picks up from the existing autoresearch commit history,
# so successive runs of this script accumulate (iter 1..20, then 21..40, ...)
# instead of clobbering prior iter numbers.
#
# Requires: claude (Claude Code CLI) on PATH, uv, git.
set -uo pipefail

ITER="${1:-${ITER:-20}}"
MARKET="${2:-${MARKET:-us}}"
LIMIT="${3:-${LIMIT:-150}}"
YEARS="${4:-${YEARS:-3}}"
SLOTS="${5:-${SLOTS:-20}}"

# Resolve the markets to evaluate per iteration (supports a "both" alias).
case "$MARKET" in
  both) MARKETS=(us india) ;;
  *)    MARKETS=("$MARKET") ;;
esac

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

# Continue iter numbering from the highest "autoresearch iter N" commit so a
# fresh invocation of this script extends the prior history rather than
# overwriting iter numbers.
PRIOR_MAX_ITER="$(git log --oneline 2>/dev/null \
  | grep -oP 'autoresearch iter \K\d+' \
  | sort -nr | head -n1 || true)"
START_ITER=$(( ${PRIOR_MAX_ITER:-0} + 1 ))
END_ITER=$(( START_ITER + ITER - 1 ))

echo "=== autoresearch: $ITER iterations (iter ${START_ITER}..${END_ITER}) | markets=${MARKETS[*]} limit=$LIMIT years=${YEARS}y slots=$SLOTS ==="
echo "journal: $JOURNAL"
echo "runlog:  $RUNLOG"
echo

START_TS="$(date -Iseconds)"
echo "--- run starting $START_TS (iter ${START_ITER}..${END_ITER}) ---" >> "$RUNLOG"

# Ensure a clean pre-mutation baseline so `git diff` is meaningful.
git add -N "$SANDBOX" 2>/dev/null || true

invoke_claude() {
  # Single Claude invocation. Accepts the prompt as the first arg, appends
  # stdout+stderr to the runlog, and prints the full agent output on stdout
  # so the caller can grep HYPOTHESIS.
  local prompt="$1"
  claude -p \
    --permission-mode=acceptEdits \
    --allowedTools=Edit,Write,Read,Bash,WebSearch,WebFetch,Grep,Glob \
    "$prompt" 2>&1 | tee -a "$RUNLOG"
}

build_leaderboard() {
  # Print a top-25 leaderboard + bottom-5 + full tried-strategies dump.
  # For each strategy: latest entry supplies score/oos (most current window),
  # earliest entry supplies the hypothesis (which describes the strategy that
  # was actually added in that iteration — later iters share a global
  # hypothesis that does NOT describe earlier strategies).
  uv run python <<'PY' 2>/dev/null || true
import json
from pathlib import Path

p = Path("journal.jsonl")
first_hyp = {}   # strategy -> hypothesis from earliest entry
latest = {}      # strategy -> latest journal entry
if p.exists():
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        name = d.get("strategy") or "?"
        if name not in first_hyp:
            first_hyp[name] = (d.get("hypothesis") or "").strip()
        latest[name] = d

if not latest:
    print("(journal is empty — this is the very first iteration)")
    raise SystemExit(0)

ranked = sorted(
    latest.values(),
    key=lambda r: (r.get("score") or -1e9),
    reverse=True,
)

def fmt(r):
    oos = r.get("oos") or {}
    name = r.get("strategy", "?")
    h = first_hyp.get(name, "").replace("\n", " ").strip()[:80]
    return (
        f"{name:<44} "
        f"score={(r.get('score') or 0):+.3f}  "
        f"dsr={(oos.get('dsr') or 0):>5.2f}  "
        f"ret={(oos.get('total_return') or 0):>+7.1%}  "
        f"mdd={(oos.get('max_drawdown') or 0):>+6.1%}  "
        f"sortino={(oos.get('sortino') or 0):>+5.2f}  "
        f"trades={int(oos.get('trade_count') or 0):>4}  "
        f"| {h}"
    )

print(f"-- TOP 25 by OOS score (out of {len(latest)} unique strategies tried) --")
for i, r in enumerate(ranked[:25], 1):
    print(f"{i:>3}. {fmt(r)}")
if len(ranked) > 25:
    print()
    print("-- BOTTOM 5 (avoid these patterns) --")
    for r in ranked[-5:]:
        print(f"     {fmt(r)}")
print()
print(f"-- ALL {len(latest)} TRIED STRATEGY NAMES (do NOT repeat) --")
print(", ".join(sorted(latest.keys())))
PY
}

compute_pending() {
  # Print comma-separated names of registered strategies that have NOT yet
  # been evaluated for today's OOS window for the given market. Empty output
  # means nothing to do.
  TARGET_MARKET="$1" uv run python <<'PY' 2>/dev/null || true
import json
import os
import sys
from datetime import date
from pathlib import Path

try:
    import autoresearch_strategies as A
    import run_pinescript_strategies as P
except Exception as e:
    print(f"[compute_pending] import failed: {e}", file=sys.stderr)
    sys.exit(0)

market = os.environ.get("TARGET_MARKET", "")
today = str(date.today())

registered = set()
for k in P.STRATEGIES:
    registered.add(f"stock:{k}")
for k in A.NEW_STRATEGIES:
    registered.add(f"new:{k}")

done = set()
p = Path("journal.jsonl")
if p.exists():
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        win = d.get("oos_window") or [None, None]
        if win[1] == today and d.get("market") == market:
            done.add(d.get("strategy") or "")

pending = sorted(registered - done)
sys.stdout.write(",".join(pending))
PY
}

for i in $(seq "$START_ITER" "$END_ITER"); do
  echo
  echo "================ iteration $i (${START_ITER}..${END_ITER}) ================"
  ITER_TS="$(date -Iseconds)"
  echo "[$ITER_TS] iter $i begin" >> "$RUNLOG"

  # Build benchmark context for the agent: leaderboard + tried-strategies.
  LEADERBOARD="$(build_leaderboard)"

  read -r -d '' PROMPT_MAIN <<PROMPT_EOF || true
You are iteration ${i}/${END_ITER} of an autoresearch loop searching for
profitable long-only trading strategies. The evaluator runs IS + OOS on every
strategy registered in NEW_STRATEGIES and has already been run for many prior
iterations — see the leaderboard below for the actual scoreboard you must beat.

HARD REQUIREMENTS — failure on any of these wastes an iteration:
  A. You MUST edit autoresearch_strategies.py using the Edit or Write tool.
     Add a NEW strat_<name> function and register it in NEW_STRATEGIES.
     The name MUST NOT match any in the "ALL TRIED STRATEGY NAMES" list below.
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

IDEA SOURCING — your idea MUST be different from the tried list. Use
WebSearch and WebFetch to mine fresh signals from primary sources. Prefer:
  - Academic / quant research papers: arxiv.org (q-fin.TR, q-fin.PM, q-fin.ST),
    SSRN, Journal of Financial Economics, Review of Financial Studies, Journal
    of Portfolio Management, Quantitative Finance.
  - Practitioner literature: AQR / Two Sigma / Man AHL / Robeco white papers,
    QuantPedia, SSRN working papers, CFA Institute Research Foundation.
  - Books / monographs: Lopez de Prado, Bouchaud, Bandy, Chan, Carver, Clenow,
    Tharp, Aronson — anything with a clean entry/exit rule you can code up.
Cite the paper / author / year in the function docstring so future iterations
can trace the lineage. Prefer ideas with a published, named edge (e.g. "Moskowitz
Ooi Pedersen 2012 time-series momentum") over generic indicator tweaks.

LEADERBOARD + TRIED LIST (real benchmark — beat the top score):
---
${LEADERBOARD}
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
     strat_<name>(df) function AND register it in NEW_STRATEGIES. The name
     MUST be one not already present in the existing NEW_STRATEGIES dict.
  2. Run: Bash: git diff --stat -- autoresearch_strategies.py
     to confirm the file changed.
  3. Print exactly one line starting with 'HYPOTHESIS: ' describing the
     idea in <=120 chars. Cite the paper/author/year if applicable.

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

  # Evaluate ONLY the strategies that haven't been scored for today's OOS
  # window yet (per market). On the first iter of a fresh day this is the
  # full registered set; on subsequent iters it's typically just the one
  # strategy claude added.
  EVAL_RC=0
  for mkt in "${MARKETS[@]}"; do
    PENDING="$(compute_pending "$mkt")"
    if [[ -z "$PENDING" ]]; then
      echo "[iter $i] market=$mkt: nothing pending (all strategies already scored for today's OOS window) — skipping eval" | tee -a "$RUNLOG"
      continue
    fi
    # Count names without spawning awk for every iter.
    PENDING_COUNT="$(echo -n "$PENDING" | tr ',' '\n' | grep -c .)"
    echo "[iter $i] evaluating market=$mkt — $PENDING_COUNT pending strategies"
    uv run python autoresearch.py evaluate \
      --market "$mkt" \
      --years "$YEARS" \
      --limit "$LIMIT" \
      --slots "$SLOTS" \
      --iteration "$i" \
      --hypothesis "$HYPOTHESIS" \
      --journal "$JOURNAL" \
      --only "$PENDING" 2>&1 | tee -a "$RUNLOG"
    rc="${PIPESTATUS[0]}"
    if [[ "$rc" -ne 0 ]]; then
      EVAL_RC="$rc"
      break
    fi
  done

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
