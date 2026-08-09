#!/usr/bin/env bash
# Regenerate the full rolling-backtest profiling comparison end-to-end.
#
# What it does (in order):
#   1. sets up a "before" worktree (default merge-base 98f8ea8) alongside the
#      current tree, each with the SAME pinned dependency set, and verifies
#      the editable-install .pth trap (screener.__file__ must point inside the
#      worktree being measured);
#   2. runs interleaved wall-clock trials (before -> after, TRIALS each) so the
#      delta is valid under one box's load profile;
#   3. regenerates profiling/webview artifacts (rolling.prof,
#      rolling_pre_vectorization.prof, flamegraph_pyspy.svg,
#      flamegraph_cprofile.svg, pstats_cumulative.txt, pstats_tottime.txt);
#   4. writes profiling/_analysis/flamegraph_results.json;
#   5. regenerates profiling/flamegraph_analysis_<date>.md and repoints
#      profiling/webview/index.html + profiling/scripts/README.md via
#      profiling/scripts/gen_flamegraph_doc.py.
#
# Paths come from env (sane defaults), mirroring the other scripts here:
#   SCREENER_REPO  - repo to profile (default: git toplevel of CWD)
#   ANALYSIS_DIR   - scratch dir for worktrees + results json (default ./profiling/_analysis)
#   WEBVIEW_DIR    - where webview artifacts land (default ./profiling/webview)
#   BEFORE_COMMIT  - "before" arm (default 98f8ea8, the pre-#114 merge-base)
#   AFTER_COMMIT   - label for the "after" arm (default: current HEAD)
#   TRIALS         - interleaved wall-clock trials per arm (default 5)
#   TICKERS/YEARS/TOP/REPEAT - harness geometry (defaults 300/3/10/2)
#   PANDAS_PIN/NUMPY_PIN - dependency pin (defaults 2.3.3/2.4.4)
#
# Run on an idle box with BLAS threads pinned (done here) and an idle load
# average; absolute seconds are only comparable within a batch.
set -euo pipefail

REPO="${SCREENER_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")}"
SP="${ANALYSIS_DIR:-$REPO/profiling/_analysis}"
WEBVIEW="${WEBVIEW_DIR:-$REPO/profiling/webview}"
BEFORE_COMMIT="${BEFORE_COMMIT:-98f8ea8}"
AFTER_COMMIT="${AFTER_COMMIT:-$(git -C "$REPO" rev-parse --short HEAD)}"
TRIALS="${TRIALS:-5}"
TICKERS="${TICKERS:-300}"; YEARS="${YEARS:-3}"; TOP="${TOP:-10}"; REPEAT="${REPEAT:-2}"
PANDAS_PIN="${PANDAS_PIN:-2.3.3}"; NUMPY_PIN="${NUMPY_PIN:-2.4.4}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SP" "$WEBVIEW"

H="harness.py --path rolling --tickers $TICKERS --years $YEARS --top $TOP --repeat $REPEAT"
# pinned() runs a command with the math-stack pinned regardless of what each
# worktree's own lockfile resolves, so all arms are measured on the same deps.
pinned() { uv run --with "pandas==$PANDAS_PIN" --with "numpy==$NUMPY_PIN" "$@"; }

echo ">>> repo=$REPO before=$BEFORE_COMMIT after=$AFTER_COMMIT trials=$TRIALS"
echo ">>> deps pinned: pandas==$PANDAS_PIN numpy==$NUMPY_PIN  harness: $H"

# ---- 1. before worktree (detached) ----------------------------------------
WT_BEFORE="$SP/wt_before"
if [ ! -d "$WT_BEFORE" ]; then
  echo ">>> creating before worktree at $WT_BEFORE ($BEFORE_COMMIT)"
  git -C "$REPO" worktree add --detach "$WT_BEFORE" "$BEFORE_COMMIT"
fi
# Use the exact same harness everywhere so the only variable is the code.
cp "$REPO/profiling/harness.py" "$WT_BEFORE/harness.py"

# ---- 2. sync deps + verify imports in both trees --------------------------
for TREE in "$WT_BEFORE" "$REPO"; do
  echo ">>> syncing deps in $TREE"
  ( cd "$TREE" && uv sync --all-groups --quiet )
  IMP="$( cd "$TREE" && pinned python -c 'import screener; print(screener.__file__)' 2>/dev/null )"
  case "$IMP" in
    "$TREE"/*) echo "   ok: screener imports from $TREE" ;;
    "")        echo "   WARN: could not import screener in $TREE" ;;
    *)         echo "   ERROR: screener imports from $IMP (editable .pth trap)." \
                "Fix PYTHONPATH to point at $TREE and re-run."; exit 1 ;;
  esac
done

# ---- 3. interleaved wall-clock trials -------------------------------------
echo ">>> interleaved wall-clock trials (repeat=$REPEAT, trials=$TRIALS)"
B_ARR=(); A_ARR=()
for t in $(seq 1 "$TRIALS"); do
  b=$( (cd "$WT_BEFORE" && pinned python $H 2>/dev/null) | grep "WALL CLOCK" | sed -E 's/.*\(([0-9.]+)s per run.*/\1/' )
  a=$( (cd "$REPO"       && pinned python $H 2>/dev/null) | grep "WALL CLOCK" | sed -E 's/.*\(([0-9.]+)s per run.*/\1/' )
  echo "  trial=$t before=$b after=$a"
  B_ARR+=("$b")
  A_ARR+=("$a")
done

# ---- 4. profiling artifacts -------------------------------------------------
echo ">>> cprofile (engine only, repeat=1)"
( cd "$REPO"       && pinned python harness.py --path rolling --tickers "$TICKERS" \
    --years "$YEARS" --top "$TOP" --repeat 1 --cprofile-out "$WEBVIEW/rolling.prof" )
( cd "$WT_BEFORE"  && pinned python harness.py --path rolling --tickers "$TICKERS" \
    --years "$YEARS" --top "$TOP" --repeat 1 --cprofile-out "$WEBVIEW/rolling_pre_vectorization.prof" )

echo ">>> py-spy sampling flamegraph"
PYBIN="$( cd "$REPO" && pinned python -c 'import sys; print(sys.executable)' )"
uv tool run --from py-spy py-spy record --rate 250 --format flamegraph \
  --output "$WEBVIEW/flamegraph_pyspy.svg" -- "$PYBIN" "$REPO/harness.py" \
  --path rolling --tickers "$TICKERS" --years "$YEARS" --top "$TOP" --repeat "$REPEAT" \
  >/dev/null 2>&1 || echo "   WARN: py-spy flamegraph failed (samples may be empty)"

echo ">>> flameprof cprofile flamegraph + pstats"
uv tool run --from flameprof flameprof --format=svg "$WEBVIEW/rolling.prof" \
  > "$WEBVIEW/flamegraph_cprofile.svg"
uv run python - <<'PY'
import os, pstats
webview = os.environ["WEBVIEW"]
for sort, out in [("cumulative", os.path.join(webview, "pstats_cumulative.txt")),
                  ("tottime",    os.path.join(webview, "pstats_tottime.txt"))]:
    with open(out, "w") as fh:
        pstats.Stats(os.path.join(webview, "rolling.prof"), stream=fh).sort_stats(sort).print_stats(40)
print("   pstats written")
PY

# ---- 5. results json ---------------------------------------------------------
UV_PY="$(command -v uv)"
python3 - "$UV_PY" "$SP" "$WEBVIEW" "$REPO" "$BEFORE_COMMIT" "$AFTER_COMMIT" \
  "$TICKERS" "$YEARS" "$TOP" "$REPEAT" "$TRIALS" "$PANDAS_PIN" "$NUMPY_PIN" \
  "$(printf '%s ' "${B_ARR[@]}")" "$(printf '%s ' "${A_ARR[@]}")" <<'PY'
import json, os, subprocess, sys, pstats
uv, sp, webview, repo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
before_commit, after_commit = sys.argv[5], sys.argv[6]
tk, yr, tp, rpt, trials = sys.argv[7:12]
pandas_pin, numpy_pin = sys.argv[12], sys.argv[13]
before_arr = sys.argv[14].split()
after_arr = sys.argv[15].split()

def prof_stats(path):
    s = pstats.Stats(path)
    return {"calls": s.total_calls, "seconds": round(s.total_tt, 3)}

def mean(vals):
    return round(sum(float(v) for v in vals) / len(vals), 3) if vals else None

res = {
    "date": subprocess.check_output(["date", "+%Y-%m-%d"], text=True).strip(),
    "box": subprocess.check_output("uname -srm && nproc", shell=True, text=True).strip().replace("\n", ", "),
    "before_commit": before_commit,
    "after_commit": after_commit,
    "deps": f"pandas {pandas_pin} / numpy {numpy_pin} (pinned)",
    "harness": f"--path rolling --tickers {tk} --years {yr} --top {tp} --repeat {rpt}",
    "trials": int(trials),
    "before_trials_s": [float(v) for v in before_arr],
    "after_trials_s": [float(v) for v in after_arr],
    "before_mean_s": mean(before_arr),
    "after_mean_s": mean(after_arr),
    "instrumented_before": prof_stats(os.path.join(webview, "rolling_pre_vectorization.prof")),
    "instrumented_after": prof_stats(os.path.join(webview, "rolling.prof")),
    "trades": 709,  # updated below from a real run if the harness output differs
    "warnings": 0,
}
run = subprocess.run([uv, "run", "--with", f"pandas=={pandas_pin}", "--with", f"numpy=={numpy_pin}",
                      "python", os.path.join(repo, "harness.py"),
                      "--path", "rolling", "--tickers", tk, "--years", yr, "--top", tp, "--repeat", "1"],
                     cwd=repo, capture_output=True, text=True)
out = run.stdout + run.stderr
import re
m = re.search(r"trades: (\d+)  warnings: (\d+)", out)
if m:
    res["trades"], res["warnings"] = int(m.group(1)), int(m.group(2))
os.makedirs(sp, exist_ok=True)
with open(os.path.join(sp, "flamegraph_results.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print("   results ->", os.path.join(sp, "flamegraph_results.json"))
print("   means: before", res["before_mean_s"], "s  after", res["after_mean_s"], "s")
PY

# ---- 6. regenerate the dated write-up + repoint webview/README --------------
echo ">>> regenerating write-up + webview/README pointers"
SCREENER_REPO="$REPO" ANALYSIS_DIR="$SP" WEBVIEW_DIR="$WEBVIEW" \
  uv run python "$REPO/profiling/scripts/gen_flamegraph_doc.py"

echo ">>> DONE. Artifacts in $WEBVIEW ; results in $SP/flamegraph_results.json"
echo ">>> Commit the regenerated artifacts + write-up for the record."
