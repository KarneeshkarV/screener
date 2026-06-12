# Plan 003 — Revive the correctness-verification test suite branch

- **Status:** TODO
- **Written against commit:** `9547d4d` on `main`
- **Category:** tests
- **Effort:** M (was L; most of the work already exists on a branch) · **Risk of fix:** low-medium
- **Depends on:** nothing. **Blocks:** plan 004 (fetch consolidation), plan 007 (rolling.py decomposition).

## Why this matters

`tests/correctness/` on `main` contains only an empty `fixtures/` dir and stale `__pycache__` — no tests. But a local branch **`test/correctness-verification-suite`** already contains a ~5,800-line independent correctness suite that was never merged:

```
$ git diff --stat main...test/correctness-verification-suite | tail
 tests/correctness/test_lookahead_blindness.py      | 480 ++++++++
 tests/correctness/test_metrics_edge_cases.py       | 298 ++++
 tests/correctness/test_metrics_golden.py           | 242 ++++
 tests/correctness/test_metrics_vs_empyrical.py     | 324 ++++
 tests/correctness/test_reference_witnesses.py      | 325 ++++
 tests/correctness/test_scoring.py                  | 729 ++++++++
 22 files changed, 5781 insertions(+), 6 deletions(-)
```

Branch commits (newest first): `83e01d3 fix: mionr`, `9a413de fix(metrics): correct CAGR off-by-one; harden test independence`, `27383df test: add independent correctness-verification suite`.

Two things make this high-leverage: (1) look-ahead/metrics correctness is the core risk of this codebase and these tests target exactly that; (2) commit `9a413de` claims a **CAGR off-by-one fix in production code** — if `main`'s `screener/backtester/metrics.py` still has that bug, every backtest report understates/overstates CAGR today.

The branch also changes `pyproject.toml` (verified):

```
 dev = [
+    "empyrical-reloaded>=0.5.12",
+    "pandas-ta-classic>=0.6.20",
      ...
+[tool.pytest.ini_options]
+markers = [
+    "network: requires live internet access (opt in with SCREENER_LIVE_TESTS=1)",
+    "requires_talib: ...",
+    "requires_quantstats: ...",
```

## Steps

Work on a new branch off current `main` (e.g. `test/revive-correctness-suite`). Do not force-push or rewrite the old branch.

### Step 1 — survey the divergence

```bash
git log --oneline main..test/correctness-verification-suite      # expect the 3 commits above
git log --oneline test/correctness-verification-suite..main      # how far main has moved since
git diff main...test/correctness-verification-suite --stat       # full 22-file list
```

Identify which of the 22 changed files touch **production code** (anything under `screener/`) vs tests/config. Pay special attention to `screener/backtester/metrics.py`.

### Step 2 — decide the CAGR fix's fate (investigate, don't assume)

Diff the branch's `metrics.py` change against current `main`:

```bash
git diff main...test/correctness-verification-suite -- screener/backtester/metrics.py
```

- If `main` has since received an equivalent fix → drop the production hunk, keep only tests.
- If `main` still has the off-by-one → port the fix, and say so prominently in your final summary (it changes reported CAGR numbers for users).
- If the diff doesn't apply cleanly because `metrics.py` was refactored → STOP on this hunk, port the *tests* only, mark the failing CAGR test `xfail` with a comment referencing this plan, and report.

### Step 3 — bring the suite over

Prefer cherry-picking the 3 commits onto the new branch and resolving conflicts (`uv.lock` conflicts: resolve by re-running `uv sync --all-groups` / `uv lock` after taking `main`'s version and re-adding the two new dev deps to `pyproject.toml`). If cherry-pick conflicts are extensive, fall back to checking out the test files directly:

```bash
git checkout test/correctness-verification-suite -- tests/correctness/
```

then hand-apply the `pyproject.toml` additions (dev deps + pytest markers) and run `uv lock`.

### Step 4 — make it pass offline

CI runs plain `uv run pytest` with no network. The branch's `network` marker exists for live tests — verify that network-marked tests are skipped by default (check for a `conftest.py` addition in the branch that implements the `SCREENER_LIVE_TESTS=1` opt-in; if the skip logic is missing, add it in `tests/correctness/conftest.py`):

```bash
uv run pytest tests/correctness -q          # all pass or skip, no network access
```

Tests that fail because **main's production code has drifted** (not network, not deps): investigate each one individually. A failing correctness test is potentially a real bug report — do not weaken the assertion to make it pass. If the expected value is genuinely stale (e.g. the test encoded pre-drift behavior that was deliberately changed on main), update it and note why; if you cannot tell, mark `xfail` with a reason and list it in your summary.

### Step 5 — full gates

```bash
uv run pytest -q
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
uv run mypy
```

Note: the new test files are excluded from mypy (`exclude = ["tests"]` in pyproject) but NOT from ruff — expect to fix lint/format in the ported files.

## Done criteria

- `uv run pytest tests/correctness -q` exits 0 with > 50 tests collected (suite is ~6 files; zero collected means the port failed).
- Full `uv run pytest` passes; ruff + mypy clean.
- Summary explicitly states: whether the CAGR fix was needed on main, every `xfail` added, and every test whose expected values were updated.

## Boundaries

- **In scope:** `tests/correctness/**`, `tests/conftest.py` (only if marker-skip logic requires it), `pyproject.toml` (dev deps + markers), `uv.lock` (via `uv lock` only), `screener/backtester/metrics.py` (only the CAGR hunk, per step 2).
- **Out of scope:** every other `screener/` module; CI workflow changes (plain `pytest` already picks up `tests/correctness/`); deleting or modifying the original branch.

## Escape hatches

- More than ~5 correctness tests fail against main's production code → STOP and report the list; that's a bug harvest needing human triage, not a test-porting task.
- The two new dev deps fail to resolve/install on Python 3.11 → port the suite without the `test_metrics_vs_empyrical.py` / TA-dependent files, skip-marker the rest, and report.

## Maintenance note

Once merged, `tests/correctness/` becomes the regression gate that plans 004 and 007 rely on. Future metric or engine changes that break a golden test should be treated as "prove the new number is right," not "update the golden."
