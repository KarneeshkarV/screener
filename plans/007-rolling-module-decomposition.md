# Plan 007 — Decompose `run_rolling_backtest` (BLOCKED until plan 003 lands)

- **Status:** BLOCKED (prerequisite: plan 003 merged — the correctness suite is the safety net for this refactor)
- **Written against commit:** `9547d4d`
- **Category:** tech-debt
- **Effort:** L · **Risk of fix:** high (behavior-preserving refactor of the hot simulation path)
- **Depends on:** plan 003. Do **not** combine with any other plan.

## Why this matters — and why not yet

`screener/backtester/rolling.py` is 729 lines, and `run_rolling_backtest` alone spans roughly lines 171-553 (~380 lines): candidate-matrix precompute, regime gating, the day loop, ledger assembly, benchmark alignment, and metrics wiring all live in one function (verified outline: `_RollingCandidateMatrices` class at 45, `_build_rolling_candidate_matrices` at 57, `_candidate_rows_for_day` at 139, `run_rolling_backtest` at 171, `backtest_rolling` CLI at 554). It was also recently performance-tuned (searchsorted vectorization, PR #38/#39) — a careless refactor can silently destroy either correctness or that performance.

Advisor verdict recorded honestly: this is real debt, but **the audit found no bug here**, and the module is the best-exercised path in the repo. The refactor only becomes positive-expected-value once `tests/correctness/` (plan 003, incl. `test_lookahead_blindness.py`) is merged and green.

## Pre-work (do first, in this order)

1. Confirm plan 003 is merged: `uv run pytest tests/correctness -q` collects > 50 tests and passes. If not, STOP — this plan is blocked.
2. **Characterization snapshot:** before touching anything, write a test (e.g. `tests/test_rolling_characterization.py`) that runs `run_rolling_backtest` on deterministic synthetic data (reuse `make_bars` + `StubPriceFetcher` from `tests/conftest.py` — see `tests/test_engine.py:21` for the import pattern) with a config exercising: ≥3 tickers, `top` < candidates (ranking matters), a stop-loss, a hold limit, and a regime gate if cheaply constructible. Assert the **exact** trade ledger (tickers, entry/exit dates, fill prices to `pytest.approx`) and final equity. Commit this test against the *unmodified* code; it must pass before and after every step below.
3. **Performance baseline:** time the characterization scenario scaled up (e.g. 100 synthetic tickers × 2 years): `uv run python -c "..."` with `time.perf_counter`. Record the number. The refactor must stay within ~10%.

## Refactor steps (each one: extract → run gates → commit)

Mechanical extraction only — no logic edits, no renames of public symbols, no signature changes to `run_rolling_backtest` or `backtest_rolling`.

1. Read `screener/backtester/rolling.py:171-553` fully and write down the phase boundaries you actually observe (the line ranges here are approximate). Expect roughly: setup/fetch, signal+filter precompute, regime gate, per-day loop, post-loop assembly.
2. Extract the per-day loop body into a private helper (e.g. `_simulate_day(...)`) taking explicit parameters — no closures over mutable locals unless passed explicitly. This is the riskiest extraction; do it first while energy is highest.
3. Extract post-loop ledger/metrics/benchmark assembly into `_assemble_results(...)`.
4. Extract setup/precompute into `_prepare_simulation(...)` returning a small dataclass (follow the existing `_RollingCandidateMatrices` style at line 45).
5. Stop there. Three helpers + the orchestrating function is the target shape; do not split into multiple files or introduce class hierarchies.

## Verification gates (after EVERY step)

```bash
uv run pytest tests/test_rolling_characterization.py tests/correctness -q   # byte-identical behavior
uv run pytest -q
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
uv run mypy
```

Plus once at the end: re-run the performance baseline; must be within ~10% of pre-refactor.

## Done criteria

- `run_rolling_backtest` body is < ~120 lines of orchestration.
- Characterization test unchanged and passing; correctness suite passing; perf within 10%.
- `git diff` shows no changes outside `screener/backtester/rolling.py` and the new test file.

## Escape hatches

- Characterization test changes value at any step → revert that step entirely; do not "fix forward."
- An extraction forces a logic change (e.g. a loop variable mutated in two phases) → leave that boundary alone, note it in the summary, extract elsewhere.
- Perf regresses > 10% and one attempt at hoisting doesn't recover it → revert the offending extraction and report.

## Maintenance note

After this, new rolling-backtest features should land in exactly one phase helper. If a change needs to touch all three, that's a design smell worth a fresh look.
