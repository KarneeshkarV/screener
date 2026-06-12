# Plan 005 — Staged re-enable of disabled mypy error codes

- **Status:** DONE
- **Written against commit:** `9547d4d`
- **Category:** tech-debt
- **Effort:** M (staged; each stage is S) · **Risk of fix:** medium
- **Depends on:** nothing. Execute stages as separate commits/PRs.

## Why this matters

`pyproject.toml` declares `strict = true` for mypy but disables 12 error codes, including the ones that catch real bugs (`arg-type`, `return-value`, `union-attr`, `attr-defined`, `operator`). The audit probed two codes: re-enabling `arg-type` alone surfaces ~44 errors across ~21 files; `no-untyped-def` ~51 (counts are from the audit run at `9547d4d` — re-measure, don't trust them blindly). In a pandas-heavy financial codebase these disabled checks are exactly where silent `None`/dtype bugs hide.

## Current state (verified)

`pyproject.toml:54-72`:

```toml
[tool.mypy]
python_version = "3.11"
strict = true
files = ["main.py", "run_pinescript_strategies.py", "screener"]
exclude = ["tests"]
disable_error_code = [
    "arg-type",
    "attr-defined",
    "call-overload",
    "index",
    "no-any-return",
    "no-untyped-call",
    "no-untyped-def",
    "operator",
    "return-value",
    "type-arg",
    "union-attr",
    "var-annotated",
]
```

CI runs `uv run mypy` (`.github/workflows/ci.yml`), and the baseline passes today.

## Strategy

Re-enable codes from cheapest/highest-signal to most expensive, one stage per commit. **Measure first, fix second.** For each code:

```bash
# measure (does not modify anything):
uv run mypy --enable-error-code <code> 2>&1 | tail -3
```

### Stage order

1. **`return-value`** and **`union-attr`** together (audit measured ~3 errors each).
2. **`var-annotated`**, **`index`**, **`operator`**, **`call-overload`** — measure each; bundle any with ≤ ~10 errors into this stage, defer the rest.
3. **`attr-defined`** and **`no-any-return`** — measure; likely dominated by pandas/yfinance dynamic attrs.
4. **`arg-type`** (~44 errors).
5. **`type-arg`**, **`no-untyped-call`**, **`no-untyped-def`** — bulk annotation work, lowest bug-catching value per fix. Optional; stop here if effort exceeds value.

### Rules for fixes (every stage)

- A fix is a **type annotation, a narrowing guard (`if x is None: ...`), or a cast with a comment** — never a behavior change. If correcting the type reveals an actual logic bug (e.g. a code path that can really receive `None` and would crash), do NOT silently change behavior: flag it in the summary and add a minimal guard that preserves current observable behavior, or STOP if that's impossible.
- Where a third-party stub gap (pandas/yfinance/click) is the cause, prefer a targeted `# type: ignore[<code>]  # <reason>` over contorting the code. Per-module overrides already exist (`pyproject.toml:74-83`) — extend that list for chronically untyped libs instead of sprinkling ignores.
- After each stage: remove the code from `disable_error_code`, fix until clean.

## Per-stage verification gates

```bash
uv run mypy                                       # clean with the newly enabled code(s)
uv run pytest -q                                  # behavior unchanged
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
```

## Done criteria (for the overall plan)

- Stages 1-4 complete: `disable_error_code` contains at most `["type-arg", "no-untyped-call", "no-untyped-def"]`.
- Zero behavior changes: full test suite passes at every stage with no test modifications (any needed test change = red flag, see escape hatches).
- Summary lists every `# type: ignore` added, with reasons.

## Boundaries

- **In scope:** type annotations/guards across `screener/`, `main.py`, `run_pinescript_strategies.py`; `pyproject.toml` mypy section.
- **Out of scope:** tests (mypy-excluded); refactors "while you're in there"; upgrading mypy or adding plugins (e.g. pandera) — note as suggestion only.

## Escape hatches

- A stage measures > ~80 errors → split it by package (fix `screener/backtester/` first, keep a per-module override for the rest) or defer it; report the measurement.
- A fix requires changing a function's runtime behavior or a public signature used by multiple commands → STOP that fix, `# type: ignore` it with a `TODO(plan-005)` comment, and report.
- If `uv run mypy` is not clean at baseline before you start, STOP and report (CI contract broken; this plan assumes a green baseline).

## Maintenance note

Each re-enabled code is a ratchet — CI now enforces it. Reviewers should reject PRs that re-add codes to `disable_error_code` without discussion.
