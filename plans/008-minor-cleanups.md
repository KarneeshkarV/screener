# Plan 008 — Minor cleanups (rename CLUADE.md, gitignore output dirs, optional [india] extra)

- **Status:** TODO
- **Written against commit:** `9547d4d`
- **Category:** dx / docs / dependencies
- **Effort:** S · **Risk of fix:** low (step 3 medium — it changes install behavior)
- **Depends on:** nothing

## Step 1 — fix the `CLUADE.md` typo

Repo root contains `CLUADE.md` (misspelled; verified contents: "use uv" + pointer to the bot repo at `/home/karneeshkar/Desktop/personal/screener_main/screener_bot/`). Claude Code only auto-loads the correctly-spelled name, so these instructions are currently dead weight.

- `git mv CLUADE.md CLAUDE.md`
- Check nothing references the old name: `grep -rn "CLUADE" --include="*.md" --include="*.toml" --include="*.yml" .` → expect zero hits afterwards.
- `AGENTS.md` already duplicates the "use uv" + bot-path facts; if the renamed file is fully redundant with `AGENTS.md`, prefer deleting it and keeping `AGENTS.md` as the single source — state which you did and why in the summary.

## Step 2 — gitignore the local output directories

`momentum_out/`, `oi_lab/`, `sweep_results/`, `reports/` exist at repo root, are **untracked** (verified via `git ls-files`), and only `*.log`/`*.html` patterns inside them are ignored. One careless `git add .` commits run artifacts.

Append to `.gitignore` (keep the file's existing comment style — it has sections like `# Generated local reports`):

```
momentum_out/
oi_lab/
sweep_results/
reports/
```

Verify: `git status --short` no longer lists those directories, and `git check-ignore -v momentum_out reports` resolves to the new lines.

## Step 3 — move india-scraper deps to an optional extra (CONFIRM WITH USER FIRST)

`jugaad-data` and `openscreener` are hard dependencies (`pyproject.toml:7-23`) but every import site is already lazy/guarded (`unusual_volume/delivery.py`, `operator/fetch.py`, `earnings_backtest/data.py`, `enrich.py:97-108`, `insiders.py`, `garp.py`). The `vectorbt` extra at `pyproject.toml:25-28` is the existing pattern. Moving them to an `india` extra trims the default venv but **breaks `uv sync` users who rely on india workflows without knowing about extras**.

This step changes install UX — it requires explicit user sign-off. If executing this plan autonomously without that sign-off, SKIP step 3 and say so in the summary.

If approved:

1. In `pyproject.toml`, move `jugaad-data>=0.33.1` and `openscreener` from `[project] dependencies` to a new `[project.optional-dependencies] india = [...]` group (keep `vectorbt` group untouched). Run `uv lock`.
2. Audit every import site (grep `jugaad\|openscreener` in `screener/`): each must catch `ImportError` and emit an actionable message naming the fix, e.g. `"jugaad-data not installed; run: uv sync --extra india"`. Add the message where missing; do not change any other behavior.
3. README: add one line under the install/run section documenting `uv sync --all-groups --extra india`.
4. CI (`.github/workflows/ci.yml`) runs `uv sync --all-groups` — tests that stub these libs may still import them. Run `uv run pytest -q` in an environment **without** the extra to prove the suite passes; if tests hard-import either lib, add the extra to CI's sync command instead of weakening tests, and note it.

## Verification gates

```bash
uv run pytest -q
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
uv run mypy
git status --short          # no untracked output dirs listed
```

## Boundaries

- **In scope:** `CLUADE.md`/`CLAUDE.md`, `.gitignore`, `pyproject.toml`, `uv.lock` (via `uv lock` only), README install section, ImportError messages at the six-ish lazy-import sites.
- **Out of scope:** deleting the contents of the output directories (user data — never delete); any other dependency changes; restructuring AGENTS.md.

## Escape hatches

- If `tests/` import `jugaad_data`/`openscreener` unconditionally in collection (not inside stubs), step 3's cost jumps — STOP step 3 and report rather than restructuring tests.
