use uv 

## Agent output mode

The CLI detects when an agent is driving it and switches to token-lean output
automatically. Verified to auto-engage under Claude Code, codex, opencode, and
pi; a plain terminal is unaffected.

- **What changes**: instead of rich tables (a 3-ticker backtest costs 113 lines
  / 11.5 KB, with every wide column ellipsis-truncated to fit 80 chars), you get
  a bounded digest plus a full-data CSV path. Digest size does not grow with the
  result set.
- **Read the CSV when the digest is not enough.** The digest answers "is this
  strategy any good"; per-trade questions (which ticker lost money, which exits
  were not time-based) need the CSV. Its path is printed on the `trades:` line.
- **Detail levels**: `--agent-detail head` (default: metrics, per-ticker PnL,
  the first 5 trades, and the CSV path), `summary` (drops the sample rows),
  `full` (every row inline, still writes the CSV). Prefer `head`; `full` is 4x
  the bytes and tends to stop agents from reading the CSV they still need,
  because an inline ledger looks complete while omitting the `pnl` column.
- **Overrides**: `--agent` / `--no-agent`, or `SCREENER_AGENT=1` / `=0`.
  Spill directory is `~/tmp`, overridable with `SCREENER_AGENT_DIR`.
- **Explicit output flags always win.** `--csv` still writes the complete CSV to
  stdout; agent mode only governs the default table path. Use it deliberately —
  it is unbounded.
- Agent mode never activates during `pytest`, so the suite behaves the same
  locally and in CI.

## Cursor Cloud specific instructions

- **Python version**: 3.11 (pinned in `.python-version`). `uv` handles this automatically.
- **Package manager**: `uv`. Install all deps (including dev): `uv sync --all-groups`
- **Run the CLI**: `uv run screener <command>` (see `uv run screener --help` for commands)
- **Lint & format**: `uv run ruff check $(git ls-files '*.py')` and `uv run ruff format --check $(git ls-files '*.py')`
- **Type check**: `uv run mypy`
- **Tests**: `uv run pytest` (1,400+ tests, all offline using stubs; coverage is behavior-oriented with a 90% floor, not a target — see `pyproject.toml`)
- **Task runner**: `just` (see `justfile` for available recipes; uses `.venv/bin/python`)
- The `--log-level` and `--config` options are global and must be placed *before* the subcommand (e.g. `uv run screener --log-level ERROR screen ...`)
- Optional env vars for extended features: `FMP_API_KEY`, `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`. The core screener and backtester work without these.
- yfinance creates a cache folder at `~/.cache/py-yfinance`; a harmless "Error creating TzCache" warning may appear on first run — it can be ignored.
