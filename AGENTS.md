use uv 
the bot code is in /home/karneeshkar/Desktop/personal/screener_bot

## Cursor Cloud specific instructions

- **Python version**: 3.11 (pinned in `.python-version`). `uv` handles this automatically.
- **Package manager**: `uv`. Install all deps (including dev): `uv sync --all-groups`
- **Run the CLI**: `uv run screener <command>` (see `uv run screener --help` for commands)
- **Lint & format**: `uv run ruff check $(git ls-files '*.py')` and `uv run ruff format --check $(git ls-files '*.py')`
- **Type check**: `uv run mypy`
- **Tests**: `uv run pytest` (177 tests, all offline using stubs)
- **Task runner**: `just` (see `justfile` for available recipes; uses `.venv/bin/python`)
- The `--log-level` and `--config` options are global and must be placed *before* the subcommand (e.g. `uv run screener --log-level ERROR screen ...`)
- Optional env vars for extended features: `FMP_API_KEY`, `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`. The core screener and backtester work without these.
- yfinance creates a cache folder at `~/.cache/py-yfinance`; a harmless "Error creating TzCache" warning may appear on first run — it can be ignored.
