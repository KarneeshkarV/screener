set positional-arguments

python := ".venv/bin/python"
screener := ".venv/bin/screener"

# List available recipes.
default:
    @just --list

# Show top-level CLI help.
help:
    @{{screener}} --help

# Show screen command help.
help-screen:
    @{{screener}} screen --help

# Show historical backtest command help.
help-backtest:
    @{{screener}} backtest-historical --help

# Show rolling backtest command help.
help-backtest-rolling:
    @{{screener}} backtest-rolling --help

# Show GARP command help.
help-garp:
    @{{screener}} garp --help

# Show promoter/insider buys command help.
help-promoter-buys:
    @{{screener}} promoter-buys --help

# Show RS breakout command help.
help-rs-breakout:
    @{{screener}} rs-breakout --help

# Show operator scan command help.
help-operator-scan:
    @{{screener}} operator-scan --help

# Show optimize command help.
help-optimize:
    @{{screener}} optimize --help

# Show cache command help.
help-cache:
    @{{screener}} cache --help

# Show conviction command help.
help-conviction:
    @{{screener}} conviction --help

# Show earnings backtest command help.
help-earnings-backtest:
    @{{screener}} earnings-backtest --help

# Show earnings PEAD command help.
help-earnings-pead:
    @{{screener}} earnings-pead --help

# Show factor tearsheet command help.
help-factor-tearsheet:
    @{{screener}} factor-tearsheet --help

# Show history command help.
help-history:
    @{{screener}} history --help

# Show history-backup command help.
help-history-backup:
    @{{screener}} history-backup --help

# Show index inclusion command help.
help-index-inclusion:
    @{{screener}} index-inclusion --help

# Show filings reader command help.
help-filings:
    @{{screener}} filings --help

# Show institutional ownership command help.
help-institutional:
    @{{screener}} institutional --help

# Show options data command help.
help-options:
    @{{screener}} options --help

# Show research report command help.
help-research-report:
    @{{screener}} optimize research-report --help

# Show seasonality command help.
help-seasonality:
    @{{screener}} seasonality --help

# Show standalone Pine strategy runner help.
help-pine:
    @{{python}} -m screener.research.pine_runner --help

# Run the screener. Example: just screen -m us -n 20 --csv
screen *args:
    @{{screener}} screen "$@"

# Run the US screener. Example: just screen-us -n 20 --detail
screen-us *args:
    @{{screener}} screen -m us "$@"

# Run the India screener. Example: just screen-india -n 20 --csv
screen-india *args:
    @{{screener}} screen -m india "$@"

# Run historical backtesting. Requires --as-of plus --entry/--strategy and a universe.
backtest *args:
    @{{screener}} backtest-historical "$@"

# Run a true daily rolling backtest over a date window.
backtest-rolling *args:
    @{{screener}} backtest-rolling "$@"

# Live US historical backtest smoke run.
backtest-smoke-us:
    @{{screener}} backtest-historical -m us --as-of 2026-03-20 --entry "close > 0" --exit false --tickers AAPL,MSFT,NVDA,AMD --hold 5 --top 2 --stop-loss 0.05 --take-profit 0.08 --trailing-stop 0.04

# Live India historical backtest smoke run.
backtest-smoke-india:
    @{{screener}} backtest-historical -m india --as-of 2026-03-20 --entry "close > 0" --exit false --tickers RELIANCE,TCS,INFY,HDFCBANK --hold 5 --top 2 --min-price 0 --min-avg-dollar-volume 0

# Run standalone Pine strategy backtests. Example: just pine --market us --years 3 --limit 50
pine *args:
    @{{python}} -m screener.research.pine_runner "$@"

# Run standalone Pine strategy backtests for the US market.
pine-us *args:
    @{{python}} -m screener.research.pine_runner --market us "$@"

# Run standalone Pine strategy backtests for the India market.
pine-india *args:
    @{{python}} -m screener.research.pine_runner --market india "$@"

# Detect unusual-volume events. Example: just unusual-volume -m us --tickers AAPL,MSFT
unusual-volume *args:
    @{{screener}} unusual-volume "$@"

# Find GARP stocks using market-specific fundamental data.
garp *args:
    @{{screener}} garp "$@"

# Find stocks where promoter/insider holding has increased.
promoter-buys *args:
    @{{screener}} promoter-buys "$@"

# Screen stocks for RS + SuperTrend + breakout/volume setups.
rs-breakout *args:
    @{{screener}} rs-breakout "$@"

# Run the NSE Operator Intent screener.
operator-scan *args:
    @{{screener}} operator-scan "$@"

# Optimize and validate backtest parameters. Example: just optimize grid --help
optimize *args:
    @{{screener}} optimize "$@"

# Show successful feature usage counts from Turso.
usage-report:
    @{{screener}} usage-report

# Inspect and prune the screener's on-disk caches. Example: just cache status
cache *args:
    @{{screener}} cache "$@"

# One composite conviction card for TICKER, fusing the screen pillars.
conviction *args:
    @{{screener}} conviction "$@"

# Backtest earnings-drift entry (E-1/E-2 -> E) with sentiment filters.
earnings-backtest *args:
    @{{screener}} earnings-backtest "$@"

# Backtest post-earnings-announcement drift (next open -> hold N days).
earnings-pead *args:
    @{{screener}} earnings-pead "$@"

# Compute factor IC and quantile tearsheet for a named strategy.
factor-tearsheet *args:
    @{{screener}} factor-tearsheet "$@"

# List persisted screen runs (replay with `backtest-historical --from-run`).
history *args:
    @{{screener}} history "$@"

# Back up screen-run history to Turso (or restore with --restore).
history-backup *args:
    @{{screener}} history-backup "$@"

# Event study of post-addition excess drift for S&P 500 additions vs SPY.
index-inclusion *args:
    @{{screener}} index-inclusion "$@"

# Read US SEC filings (list recent filings, or a 10-K/10-Q by section).
filings *args:
    @{{screener}} filings "$@"

# Show FMP institutional ownership per ticker, ranked by QoQ change.
institutional *args:
    @{{screener}} institutional "$@"

# Build, snapshot, and inspect normalized options data. Example: just options snapshot --help
options *args:
    @{{screener}} options "$@"

# Deep multi-year India options panel backfill. Example: just options-backfill 2020-01-01 2024-12-31
options-backfill start end:
    @{{python}} main.py options build-panel -m india --start {{start}} --end {{end}}

# Backfill India participant OI + VIX regime context for a range. Example: just options-backfill-context 2020-01-01 2024-12-31
options-backfill-context start end:
    @{{python}} main.py options participants --start {{start}} --end {{end}}
    @{{python}} main.py options regime -m india --start {{start}} --end {{end}}

# One-command research report: grid -> walk-forward -> Monte Carlo.
research-report *args:
    @{{screener}} optimize research-report "$@"

# Show monthly, turn-of-month and day-of-week seasonality for TICKER.
seasonality *args:
    @{{screener}} seasonality "$@"

# Show unusual-volume command help.
help-unusual-volume:
    @{{screener}} unusual-volume --help

# Compile Python files without running tests.
compile:
    @{{python}} -m compileall screener

# Run the test suite with coverage, matching CI.
test *args:
    uv run pytest --cov --cov-report=term-missing:skip-covered "$@"

# Run ruff lint over tracked Python files, matching CI.
lint:
    uv run ruff check $(git ls-files '*.py')

# Check formatting over tracked Python files, matching CI.
format-check:
    uv run ruff format --check $(git ls-files '*.py')

# Run strict mypy type checking, matching CI.
typecheck:
    uv run mypy

# Run every CI gate locally: tests, lint, format check, types.
ci: test lint format-check typecheck

# Re-generate the .codex and .opencode agent copies from the canonical
# .claude sources. `.claude/` is the single source of truth for agent
# skills/commands; never hand-edit the generated .codex/.opencode copies.
# Idempotent: running `just sync-skills` twice produces no diff.
sync-skills:
    #!/usr/bin/env python3
    import pathlib

    # `.claude/` is canonical. Codex skills use `name:` + `description:`
    # frontmatter (no argument-hint/allowed-tools); opencode commands keep
    # only `description:`. `$ARGUMENTS` is Claude/opencode command syntax,
    # so the codex skill describes its input in prose instead.
    CODEX_TECHOFUNDO_DESCRIPTION = (
        "Use when the user invokes /use techofundo or asks for technofundamental "
        "stock or portfolio analysis with stop-loss and take-profit levels. Inputs "
        "may be plain tickers, comma-separated tickers, or portfolio entries like "
        "SYMBOL:ENTRY:MARKET_VALUE. The output should combine technical and "
        "fundamental evidence from the repo/providers and explain the logic behind "
        "stop-loss and take-profit levels."
    )
    CODEX_TECHOFUNDO_INPUT = (
        "Analyze the stock or portfolio input the user supplies (plain tickers, "
        "comma-separated tickers, or `SYMBOL:ENTRY:MARKET_VALUE`)."
    )

    def write(path, text):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    # 1. screener-stock-analysis-codebase: codex uses identical name+description
    #    frontmatter, so the skill is a verbatim copy of the canonical version.
    skill = pathlib.Path(
        ".claude/skills/screener-stock-analysis-codebase/SKILL.md"
    ).read_text()
    write(".codex/skills/screener-stock-analysis-codebase/SKILL.md", skill)

    # 2. techofundo: canonical command is .claude/commands/techofundo.md.
    cmd = pathlib.Path(".claude/commands/techofundo.md").read_text()
    _, frontmatter, body = cmd.split("---\n", 2)

    # opencode command: drop argument-hint/allowed-tools, keep description + body.
    opencode_fm = [
        line
        for line in frontmatter.splitlines(keepends=True)
        if not line.startswith(("argument-hint:", "allowed-tools:"))
    ]
    write(
        ".opencode/commands/techofundo.md",
        "---\n" + "".join(opencode_fm) + "---\n" + body,
    )

    # codex skill: name+description frontmatter, prose input instead of $ARGUMENTS.
    codex_body = body.replace("# /techofundo", "# Techofundo").replace(
        "Input portfolio or stock list: `$ARGUMENTS`",
        CODEX_TECHOFUNDO_INPUT,
    )
    write(
        ".codex/skills/techofundo/SKILL.md",
        "---\nname: techofundo\ndescription: "
        + CODEX_TECHOFUNDO_DESCRIPTION
        + "\n---\n"
        + codex_body,
    )
    print("Synced .codex and .opencode copies from .claude sources.")
