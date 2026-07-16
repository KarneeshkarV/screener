---
name: screener-stock-analysis-codebase
description: Use when analyzing stocks, portfolios, screens, backtests, or strategy ideas with this workspace. Covers how to use the Python screener CLI, Telegram bot support code, and Rust migration project together without inventing data or bypassing existing providers.
---

# Screener Stock Analysis Codebase

Use this skill for codebase-backed stock analysis, portfolio reviews, signal checks, strategy research, and comparisons between the Python and Rust implementations in this workspace.

## Workspace Map

- `screener/`: primary Python CLI and research code. Use `uv` from this directory.
- `screener_bot/`: Telegram bot that wraps the Python `screener` package for portfolio checks, alerts, charts, and scheduled screen diffs.
- `screener-rs/`: Rust migration and parity/performance implementation. Use Cargo from this directory. **Currently not present in this workspace** — skip all Rust/parity guidance below unless the directory exists.

If repo guidance conflicts, follow `screener/AGENTS.md`: use `uv`; bot code lives in `../screener_bot/`. The root instruction references `RTK.md`, but this workspace currently does not contain that file.

## First Choice Tooling

Use the Python CLI for complete stock-analysis workflows because it has the richest feature surface and tests:

```bash
cd screener
uv run screener --help
uv run screener screen -m us -c ema -n 30
uv run screener screen -m india -c breakout -n 30 --detail
uv run screener rs-breakout -m india -n 50
uv run screener unusual-volume -m us --tickers AAPL,MSFT
uv run screener promoter-buys -m india --min-change 0.5
uv run screener garp -m us --universe-size 300 --workers 8
uv run screener backtest-rolling -m us --years 2 --strategy rs_breakout --top 10
uv run screener backtest-rolling -m us --years 2 --strategy rs_breakout --top 10 --sizing atr_risk --sizing-risk-pct 0.01
uv run screener backtest-rolling -m us --tickers AAPL,MSFT --start 2026-06-22 --end 2026-07-02 --interval 15m --entry "close > sma(close,5)" --hold 4
uv run screener conviction AAPL -m us
uv run screener research-report -m us --years 1 --strategy rs_breakout --top 10
```

Use Rust when the task is parity, speed, CLI migration, or checking behavior against the migration target (only if `screener-rs/` exists in the workspace):

```bash
cd screener-rs
cargo run -- screen -m us -c ema -n 30
cargo run -- rs-breakout -m india -n 50
cargo run -- backtest-rolling -m us --years 2 --strategy rs_breakout --top 10
cargo test
```

Use the bot project when the task involves Telegram command behavior, portfolio alerting, scheduled screener messages, chart rendering, authorization, or Turso-backed portfolio state:

```bash
cd screener_bot
uv run python -m screener_bot
uv run pytest
```

## Data Source Rules

- Prefer existing repo providers and caches before writing ad hoc network code.
- Use TradingView scanner data for broad technical universes in `screen`.
- Use yfinance-backed OHLCV through `screener.backtester.data.build_price_fetcher()` for time-series analysis.
- Intraday bars: `backtest-rolling` / `backtest-historical` accept `--interval` (`1d` default, `1h`, `30m`, `15m`, `5m`, `1m`); `build_price_fetcher(interval=...)` threads it through. yfinance caps history (1m ~30d, 5m-30m ~60d, 1h ~730d; no chunking yet), FMP serves intraday via `historical-chart` when `FMP_API_KEY` is set (raw, unadjusted bars). All intraday timestamps are canonical naive UTC across providers; intraday caches are namespaced per interval (`AAPL__15m`, `fmp_AAPL__15m`) so daily parquet files are never polluted. Metrics annualize by bars-per-year for the interval. Long-warmup strategies (e.g. SMA200) usually cannot fill their lookback inside the capped intraday windows.
- Use FMP only when `FMP_API_KEY` is present, mostly for US insider/fundamental/event context (plus intraday/daily price fallback).
- US SEC filings: list recent filings (10-K/10-Q/8-K) and read a filed 10-K/10-Q by section via FMP (`filings` command / `screener/filings.py`); needs `FMP_API_KEY`.
- India EPS surprise for PEAD backtests comes from FMP's historical earnings calendar (needs `FMP_API_KEY`): real NSE announcement dates are preferred and enriched with FMP surprise; quarters NSE lacks fall back to FMP's own point-in-time dates. The default India earnings path (NSE + openscreener with filing-lag floor) is unchanged.
- Use screener.in / openscreener for Indian fundamentals and promoter/shareholding context.
- Use NSE cash/F&O bhavcopy and option-chain helpers for India delivery, operator intent, and unusual-volume overlays.
- When giving current stock advice, verify the latest available data timestamp and state it. Do not invent fundamentals, analyst targets, earnings dates, or promoter changes.
- Treat stale, missing, or conflicting data as an analysis finding, not a reason to fill gaps from memory.

## Python Analysis Paths

Use these modules instead of recreating logic:

- Technical screen: `screener/screener/commands/screen.py`, `screener/screener/scanner.py`, `screener/screener/criteria/plugins/`.
- Custom criteria: add a plugin in `screener/screener/criteria/plugins/` with `@criterion("name")`; use `pipeline=True` only when the scan needs enrichment/history/external providers.
- Backtests: `screener/screener/backtester/historical.py`, `rolling.py`, `core.py`, `models.py`, `metrics.py`.
- Position sizing: `screener/screener/backtester/sizing.py` (`@sizer` registry: `equal_slot`, `fixed_fraction`, `fixed_risk`, `atr_risk`, `inverse_vol`); exposed as `--sizing`/`--sizing-*` on both backtest commands. Default `equal_slot` is bit-identical to the legacy engine; other rules clamp to the slot budget and read only up to the signal bar.
- Price data: `screener/screener/backtester/data.py`; use `tv_to_yf()` for symbol mapping and injected `PriceFetcher` for tests. Interval-aware: pass `interval=` to the fetcher constructors, never mix intervals in one cache key.
- Pine-like expressions: `screener/screener/backtester/pine.py`.
- Named strategies: `screener/screener/strategies/plugins/` with `@strategy(...)`; expressions flow through `screener/strategies/expressions.py`.
- GARP: `screener/screener/garp.py` and `screener/screener/commands/garp.py`.
- Promoter/insider buys: `screener/screener/insiders.py` and `screener/screener/commands/insiders.py`.
- Relative-strength breakout: `screener/screener/rs_breakout.py` and `screener/screener/commands/rs_breakout.py`.
- Unusual volume: `screener/screener/unusual_volume/`.
- Operator scan: `screener/screener/operator/`.
- Optimization: `screener/screener/backtester/optimization/`.
- Earnings-event backtests: `screener/screener/earnings_backtest/` (`earnings_dates.py` for providers/surprise, `pead.py` for PEAD simulation, `cli.py` for `earnings-backtest` / `earnings-pead`).
- Options data layer: `screener/screener/options/` (NSE/CBOE/yfinance chains, panels, greeks, PCR/IV/max-pain, point-in-time backtest fields) plus options screen criteria (`unusual_options`, `bullish_oi_buildup`, `high_iv_rank`, `low_iv_rank`, `cheap_earnings_vol`); see `screener/docs/options.md`.
- Conviction card: `screener/screener/conviction.py` and `screener/screener/commands/conviction.py` (composite pillar score, PIT-aware pillar skipping).
- Seasonality: `screener/screener/seasonality.py`; index-inclusion event study: `screener/screener/index_inclusion.py`; US institutional (13F): `screener/screener/institutional.py`.
- US SEC filings reader: `screener/screener/filings.py` and `screener/screener/commands/filings.py` (FMP filings index + 10-K/10-Q section JSON, needs `FMP_API_KEY`).
- Factor research: `screener/screener/backtester/factor_tearsheet.py` (`factor-tearsheet`), `vbt_sweep.py` (`vbt-sweep`); one-command pipeline: `research-report` (grid → walk-forward → Monte Carlo) in `screener/screener/backtester/optimization/`.
- Screen-run history: `screener/screener/history.py` (SQLite `~/.screener/history.db`), replay via `backtest-historical --from-run`, Turso mirror via `screener/screener/history_sync.py` (`history-backup`).

For quick per-symbol technical detail, it is often easier to import bot logic:

```python
from screener_bot.technical import TechnicalService
```

but remember `screener_bot` normally depends on bot config and portfolio objects, so CLI/import scripts in `screener/` are cleaner for one-off research.

## Rust Analysis Paths

(Only applicable when `screener-rs/` is present in the workspace.) Rust mirrors many Python concepts but is not just a wrapper:

- CLI entrypoint and command wiring: `screener-rs/src/main.rs`.
- Backtest engine and rolling/historical simulation: `screener-rs/src/backtester/engine.rs`.
- Data models: `screener-rs/src/backtester/models.rs`, `screener-rs/src/data.rs`.
- Pine expression evaluator: `screener-rs/src/pine.rs`.
- Providers: `screener-rs/src/providers/` for Yahoo, TradingView, NSE, screener.in, cache, resilience, fundamentals.
- Screeners: `screener-rs/src/screeners/`.
- Criteria parity files: `screener-rs/src/screeners/criteria/`.
- Parity tests: `screener-rs/tests/parity.rs`, `parity_test.rs`.

Rust config intentionally supports YAML strategy and criteria aliases through `strategies:` and `criteria:`. JSON config support is intentionally not part of the migration.

## Recommended Stock Analysis Workflow

1. Normalize the input:
   - US symbols: `AAPL`, `MSFT`, `SPY`.
   - India symbols: prefer `NSE:RELIANCE` or plain `RELIANCE` with `market=india`; `tv_to_yf()` maps to `.NS`.
   - Portfolio entries may include entry price, market value, timeframe, and risk style.
2. Pull current/recent OHLCV through repo providers and record the last bar date.
3. Run a technical pass:
   - Trend: EMA/SMA 20/50/200 alignment.
   - Momentum: RSI, price change, relative strength versus `SPY` for US or `^NSEI` for India.
   - Risk: ATR, recent swing lows/highs, gap risk, volume/liquidity.
   - Confirmation: breakout, unusual volume, delivery, SuperTrend, operator labels when relevant.
4. Run a fundamental/context pass:
   - GARP / quality / valuation when available.
   - Promoter/insider/ownership changes where supported.
   - Earnings or event risk if discoverable from repo providers or verified external sources.
5. Use backtests for strategy claims:
   - Prefer `backtest-rolling` for live-like selection behavior.
   - Use `backtest-historical` for point-in-time candidate checks.
   - Use `vbt-sweep` for fast triage only; validate promising ideas with rolling backtests.
   - Use `earnings-backtest` / `earnings-pead` for earnings-event claims. `earnings-pead --exit-mode fixed` (default) holds `--hold-days` sessions; `--exit-mode dynamic` holds a beat (`--min-surprise`) until a later report misses, tagging trades with `exit_reason`.
   - For sizing-sensitive claims, state the `--sizing` rule (default `equal_slot`); risk-based rules (`fixed_risk`, `atr_risk`, `inverse_vol`) change per-trade exposure and drawdown, not just returns.
   - Include slippage, commission, liquidity filters, benchmark, and clear start/end dates.
6. Convert evidence into action levels only after the above:
   - Close-based stop.
   - Hard invalidation stop.
   - TP1/TP2 or trim zones.
   - Add/reclaim level if setup is not yet confirmed.

## Best Practices And Pitfalls

- Do not bypass the registries. Add criteria under `criteria/plugins/` and strategies under `strategies/plugins/`.
- Keep data loading separate from signal math so tests can use stub price fetchers.
- Avoid lookahead. For volume averages and rolling highs/lows, use prior-bar baselines when the code already does so.
- Prefer CSV/JSON/Markdown output flags for repeatable analysis artifacts.
- Use `--refresh` only when cache freshness matters; otherwise respect cache TTLs.
- Use global Click options before subcommands: `uv run screener --log-level ERROR screen ...`.
- For India analysis, distinguish NSE cash delivery, F&O OI, promoter shareholding, and yfinance data; they answer different questions.
- For US insider buys, FMP Form 4 data is better when configured; yfinance is fallback context.
- For current financial facts, browse or use live providers when there is any chance the answer changed recently.
- Always state uncertainty: stale prices, missing FMP key, unavailable promoter table, failed provider, or thin liquidity.

## Validation

Python:

```bash
cd screener
uv run pytest
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
uv run mypy
```

Bot:

```bash
cd screener_bot
uv run pytest
uv run ruff check $(git ls-files '*.py')
uv run mypy
```

Rust:

```bash
cd screener-rs
cargo test
cargo fmt --check
cargo clippy --all-targets --all-features
```

For parity-sensitive changes, run both Python and Rust on the same explicit tickers, dates, strategy, and offline CSV if possible.

## Output Standard

For analysis responses, include:

- Data date/time and sources.
- Market, ticker normalization, and commands or modules used.
- A concise table of signals or holdings.
- Short reasoning for each symbol.
- Risk levels and invalidation logic when asked for trading decisions.
- A clear note when the output is research, not financial advice.

Do not present a screen or backtest as a recommendation without explaining assumptions, data freshness, and the main failure modes.
