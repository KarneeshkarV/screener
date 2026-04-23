# screener

Stock screener and historical backtester for the US and Indian markets.

- **`screener screen`** — live TradingView scan with named criteria (EMA
  stack, 52-week breakout, value, quality, …) and a local composite
  `setup_score` for ranking.
- **`screener backtest-historical`** — Pine-like entry/exit expressions run
  over an explicit as-of universe, with slippage models, gap-aware fills,
  partial exits, re-entry, and multiple entry-order types.
- **`run_pinescript_strategies.py`** — batch-run a catalogue of ported Pine
  strategies (supertrend, MACD+RSI, MA cross, Bollinger breakout, …)
  against a supplied universe.

## Install

```bash
uv sync
```

Python ≥ 3.11.

## Quickstart

```bash
# Live US screen, composite score ranking
just screen-us -n 20

# Live India screen with fundamental detail columns
just screen-india -n 20 --detail

# Historical backtest smoke run (US)
just backtest-smoke-us

# Run the Pine-strategy catalogue against a custom universe
just pine --market us --years 3 --universe-file my_us_universe.txt
```

A universe file is a plain newline-separated list of TradingView-style
tickers:

```
AAPL
NASDAQ:NVDA
NSE:RELIANCE
# lines starting with # are ignored
```

The backtester will **not** fall back to a live TradingView scan for the
historical universe. A current top-N-by-volume list injects survivorship
bias into historical results; you must supply the universe explicitly via
`--tickers` or `--universe-file`.

## Testing

```bash
uv run pytest -q
```

Tests run offline via a stub price fetcher injected through Click's
context object — no yfinance calls.

## Layout

| Path | Purpose |
| --- | --- |
| `screener/cli.py` | Click entrypoint |
| `screener/scanner.py`, `criteria.py`, `display.py`, `history.py` | Live screener |
| `screener/indicators.py` | Shared SMA/EMA/RSI/ATR/crossover primitives |
| `screener/backtester/` | Historical backtest engine |
| `run_pinescript_strategies.py` | Standalone Pine strategy runner |
| `tests/` | Pytest suite |
