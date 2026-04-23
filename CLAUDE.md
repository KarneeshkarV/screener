# Claude notes

Package manager: `uv`. Run commands via `just <recipe>` (see `justfile`) or
`uv run <command>`.

## Layout

- `screener/` — library package.
  - `cli.py` — Click entrypoint (`screener = "screener.cli:cli"`).
  - `scanner.py`, `criteria.py`, `display.py`, `history.py` — live TradingView
    screener.
  - `indicators.py` — single source of truth for SMA/EMA/RSI/ATR/crossovers.
    Both the Pine AST evaluator and the standalone strategy runner delegate
    here.
  - `backtester/` — historical backtest engine.
- `run_pinescript_strategies.py` — standalone runner for a catalogue of Pine
  strategy ports. Uses `screener.indicators`.
- `tests/` — pytest suite; offline via `StubPriceFetcher` in `conftest.py`.

## Conventions

- **No live TradingView scan as the as-of universe for a historical
  backtest.** The current top-N-by-volume list injects survivorship bias
  (delisted / halted / acquired tickers as of today are silently excluded).
  Either require `--universe-file` or hard-fail. See
  `screener/backtester/engine.py::_resolve_universe` for the guard.
- The backtest's equity curve is reconstructed by `build_equity_curve` from
  the trade ledger. `Portfolio._cash` is load-bearing during sizing but is
  not the source of truth for reported metrics.
- Indicators live in `screener/indicators.py`. When adding a new primitive,
  add it there and delegate from both call sites. Keep Pine semantics where
  they apply (Wilder's RMA for RSI/ATR, α = 2/(n+1) for EMA).

## Testing

```
uv run pytest -q
```

Tests are offline: `tests/conftest.py::StubPriceFetcher` feeds synthetic
OHLCV frames via `click.get_current_context().obj` so `main.py::backtest`
never hits yfinance. Prefer the stub over mocking.
