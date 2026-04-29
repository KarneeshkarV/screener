# Backtest Validation Prompt

Validate that the backtester in `screener/backtester/` produces accurate results by comparing it against independently verifiable ground truth.

Write a **standalone runnable script** (not pytest — something I can `python validate_backtest.py` and read the output of) at the repo root that:

1. **Runs buy-and-hold benchmarks through our engine** for a handful of liquid tickers over a fixed historical window (e.g. AAPL, MSFT, SPY, QQQ from 2015-01-01 to 2024-12-31). Implement "buy and hold" as the simplest possible strategy in our `Strategy` framework — buy on day 1, never sell.
2. **Computes ground truth independently** for the same window using raw price data (e.g. yfinance / the same data source the engine pulls from, but called directly without going through the engine). Compute total return, CAGR, max drawdown, and Sharpe directly from the price series.
3. **Compares the two** side-by-side in a printed table: engine result, ground-truth result, absolute diff, % diff. Flag any metric that diverges by more than a small tolerance (e.g. 0.5% on returns, 1% on Sharpe).
4. **Adds at least one indicator-based sanity check** — pick a well-known strategy with a publicly known result (e.g. SPY 200-day SMA crossover, or golden/death cross on AAPL) and compare our engine's output against a hand-rolled numpy/pandas implementation of the same rule on the same data.

Before writing code, **read these files** to understand the engine's API and conventions: `screener/backtester/engine.py`, `screener/backtester/strategies.py`, `screener/backtester/metrics.py`, `screener/backtester/data.py`, and `tests/test_engine.py` (for examples of how strategies are wired up).

Report findings as a short summary at the end: which metrics matched, which diverged, and a hypothesis for any divergence (slippage model? fee model? off-by-one on entry day? dividend handling?).
