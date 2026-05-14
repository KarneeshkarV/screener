"""Historical backtester with Pine-like expression support.

Engine choice:
  * Core event engine: use for production-accuracy runs and any feature with
    path-dependent fills, including half-spread / volume-impact / composite
    slippage, gap-aware stop/target fills, limit or MOC entries, re-entry, and
    partial exits. This is the canonical correctness path.
  * vectorbt fast path: use for quick long-only signal experiments where the
    requested features overlap with ``Portfolio.from_signals``: boolean Pine
    entries/exits, hold caps, fixed-bps slippage, commissions, and simple
    stop/target/trailing exits. Unsupported coverage should fall back to core.
  * Monte Carlo: run after either engine because it consumes the normalized
    ``BacktestResult`` trade ledger and equity curve.
"""

from screener.backtester.historical import run_backtest
from screener.backtester.monte_carlo import MonteCarloResult, run_monte_carlo
from screener.backtester.models import (
    BacktestConfig,
    BacktestResult,
    Position,
    Trade,
)
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.vbt_adapter import run_vbt

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "MonteCarloResult",
    "Position",
    "Trade",
    "run_backtest",
    "run_monte_carlo",
    "run_rolling_backtest",
    "run_vbt",
]
