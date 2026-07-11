"""Public backtester models and engine entry points.

Engine modules are imported only when their entry point is called. This keeps
lightweight imports such as ``screener.backtester.data`` from initializing the
CLI/market graph and avoids package-import cycles.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from screener.backtester.models import (
    BacktestConfig,
    BacktestResult,
    Position,
    Trade,
)

if TYPE_CHECKING:
    from screener.backtester.data import PriceFetcher


def run_backtest(cfg: BacktestConfig, fetcher: PriceFetcher) -> BacktestResult:
    from screener.backtester.historical import run_backtest as run

    return run(cfg, fetcher)


def run_rolling_backtest(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    *,
    start_date: date,
    end_date: date,
) -> BacktestResult:
    from screener.backtester.rolling_simulation import run_rolling_backtest as run

    return run(cfg, fetcher, start_date=start_date, end_date=end_date)


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Position",
    "Trade",
    "run_backtest",
    "run_rolling_backtest",
]
