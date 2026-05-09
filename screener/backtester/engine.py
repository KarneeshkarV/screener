"""Compatibility exports for the backtester split."""

from screener.backtester.core import (
    _resolve_stop_fill,
    _resolve_target_fill,
    _resolve_universe,
    simulate_ticker,
)
from screener.backtester.historical import run_backtest, select_candidates
from screener.backtester.rolling import run_rolling_backtest

__all__ = [
    "_resolve_universe",
    "_resolve_stop_fill",
    "_resolve_target_fill",
    "run_backtest",
    "run_rolling_backtest",
    "select_candidates",
    "simulate_ticker",
]
