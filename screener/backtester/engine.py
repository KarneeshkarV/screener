"""Compatibility exports for the backtester split."""

from screener.backtester.core import (
    _resolve_universe,
    simulate_ticker,
)

# _resolve_*_fill live in fills.py; import from the source so the re-export is
# explicit (mypy never implicitly re-exports underscore-prefixed names).
from screener.backtester.fills import _resolve_stop_fill, _resolve_target_fill
from screener.backtester.day_loop import DayLoop, FreedSlot
from screener.backtester.historical import run_backtest, select_candidates
from screener.backtester.rolling import run_rolling_backtest

__all__ = [
    "_resolve_universe",
    "_resolve_stop_fill",
    "_resolve_target_fill",
    "DayLoop",
    "FreedSlot",
    "run_backtest",
    "run_rolling_backtest",
    "select_candidates",
    "simulate_ticker",
]
