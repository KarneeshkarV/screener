"""Trend-following Pine strategy ports.

Implementations live in ``screener.strategies.plugins``. This module re-exports
them for callers that imported from ``pine_ports``.
"""

from __future__ import annotations

from screener.strategies.plugins.ma_cross import strat_ma_cross
from screener.strategies.plugins.ma_cross_regime import strat_ma_cross_regime
from screener.strategies.plugins.ma_cross_st_entry import strat_ma_cross_st_entry
from screener.strategies.plugins.ma_cross_st_exit import strat_ma_cross_st_exit
from screener.strategies.plugins.supertrend import strat_supertrend
from screener.strategies.plugins.supertrend_rsi import strat_supertrend_rsi

__all__ = [
    "strat_ma_cross",
    "strat_ma_cross_regime",
    "strat_ma_cross_st_entry",
    "strat_ma_cross_st_exit",
    "strat_supertrend",
    "strat_supertrend_rsi",
]
