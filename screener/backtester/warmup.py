"""Warmup-window arithmetic shared by backtest engines."""

from __future__ import annotations

import numpy as np

from screener.backtester.metrics import periods_per_year_for_interval


def _warmup_days_for_interval(lookback: int, interval: str) -> int:
    """Return calendar days for the shared ``3 * lookback + 30`` warmup."""
    warmup_bars = lookback * 3 + 30
    if interval == "1d":
        return max(warmup_bars, 365)
    bars_per_day = max(periods_per_year_for_interval(interval) // 252, 1)
    return int(np.ceil(warmup_bars / bars_per_day) * 1.6) + 5
