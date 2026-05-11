"""Supertrend direction (matches Pine ``ta.supertrend`` semantics)."""

from __future__ import annotations

import numpy as np

from screener.indicators.plugins.atr import atr
from screener.indicators.registry import indicator


@indicator("supertrend_dir")
def supertrend_dir(high, low, close, period: int = 10, mult: float = 3.0) -> np.ndarray:
    """direction < 0 means uptrend; direction > 0 means downtrend."""
    n = len(close)
    hl2 = (high + low) / 2.0
    atr_v = atr(high, low, close, period)
    upper_b = hl2 + mult * atr_v
    lower_b = hl2 - mult * atr_v
    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    direction = np.ones(n, dtype=np.int8)

    for i in range(n):
        if np.isnan(atr_v[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = upper_b[i]
            final_lower[i] = lower_b[i]
            continue
        if upper_b[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_b[i]
        else:
            final_upper[i] = final_upper[i - 1]
        if lower_b[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_b[i]
        else:
            final_lower[i] = final_lower[i - 1]
        if close[i] > final_upper[i - 1]:
            direction[i] = -1
        elif close[i] < final_lower[i - 1]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
    return direction
