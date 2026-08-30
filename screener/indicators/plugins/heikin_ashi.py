"""Heikin-Ashi candle transform, shared by every consumer that needs it.

``ha_open`` is recursive (``ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2``),
so it is seeded from the first bar's real open. That seed's influence halves
every bar, so after a few dozen bars it is negligible; callers that start
scoring only after a long lookback (e.g. 252 bars) never see it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from screener.indicators.registry import indicator


@indicator("heikin_ashi")
def heikin_ashi_ohlc(
    open_: NDArray[np.float64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Return ``(ha_open, ha_high, ha_low, ha_close)`` for one symbol's bars."""
    n = len(close)
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = np.zeros_like(close)
    if n > 0:
        ha_open[0] = open_[0]
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([ha_open, ha_close, high])
    ha_low = np.minimum.reduce([ha_open, ha_close, low])
    return ha_open, ha_high, ha_low, ha_close
