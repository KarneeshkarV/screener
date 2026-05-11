"""Average True Range via Wilder smoothing."""

from __future__ import annotations

import numpy as np

from screener.indicators.plugins.rma import rma
from screener.indicators.registry import indicator


@indicator("atr")
def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14
) -> np.ndarray:
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    return rma(tr, n)
