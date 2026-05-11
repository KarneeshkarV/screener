"""Exponential moving average."""

from __future__ import annotations

import numpy as np

from screener.indicators.registry import indicator


@indicator("ema")
def ema(x: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1)
    out = np.empty(len(x), dtype=np.float64)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out
