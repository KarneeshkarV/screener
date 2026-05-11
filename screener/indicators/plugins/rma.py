"""Wilder's RMA, matching Pine ``ta.rma``."""

from __future__ import annotations

import numpy as np

from screener.indicators.registry import indicator


@indicator("rma")
def rma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < n:
        return out
    alpha = 1.0 / n
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out
