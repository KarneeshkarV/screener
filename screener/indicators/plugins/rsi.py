"""Wilder RSI."""

from __future__ import annotations

import numpy as np

from screener.indicators.plugins.rma import rma
from screener.indicators.registry import indicator


@indicator("rsi")
def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    rma_up = rma(up, n)
    rma_dn = rma(dn, n)
    rs = np.where(rma_dn > 0, rma_up / np.maximum(rma_dn, 1e-12), np.inf)
    out = 100 - 100 / (1 + rs)
    out[rma_dn == 0] = 100
    return out
