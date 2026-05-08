"""Numpy/Pandas indicator helpers shared by strategy research code."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's RMA, matching Pine ta.rma."""
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < n:
        return out
    alpha = 1.0 / n
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1)
    out = np.empty(len(x), dtype=np.float64)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _stdev(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    rma_up = _rma(up, n)
    rma_dn = _rma(dn, n)
    rs = np.where(rma_dn > 0, rma_up / np.maximum(rma_dn, 1e-12), np.inf)
    rsi = 100 - 100 / (1 + rs)
    rsi[rma_dn == 0] = 100
    return rsi


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    return _rma(tr, n)


def _supertrend_dir(high, low, close, period=10, mult=3.0) -> np.ndarray:
    """Return direction array matching Pine ta.supertrend semantics.

    direction < 0 means uptrend; direction > 0 means downtrend.
    """
    n = len(close)
    hl2 = (high + low) / 2.0
    atr = _atr(high, low, close, period)
    upper_b = hl2 + mult * atr
    lower_b = hl2 - mult * atr
    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    direction = np.ones(n, dtype=np.int8)

    for i in range(n):
        if np.isnan(atr[i]):
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
