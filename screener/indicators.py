"""Technical indicator primitives shared by the Pine evaluator and the
standalone strategy runner.

Each function accepts either a :class:`numpy.ndarray` or a :class:`pandas.Series`
and returns the same type as its input, so both numeric loops (the Pine-port
script) and AST evaluation (the backtester engine) can call the same code.

Semantics match TradingView Pine ``ta.*`` where applicable:
  * :func:`ema` uses ``α = 2 / (length + 1)`` with ``adjust=False`` and masks
    the first ``length - 1`` bars as NaN (``ta.ema``).
  * :func:`rsi` and :func:`atr` use Wilder's smoothing seeded with the
    arithmetic mean of the first ``length`` values (``ta.rma`` / ``ta.rsi`` /
    ``ta.atr``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series


def _to_ndarray(x: ArrayLike) -> tuple[np.ndarray, pd.Index | None]:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, copy=False), x.index
    return np.asarray(x, dtype=float), None


def _wrap(result: np.ndarray, index: pd.Index | None) -> ArrayLike:
    if index is None:
        return result
    return pd.Series(result, index=index)


def sma(x: ArrayLike, length: int) -> ArrayLike:
    arr, idx = _to_ndarray(x)
    out = pd.Series(arr).rolling(length, min_periods=length).mean().to_numpy()
    return _wrap(out, idx)


def stdev(x: ArrayLike, length: int) -> ArrayLike:
    arr, idx = _to_ndarray(x)
    out = (
        pd.Series(arr)
        .rolling(length, min_periods=length)
        .std(ddof=0)
        .to_numpy()
    )
    return _wrap(out, idx)


def highest(x: ArrayLike, length: int) -> ArrayLike:
    arr, idx = _to_ndarray(x)
    out = pd.Series(arr).rolling(length, min_periods=length).max().to_numpy()
    return _wrap(out, idx)


def lowest(x: ArrayLike, length: int) -> ArrayLike:
    arr, idx = _to_ndarray(x)
    out = pd.Series(arr).rolling(length, min_periods=length).min().to_numpy()
    return _wrap(out, idx)


def ema(x: ArrayLike, length: int) -> ArrayLike:
    arr, idx = _to_ndarray(x)
    out = (
        pd.Series(arr)
        .ewm(span=length, adjust=False, min_periods=length)
        .mean()
        .to_numpy()
    )
    return _wrap(out, idx)


def _rma_ndarray(arr: np.ndarray, length: int) -> np.ndarray:
    """Wilder's RMA: seed = arithmetic mean of first ``length`` values,
    then ``y_t = (1 - α) y_{t-1} + α x_t`` with ``α = 1 / length``.
    """
    m = len(arr)
    out = np.full(m, np.nan, dtype=np.float64)
    if m < length:
        return out
    seed_window = arr[:length]
    if np.isnan(seed_window).any():
        return out
    out[length - 1] = float(np.mean(seed_window))
    alpha = 1.0 / length
    for i in range(length, m):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def wilder_rma(x: ArrayLike, length: int) -> ArrayLike:
    arr, idx = _to_ndarray(x)
    return _wrap(_rma_ndarray(arr, length), idx)


def rsi(close: ArrayLike, length: int = 14) -> ArrayLike:
    arr, idx = _to_ndarray(close)
    m = len(arr)
    if m == 0:
        return _wrap(np.array([], dtype=float), idx)
    diff = np.diff(arr, prepend=arr[0])
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = _rma_ndarray(gains, length)
    avg_loss = _rma_ndarray(losses, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Wilder convention: zero losses over the window → RSI = 100.
    out = np.where(avg_loss == 0, 100.0, out)
    nan_mask = np.isnan(avg_gain) | np.isnan(avg_loss)
    out = np.where(nan_mask, np.nan, out)
    return _wrap(out, idx)


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int = 14) -> ArrayLike:
    h, idx = _to_ndarray(high)
    lo, _ = _to_ndarray(low)
    c, _ = _to_ndarray(close)
    if len(c) == 0:
        return _wrap(np.array([], dtype=float), idx)
    prev_close = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce(
        [h - lo, np.abs(h - prev_close), np.abs(lo - prev_close)]
    )
    return _wrap(_rma_ndarray(tr, length), idx)


def crossover(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    return _cross(a, b, direction="over")


def crossunder(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    return _cross(a, b, direction="under")


def _cross(a: ArrayLike, b: ArrayLike, direction: str) -> ArrayLike:
    a_arr, idx = _to_ndarray(a)
    b_arr, _ = _to_ndarray(b)
    if len(a_arr) == 0:
        return _wrap(np.array([], dtype=bool), idx)
    a_prev = np.concatenate(([np.nan], a_arr[:-1]))
    b_prev = np.concatenate(([np.nan], b_arr[:-1]))
    if direction == "over":
        out = (a_arr > b_arr) & (a_prev <= b_prev)
    else:
        out = (a_arr < b_arr) & (a_prev >= b_prev)
    out = np.where(np.isnan(a_prev) | np.isnan(b_prev), False, out).astype(bool)
    return _wrap(out, idx)
