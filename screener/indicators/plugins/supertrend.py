"""Supertrend direction (matches Pine ``ta.supertrend`` semantics)."""

from __future__ import annotations

import numpy as np

from screener.indicators.plugins.atr import atr
from screener.indicators.registry import indicator


@indicator("supertrend_dir")
def supertrend_dir(high, low, close, period: int = 10, mult: float = 3.0) -> np.ndarray:
    """direction < 0 means uptrend; direction > 0 means downtrend.

    The inputs may be one series each or a ``(bars, symbols)`` panel of them.
    The band recursion cannot be vectorised along time, but it is independent
    across symbols, so the panel form runs the loop over *bars* only and does
    the per-symbol work in numpy: a whole field then costs about what one
    ticker used to. The two forms agree exactly, which
    ``tests/test_bar_column_panel.py`` pins; they are written out separately
    because a row of numpy work per bar is worth it for a panel and several
    times too expensive for a lone series.
    """
    if np.ndim(close) > 1:
        return _supertrend_dir_panel(high, low, close, period, mult)
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


def _supertrend_dir_panel(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, mult: float
) -> np.ndarray:
    """:func:`supertrend_dir` over a ``(bars, symbols)`` panel.

    A column whose history starts late arrives padded with leading NaN. Its
    ATR is undefined there, so those bars are skipped exactly as bar 0 of a
    short frame is, and the column gets what its own frame would have given.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    hl2 = (high + low) / 2.0
    atr_v = atr(high, low, close, period)
    upper_b = hl2 + mult * atr_v
    lower_b = hl2 - mult * atr_v
    final_upper = np.full(close.shape, np.nan, dtype=np.float64)
    final_lower = np.full(close.shape, np.nan, dtype=np.float64)
    direction = np.ones(close.shape, dtype=np.int8)
    undefined = np.full(close.shape[1], np.nan, dtype=np.float64)

    for i in range(close.shape[0]):
        # A symbol whose ATR is still undefined is skipped entirely: its bands
        # stay NaN and its direction stays at the initial 1.
        valid = ~np.isnan(atr_v[i])
        if not valid.any():
            continue
        previous_upper = final_upper[i - 1] if i else undefined
        previous_lower = final_lower[i - 1] if i else undefined
        seed = valid & np.isnan(previous_upper)
        carry = valid & ~seed
        keep_upper = (upper_b[i] < previous_upper) | (close[i - 1] > previous_upper)
        keep_lower = (lower_b[i] > previous_lower) | (close[i - 1] < previous_lower)
        final_upper[i] = np.where(
            seed,
            upper_b[i],
            np.where(carry, np.where(keep_upper, upper_b[i], previous_upper), np.nan),
        )
        final_lower[i] = np.where(
            seed,
            lower_b[i],
            np.where(carry, np.where(keep_lower, lower_b[i], previous_lower), np.nan),
        )
        if carry.any():
            crossed = np.where(
                close[i] > previous_upper,
                -1,
                np.where(close[i] < previous_lower, 1, direction[i - 1]),
            )
            direction[i] = np.where(carry, crossed, 1)
    return direction
