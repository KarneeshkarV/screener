"""Parabolic SAR (Stop and Reverse) indicator."""

from __future__ import annotations

import numpy as np

from screener.indicators.registry import indicator

INITIAL_AF = 0.02
STEP_AF = 0.02
END_AF = 0.2


@indicator("sar")
def sar(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Parabolic SAR over one series each, or a ``(bars, symbols)`` panel of them.

    The recursion cannot be vectorised along time, but it is independent across
    symbols, so the panel form runs the loop over *bars* only. The two forms
    agree exactly, which ``tests/test_bar_column_panel.py`` pins; they are
    written out separately because a row of numpy work per bar is worth it for
    a panel and several times too expensive for a lone series.
    """
    if np.ndim(close) > 1:
        return _sar_panel(high, low, close)
    n_len = len(close)
    real_sar = np.zeros(n_len, dtype=np.float64)
    if n_len < 2:
        return real_sar

    trend = np.zeros(n_len, dtype=np.int32)
    calc_sar = np.zeros(n_len, dtype=np.float64)
    ep = np.zeros(n_len, dtype=np.float64)
    af = np.zeros(n_len, dtype=np.float64)

    trend[1] = 1 if close[1] > close[0] else -1
    calc_sar[1] = high[0] if trend[1] > 0 else low[0]
    real_sar[1] = calc_sar[1]
    ep[1] = high[1] if trend[1] > 0 else low[1]
    af[1] = INITIAL_AF

    for i in range(2, n_len):
        temp = calc_sar[i - 1] + af[i - 1] * (ep[i - 1] - calc_sar[i - 1])
        if trend[i - 1] < 0:
            calc_sar[i] = max(temp, high[i - 1], high[i - 2])
            trend[i] = 1 if calc_sar[i] < high[i] else trend[i - 1] - 1
        else:
            calc_sar[i] = min(temp, low[i - 1], low[i - 2])
            trend[i] = -1 if calc_sar[i] > low[i] else trend[i - 1] + 1

        if trend[i] < 0:
            ep[i] = min(low[i], ep[i - 1]) if trend[i] != -1 else low[i]
        else:
            ep[i] = max(high[i], ep[i - 1]) if trend[i] != 1 else high[i]

        if abs(trend[i]) == 1:
            real_sar[i] = ep[i - 1]
            af[i] = INITIAL_AF
        else:
            real_sar[i] = calc_sar[i]
            if ep[i] == ep[i - 1]:
                af[i] = af[i - 1]
            else:
                af[i] = min(END_AF, af[i - 1] + STEP_AF)

    return real_sar


def _sar_panel(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """:func:`sar` over a ``(bars, symbols)`` panel.

    Each column starts at its own first observed bar, not at row 0: a panel
    holds symbols whose histories begin on different dates, so the shorter ones
    arrive padded with leading NaN, and starting where the data starts is what
    gives such a column exactly what its own frame would have given. Rows
    outside a column's history stay 0, as the bars before the seed are in the
    one-series form.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n_len = close.shape[0]
    real_sar = np.zeros(close.shape, dtype=np.float64)
    if n_len < 2:
        return real_sar

    trend = np.zeros(close.shape, dtype=np.int32)
    calc_sar = np.zeros(close.shape, dtype=np.float64)
    ep = np.zeros(close.shape, dtype=np.float64)
    af = np.zeros(close.shape, dtype=np.float64)

    blank = np.zeros(close.shape[1], dtype=np.float64)
    observed = ~np.isnan(close)
    has_bars = observed.any(axis=0)
    first_bar = np.argmax(observed, axis=0)

    for i in range(1, n_len):
        # The second bar of a column seeds its trend; the third onwards runs
        # the recursion.
        seeding = has_bars & (first_bar == i - 1)
        running = has_bars & (first_bar <= i - 2)
        rising = close[i] > close[i - 1]
        if running.any():
            temp = calc_sar[i - 1] + af[i - 1] * (ep[i - 1] - calc_sar[i - 1])
            falling = trend[i - 1] < 0
            step_sar = np.where(
                falling,
                np.maximum(np.maximum(temp, high[i - 1]), high[i - 2]),
                np.minimum(np.minimum(temp, low[i - 1]), low[i - 2]),
            )
            step_trend = np.where(
                falling,
                np.where(step_sar < high[i], 1, trend[i - 1] - 1),
                np.where(step_sar > low[i], -1, trend[i - 1] + 1),
            )
            step_ep = np.where(
                step_trend < 0,
                np.where(step_trend != -1, np.minimum(low[i], ep[i - 1]), low[i]),
                np.where(step_trend != 1, np.maximum(high[i], ep[i - 1]), high[i]),
            )
            reversing = np.abs(step_trend) == 1
            step_real = np.where(reversing, ep[i - 1], step_sar)
            step_af = np.where(
                reversing,
                INITIAL_AF,
                np.where(
                    step_ep == ep[i - 1],
                    af[i - 1],
                    np.minimum(END_AF, af[i - 1] + STEP_AF),
                ),
            )
        else:
            step_sar = step_trend = step_ep = step_real = step_af = blank

        seed_sar = np.where(rising, high[i - 1], low[i - 1])
        trend[i] = np.where(
            seeding, np.where(rising, 1, -1), np.where(running, step_trend, 0)
        )
        calc_sar[i] = np.where(seeding, seed_sar, np.where(running, step_sar, 0.0))
        real_sar[i] = np.where(seeding, seed_sar, np.where(running, step_real, 0.0))
        ep[i] = np.where(
            seeding, np.where(rising, high[i], low[i]), np.where(running, step_ep, 0.0)
        )
        af[i] = np.where(seeding, INITIAL_AF, np.where(running, step_af, 0.0))

    return real_sar
