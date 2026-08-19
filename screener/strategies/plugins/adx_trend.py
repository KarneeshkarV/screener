"""ADX trend filter (J. Welles Wilder).

Source: Wilder, "New Concepts in Technical Trading Systems", 1978, Trend
Research. The ADX + Directional Movement system is Wilder's trend-strength
framework; ADX(14) > 25 marks a strong trend, +DI > -DI marks its direction.

Indicators are not expressible in the engine's Pine (Wilder RMA smoothing), so
``prepare_bars`` computes them from the repo's ``rma`` indicator:

    tr    = max(high-low, |high - close_prev|, |low - close_prev|)
    +DM   = high - high_prev  when that exceeds low_prev - low and is positive
    -DM   = low_prev - low    when that exceeds high - high_prev and is positive
    +DI   = 100 * rma(+DM, 14) / rma(tr, 14)
    -DI   = 100 * rma(-DM, 14) / rma(tr, 14)
    ADX   = rma(100 * |+DI - -DI| / (+DI + -DI), 14)

Rule: long when ADX confirms a strong up-trend (+DI > -DI and ADX > 25), exit
when the direction flips (-DI overtakes +DI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.plugins.rma import rma as _rma
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 14
_MIN_ADX = 25.0


def _prepare_adx(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        high = frame["high"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float)
        close = frame["close"].to_numpy(dtype=float)

        prev_close = np.concatenate(([close[0]], close[:-1]))
        prev_high = np.concatenate(([high[0]], high[:-1]))
        prev_low = np.concatenate(([low[0]], low[:-1]))

        tr = np.maximum(
            high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_rma = _rma(tr, _WINDOW)
        plus_rma = _rma(plus_dm, _WINDOW)
        minus_rma = _rma(minus_dm, _WINDOW)

        with np.errstate(divide="ignore", invalid="ignore"):
            di_plus = 100.0 * plus_rma / tr_rma
            di_minus = 100.0 * minus_rma / tr_rma
            dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = _rma(dx, _WINDOW)

        frame["di_plus"] = di_plus
        frame["di_minus"] = di_minus
        frame["adx_14"] = adx
        out[tv] = frame
    return out


def _lookback() -> int:
    # tr needs 1 prior close, then two Wilder seeds (2 * _WINDOW).
    return 2 * _WINDOW + 1


register_expression_strategy(
    "adx_trend",
    entry=f"adx_14 > {_MIN_ADX} and di_plus > di_minus",
    exit="di_plus < di_minus",
    prepare_bars=_prepare_adx,
    required_lookback=_lookback,
)
