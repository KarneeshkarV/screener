"""Bollinger Band mean reversion: buy the lower-band tag, sell the middle band.

Source: Bollinger, "Bollinger on Bollinger Bands", 2001 (McGraw-Hill). Bands
are SMA20 ± 2× the 20-day population standard deviation; roughly 95% of closes
sit inside the bands, so a close below the lower band is a statistical
overreaction that tends to snap back to the mean (middle band).

Signal (as-of bar ``t``, computed via ``prepare_bars`` because Pine in this
engine has no ``stdev`` function):

    mid   = sma(close, 20)
    lower = mid - 2 * stdev_pop(close, 20)
    entry = close < lower                       # overshoot below the band
    exit  = close > mid                         # revert to the mean

Mean-reversion profile: frequent trades, high win rate, small wins, and the
classic failure mode — buying a falling knife during a strong downtrend where
the price keeps printing closes below the band.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 20
_MULT = 2.0


def _prepare_bb(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        mid = close.rolling(_WINDOW, min_periods=_WINDOW).mean()
        # Population std (ddof=0) matches the repo's bollinger_bands indicator.
        std = close.rolling(_WINDOW, min_periods=_WINDOW).std(ddof=0)
        frame["bb_mid"] = mid
        frame["bb_lower"] = mid - _MULT * std
        out[tv] = frame
    return out


def _lookback() -> int:
    return _WINDOW


register_expression_strategy(
    "bollinger_mean_reversion",
    entry="close < bb_lower",
    exit="close > bb_mid",
    prepare_bars=_prepare_bb,
    required_lookback=_lookback,
)
