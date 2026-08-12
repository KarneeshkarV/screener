"""Donchian breakout entry with a chandelier (ATR) trailing exit.

Combines two classic trend-following building blocks:

* Donchian breakout entry (Turtle system; also the basis of time-series
  momentum / trend-following evidence, e.g. Moskowitz, Ooi & Pedersen 2012):
  enter when today's close clears the highest high of the prior 55 sessions.
* Chandelier exit (Chuck LeBeau): trail a stop at the highest high over the
  last 22 sessions minus 3 x ATR(22). The position is only closed when the
  stop is actually breached on a close.

The volatility-scaled trailing stop gives winners room while cutting losers
fast, which is the documented edge of trend-following systems.

Rules:
    entry: close > highest(high, 55) shifted by one bar (fresh breakout)
    exit : crossunder(close, highest(high, 22) - 3 * atr(22))
    --hold caps the maximum holding period.
"""

from __future__ import annotations

import pandas as pd

from screener.indicators.frames import wilder_atr
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_BREAKOUT = 55
_CHAND_LOOKBACK = 22
_CHAND_ATR = 22
_CHAND_MULT = 3.0


def _prepare_chandelier(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        # Prior-55-day high, today excluded (fresh-breakout semantics).
        frame["breakout_high"] = high.rolling(_BREAKOUT).max().shift(1)
        atr = wilder_atr(high, low, close, _CHAND_ATR, first_bar="nan")
        frame["chand_exit"] = high.rolling(_CHAND_LOOKBACK).max() - _CHAND_MULT * atr
        out[tv] = frame
    return out


def _lookback() -> int:
    return max(_BREAKOUT, _CHAND_ATR * 2)


register_expression_strategy(
    "chandelier_breakout",
    entry="close > breakout_high",
    exit="crossunder(close, chand_exit)",
    prepare_bars=_prepare_chandelier,
    required_lookback=_lookback,
)
