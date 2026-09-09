"""Anchored-VWAP trend continuation (institutional benchmark flavour).

VWAP (volume-weighted average price) is the reference execution benchmark of
institutional desks worldwide and a staple of Indian equity trading. The
anchored VWAP here accumulates typical price x volume from the first bar of
the fetched history (the warmup window, so the anchor is effectively the
available listing history):

    vwap[t] = cumsum(typical_price * volume) / cumsum(volume)

A stock trading above its anchored VWAP with a rising long-term trend is in
institutional-favoured territory; a close back below VWAP signals distribution.

Rules:
    entry: close > vwap and close > sma(close, 200)   (uptrend + above VWAP)
    exit : crossunder(close, vwap)                     (VWAP breakdown)
    --hold caps the maximum holding period.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_TREND_SMA = 200


def _prepare_vwap(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        volume = frame["volume"].astype(float)
        cum_pv = (typical * volume).cumsum()
        cum_v = volume.cumsum()
        frame["vwap"] = cum_pv / cum_v.where(cum_v > 0)
        out[tv] = frame
    return out


def _lookback() -> int:
    return _TREND_SMA


register_expression_strategy(
    "vwap_trend",
    entry=f"close > vwap and close > sma(close, {_TREND_SMA})",
    exit="crossunder(close, vwap)",
    prepare_bars=_prepare_vwap,
    required_lookback=_lookback,
)
