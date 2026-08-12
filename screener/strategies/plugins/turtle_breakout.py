"""Turtle-style Donchian breakout with the original dual-channel exits.

The Turtle system (Richard Dennis / William Eckhardt, 1983) is the canonical
trend-following system: enter on a 20-day high breakout, exit on a 10-day low
breakthrough (System 1). The edge is the same time-series-momentum persistence
documented by Moskowitz, Ooi & Pedersen (2012) and in the managed-futures
literature. Long-only adaptation for equity universes.

Rules (as-of bar ``t``, prior-bar references so today is never self-included):
    entry: close > highest(high, 20) shifted by one bar
    exit : close < lowest(low, 10) shifted by one bar
    --hold caps the maximum holding period.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_ENTRY_WINDOW = 20
_EXIT_WINDOW = 10


def _prepare_turtle(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        frame["turtle_high"] = (
            frame["high"].astype(float).rolling(_ENTRY_WINDOW).max().shift(1)
        )
        frame["turtle_low"] = (
            frame["low"].astype(float).rolling(_EXIT_WINDOW).min().shift(1)
        )
        out[tv] = frame
    return out


def _lookback() -> int:
    return _ENTRY_WINDOW


register_expression_strategy(
    "turtle_breakout",
    entry="close > turtle_high",
    exit="close < turtle_low",
    prepare_bars=_prepare_turtle,
    required_lookback=_lookback,
)
