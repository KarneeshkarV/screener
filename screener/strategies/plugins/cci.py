"""Commodity Channel Index (CCI) mean reversion (Donald Lambert).

Source: Lambert, "Commodity Channel Index: Tool for Trading Cyclic Trends",
Commodities magazine (now Futures), 1980. Modern reference:
https://school.stockcharts.com/doku.php?id=technical_indicators:commodity_channel_index_cci

CCI measures how far the typical price deviates from its 20-period mean,
normalized by the mean absolute deviation:

    tp     = (high + low + close) / 3
    cci    = (tp - sma(tp, 20)) / (0.015 * mean(|tp - sma(tp, 20)|, 20))

Lambert's rule: buy when CCI falls below -100 (statistical overshoot), sell
when it rises above +100. The engine's Pine has no absolute value or mean
deviation, so ``prepare_bars`` precomputes the ``cci_20`` column.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 20
_MULT = 0.015


def _prepare_cci(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        tp = (
            frame["high"].astype(float)
            + frame["low"].astype(float)
            + frame["close"].astype(float)
        ) / 3.0
        sma_tp = tp.rolling(_WINDOW, min_periods=_WINDOW).mean()
        mean_dev = (tp - sma_tp).abs().rolling(_WINDOW, min_periods=_WINDOW).mean()
        frame["cci_20"] = (tp - sma_tp) / (_MULT * mean_dev)
        out[tv] = frame
    return out


def _lookback() -> int:
    return _WINDOW


register_expression_strategy(
    "cci_reversion",
    entry="cci_20 < -100",
    exit="cci_20 > 100",
    prepare_bars=_prepare_cci,
    required_lookback=_lookback,
)
