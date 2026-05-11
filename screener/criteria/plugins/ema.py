"""EMA5 > EMA20 > EMA100 > EMA200 (bullish stacking)."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("ema")
def ema_bullish_stack() -> list:
    return [
        col("EMA5") > col("EMA20"),
        col("EMA20") > col("EMA100"),
        col("EMA100") > col("EMA200"),
        col("EMA200") > 0,
    ]
