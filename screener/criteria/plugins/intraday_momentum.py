"""Liquid movers with relative-volume surge and clean trend."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("intraday_momentum")
def intraday_momentum() -> list:
    """Designed for intraday trading: filters for above-average current volume
    vs. 10d average, today moving meaningfully, price riding above the
    short EMA, and RSI in trend-strong territory.
    """
    return [
        col("relative_volume_10d_calc") >= 1.5,
        col("volume") >= 200_000,
        col("close") >= col("EMA20"),
        col("EMA20") > col("EMA200"),
        col("RSI") >= 55,
        col("RSI") <= 80,
        col("change") >= 1.0,
    ]
