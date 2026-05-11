"""Stocks breaking through 52w high intraday on volume surge."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("intraday_breakout")
def intraday_breakout() -> list:
    return [
        col("close").above_pct("price_52_week_high", 0.97),
        col("relative_volume_10d_calc") >= 2.0,
        col("change") >= 1.5,
        col("EMA5") > col("EMA20"),
    ]
