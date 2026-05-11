"""Close within 10% of 52-week high with above-average volume."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("breakout")
def near_52w_breakout() -> list:
    return [
        col("close").above_pct("price_52_week_high", 0.9),
        col("volume") > col("average_volume_10d_calc"),
    ]
