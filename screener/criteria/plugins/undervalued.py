"""Deep value: P/E <12, positive earnings, above-average volume."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("undervalued")
def undervalued() -> list:
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 12,
        col("volume") > col("average_volume_10d_calc"),
    ]
