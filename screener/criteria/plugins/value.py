"""Low P/E (<20) with positive earnings."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("value")
def value() -> list:
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 20,
    ]
