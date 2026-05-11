"""Value + Quality: P/E <20, ROE >15%, low debt, bullish trend."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("cheap_quality")
def cheap_quality() -> list:
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 20,
        col("return_on_equity") > 15,
        col("debt_to_equity") < 1,
        col("EMA20") > col("EMA200"),
    ]
