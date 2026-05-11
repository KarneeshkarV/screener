"""Dividend yield >3% with positive earnings and low debt."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("dividend")
def dividend() -> list:
    return [
        col("dividend_yield_recent") > 3,
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 25,
        col("debt_to_equity") < 1.5,
    ]
