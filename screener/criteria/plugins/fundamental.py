"""Valuation, quality, and dividend screening criteria."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("cheap_quality")
def cheap_quality() -> list:
    """Value + Quality: P/E <20, ROE >15%, low debt, bullish trend."""
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 20,
        col("return_on_equity") > 15,
        col("debt_to_equity") < 1,
        col("EMA20") > col("EMA200"),
    ]


@criterion("dividend")
def dividend() -> list:
    """Dividend yield >3% with positive earnings and low debt."""
    return [
        col("dividend_yield_recent") > 3,
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 25,
        col("debt_to_equity") < 1.5,
    ]


@criterion("quality")
def quality() -> list:
    """High ROE (>15%) with low debt."""
    return [
        col("return_on_equity") > 15,
        col("debt_to_equity") < 1,
    ]


@criterion("undervalued")
def undervalued() -> list:
    """Deep value: P/E <12, positive earnings, above-average volume."""
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 12,
        col("volume") > col("average_volume_10d_calc"),
    ]


@criterion("value")
def value() -> list:
    """Low P/E (<20) with positive earnings."""
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 20,
    ]
