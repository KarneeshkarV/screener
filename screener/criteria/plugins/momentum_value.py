"""Cheap stocks breaking out: P/E <25, RSI 50-70, EMA bullish."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("momentum_value")
def momentum_value() -> list:
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 25,
        col("RSI") >= 50,
        col("RSI") <= 70,
        col("EMA5") > col("EMA20"),
        col("EMA20") > col("EMA200"),
    ]
