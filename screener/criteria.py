from tradingview_screener import col


def ema_bullish_stack():
    """EMA5 > EMA20 > EMA100 > EMA200 (bullish stacking)."""
    return [
        col("EMA5") > col("EMA20"),
        col("EMA20") > col("EMA100"),
        col("EMA100") > col("EMA200"),
        col("EMA200") > 0,
    ]


CRITERIA = {
    "ema": ema_bullish_stack,
}
