from tradingview_screener import col


# ── filter building blocks ──────────────────────────────────────────


def ema_bullish_stack():
    """EMA5 > EMA20 > EMA100 > EMA200 (bullish stacking)."""
    return [
        col("EMA5") > col("EMA20"),
        col("EMA20") > col("EMA100"),
        col("EMA100") > col("EMA200"),
        col("EMA200") > 0,
    ]


def near_52w_breakout():
    """Close within 10% of 52-week high with above-average volume."""
    return [
        col("close").above_pct("price_52_week_high", 0.9),
        col("volume") > col("average_volume_10d_calc"),
    ]


# ── composition helper ──────────────────────────────────────────────


def combine(*filter_fns):
    """Return a function that merges filters from all given filter functions."""
    def combined():
        filters = []
        for fn in filter_fns:
            filters.extend(fn())
        return filters
    return combined


# ── registered shortlists ──────────────────────────────────────────

CRITERIA = {
    "ema": ema_bullish_stack,
    "breakout": near_52w_breakout,
    "ema_breakout": combine(ema_bullish_stack, near_52w_breakout),
}
