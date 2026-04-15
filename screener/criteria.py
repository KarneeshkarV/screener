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


def value():
    """Low P/E (<20) with positive earnings."""
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 20,
    ]


def quality():
    """High ROE (>15%) with low debt."""
    return [
        col("return_on_equity") > 15,
        col("debt_to_equity") < 1,
    ]


def cheap_quality():
    """Value + Quality: P/E <20, ROE >15%, low debt, bullish trend."""
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 20,
        col("return_on_equity") > 15,
        col("debt_to_equity") < 1,
        col("EMA20") > col("EMA200"),
    ]


def undervalued():
    """Deep value: P/E <12, positive earnings, above-average volume."""
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 12,
        col("volume") > col("average_volume_10d_calc"),
    ]


def dividend():
    """Dividend yield >3% with positive earnings and low debt."""
    return [
        col("dividend_yield_recent") > 3,
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 25,
        col("debt_to_equity") < 1.5,
    ]


def momentum_value():
    """Cheap stocks breaking out: P/E <25, RSI 50-70, EMA bullish."""
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 25,
        col("RSI") >= 50,
        col("RSI") <= 70,
        col("EMA5") > col("EMA20"),
        col("EMA20") > col("EMA200"),
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
    "value": value,
    "quality": quality,
    "cheap_quality": cheap_quality,
    "undervalued": undervalued,
    "dividend": dividend,
    "momentum_value": momentum_value,
}
