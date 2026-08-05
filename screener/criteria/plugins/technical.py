"""Chart/price/volume-based screening criteria."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import combine, criterion


@criterion("breakout")
def near_52w_breakout() -> list:
    """Close within 10% of 52-week high with above-average volume."""
    return [
        col("close").above_pct("price_52_week_high", 0.9),
        col("volume") > col("average_volume_10d_calc"),
    ]


@criterion("ema")
def ema_bullish_stack() -> list:
    """EMA5 > EMA20 > EMA100 > EMA200 (bullish stacking)."""
    return [
        col("EMA5") > col("EMA20"),
        col("EMA20") > col("EMA100"),
        col("EMA100") > col("EMA200"),
        col("EMA200") > 0,
    ]


@criterion("ema_breakout")
def ema_with_breakout() -> list:
    """EMA bullish stack + 52-week breakout — composition of two criteria."""
    return combine(ema_bullish_stack, near_52w_breakout)()


@criterion("intraday_breakout")
def intraday_breakout() -> list:
    """Stocks breaking through 52w high intraday on volume surge."""
    return [
        col("close").above_pct("price_52_week_high", 0.97),
        col("relative_volume_10d_calc") >= 2.0,
        col("change") >= 1.5,
        col("EMA5") > col("EMA20"),
    ]


@criterion("intraday_momentum")
def intraday_momentum() -> list:
    """Liquid movers with relative-volume surge and clean trend.

    Designed for intraday trading: filters for above-average current volume
    vs. 10d average, today moving meaningfully, price riding above the
    short EMA, and RSI in trend-strong territory.
    """
    return [
        col("relative_volume_10d_calc") >= 1.5,
        col("volume") >= 200_000,
        col("close") >= col("EMA20"),
        col("EMA20") > col("EMA200"),
        col("RSI") >= 55,
        col("RSI") <= 80,
        col("change") >= 1.0,
    ]


@criterion("momentum_value")
def momentum_value() -> list:
    """Cheap stocks breaking out: P/E <25, RSI 50-70, EMA bullish."""
    return [
        col("price_earnings_ttm") > 0,
        col("price_earnings_ttm") <= 25,
        col("RSI") >= 50,
        col("RSI") <= 70,
        col("EMA5") > col("EMA20"),
        col("EMA20") > col("EMA200"),
    ]


@criterion("near_52_high")
def near_52_week_high() -> list:
    """Between 80–100% of the 52-week high but strictly below it (under resistance)."""
    return [
        col("close").between_pct("price_52_week_high", 0.8, 1),
        col("close") < col("price_52_week_high"),
    ]


@criterion("above_20ema")
def above_20ema() -> list:
    """Close above 20-day EMA."""
    return [col("close") > col("EMA20")]


@criterion("above_200ema")
def above_200ema() -> list:
    """Close above 200-day EMA."""
    return [col("close") > col("EMA200")]
