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


@criterion("momentum_12_1")
def momentum_12_1() -> list:
    """Jegadeesh-Titman 12-1 momentum: 12-month return excluding the last month.

    TradingView exposes yearly (``Perf.Y``) and monthly (``Perf.1M``) trailing
    performance, so the causal 12-1 momentum
    ``(1 + Perf.Y) / (1 + Perf.1M) - 1 > 0`` reduces to ``Perf.Y > Perf.1M``.

    This snapshot comparison is a coarse pre-filter that keeps the TradingView
    field momentum-shaped, so the later bar download stays small. It is not
    the eligibility rule. TradingView's ``Column`` type has no arithmetic, so
    a slack form such as ``Perf.Y > Perf.1M - 5`` cannot be sent to the vendor.
    The exact gate is the recipe's ``eligible_above=0`` floor, applied after
    bar scoring as ``mom_12_1 > 0`` (the same expression the backtest entry
    uses). Names whose 12-1 return sits near zero can still flip across this
    snapshot diagonal: TradingView uses calendar-month anchors and its own
    adjusted series, while the bar recipe uses 252/21 sessions and the
    yfinance close. Those names rank at the bottom of the positive set, so a
    top-N screen does not select them.
    """
    return [
        col("Perf.Y") > col("Perf.1M"),
    ]


@criterion("mark_minervini")
def mark_minervini() -> list:
    """Mark Minervini Trend Template (TradingView approximation).

    Matches ``MINERVINI_ENTRY_EXPR`` wherever TradingView can express the
    condition (SMA50/150/200 stack, price above 52-week low by 30% and within
    25% of the 52-week high). The SMA200-rising and cross-sectional RS-rank
    legs have no TradingView column, so they are dropped here.
    """
    return [
        col("close") > col("SMA150"),
        col("close") > col("SMA200"),
        col("SMA150") > col("SMA200"),
        col("SMA50") > col("SMA150"),
        col("SMA50") > col("SMA200"),
        col("close") > col("SMA50"),
        col("close").above_pct("price_52_week_low", 1.3),
        col("close").above_pct("price_52_week_high", 0.75),
    ]


@criterion("near_52_high")
def near_52_week_high() -> list:
    """Between 80–100% of the 52-week high but strictly below it (under resistance)."""
    return [
        col("close").between_pct("price_52_week_high", 0.8, 1),
        col("close") < col("price_52_week_high"),
    ]
