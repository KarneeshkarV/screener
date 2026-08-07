"""Valuation, quality, and dividend ranking recipes."""

from __future__ import annotations

import pandas as pd

from screener.scoring import scorer
from screener.scoring.components import (
    above_flag,
    inv_percentile,
    liquidity_from_dollar_volume,
    log_percentile,
    momentum_change,
    numeric,
    percentile,
    rsi_quality,
    trend_stack_strength,
)

_VALUE_COLUMNS = ("price_earnings_ttm",)
_QUALITY_COLUMNS = ("return_on_equity", "debt_to_equity", "EMA20", "EMA200")
_CHEAP_QUALITY_COLUMNS = (
    "price_earnings_ttm",
    "return_on_equity",
    "debt_to_equity",
    "EMA20",
    "EMA200",
)
_DIVIDEND_COLUMNS = (
    "dividend_yield_recent",
    "price_earnings_ttm",
    "debt_to_equity",
)
_MOMENTUM_VALUE_COLUMNS = (
    "price_earnings_ttm",
    "RSI",
    "EMA5",
    "EMA20",
    "EMA100",
    "EMA200",
)


def _score_value_core(df: pd.DataFrame) -> pd.Series:
    """Cheapness with liquidity sanity (P/E > 0 only)."""
    close = numeric(df, "close")
    pe = numeric(df, "price_earnings_ttm")
    volume = numeric(df, "volume")
    mcap = numeric(df, "market_cap_basic")

    # ``positive_only`` already zeroes non-positive P/E, so there is no separate
    # "has earnings" term: it would double-penalize loss-makers and act as a
    # constant for the criteria whose filters already require P/E > 0.
    cheap = inv_percentile(pe, positive_only=True)
    liquidity = liquidity_from_dollar_volume(volume, close)
    market_cap = log_percentile(mcap)

    return (65 * cheap + 20 * liquidity + 15 * market_cap).round(2)


@scorer(
    "value",
    columns=_VALUE_COLUMNS,
    description="Lower P/E (positive only) + liquidity",
)
def score_value(df: pd.DataFrame) -> pd.Series:
    return _score_value_core(df)


@scorer(
    "undervalued",
    columns=_VALUE_COLUMNS,
    description="Deep-value: cheapest positive P/E with volume/liquidity",
)
def score_undervalued(df: pd.DataFrame) -> pd.Series:
    # Same core as value; undervalued filter already enforces deeper PE cut.
    return _score_value_core(df)


def _score_quality_core(df: pd.DataFrame) -> pd.Series:
    close = numeric(df, "close")
    roe = numeric(df, "return_on_equity")
    de = numeric(df, "debt_to_equity")
    volume = numeric(df, "volume")
    ema20 = numeric(df, "EMA20")
    ema200 = numeric(df, "EMA200")

    roe_rank = percentile(roe)
    # Lower D/E better, but only down to zero: negative D/E means negative
    # shareholder equity, which must not earn the top low-debt rank.
    low_debt = inv_percentile(de, lower_bound=0.0)
    liquidity = liquidity_from_dollar_volume(volume, close)
    trend = above_flag(ema20, ema200)

    return (40 * roe_rank + 30 * low_debt + 15 * liquidity + 15 * trend).round(2)


@scorer(
    "quality",
    columns=_QUALITY_COLUMNS,
    description="High ROE + low debt + mild trend",
)
def score_quality(df: pd.DataFrame) -> pd.Series:
    return _score_quality_core(df)


@scorer(
    "cheap_quality",
    columns=_CHEAP_QUALITY_COLUMNS,
    description="Blend of cheapness, franchise quality, and mild trend",
)
def score_cheap_quality(df: pd.DataFrame) -> pd.Series:
    value = _score_value_core(df)
    quality = _score_quality_core(df)
    return ((value + quality) / 2.0).round(2)


@scorer(
    "dividend",
    columns=_DIVIDEND_COLUMNS,
    description="Yield + sane valuation + balance sheet",
)
def score_dividend(df: pd.DataFrame) -> pd.Series:
    close = numeric(df, "close")
    yield_ = numeric(df, "dividend_yield_recent")
    pe = numeric(df, "price_earnings_ttm")
    de = numeric(df, "debt_to_equity")
    volume = numeric(df, "volume")

    yield_rank = percentile(yield_)
    cheap = inv_percentile(pe, positive_only=True)
    low_debt = inv_percentile(de, lower_bound=0.0)
    liquidity = liquidity_from_dollar_volume(volume, close)

    return (40 * yield_rank + 25 * cheap + 20 * low_debt + 15 * liquidity).round(2)


@scorer(
    "momentum_value",
    columns=_MOMENTUM_VALUE_COLUMNS,
    description="Cheap + RSI in 50–70 band + short/long EMA support",
)
def score_momentum_value(df: pd.DataFrame) -> pd.Series:
    close = numeric(df, "close")
    pe = numeric(df, "price_earnings_ttm")
    rsi = numeric(df, "RSI")
    ema5 = numeric(df, "EMA5")
    ema20 = numeric(df, "EMA20")
    ema200 = numeric(df, "EMA200")
    change = numeric(df, "change")
    volume = numeric(df, "volume")

    cheap = inv_percentile(pe, positive_only=True)
    # Center 60 with half-width 20 → 0 at 40/80, peak at 60 (covers 50–70 well).
    rsi_q = rsi_quality(rsi, center=60.0, half_width=20.0)
    short_stack = above_flag(ema5, ema20)
    long_trend = above_flag(ema20, ema200)
    liquidity = liquidity_from_dollar_volume(volume, close)
    # EMA100 is declared in this scorer's columns, so the full stack spread is
    # the normal path; the flag average is a fallback for markets that return
    # no EMA100 at all. Keyed off the data, not off which other criteria were
    # combined — otherwise ``-c momentum_value`` and ``-c momentum_value -c ema``
    # would score the same stock differently.
    ema100 = numeric(df, "EMA100")
    if ema100.notna().any():
        trend = trend_stack_strength(close, ema5, ema20, ema100, ema200)
    else:
        trend = (short_stack + long_trend) / 2.0

    return (
        30 * cheap
        + 25 * rsi_q
        + 20 * trend
        + 15 * liquidity
        + 10 * momentum_change(change)
    ).round(2)


__all__ = [
    "score_cheap_quality",
    "score_dividend",
    "score_momentum_value",
    "score_quality",
    "score_undervalued",
    "score_value",
]
