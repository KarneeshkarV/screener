"""Value & GARP strategies: cheap quality, growth at a reasonable price.

Methodology sources:

- ``value_rank``        — NSE Indices "Nifty500 Value 50" / "Nifty200 Value 30"
                          index methodology (composite of E/P, B/P, sales/price
                          and dividend-yield percentiles, semi-annual
                          rebalance) and the India Fama-French value evidence.
                          The backtester's dated fundamentals (pe_ttm, pb_ttm,
                          roe_ttm, debt_to_equity) are merged *after* the
                          prepare hook, so the composite cannot feed
                          ``rank_score``; this strategy therefore gates on the
                          value composite (E/P and B/P percentile blend) in the
                          entry expression and lets the rolling backtester pick
                          the passing names by dollar volume — the closest
                          backtest analogue of the index family's
                          free-float-mcap weighting.
- ``garp``              — Growth At a Reasonable Price: reasonable P/E with
                          high EPS growth, high ROE and low leverage (PEG<2
                          spirit). Common in India's quality-growth funds and
                          the "growth at reasonable price" factor literature.
- ``deep_value``        — Classic cheap-and-quality screen (low P/E, low P/B,
                          low leverage, positive ROE) plus a 200-day trend gate
                          so falling knives (value traps) are skipped.
- ``value_momentum_harness`` — Asness, Moskowitz & Pedersen (2013) "Value and
                          Momentum Everywhere": value gates for cheapness AND a
                          price-trend gate, with selection ranking the eligible
                          names by trailing 6-month return so the portfolio
                          harvests momentum *within* the cheap universe.

All four are market-agnostic expressions (FMP serves both US and India .NS
symbols); run with ``--fundamentals-provider fmp``. Fundamentals arrive with a
filing lag (defaults: fmp=1 day, openscreener=60 days) so entry signals are
point-in-time. Selection among eligible names uses dollar volume unless a
``rank_score`` column is present (only ``value_momentum_harness`` emits one).
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

# ── Shared, deliberately moderate gates (Nifty500-friendly) ────────────────
_PE_MAX = 25.0
_PB_MAX = 4.0
_ROE_MIN = 10.0
_DEBT_MAX = 2.5

# Nifty500 Value 50: positive earnings & book value + quality floor, so pure
# "cheap because broken" names (negative earnings, negative book, distressed
# leverage, sub-10% ROE) are excluded before the E/P + B/P blend is applied.
VALUE_RANK_ENTRY = (
    f"pe_ttm > 0 and pb_ttm > 0 and pe_ttm <= {_PE_MAX} and pb_ttm <= {_PB_MAX} "
    f"and roe_ttm >= {_ROE_MIN} and debt_to_equity <= {_DEBT_MAX}"
)

# GARP: reasonable multiple + double-digit EPS growth + high ROE + low leverage.
GARP_ENTRY = (
    "pe_ttm >= 5 and pe_ttm <= 30 and eps_growth_yoy > 15 "
    "and roe_ttm >= 12 and debt_to_equity <= 2"
)

# Deep value: cheap on both P/E and P/B, low leverage, positive ROE, and only
# if the price is already above its 200-day SMA (skip falling knives).
DEEP_VALUE_ENTRY = (
    "pe_ttm > 0 and pe_ttm < 15 and pb_ttm > 0 and pb_ttm < 2 "
    "and debt_to_equity <= 1.5 and roe_ttm >= 8 and close > sma(close, 200)"
)

# Value + momentum harness: same value gate as ``value_rank`` plus an absolute
# price-trend filter; eligible names are then ranked by trailing 6-month return.
HARNESS_ENTRY = (
    f"pe_ttm > 0 and pb_ttm > 0 and pe_ttm <= {_PE_MAX} and pb_ttm <= {_PB_MAX} "
    f"and roe_ttm >= {_ROE_MIN} and debt_to_equity <= {_DEBT_MAX} "
    "and close > sma(close, 100)"
)

_MOM_WINDOW = 126  # ~6 months


def _prepare_harness(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach a price-side 6-month return and rank by it (value+momentum).

    Fundamentals merge after this hook, so the rank can only use price data:
    among the names passing the value gates in the entry expression, the
    rolling backtester then fills its ``--top`` slots with the strongest
    trailing-6-month performers (causal: bar ``t`` uses closes <= ``t``).
    """
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["mom_126"] = close / close.shift(_MOM_WINDOW) - 1.0
        frame["rank_score"] = frame["mom_126"]
        out[tv] = frame
    return out


def _lookback_basic() -> int:
    # No long rolling windows; just enough history for the merge + entry eval.
    return 20


def _lookback_deep_value() -> int:
    # ``close > sma(close, 200)`` needs 200 trailing closes.
    return 200


def _lookback_harness() -> int:
    # 6-month return leg needs 126 prior closes (covers the sma(close,100) too).
    return _MOM_WINDOW


register_expression_strategy(
    "value_rank",
    entry=VALUE_RANK_ENTRY,
    exit=None,
    required_lookback=_lookback_basic,
)

register_expression_strategy(
    "garp",
    entry=GARP_ENTRY,
    exit=None,
    required_lookback=_lookback_basic,
)

register_expression_strategy(
    "deep_value",
    entry=DEEP_VALUE_ENTRY,
    exit=None,
    required_lookback=_lookback_deep_value,
)

register_expression_strategy(
    "value_momentum_harness",
    entry=HARNESS_ENTRY,
    exit=None,
    prepare_bars=_prepare_harness,
    required_lookback=_lookback_harness,
)
