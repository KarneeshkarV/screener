"""Dual momentum: relative selection plus an absolute risk-off gate.

Antonacci's design is two decisions, not one. Relative momentum picks the
strongest of the risky alternatives; absolute momentum then asks whether that
winner beat Treasury bills over the same twelve months, and routes the
allocation to the defensive asset when it did not. His own table is the cleanest
evidence for which half matters: dropping the T-bill gate moves the 1974-2011
equity module from 0.73 Sharpe / -23.01% drawdown to 0.45 / -54.56%. Almost all
of the drawdown advantage is the gate, not the ranking.

Long-only approximation
-----------------------
The published strategies are index-level tactical allocations that rotate
between two equity indices and a bond or bill sleeve. This engine trades single
stocks from a universe and holds cash when no candidate qualifies, so the
translation is: relative momentum ranks *stocks*, and the absolute gate decides
whether stocks are held at all. Cash earns nothing here, whereas Antonacci's
defensive leg earns the bond return, so these variants give up the defensive
sleeve's carry - a real and consistently negative difference against the
published record, largest in the periods spent risk-off.

Every gate below is both an entry condition and an exit: the published rules
re-decide the allocation each month and *move out* of equities when the test
fails, so a gate that only blocked new entries would keep the portfolio invested
through the decline it exists to avoid.

Variants
--------
``dual_momentum_gem``
    The gate applied per name: a stock is eligible only if its own 12-1 momentum
    cleared the T-bill return over the same window. The portfolio drifts to cash
    on its own when no winner clears bills, which is what happens in a broad
    decline. Closest to Antonacci's rule mechanically; least like it in effect,
    because a top-decile winner nearly always clears a bill hurdle.

``dual_momentum_market``
    The gate applied to the market: stocks are held only while the benchmark's
    own trailing 12-month return beat bills. This is the equity/bill switch that
    produces GEM's drawdown reduction, used as an overlay on stock momentum.

``dual_momentum_paa``
    Keller & Keuning's Protective Asset Allocation. Its crash protection is
    breadth-driven: with protection factor a=2, the cash fraction reaches 1 once
    half the universe has non-positive momentum. Fixed slots cannot hold a
    fractional cash weight, so the gate is binary at that same boundary - fully
    in cash while half or more of the universe is below its 12-month average.
    Breadth uses the paper's SMA-based momentum (price / 12-month SMA - 1)
    rather than the 12-1 return.

``dual_momentum_daa``
    Keller & Keuning's Defensive Asset Allocation, which separates the assets
    that *detect* trouble from the assets that are traded. The paper's canary is
    two ETFs (emerging equities and aggregate bonds); only the equity half has
    an equivalent here, so the canary reduces to the benchmark scored with the
    paper's 13612W momentum - the weighted blend of 1-, 3-, 6- and 12-month
    returns. A single-asset canary fires less often than a two-asset one, so
    this is the more permissive of the two market-level gates.
"""

from __future__ import annotations

import pandas as pd

from screener.risk_free import annualized_rate, compounded_hurdle
from screener.strategies.cross_section import attach_column, close_panel, positive_share
from screener.strategies.plugins.momentum_12_1 import momentum_12_1_score
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_YEAR = 252
_MONTH = 21
# 13612W weights: 12 * r1 + 4 * r3 + 2 * r6 + 1 * r12, all annualized in the
# paper by construction of the multipliers.
_W13612 = ((1, 12.0), (3, 4.0), (6, 2.0), (12, 1.0))
# PAA protection factor a=2: full cash once half the universe is negative.
PAA_BREADTH_FLOOR = 0.5

ENTRY_GEM = "mom_12_1 > rf_hurdle"
ENTRY_MARKET = "mom_12_1 > 0 and market_excess_mom > 0"
ENTRY_PAA = f"mom_12_1 > 0 and paa_breadth > {PAA_BREADTH_FLOOR}"
ENTRY_DAA = "mom_12_1 > 0 and canary_risk_on"

# Each exit is the negation of its gate. Antonacci's monthly decision moves the
# allocation *out* of equities when the absolute-momentum test fails, and PAA and
# DAA raise their cash weight the same way, so a gate that only blocked new
# entries would not be the published strategy: it would leave the portfolio fully
# invested through exactly the decline the gate exists to avoid.
EXIT_GEM = "mom_12_1 <= rf_hurdle"
EXIT_MARKET = "market_excess_mom <= 0"
EXIT_PAA = f"paa_breadth <= {PAA_BREADTH_FLOOR}"
EXIT_DAA = "not canary_risk_on"


def momentum_13612w(close: pd.Series) -> pd.Series:
    """Keller & Keuning's 13612W momentum: weighted 1/3/6/12-month returns."""
    close = close.astype(float)
    total = pd.Series(0.0, index=close.index, dtype=float)
    # A leg without enough history is NaN, which propagates: the blend is
    # undefined until every horizon is available, rather than counting a missing
    # leg as a zero contribution.
    for months, weight in _W13612:
        total = total + weight * (close / close.shift(months * _MONTH) - 1.0)
    return total


def sma_momentum(close: pd.Series, window: int = _YEAR) -> pd.Series:
    """PAA's momentum: price relative to its own 12-month simple average."""
    close = close.astype(float)
    average = close.rolling(window, min_periods=window).mean()
    return close / average - 1.0


def _base_frames(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach 12-1 momentum and rank by it, the relative-momentum leg."""
    prepared: dict[str, pd.DataFrame] = {}
    for symbol, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            prepared[symbol] = bars
            continue
        frame = bars.copy()
        mom = momentum_12_1_score(frame["close"])
        frame["mom_12_1"] = mom
        frame["rank_score"] = mom
        prepared[symbol] = frame
    return prepared


def _panel_index(ctx: PrepareCtx) -> pd.DatetimeIndex:
    closes = close_panel(ctx.bars_by_tv)
    if not closes.empty:
        return pd.DatetimeIndex(closes.index)
    return pd.DatetimeIndex([])


def _benchmark_close(ctx: PrepareCtx, strategy: str) -> pd.Series:
    bars = ctx.price_panel.get(ctx.benchmark)
    if bars is None or bars.empty:
        ctx.warnings.append(
            f"benchmark data unavailable for {strategy}: {ctx.benchmark}"
        )
        return pd.Series(dtype=float)
    return bars["close"].astype(float)


def _prepare_gem(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    prepared = _base_frames(ctx)
    index = _panel_index(ctx)
    rate = annualized_rate(ctx.market, index, ctx.fetcher, ctx.start, ctx.end)
    return attach_column(prepared, compounded_hurdle(rate), "rf_hurdle", 0.0)


def _prepare_market(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    prepared = _base_frames(ctx)
    close = _benchmark_close(ctx, "dual_momentum_market")
    if close.empty:
        # No benchmark means no defensible risk-off call; stay flat rather than
        # silently degrading to plain momentum.
        return attach_column(
            prepared, pd.Series(dtype=float), "market_excess_mom", -1.0
        )
    index = pd.DatetimeIndex(close.index)
    hurdle = compounded_hurdle(
        annualized_rate(ctx.market, index, ctx.fetcher, ctx.start, ctx.end)
    )
    market_return = close / close.shift(_YEAR) - 1.0
    excess = market_return - hurdle
    return attach_column(prepared, excess, "market_excess_mom", -1.0)


def _prepare_paa(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    prepared = _base_frames(ctx)
    closes = close_panel(ctx.bars_by_tv)
    if closes.empty:
        breadth = pd.Series(dtype=float)
    else:
        breadth = positive_share(closes.apply(sma_momentum))
    return attach_column(prepared, breadth, "paa_breadth", 0.0)


def _prepare_daa(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    prepared = _base_frames(ctx)
    close = _benchmark_close(ctx, "dual_momentum_daa")
    canary = (
        pd.Series(dtype=bool)
        if close.empty
        else (momentum_13612w(close) > 0).fillna(False)
    )
    return attach_column(prepared, canary, "canary_risk_on", False)


def _year_lookback() -> int:
    return _YEAR


register_expression_strategy(
    "dual_momentum_gem",
    entry=ENTRY_GEM,
    exit=EXIT_GEM,
    prepare_bars=_prepare_gem,
    required_lookback=_year_lookback,
)

register_expression_strategy(
    "dual_momentum_market",
    entry=ENTRY_MARKET,
    exit=EXIT_MARKET,
    prepare_bars=_prepare_market,
    required_lookback=_year_lookback,
)

register_expression_strategy(
    "dual_momentum_paa",
    entry=ENTRY_PAA,
    exit=EXIT_PAA,
    prepare_bars=_prepare_paa,
    required_lookback=_year_lookback,
)

register_expression_strategy(
    "dual_momentum_daa",
    entry=ENTRY_DAA,
    exit=EXIT_DAA,
    prepare_bars=_prepare_daa,
    required_lookback=_year_lookback,
)


__all__ = ["PAA_BREADTH_FLOOR", "momentum_13612w", "sma_momentum"]
