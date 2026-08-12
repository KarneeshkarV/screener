"""Earnings momentum, PEAD and cash-flow yield strategies.

Methodology sources (see README of this family for the full literature map):

- ``pead_drift``        — Bernard & Thomas (1989), "Post-Earnings-Announcement
                          Drift: Delayed Price Response or Risk Premium?",
                          Journal of Accounting Research 27 (Supplement).
                          Stock prices keep drifting in the direction of
                          quarterly earnings surprises for up to ~6 months
                          because investors under-react to the predictable
                          serial correlation in quarterly earnings. The data
                          here carry no analyst estimates, so the surprise is
                          proxied by *realized* growth: ``revenue_up_3q``
                          (revenue grew three consecutive quarters) plus
                          positive YoY EPS growth, both known only with a
                          filing lag (point-in-time merge). The price drift
                          that follows is harvested by ranking the passing
                          names on trailing 6-month price momentum.
- ``earnings_momentum``  — Chan, Jegadeesh & Lakonishok (1996), "Momentum
                          Strategies", Journal of Finance 51(5): combining past
                          6-12 month price momentum with earnings momentum
                          (SUE / analyst revisions) is stronger than either
                          signal alone; earnings momentum adds information
                          beyond price momentum. Also Novy-Marx (2015),
                          "Fundamentally, Momentum is Fundamental Momentum",
                          NBER WP 20905: *changes in fundamentals* (sales
                          growth, margins) forecast returns independently of
                          price momentum. Proxy here: positive EPS growth AND
                          positive revenue growth (both lines expanding) with
                          a profitability floor, ranked by volatility-adjusted
                          6-month price momentum (the CJL price leg).
- ``fcf_yield_value``    — Lakonishok, Shleifer & Vishny (1994), "Contrarian
                          Investment, Extrapolation and Risk", Journal of
                          Finance 49(5): cash-flow yield (CF/price) is a
                          superior value signal to E/P or B/P — high CF/P
                          names beat low CF/P names because investors
                          extrapolate past growth into the future. Long-only
                          screen: free-cash-flow yield above ~3% (fraction),
                          profitable, moderate P/E and low leverage. Selection
                          is by dollar volume (no ``rank_score``), the same
                          convention as ``value_rank``.
- ``qmj_quality``        — Asness, Frazzini & Pedersen (2019), "Quality Minus
                          Junk", Review of Accounting Studies 24(1): quality =
                          profitability, growth, safety and payout; high-
                          quality stocks earn a positive premium in 24
                          countries and QMJ is complementary to value and
                          momentum. Fundamentals merge after ``prepare_bars``,
                          so the QMJ composite cannot feed ``rank_score``;
                          this strategy therefore gates on the fundamental
                          composite (ROE, gross margin, leverage, EPS growth)
                          plus a price-side safety gate (beta < 1) in the
                          entry expression and ranks the passing names by
                          lowest trailing realized volatility. Payout
                          (``dividend_yield_ttm``) is one of the four QMJ
                          legs; it is deliberately NOT required here so
                          non-dividend growers stay eligible (an optional
                          tightening is documented below).

All four are market-agnostic expressions (FMP serves both US and India .NS
symbols); run with ``--fundamentals-provider fmp``. Fundamentals arrive with a
filing lag (defaults: fmp=1 day, openscreener=60 days) so entry signals are
point-in-time. NaN fundamentals fail their gates closed (missing data excludes
a name), which is the desired behavior.

Recommended backtest config (parent runs these):

    pead_drift         --top 10 --hold 63   fields: revenue_up_3q eps_growth_yoy roe_ttm
    earnings_momentum  --top 10 --hold 126  fields: eps_growth_yoy revenue_growth_yoy roe_ttm
    fcf_yield_value    --top 10 --hold 126  fields: fcf_yield pe_ttm roe_ttm debt_to_equity
    qmj_quality        --top 10 --hold 126  fields: roe_ttm gross_margin_ttm debt_to_equity eps_growth_yoy

Only ``fcf_yield`` and ``gross_margin_ttm`` are non-default fundamental fields
(see ``screener.backtester.fundamentals.DEFAULT_FUNDAMENTAL_FIELDS``); all other
fields above are fetched by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.plugins.quality_defensive import rolling_beta
from screener.strategies.spec import PrepareCtx, register_expression_strategy

# ── Shared windows ───────────────────────────────────────────────────────────
_MOM_WINDOW = 126  # ~6 months of trading days (CJL price-momentum leg / drift window)
_VOL_WINDOW = 252  # realized-vol / beta estimation window (~12 months)
_BETA_WINDOW = 252

# ── Shared, moderate gates (Nifty500 + SP500 friendly) ───────────────────────
_ROE_FLOOR = 10.0  # earnings-momentum profitability floor (percent)
_FCF_YIELD_MIN = 0.03  # LSV cash-flow yield lower bound (fraction)
_FCF_YIELD_MAX = 0.5  # sanity cap: >50% FCF yield is data noise, not value
_PE_MAX = 40.0
_DEBT_MAX = 2.0
_QMJ_ROE_MIN = 15.0  # QMJ profitability leg (percent)
_QMJ_GROSS_MARGIN_MIN = 20.0  # QMJ profitability leg (percent)
_QMJ_DEBT_MAX = 1.0  # QMJ safety leg (leverage)
_QMJ_BETA_MAX = 1.0  # QMJ safety leg (market beta)

# PEAD: 3 consecutive quarters of revenue growth (sustained positive surprise)
# + positive YoY EPS growth + profitable, with the price already drifting up.
PEAD_ENTRY = "revenue_up_3q == 1 and eps_growth_yoy > 0 and roe_ttm > 0 and mom_126 > 0"

# Earnings momentum: both lines expanding, profitable, price drifting up.
# (Novy-Marx fundamental momentum: the *change* in fundamentals — here realized
# top-line + bottom-line growth — carries information beyond price momentum.)
EARNINGS_MOM_ENTRY = (
    f"eps_growth_yoy > 0 and revenue_growth_yoy > 0 and roe_ttm >= {_ROE_FLOOR} "
    "and mom_126 > 0 and vol_252 > 0"
)

# LSV cash-flow yield: high FCF yield, profitable, moderate multiple, low
# leverage. No trend filter — contrarian value buys cheap names as they are.
FCF_YIELD_ENTRY = (
    f"fcf_yield >= {_FCF_YIELD_MIN} and fcf_yield <= {_FCF_YIELD_MAX} "
    f"and roe_ttm > 0 and pe_ttm > 0 and pe_ttm <= {_PE_MAX} "
    f"and debt_to_equity <= {_DEBT_MAX}"
)

# QMJ composite gate: profitability (ROE + gross margin), growth (EPS), safety
# (leverage + beta < 1). Payout is documented but optional — requiring
# ``dividend_yield_ttm > 0`` here would exclude every non-dividend grower.
QMJ_ENTRY = (
    f"roe_ttm >= {_QMJ_ROE_MIN} and gross_margin_ttm >= {_QMJ_GROSS_MARGIN_MIN} "
    f"and debt_to_equity <= {_QMJ_DEBT_MAX} and eps_growth_yoy > 0 "
    f"and beta_252 < {_QMJ_BETA_MAX} and vol_252 > 0"
)


def _momentum_6m(close: pd.Series) -> pd.Series:
    """Causal trailing-6-month return: ``close[t] / close[t-126] - 1``."""
    close = close.astype(float)
    return close / close.shift(_MOM_WINDOW) - 1.0


def _prepare_drift(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach the 6-month return and rank by it (the PEAD price drift)."""
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        frame["mom_126"] = _momentum_6m(frame["close"])
        frame["rank_score"] = frame["mom_126"]
        out[tv] = frame
    return out


def _prepare_earnings_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach 6-month return, realized vol and rank by momentum/vol.

    Volatility-adjusted momentum (Sharpe-like) demotes the crashy high-vol
    winners; the entry gate ``vol_252 > 0`` keeps ranking well-defined.
    """
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["mom_126"] = _momentum_6m(close)
        frame["vol_252"] = realized_volatility(close)
        score = frame["mom_126"] / frame["vol_252"]
        frame["rank_score"] = score.where(frame["vol_252"] > 0)
        out[tv] = frame
    return out


def _prepare_qmj(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach realized vol + benchmark beta and rank by lowest volatility.

    ``beta_252`` is the price-side safety gate used by the entry expression;
    ``rank_score = -vol_252`` makes the rolling backtester fill ``--top``
    slots with the calmest names among the quality-eligible ones.
    """
    benchmark_bars = ctx.price_panel.get(ctx.benchmark, pd.DataFrame())
    benchmark_close = (
        benchmark_bars["close"]
        if benchmark_bars is not None and not benchmark_bars.empty
        else None
    )
    if benchmark_close is None:
        ctx.warnings.append(
            f"benchmark data unavailable for qmj beta column: {ctx.benchmark}"
        )

    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["vol_252"] = realized_volatility(close)
        if benchmark_close is not None:
            aligned = benchmark_close.reindex(frame.index).ffill()
            frame["beta_252"] = rolling_beta(close, aligned, window=_BETA_WINDOW)
        else:
            frame["beta_252"] = np.nan
        frame["rank_score"] = -frame["vol_252"]
        out[tv] = frame
    return out


def _lookback_momentum_only() -> int:
    # 6-month return leg needs 126 prior closes; nothing longer.
    return _MOM_WINDOW


def _lookback_vol() -> int:
    # pct_change consumes one bar, then the rolling std needs ``_VOL_WINDOW``
    # returns (same convention as ``low_volatility`` / ``quality_defensive``).
    return _VOL_WINDOW + 1


def _lookback_basic() -> int:
    # No rolling price windows; just enough history for the merge + entry eval.
    return 20


register_expression_strategy(
    "pead_drift",
    entry=PEAD_ENTRY,
    exit=None,
    prepare_bars=_prepare_drift,
    required_lookback=_lookback_momentum_only,
)

register_expression_strategy(
    "earnings_momentum",
    entry=EARNINGS_MOM_ENTRY,
    exit=None,
    prepare_bars=_prepare_earnings_momentum,
    required_lookback=_lookback_vol,
)

register_expression_strategy(
    "fcf_yield_value",
    entry=FCF_YIELD_ENTRY,
    exit=None,
    prepare_bars=None,
    required_lookback=_lookback_basic,
)

register_expression_strategy(
    "qmj_quality",
    entry=QMJ_ENTRY,
    exit=None,
    prepare_bars=_prepare_qmj,
    required_lookback=_lookback_vol,
)
