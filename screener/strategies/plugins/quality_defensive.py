"""Quality & defensive factor strategies (non-momentum).

The family brief asks for NON-momentum strategies with genuine evidence that
can match the momentum benchmarks (US ``nifty_momentum`` ~1.3-1.6 Sharpe;
India ``momentum_quality_pe60`` 0.54/0.88/1.81/1.15 across 1/2/3/5y).

This module implements the official "quality index" recipe as a *gate* plus a
defensive *ranking*, because the backtester merges fundamental columns AFTER
``prepare_bars`` (see ``screener.backtester.price_panel.build_price_panel``),
so ``rank_score`` can only use price-derived columns while fundamentals
(``roe_ttm``, ``debt_to_equity``, ``eps_growth_yoy``, ...) must live in the
entry/exit expressions. The construction therefore mirrors how NSE/MSCI build
their quality indices: a quality screen first (ROE, leverage, earnings
growth), then ordering by a defensive attribute — never by price momentum.

Quality gate (moderate, India Nifty 500 friendly — mid/small caps survive):

    roe_ttm >= 10  and  debt_to_equity <= 2.5  and  eps_growth_yoy > 0

Sources:

* NSE Indices, "Nifty100 / Nifty200 Quality 30" methodology — quality score of
  return on equity (high), financial leverage (low) and EPS growth stability
  over 5 years; rebalanced semi-annually. The ROE / leverage / growth triple
  here is the same screen with the backtester's dated fundamentals.
* NSE Indices, "Nifty Alpha Quality Low-Volatility 30" (AQLV) — composite of
  alpha, quality and low-volatility scores. ``quality_lowvol`` is the AQLV
  recipe with the alpha leg removed, i.e. the pure quality+low-vol defensive
  blend the index family documents.
* Ang, Hodrick, Xing & Zhang (2006), "The Cross-Section of Volatility and
  Expected Returns", J. Finance 61(1) — high-idiosyncratic-vol stocks earn
  abnormally low returns; low-vol stocks earn a risk-adjusted premium.
* Blitz & van Vliet (2007), "The Volatility Effect", J. Portfolio Mgmt — the
  volatility effect is strongest in the least-volatile decile; it is a
  persistent, cross-country anomaly (emerging markets included).
* Black, Jensen & Scholes (1972) and Frazzini & Pedersen (2014), "Betting
  Against Beta", J. Financial Economics 111(1) — the low-beta anomaly: high
  beta underperforms its CAPM promise, so leveraged long-low-beta portfolios
  earn abnormal returns; BAB is profitable in every equity market tested.
* "Does profitability explain the low-risk anomaly in India?" (2025) — the
  low-volatility / low-beta premium is present in India and is NOT fully
  explained by profitability, i.e. a quality x low-risk combination should not
  be redundant (the two legs contribute separately).
* Asness, Frazzini & Pedersen (2019), "Quality Minus Junk", Rev. Accounting
  Studies 24(1) — "quality" (profitable, growing, safe) earns a positive
  premium in 24 countries; quality and value are complementary, which is the
  basis of the ``quality_value`` GARP variant.
* Novy-Marx (2013), "The Other Side of Value", J. Financial Economics 108(1) —
  profitability premium; Fama & French (2015) five-factor model adds
  profitability (RMW) and investment as priced factors.
* Repo's own finding: ``momentum_quality_pe60`` (quality gates + PE<=60 cap)
  is the best India recipe to date (Sharpe 1.81 / 1.15 at 3y/5y, positive in
  all four windows) — valuation-capping a quality screen helped, which is the
  hypothesis ``quality_value`` isolates here without any momentum leg.

Strategies (all long-only, quality-gated, defensive-ranked):

``quality_lowvol``
    Quality gate + rank by lowest trailing 252-day realized volatility
    (AQLV-minus-alpha / volatility effect among profitable low-leverage
    growers). Defensive, low-turnover core holding.
``quality_lowbeta``
    Quality gate + rank by lowest trailing 252-day beta vs the benchmark
    (Betting Against Beta among quality names; India low-risk anomaly).
``quality_stability``
    Quality gate + 3-consecutive-quarter revenue growth (revenue_up_3q, the
    closest available proxy for Nifty Quality 30's EPS-growth-stability
    criterion) + price within 40% of its 52-week high (dd_252 > -0.40) +
    rank by lowest downside deviation (Sortino-style; Ang, Chen & Xing 2006
    "Downside Risk" — low downside-risk stocks are defensive).
``quality_value``
    Quality gate + valuation caps (pe_ttm <= 40 and pb_ttm <= 8) + rank by
    lowest realized volatility: GARP-style defensive value (QMJ quality x
    value complementarity; the repo's PE-capped momentum-quality result).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_VOL_WINDOW = 252  # realized-vol / beta estimation window (~12 months)
_BETA_WINDOW = 252
_DD_WINDOW = 252  # 52-week high drawdown window
_DOWNSIDE_WINDOW = 126  # 6-month downside deviation
_DD_MAX = -0.40  # keep names within 40% of their 52-week high
_PE_MAX = 40.0
_PB_MAX = 8.0

#: Moderate quality screen shared by all four strategies (India-friendly).
QUALITY_GATE = "roe_ttm >= 10 and debt_to_equity <= 2.5 and eps_growth_yoy > 0"

ENTRY_LOWVOL = f"{QUALITY_GATE} and vol_252 > 0"
ENTRY_LOWBETA = f"{QUALITY_GATE} and beta_252 > 0"
ENTRY_STABILITY = (
    f"{QUALITY_GATE} and revenue_up_3q == 1 and dd_252 > {_DD_MAX} and dd_126 > 0"
)
ENTRY_VALUE = (
    f"{QUALITY_GATE} and pe_ttm <= {_PE_MAX} and pb_ttm <= {_PB_MAX} and vol_252 > 0"
)


def rolling_beta(
    close: pd.Series, benchmark_close: pd.Series, window: int = _BETA_WINDOW
) -> pd.Series:
    """Trailing-``window`` OLS beta of ``close`` vs ``benchmark_close``.

    Causal: the beta at bar ``t`` uses daily returns over bars ``<= t`` only.
    ``benchmark_close`` must already be aligned/reindexed to ``close.index``
    (the caller does ``reindex(...).ffill()``); both series are assumed to
    share the same DatetimeIndex.
    """
    stock_ret = close.astype(float).pct_change()
    bench_ret = benchmark_close.astype(float).pct_change()
    cov = stock_ret.rolling(window, min_periods=window).cov(bench_ret)
    var = bench_ret.rolling(window, min_periods=window).var()
    return pd.Series(cov / var, index=close.index)


def downside_deviation(close: pd.Series, window: int = _DOWNSIDE_WINDOW) -> pd.Series:
    """Trailing-``window`` downside deviation (sqrt mean of squared losses)."""
    ret = close.astype(float).pct_change()
    neg = ret.where(ret < 0, 0.0)
    values = np.sqrt((neg**2).rolling(window, min_periods=window).mean())
    return pd.Series(values, index=close.index)


def drawdown_from_high(close: pd.Series, window: int = _DD_WINDOW) -> pd.Series:
    """Trailing-``window`` drawdown: close / rolling_max - 1 (<= 0)."""
    return (
        close.astype(float)
        / close.astype(float).rolling(window, min_periods=window).max()
        - 1.0
    )


def _prepare_quality_defensive(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Compute the defensive indicator columns; no rank_score yet."""
    benchmark_bars = ctx.price_panel.get(ctx.benchmark, pd.DataFrame())
    benchmark_close = (
        benchmark_bars["close"]
        if benchmark_bars is not None and not benchmark_bars.empty
        else None
    )
    if benchmark_close is None:
        ctx.warnings.append(
            f"benchmark data unavailable for quality beta column: {ctx.benchmark}"
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
            frame["beta_252"] = rolling_beta(close, aligned)
        else:
            frame["beta_252"] = np.nan
        frame["dd_252"] = drawdown_from_high(close)
        frame["dd_126"] = downside_deviation(close)
        out[tv] = frame
    return out


def _rank_by(frames: dict[str, pd.DataFrame], column: str) -> dict[str, pd.DataFrame]:
    """Set ``rank_score = -column`` so the backtester's descending ranker picks
    the LOWEST values (calmest / least risky) first."""
    for tv, frame in frames.items():
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        frame["rank_score"] = -frame[column].astype(float)
        frames[tv] = frame
    return frames


def _prepare_lowvol(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return _rank_by(_prepare_quality_defensive(ctx), "vol_252")


def _prepare_lowbeta(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return _rank_by(_prepare_quality_defensive(ctx), "beta_252")


def _prepare_stability(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return _rank_by(_prepare_quality_defensive(ctx), "dd_126")


def _prepare_value(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return _rank_by(_prepare_quality_defensive(ctx), "vol_252")


def _lookback() -> int:
    # pct_change consumes one bar, then the longest rolling window is 252.
    return max(_VOL_WINDOW, _BETA_WINDOW, _DD_WINDOW) + 1


register_expression_strategy(
    "quality_lowvol",
    entry=ENTRY_LOWVOL,
    exit=None,
    prepare_bars=_prepare_lowvol,
    required_lookback=_lookback,
)

register_expression_strategy(
    "quality_lowbeta",
    entry=ENTRY_LOWBETA,
    exit=None,
    prepare_bars=_prepare_lowbeta,
    required_lookback=_lookback,
)

register_expression_strategy(
    "quality_stability",
    entry=ENTRY_STABILITY,
    exit=None,
    prepare_bars=_prepare_stability,
    required_lookback=_lookback,
)

register_expression_strategy(
    "quality_value",
    entry=ENTRY_VALUE,
    exit=None,
    prepare_bars=_prepare_value,
    required_lookback=_lookback,
)
