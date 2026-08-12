"""Volatility, beta & downside-risk factor strategies (defensive long-only).

Methodology sources:

- ``low_idio_vol``  — Ang, Hodrick, Xing & Zhang (2006), "The Cross-Section of
                      Volatility and Expected Returns", Journal of Finance
                      61(1), 259-299: stocks with high *idiosyncratic*
                      volatility earn abnormally low subsequent returns (the
                      IVOL anomaly; strongest among lottery-like names). Long-
                      only proxy: rank by LOW trailing 252-day residual
                      volatility from a 1-factor regression on the benchmark.
- ``betting_against_beta`` — Frazzini & Pedersen (2014), "Betting Against
                      Beta", Journal of Financial Economics 111(1), 1-25:
                      high-beta assets earn lower risk-adjusted returns than
                      their CAPM promise; low-beta assets are the long leg of
                      the BAB factor. Long-only proxy: rank by LOW trailing
                      252-day *price* beta vs the benchmark (no fundamentals —
                      unlike round-1 ``quality_lowbeta``).
- ``downside_risk``  — Ang, Chen & Xing (2006), "Downside Risk", Review of
                      Financial Studies 19(4), 1191-1239: downside beta (the
                      market beta estimated only on down-market days) is priced
                      in the cross-section; high-downside-beta names carry the
                      lottery / drawdown tail. Long-only defensive proxy: rank
                      by LOW downside beta, i.e. least sensitive to market
                      drawdowns.
- ``max_avoidance``  — Bali, Cakici & Whitelaw (2011), "Maxing Out: Stocks as
                      Lotteries", Journal of Financial Economics 99(2),
                      427-446: stocks with extreme positive daily returns
                      (high MAX) earn LOW subsequent returns (investor
                      preference for lottery payoffs). Long-only proxy: a
                      moderate value/quality gate plus ranking by LOW
                      trailing-21-day maximum daily return — "cheap and
                      boring" (value x lottery-avoidance).
- Sizing lever       — Moreira & Muir (2017), "Volatility-Managed Portfolios",
                      Journal of Finance 72(4), 1611-1644: scaling exposure by
                      inverse realized volatility raises Sharpe ratios and
                      alphas. The repo ships this as ``--sizing inverse_vol``
                      with ``--sizing-risk-pct``; pair it with any strategy
                      below instead of hand-rolling a vol-scaled variant.

Construction notes
------------------
All four are long-only cross-sectional factor portfolios. ``prepare_bars``
computes each risk statistic causally per ticker (bar ``t`` uses only data
``<= t``) and emits ``rank_score = -statistic``, so the rolling backtester
fills its ``--top`` slots with the LOWEST-risk names. Entry expressions are
eligibility gates only (statistic defined and sane); ranking does the work —
the same design as round-1 ``low_volatility`` / ``quality_lowbeta``. The first
three strategies need only price data (the benchmark close comes from
``ctx.price_panel``); ``max_avoidance`` additionally gates on dated
fundamentals (run with ``--fundamentals-provider fmp``). Missing statistics or
missing fundamentals fail the gate (NaN comparisons are False), excluding
thin-history and uncovered names by construction.

These are deliberately defensive: they should hold up in India's flat/bear
1y/2y windows where momentum dies, so no trend gates are applied (a 200-day
SMA filter would defeat the purpose). Benchmarks: India ``^NSEI`` / US ``SPY``
are read from the panel; if unavailable the beta columns stay NaN and the
strategies simply produce no candidates (a warning is recorded).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

# ── Estimation windows ──────────────────────────────────────────────────────
_WINDOW = 252  # beta / IVOL / downside-beta estimation window (~12 months)
_MIN_DOWN_DAYS = 60  # min down-market days inside the window for downside beta
_MAX_WINDOW = 21  # BCW "MAX" window (~1 month of trading days)
_BETA_CAP = 2.0  # sanity cap on estimated betas (excludes data errors)

# ── Value/quality gate for ``max_avoidance`` (moderate, Nifty500-friendly) ──
_PE_MAX = 30.0
_PB_MAX = 6.0
_ROE_MIN = 8.0
_DEBT_MAX = 3.0

#: Eligibility gates. Each is satisfied only when the risk statistic is defined
#: (NaN comparisons are False) and within a sane range; cross-sectional ranking
#: via ``rank_score`` does the actual selection.
ENTRY_IDIO_VOL = "ivol_252 > 0"
ENTRY_BAB = f"beta_252 > 0 and beta_252 < {_BETA_CAP}"
ENTRY_DOWNSIDE = f"dbeta_252 > 0 and dbeta_252 < {_BETA_CAP}"
MAX_VALUE_GATE = (
    f"pe_ttm > 0 and pe_ttm <= {_PE_MAX} and pb_ttm > 0 and pb_ttm <= {_PB_MAX} "
    f"and roe_ttm >= {_ROE_MIN} and debt_to_equity <= {_DEBT_MAX}"
)
ENTRY_MAX = f"{MAX_VALUE_GATE} and max_21 > 0"


def _benchmark_close(ctx: PrepareCtx) -> pd.Series | None:
    """Benchmark close series from the panel, or None (with a warning) if absent."""
    benchmark_bars = ctx.price_panel.get(ctx.benchmark, pd.DataFrame())
    if benchmark_bars is None or benchmark_bars.empty:
        ctx.warnings.append(
            f"benchmark data unavailable for beta_volatility columns: {ctx.benchmark}"
        )
        return None
    return benchmark_bars["close"].astype(float)


def _finite(series: pd.Series) -> pd.Series:
    """Replace inf/nan produced by degenerate windows with NaN (→ gate fails)."""
    return series.where(np.isfinite(series))


def rolling_beta(
    close: pd.Series, benchmark_close: pd.Series, window: int = _WINDOW
) -> pd.Series:
    """Trailing-``window`` OLS beta of ``close`` vs ``benchmark_close`` (causal)."""
    stock_ret = close.astype(float).pct_change()
    bench_ret = benchmark_close.astype(float).pct_change()
    cov = stock_ret.rolling(window, min_periods=window).cov(bench_ret)
    var = bench_ret.rolling(window, min_periods=window).var()
    return _finite(cov / var)


def rolling_ivol(
    close: pd.Series, benchmark_close: pd.Series, window: int = _WINDOW
) -> pd.Series:
    """Trailing-``window`` idiosyncratic vol: stdev of 1-factor residuals.

    Computed via the exact OLS identity ``Var(resid) = Var(y) - Cov(y,x)^2 /
    Var(x)`` on the trailing window, so the column is defined at the same bar
    as ``rolling_beta`` (252 returns) instead of 252 bars later. All three
    rolling moments are causal (bar ``t`` uses returns ``<= t`` only).
    """
    stock_ret = close.astype(float).pct_change()
    bench_ret = benchmark_close.astype(float).pct_change()
    var_i = stock_ret.rolling(window, min_periods=window).var()
    var_m = bench_ret.rolling(window, min_periods=window).var()
    cov = stock_ret.rolling(window, min_periods=window).cov(bench_ret)
    resid_var = var_i - cov**2 / var_m
    # clip guards tiny negative values from floating-point cancellation.
    return _finite(resid_var.clip(lower=0.0).pow(0.5))


def rolling_downside_beta(
    close: pd.Series,
    benchmark_close: pd.Series,
    window: int = _WINDOW,
    min_down_days: int = _MIN_DOWN_DAYS,
) -> pd.Series:
    """Trailing-``window`` downside beta: market beta on down-market days only.

    Vectorised causal estimate: within each trailing window, restrict daily
    returns to days where the benchmark return is negative and compute
    ``Cov(stock, market | market<0) / Var(market | market<0)`` via rolling
    sums. The ``(n-1)`` sample denominators cancel in the ratio, so the
    raw-sum form is exact. NaN until at least ``min_down_days`` down days
    exist inside the window (~100 of 252 typical, so ~40% of the window).
    """
    stock_ret = close.astype(float).pct_change()
    bench_ret = benchmark_close.astype(float).pct_change()
    down = bench_ret < 0
    stock_down = stock_ret.where(down)
    bench_down = bench_ret.where(down)
    count = down.astype(float).rolling(window, min_periods=min_down_days).sum()
    sum_s = stock_down.rolling(window, min_periods=min_down_days).sum()
    sum_b = bench_down.rolling(window, min_periods=min_down_days).sum()
    sum_bb = (bench_down * bench_down).rolling(window, min_periods=min_down_days).sum()
    sum_sb = (stock_down * bench_down).rolling(window, min_periods=min_down_days).sum()
    cov = (sum_sb - sum_s * sum_b / count) / (count - 1)
    var = (sum_bb - sum_b * sum_b / count) / (count - 1)
    return _finite(cov / var)


def max_daily_return(close: pd.Series, window: int = _MAX_WINDOW) -> pd.Series:
    """Trailing-``window`` maximum daily return (BCW 2011 MAX)."""
    returns = close.astype(float).pct_change()
    return returns.rolling(window, min_periods=window).max()


def _prepare_benchmark_factor(
    ctx: PrepareCtx,
    column: str,
    factor_fn: Callable[[pd.Series, pd.Series], pd.Series],
) -> dict[str, pd.DataFrame]:
    """Compute a benchmark-relative risk factor and rank by its LOW values.

    Emits ``column`` plus ``rank_score = -column`` so the backtester's
    descending ranker picks the lowest-risk names first. When the benchmark is
    unavailable the column is all-NaN and every name fails the entry gate.
    """
    benchmark_close = _benchmark_close(ctx)
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        if benchmark_close is None:
            frame[column] = np.nan
        else:
            aligned = benchmark_close.reindex(frame.index).ffill()
            frame[column] = factor_fn(frame["close"], aligned)
        frame["rank_score"] = -frame[column].astype(float)
        out[tv] = frame
    return out


def _prepare_idio_vol(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return _prepare_benchmark_factor(ctx, "ivol_252", rolling_ivol)


def _prepare_bab(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return _prepare_benchmark_factor(ctx, "beta_252", rolling_beta)


def _prepare_downside(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return _prepare_benchmark_factor(ctx, "dbeta_252", rolling_downside_beta)


def _prepare_max(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Emit trailing-21-day MAX and rank by its LOW values (no benchmark needed)."""
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        frame["max_21"] = max_daily_return(frame["close"])
        frame["rank_score"] = -frame["max_21"].astype(float)
        out[tv] = frame
    return out


def _lookback_252() -> int:
    # pct_change consumes one bar, then the 252-bar rolling window.
    return _WINDOW + 1


def _lookback_max() -> int:
    # pct_change consumes one bar, then the 21-bar MAX window.
    return _MAX_WINDOW + 1


register_expression_strategy(
    "low_idio_vol",
    entry=ENTRY_IDIO_VOL,
    exit=None,
    prepare_bars=_prepare_idio_vol,
    required_lookback=_lookback_252,
)

register_expression_strategy(
    "betting_against_beta",
    entry=ENTRY_BAB,
    exit=None,
    prepare_bars=_prepare_bab,
    required_lookback=_lookback_252,
)

register_expression_strategy(
    "downside_risk",
    entry=ENTRY_DOWNSIDE,
    exit=None,
    prepare_bars=_prepare_downside,
    required_lookback=_lookback_252,
)

register_expression_strategy(
    "max_avoidance",
    entry=ENTRY_MAX,
    exit=None,
    prepare_bars=_prepare_max,
    required_lookback=_lookback_max,
)
