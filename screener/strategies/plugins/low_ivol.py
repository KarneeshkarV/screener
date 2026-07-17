"""Ang-Hodrick-Xing-Zhang (2006) idiosyncratic-volatility anomaly.

Paper: Ang, Hodrick, Xing & Zhang, "The Cross-Section of Volatility and Expected
Returns", Journal of Finance 61(1), 2006. Stocks with *low idiosyncratic*
volatility earn higher risk-adjusted returns than high-IVOL stocks — an anomaly
also documented on the NSE. This is distinct from ``low_volatility`` (which uses
*total* return volatility) and ``low_beta`` (which uses the market beta): IVOL is
the volatility of the market-model *residual*, i.e. the return variation left
after stripping out the benchmark-explained (systematic) component.

Signal (causal, as-of bar ``t``, over a trailing ``W=252`` daily-return window):

    r_i = close.pct_change()                 # symbol daily returns
    r_m = bench_close.pct_change()           # benchmark daily returns (aligned)
    beta[t]      = rolling_cov(r_i, r_m, W) / rolling_var(r_m, W)
    resid_var[t] = rolling_var(r_i, W) - beta[t]^2 * rolling_var(r_m, W)
    ivol[t]      = sqrt( max(resid_var[t], 0) ) * sqrt(252)

``resid_var`` is the OLS residual-variance identity for the window regression of
``r_i`` on ``r_m``: ``Var(r_i) - beta^2 * Var(r_m)`` equals the mean-squared
residual, so no per-row regression loop and no explicit ``alpha`` term is needed
(the intercept cancels out of the residual variance). All moments are rolling
over ``W`` with pandas-default ``ddof=1``; tiny negative ``resid_var`` values
from float error are clipped to 0 before the square root. IVOL is annualized by
``sqrt(252)``.

Approximation note: Ang et al.'s original measures IVOL as the std of the daily
residuals from a *Fama-French 3-factor* regression estimated over ~1 month. Here
we approximate with a single-factor market model (the benchmark) over 252 days —
chosen for stability and because Fama-French factors are not available for India
in-repo. The single-factor residual is a well-established proxy for total IVOL.

Selection: ``rank_score = -ivol`` so the descending ranker fills its ``--top``
slots with the *lowest*-IVOL names first. The entry gate ``ivol > 0`` keeps only
names with a defined, positive idiosyncratic vol eligible (a "has enough history"
gate; residual vol is positive on non-degenerate data).

Benchmark acquisition reuses ``low_beta._benchmark_closes`` (panel -> universe ->
fetch ladder). If the benchmark cannot be located, a warning is recorded and
every name's ``ivol`` / ``rank_score`` is set to NaN (the entry gate then never
fires, so the strategy produces no entries) — mirroring ``low_beta``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.plugins.low_beta import _benchmark_closes
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 252  # ~12 months of trading days
_TRADING_DAYS = 252  # annualization factor


def idiosyncratic_volatility(
    close: pd.Series, benchmark_returns: pd.Series
) -> pd.Series:
    """Causal trailing-``_WINDOW`` annualized market-model residual volatility.

    ``benchmark_returns`` is aligned to ``close``'s index (only exactly-matching
    dates contribute; missing benchmark days become NaN returns that pandas'
    rolling moments skip). Vectorized rolling moments — no per-row regression.
    """
    r_i = close.astype(float).pct_change()
    r_m = benchmark_returns.reindex(r_i.index)
    cov = r_i.rolling(_WINDOW, min_periods=_WINDOW).cov(r_m)
    var_m = r_m.rolling(_WINDOW, min_periods=_WINDOW).var()
    var_i = r_i.rolling(_WINDOW, min_periods=_WINDOW).var()
    beta = cov / var_m
    # OLS residual-variance identity over the window (intercept cancels).
    resid_var = (var_i - beta**2 * var_m).clip(lower=0.0)
    ivol: pd.Series = np.sqrt(resid_var) * np.sqrt(_TRADING_DAYS)
    return ivol


def _prepare_low_ivol(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    bench_close = _benchmark_closes(ctx)
    out: dict[str, pd.DataFrame] = {}

    if bench_close is None:
        ctx.warnings.append(
            f"low_ivol: benchmark {ctx.benchmark!r} unavailable; no idiosyncratic "
            "volatility can be computed, so the strategy yields no entries."
        )
        for tv, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[tv] = bars
                continue
            frame = bars.copy()
            frame["ivol"] = np.nan
            frame["rank_score"] = np.nan
            out[tv] = frame
        return out

    bench_returns = bench_close.pct_change()
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        ivol = idiosyncratic_volatility(frame["close"], bench_returns)
        frame["ivol"] = ivol
        # Lowest idiosyncratic vol ranks highest -> negate for the ranker.
        frame["rank_score"] = -ivol
        out[tv] = frame
    return out


def _low_ivol_lookback() -> int:
    # pct_change consumes one bar, then the 252-return rolling window needs 252
    # returns: 252 + 1 = 253. The benchmark is aligned from the same warmup-padded
    # panel window, so no additional bars are required for alignment.
    return _WINDOW + 1


register_expression_strategy(
    "low_ivol",
    entry="ivol > 0",
    exit=None,
    prepare_bars=_prepare_low_ivol,
    required_lookback=_low_ivol_lookback,
)
