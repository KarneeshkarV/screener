"""Blitz-Huij-Martens (2011) residual momentum.

Paper: Blitz, Huij & Martens, "Residual Momentum", Journal of Empirical
Finance 18(3), 2011. Ranking stocks on their stock-specific return component
retains momentum's return continuation while reducing the market exposure that
contributes to conventional momentum crashes.

This price-only implementation uses a single-factor rolling market model because
Fama-French factor histories are not available in-repo. For stock return ``r_i``
and benchmark return ``r_m``, the causal parameters as of bar ``t`` are:

    beta[t]  = rolling_cov(r_i, r_m, 252) / rolling_var(r_m, 252)
    alpha[t] = rolling_mean(r_i, 252) - beta[t] * rolling_mean(r_m, 252)
    eps[t]   = r_i[t] - (alpha[t] + beta[t] * r_m[t])

The score is the standardized mean residual over the 12-1 formation window:

    resid_mom[t] = mean(eps[t-251..t-21]) / std(eps[t-251..t-21])

implemented as a trailing 231-residual rolling statistic shifted by 21 trading
days. Every parameter and residual at ``t`` uses only returns available through
``t``; the formation shift then excludes the most recent month.

Selection: ``rank_score = resid_mom`` so the descending factor ranker fills
``--top`` slots with the strongest positive residual momentum. The entry gate
keeps only positive-score names eligible. Benchmark closes use the same
acquisition ladder as ``low_beta``. If no benchmark is available, every non-empty
frame receives NaN scores and the strategy yields no entries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.plugins.low_beta import _benchmark_closes
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_BETA_WINDOW = 252
_FORMATION_WINDOW = 231
_SKIP = 21


def residual_momentum_score(
    close: pd.Series, benchmark_returns: pd.Series
) -> pd.Series:
    """Return the causal standardized 12-1 residual-momentum score."""
    returns = close.astype(float).pct_change()
    bench = benchmark_returns.reindex(returns.index)

    stock_window = returns.rolling(_BETA_WINDOW, min_periods=_BETA_WINDOW)
    bench_window = bench.rolling(_BETA_WINDOW, min_periods=_BETA_WINDOW)
    beta = stock_window.cov(bench) / bench_window.var()
    alpha = stock_window.mean() - beta * bench_window.mean()
    residual = returns - (alpha + beta * bench)

    formation = residual.rolling(_FORMATION_WINDOW, min_periods=_FORMATION_WINDOW)
    return (formation.mean() / formation.std()).shift(_SKIP)


def _prepare_residual_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    bench_close = _benchmark_closes(ctx)
    out: dict[str, pd.DataFrame] = {}

    if bench_close is None:
        ctx.warnings.append(
            f"residual_momentum: benchmark {ctx.benchmark!r} unavailable; no "
            "residual momentum can be computed, so the strategy yields no "
            "entries."
        )
        for tv, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[tv] = bars
                continue
            frame = bars.copy()
            frame["resid_mom"] = np.nan
            frame["rank_score"] = np.nan
            out[tv] = frame
        return out

    benchmark_returns = bench_close.pct_change()
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        score = residual_momentum_score(frame["close"], benchmark_returns)
        frame["resid_mom"] = score
        frame["rank_score"] = score
        out[tv] = frame
    return out


def _residual_momentum_lookback() -> int:
    # Conservative history padding: 252 returns for the first market-model
    # estimate consume one source close via pct_change; the score then needs a
    # 231-residual formation window shifted by 21 bars. Thus
    # 252 + 1 + 231 + 21 = 505 close bars of warmup.
    return _BETA_WINDOW + 1 + _FORMATION_WINDOW + _SKIP


register_expression_strategy(
    "residual_momentum",
    entry="resid_mom > 0",
    exit=None,
    prepare_bars=_prepare_residual_momentum,
    required_lookback=_residual_momentum_lookback,
)
