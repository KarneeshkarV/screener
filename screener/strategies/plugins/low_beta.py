"""Frazzini-Pedersen (2014) "Betting Against Beta" — long leg.

Paper: Frazzini & Pedersen, "Betting Against Beta", Journal of Financial
Economics 111(1), 2014. Because leverage-constrained investors bid up high-beta
assets, low-beta assets deliver higher risk-adjusted returns. The full BAB factor
is long low-beta / short high-beta and beta-neutral; a long-only tilt simply
holds the *lowest*-beta names.

Signal (causal, as-of bar ``t``):

    beta_252[t] = cov( r_i[t-251..t], r_m[t-251..t] ) / var( r_m[t-251..t] )

the OLS slope of the name's daily returns on the benchmark's daily returns over a
trailing 252-day window. Windows end at ``t`` using returns available through
``t`` (``rolling`` is backward-looking), so the signal is causal. No shrinkage is
applied — the raw rolling OLS beta is used so the ``beta < 1.0`` eligibility gate
has a clean interpretation. (Frazzini-Pedersen shrink toward 1 via
``0.6*beta + 0.4*1.0``; that is optional and intentionally omitted here.)

Selection: ``rank_score = -beta_252`` so the descending ranker fills its
``--top`` slots with the calmest (lowest-beta) names first. The entry gate
``beta_252 < 1.0`` keeps only below-market-beta names eligible.

Benchmark acquisition (documented design decision): the daily benchmark closes
are taken, in priority order, from (1) ``ctx.price_panel`` under the benchmark
symbol — the rolling engine fetches the benchmark into the panel alongside the
portfolio symbols over the *same* warmup-padded window, so this is the normal
path and needs no extra history; (2) ``ctx.bars_by_tv`` in case the benchmark is
carried as a universe member; (3) a direct ``ctx.fetcher`` fetch over
``[ctx.start - buffer, ctx.end]`` as a last resort. ``ctx.benchmark`` is already
the yfinance-style symbol used as the panel key (the engine adds it to the fetch
list without re-mapping), so it is passed to the fetcher as-is. If none of these
yields benchmark closes, a warning is recorded and every name's ``beta_252`` /
``rank_score`` is set to NaN (the entry gate then never fires, so the strategy
produces no entries) — the frames are otherwise returned unchanged.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 252  # ~12 months of trading days
# Extra calendar days requested when the benchmark must be fetched fresh, so the
# 252-return rolling window is fully warmed at ``ctx.start``.
_FETCH_BUFFER_DAYS = int(_WINDOW * 1.7) + 10


def rolling_beta(close: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """Return the causal trailing-``_WINDOW`` OLS beta vs the benchmark.

    ``benchmark_returns`` is aligned to ``close``'s index (forward-fill is not
    used — only exactly-matching dates contribute, and missing benchmark days
    become NaN returns that pandas' rolling cov/var skip).
    """
    returns = close.astype(float).pct_change()
    bench = benchmark_returns.reindex(returns.index)
    cov = returns.rolling(_WINDOW, min_periods=_WINDOW).cov(bench)
    var = bench.rolling(_WINDOW, min_periods=_WINDOW).var()
    return cov / var


def _benchmark_closes(ctx: PrepareCtx) -> pd.Series | None:
    """Locate the benchmark daily closes via panel -> universe -> fetch."""
    symbol = ctx.benchmark
    frame = ctx.price_panel.get(symbol)
    if frame is not None and not frame.empty and "close" in frame.columns:
        return frame["close"].astype(float)

    frame = ctx.bars_by_tv.get(symbol)
    if frame is not None and not frame.empty and "close" in frame.columns:
        return frame["close"].astype(float)

    fetch_start = ctx.start - timedelta(days=_FETCH_BUFFER_DAYS)
    fetched = ctx.fetcher.fetch([symbol], fetch_start, ctx.end)
    frame = fetched.get(symbol)
    if frame is not None and not frame.empty and "close" in frame.columns:
        return frame["close"].astype(float)
    return None


def _prepare_low_beta(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    bench_close = _benchmark_closes(ctx)
    out: dict[str, pd.DataFrame] = {}

    if bench_close is None:
        ctx.warnings.append(
            f"low_beta: benchmark {ctx.benchmark!r} unavailable; no beta can be "
            "computed, so the strategy yields no entries."
        )
        for tv, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[tv] = bars
                continue
            frame = bars.copy()
            frame["beta_252"] = np.nan
            frame["rank_score"] = np.nan
            out[tv] = frame
        return out

    bench_returns = bench_close.pct_change()
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        beta = rolling_beta(frame["close"], bench_returns)
        frame["beta_252"] = beta
        # Lowest beta ranks highest -> negate for the descending ranker.
        frame["rank_score"] = -beta
        out[tv] = frame
    return out


def _low_beta_lookback() -> int:
    # pct_change consumes one bar, then the 252-return rolling window needs 252
    # returns: 252 + 1 = 253. The benchmark is aligned from the same warmup-padded
    # panel window, so no additional bars are required for alignment.
    return _WINDOW + 1


register_expression_strategy(
    "low_beta",
    entry="beta_252 < 1.0",
    exit=None,
    prepare_bars=_prepare_low_beta,
    required_lookback=_low_beta_lookback,
)
