"""Volatility-managed low-volatility factor (market-vol regime gated).

Combines the two most heavily documented "non-momentum" anomalies into one
defensive book:

* Low-volatility anomaly — Ang, Hodrick, Xing & Zhang, "The Cross-Section of
  Volatility and Expected Returns", Journal of Finance 61(1), 2006: high-idio-vol
  stocks underperform; the calmest names earn the best risk-adjusted returns.
* Volatility-managed exposure — Moreira & Muir, "Volatility-Managed Portfolios",
  Journal of Finance 72(4), 2017: scaling exposure DOWN when recent realized
  volatility is high raises Sharpe ratios and alphas across equity markets,
  anomalies and asset classes, because volatility spikes are followed by low
  returns (negative vol-of-returns relation). Harvey et al. (2018, JPM,
  "The Impact of Volatility Targeting") reach the same conclusion: vol targeting
  improves Sharpe and cuts left-tail risk.

Signal (causal, as-of bar ``t``):

    vol_63[t]        = stdev(daily returns, 63)            # stock realized vol
    bench_vol_21[t]  = stdev(benchmark daily returns, 21)  # market realized vol
    bench_vol_pct[t] = percentile rank of bench_vol_21[t] within the trailing
                       252 days of market volatility        (0..1, high = stressed)

Selection: cross-sectional factor — ``rank_score = -vol_63`` so the rolling
backtester fills its ``--top`` slots with the *calmest* names (AHXZ low-vol).
Entry is gated on the market regime: only buy when the market's short-term
realized volatility is in its own low 75th percentile (calm regime). If the
market vol percentile spikes above 0.90 the book de-risks (exit), mimicking
Moreira-Muir's inverse-volatility scaling with a discrete regime switch. The
benchmark column is replicated per stock so the gate is market-wide, not
per-name.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_STOCK_VOL = 63  # trailing days of per-stock realized vol
_BENCH_VOL = 21  # trailing days of market realized vol
_BENCH_RANK = 252  # window over which today's market vol is percentile-ranked
_ENTRY_PCT = 0.75  # enter only when market vol percentile < this (calm)
_EXIT_PCT = 0.90  # exit when market vol percentile exceeds this (stress)


def _prepare_vol_target(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    benchmark_bars = ctx.price_panel.get(ctx.benchmark, pd.DataFrame())
    if benchmark_bars is None or benchmark_bars.empty:
        ctx.warnings.append(
            f"benchmark data unavailable for vol_target_lowvol: {ctx.benchmark}"
        )
        return ctx.bars_by_tv

    bench_close = benchmark_bars["close"].astype(float)
    bench_ret = bench_close.pct_change()
    bench_vol = bench_ret.rolling(_BENCH_VOL, min_periods=_BENCH_VOL).std()
    # Causal percentile of today's market vol within its trailing window.
    bench_vol_pct = bench_vol.rolling(_BENCH_RANK, min_periods=_BENCH_RANK // 2).rank(
        pct=True
    )

    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        vol = close.pct_change().rolling(_STOCK_VOL, min_periods=_STOCK_VOL).std()
        frame["vol_63"] = vol
        frame["rank_score"] = -vol  # calmest names rank first
        frame["bench_vol_pct"] = bench_vol_pct.reindex(frame.index).ffill()
        out[tv] = frame
    return out


def _lookback() -> int:
    # pct_change consumes a bar; the percentile window is the long pole.
    return _BENCH_RANK + 1


register_expression_strategy(
    "vol_target_lowvol",
    entry=(f"vol_63 > 0 and bench_vol_pct < {_ENTRY_PCT}"),
    exit=f"bench_vol_pct > {_EXIT_PCT}",
    prepare_bars=_prepare_vol_target,
    required_lookback=_lookback,
)
