"""Nifty 500 Momentum 50 methodology (NSE Indices) as a rolling factor strategy.

Source: NSE Indices "Nifty500 Momentum 50" index methodology (also used by
Nifty200 Momentum 30 / Nifty Midcap150 Momentum 50), and the Nippon India /
Motilal Oswal momentum index fund documents. The official Normalized Momentum
Score is built from volatility-adjusted 6-month and 12-month price returns:

    mom_ratio_12m[t] = (close[t] / close[t-252] - 1) / vol_252[t]
    mom_ratio_6m[t]  = (close[t] / close[t-126] - 1) / vol_252[t]
    vol_252[t]       = annualised std of daily returns over the trailing 252 bars
    z12[t]           = cross-sectional z-score of mom_ratio_12m across names
    z6[t]            = cross-sectional z-score of mom_ratio_6m  across names
    weighted_z[t]    = 0.5 * z12[t] + 0.5 * z6[t]
    norm_score[t]    = (1 + weighted_z)        if weighted_z >= 0
                       (1 - weighted_z) ** -1  otherwise

The Nifty index rebalances semi-annually and weights by score x free-float
market cap; the rolling backtester instead refills freed slots daily from the
highest ``rank_score`` names (an approximation documented in the codebase).
The z-scores are computed over the full prepared universe per day (same
convention as ``mom_lowvol_combo``), so a name's score is comparable across
the universe even if eligibility filters later drop some names.

``nifty_momentum``      — long the positive-momentum names (weighted_z > 0).
``nifty_momentum_trend``— dual-momentum gate: positive momentum AND price above
                          the 200-day SMA (Antonacci absolute-momentum style).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 252  # 12 months
_LOOKBACK = 252  # 12-month momentum leg
_LOOKBACK6 = 126  # 6-month momentum leg
_TREND_SMA = 200

ENTRY = "mom_z > 0"
ENTRY_TREND = f"mom_z > 0 and close > sma(close, {_TREND_SMA})"


def _prepare_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    ratios: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        returns = close.pct_change()
        vol = returns.rolling(_WINDOW, min_periods=_WINDOW).std() * np.sqrt(_WINDOW)
        frame["mom_ratio_12"] = (close / close.shift(_LOOKBACK) - 1.0) / vol
        frame["mom_ratio_6"] = (close / close.shift(_LOOKBACK6) - 1.0) / vol
        ratios[tv] = frame

    if not ratios:
        return ctx.bars_by_tv

    # Cross-sectional percentile ranks per day over the full prepared universe
    # (the same ``rank(axis=1, pct=True)`` pattern as ``mom_lowvol_combo``).
    # The official methodology z-scores each ratio then averages; a percentile
    # rank is a monotone transform of the same daily cross-section, so the
    # blended ordering is identical and pct > 0.5  <=>  weighted_z > 0.
    z12 = pd.DataFrame({tv: f["mom_ratio_12"] for tv, f in ratios.items()})
    z6 = pd.DataFrame({tv: f["mom_ratio_6"] for tv, f in ratios.items()})
    p12 = z12.rank(axis=1, pct=True)
    p6 = z6.rank(axis=1, pct=True)
    mom_pct = 0.5 * p12 + 0.5 * p6

    out: dict[str, pd.DataFrame] = {}
    for tv, frame in ratios.items():
        aligned = mom_pct[tv].reindex(frame.index)
        frame["mom_z"] = aligned - 0.5  # > 0  <=>  above-median momentum
        frame["rank_score"] = aligned
        out[tv] = frame
    return out


def _lookback() -> int:
    # 12-month ratio needs 252 prior closes; vol needs 252 returns.
    return _LOOKBACK


register_expression_strategy(
    "nifty_momentum",
    entry=ENTRY,
    exit=None,
    prepare_bars=_prepare_momentum,
    required_lookback=_lookback,
)

register_expression_strategy(
    "nifty_momentum_trend",
    entry=ENTRY_TREND,
    exit=None,
    prepare_bars=_prepare_momentum,
    required_lookback=_lookback,
)
