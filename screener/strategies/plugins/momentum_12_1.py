"""Jegadeesh-Titman (1993) 12-1 cross-sectional momentum.

Paper: Jegadeesh & Titman, "Returns to Buying Winners and Selling Losers",
Journal of Finance 48(1), 1993. Long the past-12-month winners while skipping
the most recent month (the 1-month-reversal window).

Signal (causal, as-of bar ``t``):

    mom_12_1[t] = close[t-21] / close[t-252] - 1

i.e. the cumulative return from ~12 months ago to ~1 month ago. The skip of the
last 21 trading days avoids the short-term reversal that contaminates raw 12m
momentum.

Selection: this is a real cross-sectional factor portfolio, so the prepared bars
carry a ``rank_score`` column equal to the momentum value. The rolling
backtester then fills its ``--top`` slots with the highest-momentum names rather
than the most liquid ones (see ``screener.backtester.rolling``). The entry
expression only gates *eligibility* (positive momentum -> long winners only).

Variants
--------
``momentum_12_1``
    Pure relative + absolute 12-1 gate: ``mom_12_1 > 0``. Ranking by raw
    ``mom_12_1``.

``momentum_12_1_trend``
    Dual-momentum style *eligibility* filter (Antonacci absolute momentum):
    ``mom_12_1 > 0 and close > sma(close, 200)``. Ranking still by raw
    ``mom_12_1``; the SMA only skips winners that have broken the long-term
    trend.

``momentum_12_1_riskadj``
    Risk-adjusted (Sharpe-like) *ranking* filter: same positive-momentum
    eligibility, but ``rank_score = mom_12_1 / vol_252``. High-vol crashy
    winners are demoted without a hard trend cut — closer to
    volatility-scaled / risk-adjusted momentum than to dual-momentum SMA
    gates. Prefer this when pure momentum's tail winners are too violent.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.spec import PrepareCtx, register_expression_strategy

# Trading-day windows. 252 ~ 12 months, 21 ~ 1 month (the skipped reversal leg).
_LOOKBACK = 252
_SKIP = 21
_TREND_SMA = 200

# Pure JT eligibility vs dual-momentum (absolute trend) eligibility.
ENTRY_PURE = "mom_12_1 > 0"
ENTRY_TREND = f"mom_12_1 > 0 and close > sma(close, {_TREND_SMA})"
# Risk-adj needs defined vol; vol_252 > 0 is the history/non-degenerate gate.
ENTRY_RISKADJ = "mom_12_1 > 0 and vol_252 > 0"


def momentum_12_1_score(close: pd.Series) -> pd.Series:
    """Return the causal 12-1 momentum series for one symbol's closes."""
    close = close.astype(float)
    return close.shift(_SKIP) / close.shift(_LOOKBACK) - 1.0


def risk_adjusted_momentum(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(mom_12_1, vol_252, mom/vol)`` for one symbol's closes.

    ``mom/vol`` is NaN wherever either leg is missing or vol is non-positive.
    """
    mom = momentum_12_1_score(close)
    vol = realized_volatility(close)
    score = mom / vol
    score = score.where(vol > 0)
    return mom, vol, score


def _prepare_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        mom = momentum_12_1_score(frame["close"])
        frame["mom_12_1"] = mom
        frame["rank_score"] = mom
        out[tv] = frame
    return out


def _prepare_riskadj(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        mom, vol, score = risk_adjusted_momentum(frame["close"])
        frame["mom_12_1"] = mom
        frame["vol_252"] = vol
        frame["rank_score"] = score
        out[tv] = frame
    return out


def _momentum_lookback() -> int:
    # Need ``_LOOKBACK`` prior closes for the oldest leg of the ratio.
    # SMA200 is shorter, so 252 still covers pure and trend variants.
    return _LOOKBACK


def _riskadj_lookback() -> int:
    # pct_change + 252-day vol window (same as low_volatility).
    return _LOOKBACK + 1


register_expression_strategy(
    "momentum_12_1",
    entry=ENTRY_PURE,
    exit=None,
    prepare_bars=_prepare_momentum,
    required_lookback=_momentum_lookback,
)

register_expression_strategy(
    "momentum_12_1_trend",
    entry=ENTRY_TREND,
    exit=None,
    prepare_bars=_prepare_momentum,
    required_lookback=_momentum_lookback,
)

register_expression_strategy(
    "momentum_12_1_riskadj",
    entry=ENTRY_RISKADJ,
    exit=None,
    prepare_bars=_prepare_riskadj,
    required_lookback=_riskadj_lookback,
)
