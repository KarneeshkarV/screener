"""The price-only score recipes themselves, defined exactly once.

Each function here is causal and reads nothing but the bar series on
:class:`screener.factors.BarFeatures`. Registration makes the recipe available
to both adapters, so ``momentum_12_1`` in the screen and ``momentum_12_1`` in
the backtest are the *same* number rather than two formulas sharing a name.

Deliberately *not* here: anything that needs a TradingView snapshot column
(``Perf.Y``, ``RSI``, ``market_cap_basic``, ``relative_volume_10d_calc``).
Those stay in ``screener.scoring.plugins`` because a snapshot carries only
today's value and replaying it through history would be lookahead.
"""

from __future__ import annotations

import pandas as pd

from screener.factors import BarFeatures, price_score

# Trading-day windows. 252 ~ 12 months, 21 ~ 1 month (the skipped reversal leg).
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21


def momentum_12_1(
    close: pd.Series,
    *,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.Series:
    """Jegadeesh-Titman 12-1 momentum, as of bar ``t``.

    ``close[t - skip] / close[t - lookback] - 1``: the cumulative return from
    ~12 months ago to ~1 month ago. Skipping the last 21 sessions avoids the
    short-term reversal that contaminates raw 12-month momentum.

    Stays NaN until ``lookback`` prior closes exist. That NaN is the
    eligibility signal for both adapters - it is never filled with 0.
    """
    values = pd.to_numeric(close, errors="coerce").astype(float)
    return values.shift(skip) / values.shift(lookback) - 1.0


@price_score(
    "momentum_12_1",
    required_lookback=MOMENTUM_LOOKBACK,
    description="Jegadeesh-Titman 12-1 momentum: close[t-21]/close[t-252] - 1",
    aux_column="mom_12_1",
    # Jegadeesh-Titman buys winners, so a non-positive 12-1 return is not a
    # candidate at all. Declaring the floor with the recipe is what lets the
    # screen and the backtest gate on one rule instead of two: the screen
    # filters on it after bar scoring, and the backtest's ``ENTRY_PURE``
    # expression is rendered from it.
    eligible_above=0.0,
)
def score_momentum_12_1(features: BarFeatures) -> pd.Series:
    return momentum_12_1(features.close)


__all__ = [
    "MOMENTUM_LOOKBACK",
    "MOMENTUM_SKIP",
    "momentum_12_1",
    "score_momentum_12_1",
]
