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
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

# Trading-day windows. 252 ~ 12 months, 21 ~ 1 month (the skipped reversal leg).
_LOOKBACK = 252
_SKIP = 21


def momentum_12_1_score(close: pd.Series) -> pd.Series:
    """Return the causal 12-1 momentum series for one symbol's closes."""
    close = close.astype(float)
    return close.shift(_SKIP) / close.shift(_LOOKBACK) - 1.0


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


def _momentum_lookback() -> int:
    # Need ``_LOOKBACK`` prior closes for the oldest leg of the ratio.
    return _LOOKBACK


register_expression_strategy(
    "momentum_12_1",
    entry="mom_12_1 > 0",
    exit=None,
    prepare_bars=_prepare_momentum,
    required_lookback=_momentum_lookback,
)
