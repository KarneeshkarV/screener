"""Time-series momentum / trend following (Moskowitz-Ooi-Pedersen; Hurst et al.).

Time-series momentum judges each market against its own history rather than
against its peers: long when the instrument's own trailing excess return is
positive, short when negative, sized inversely to its volatility, and combined
across many weakly correlated markets. Moskowitz, Ooi & Pedersen find all 58 of
their futures contracts have positive 12-month TSMOM returns; Hurst, Ooi &
Pedersen blend 1-, 3- and 12-month signals and target 10% portfolio volatility.

Long-only approximation
-----------------------
Three things separate the published record from what this engine can run, and
all three cut the same way:

- **No short leg.** Negative-trend names are skipped, not sold. Trend following's
  crisis-diversification property comes largely from being short during
  sustained declines, so that property is absent here.
- **No cross-asset diversification.** The papers' headline Sharpe ratios are
  portfolio results across equities, bonds, currencies and commodities. Hurst et
  al. put their equity-index sleeve alone at 0.78 against 1.79 for the
  diversified portfolio, so the equity-only sleeve is the honest comparison for
  these variants - and this is a single-country stock sleeve, narrower still.
- **No volatility targeting.** Inverse-volatility sizing is available as
  ``--sizing inverse_vol``, but the aggregate portfolio is not scaled to a fixed
  volatility target.

What survives is the signal itself: own-trend timing on each name, with slots
allocated to the strongest trends per unit of risk. A position is closed as soon
as its own trend turns negative, which is the long-only half of the paper's
sign-flip - the short half is what this cannot do.

Variants
--------
``tsmom_12``
    Moskowitz-Ooi-Pedersen's canonical rule: long while the trailing 12-month
    return is positive. No skip month - unlike cross-sectional momentum, the
    time-series signal is not contaminated by the one-month reversal, and the
    paper uses the full window. Slots are ranked by return per unit of
    volatility so the fixed number of positions goes to the best risk-adjusted
    trends, standing in for the paper's inverse-volatility weighting.

``tsmom_blend``
    Hurst-Ooi-Pedersen's 1/3/12-month blend. Each horizon votes with the sign of
    its own return; the blend is the average vote, so it runs over
    {-1, -1/3, +1/3, +1}. Entry requires a positive blend, i.e. at least two of
    the three horizons trending up. Ranking is the blend divided by volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_MONTH = 21
_YEAR = 252
# Hurst et al. weight the three horizons equally.
_BLEND_WINDOWS = (1 * _MONTH, 3 * _MONTH, 12 * _MONTH)

ENTRY_TSMOM = "tsmom_12 > 0 and vol_252 > 0"
ENTRY_BLEND = "tsmom_blend > 0 and vol_252 > 0"
# Time-series momentum flips its position when the sign flips; long-only, that
# is an exit rather than a short. Without it a position would be held to the end
# of its window regardless of the trend it was opened on.
EXIT_TSMOM = "tsmom_12 <= 0"
EXIT_BLEND = "tsmom_blend <= 0"


def trailing_return(close: pd.Series, window: int) -> pd.Series:
    """Return the ``window``-bar trailing return, NaN before enough history."""
    close = close.astype(float)
    return close / close.shift(window) - 1.0


def trend_blend(close: pd.Series) -> pd.Series:
    """Average of the 1-, 3- and 12-month trend signs, in [-1, 1].

    ``sign`` maps a flat window to 0, so a dead-flat market contributes neither
    a long nor a short vote instead of being counted as an uptrend.
    """
    votes = [np.sign(trailing_return(close, window)) for window in _BLEND_WINDOWS]
    stacked = pd.concat(votes, axis=1)
    # Any missing horizon leaves the blend undefined rather than letting the
    # shorter horizons outvote a 12-month leg that has not warmed up yet.
    return pd.Series(stacked.mean(axis=1).where(stacked.notna().all(axis=1)))


def _prepare_tsmom(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[symbol] = bars
            continue
        frame = bars.copy()
        signal = trailing_return(frame["close"], _YEAR)
        vol = realized_volatility(frame["close"])
        frame["tsmom_12"] = signal
        frame["vol_252"] = vol
        frame["rank_score"] = (signal / vol).where(vol > 0)
        out[symbol] = frame
    return out


def _prepare_blend(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[symbol] = bars
            continue
        frame = bars.copy()
        blend = trend_blend(frame["close"])
        vol = realized_volatility(frame["close"])
        frame["tsmom_blend"] = blend
        frame["vol_252"] = vol
        frame["rank_score"] = (blend / vol).where(vol > 0)
        out[symbol] = frame
    return out


def _tsmom_lookback() -> int:
    # 252-bar trend window and the 252-bar volatility window, the latter needing
    # one extra bar for its first percentage change.
    return _YEAR + 1


register_expression_strategy(
    "tsmom_12",
    entry=ENTRY_TSMOM,
    exit=EXIT_TSMOM,
    prepare_bars=_prepare_tsmom,
    required_lookback=_tsmom_lookback,
)

register_expression_strategy(
    "tsmom_blend",
    entry=ENTRY_BLEND,
    exit=EXIT_BLEND,
    prepare_bars=_prepare_blend,
    required_lookback=_tsmom_lookback,
)


__all__ = ["trailing_return", "trend_blend"]
