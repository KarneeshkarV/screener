"""12-1 cross-sectional momentum, confirmed by time-series momentum.

Two different questions about the same name:

* Cross-sectional (Jegadeesh & Titman, 1993): is this one of the strongest
  names in the universe? Ranked by ``mom_12_1 = close[t-21] / close[t-252] - 1``,
  which skips the most recent month (the 1-month reversal window).
* Time-series (Moskowitz, Ooi & Pedersen, 2012): is this name's own trend up?
  Measured with plain trailing returns that do *not* skip the last month,
  ``tsmom_N = close[t] / close[t-N] - 1``, so a winner that has rolled over in
  the last few weeks fails the check even while its 12-1 rank is still high.

Selection: on each bar the universe is ranked by ``mom_12_1`` (``mom_xs_rank``,
1 = best, over the dated index members when membership history exists). Only
the top ``_XS_TOP`` names are candidates, and of those only the ones with
positive 12-month time-series momentum may enter. ``rank_score`` stays the raw
``mom_12_1`` so ``--top`` slots still fill with the strongest survivors.

Prepared columns, all usable from ``--entry`` to test other confirmations:

``mom_12_1``     the 12-1 ranking value (shared recipe)
``mom_xs_rank``  cross-sectional rank of ``mom_12_1``, 1 = best
``tsmom_12``     ``close[t] / close[t-252] - 1``
``tsmom_6``      ``close[t] / close[t-126] - 1``
``tsmom_3``      ``close[t] / close[t-63] - 1``
``tsmom_1``      ``close[t] / close[t-21] - 1``
``vol_252``      trailing realized daily-return volatility
``tsmom_12_vol`` ``tsmom_12 / (vol_252 * sqrt(252))``, a volatility-scaled trend
"""

from __future__ import annotations

import math

import pandas as pd

from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.plugins.momentum_12_1 import ENTRY_PURE, momentum_12_1_score
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    PrepareCtx,
    register_expression_strategy,
)
from screener.tradeable import tradeable_window

# Cross-sectional cut: candidates must sit in the top 20 by 12-1 momentum.
_XS_TOP = 20
# Trailing time-series windows in trading days (12, 6, 3 and 1 months).
_TS_WINDOWS = {"tsmom_12": 252, "tsmom_6": 126, "tsmom_3": 63, "tsmom_1": 21}

ENTRY_TSMOM = f"{ENTRY_PURE} and mom_xs_rank <= {_XS_TOP} and tsmom_12 > 0"


def trailing_return(
    close: pd.Series, volume: pd.Series | None, lookback: int
) -> pd.Series:
    """Causal ``close[t] / close[t-lookback] - 1`` with no skipped month.

    NaN where the window does not rest on real, traded prices, the same
    tradeability rule the 12-1 recipe applies with ``skip=21``.
    """
    values = pd.to_numeric(close, errors="coerce").astype(float)
    raw = values / values.shift(lookback) - 1.0
    return raw.where(tradeable_window(values, volume, lookback=lookback))


def _prepare_tsmom(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = dict(ctx.bars_by_tv)
    frames: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        frame = bars.copy()
        volume = frame["volume"] if "volume" in frame.columns else None
        frame["mom_12_1"] = momentum_12_1_score(frame["close"], volume)
        for column, lookback in _TS_WINDOWS.items():
            frame[column] = trailing_return(frame["close"], volume, lookback)
        frame["vol_252"] = realized_volatility(frame["close"], volume)
        annual_vol = frame["vol_252"] * math.sqrt(252)
        frame["tsmom_12_vol"] = (frame["tsmom_12"] / annual_vol).where(annual_vol > 0)
        frame["rank_score"] = frame["mom_12_1"]
        frames[tv] = frame
    if not frames:
        return out

    # Rank across names per bar. NaN momentum stays unranked (NaN rank), so a
    # name without 12-1 history can never pass ``mom_xs_rank <= N``.
    mom = ctx.mask_rank_reference(
        pd.DataFrame({tv: frame["mom_12_1"] for tv, frame in frames.items()})
    )
    xs_rank = mom.rank(axis=1, ascending=False, method="min")
    for tv, frame in frames.items():
        frame["mom_xs_rank"] = xs_rank[tv].reindex(frame.index)
        out[tv] = frame
    return out


def _tsmom_lookback() -> int:
    # pct_change + 252-day vol window, which also covers every return leg.
    return max(_TS_WINDOWS.values()) + 1


register_expression_strategy(
    "momentum_12_1_tsmom",
    entry=ENTRY_TSMOM,
    exit=None,
    prepare_bars=_prepare_tsmom,
    required_lookback=_tsmom_lookback,
    profile=DEFAULT_STRATEGY_PROFILE,
)
