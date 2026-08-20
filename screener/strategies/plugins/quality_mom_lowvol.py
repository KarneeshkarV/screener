"""Three-factor India blend: quality-gated momentum with a low-vol brake.

Combines the three factors with the strongest documented evidence in India
(official NSE strategy indices for each: Nifty500 Momentum 50, Nifty Low
Volatility 50, and the Momentum Quality family):

* momentum — the Nifty Normalized Momentum Score from ``nifty_momentum``.
* low volatility — trailing 252-day realized vol (Ang-Hodrick-Xing-Zhang 2006).
* quality gate — ROE, leverage and EPS growth via the backtester's dated
  fundamental columns (merged after prepare; FMP for US/India).

The cross-sectional rank score is the 50/50 momentum/low-vol percentile blend
(the ``mom_lowvol_combo`` recipe), while the entry gate requires positive
momentum AND a quality screen, so the portfolio is "quality winners with a
volatility brake". Fundamentals are optional at runtime: if the fundamental
columns are absent the expression evaluates False and the strategy degenerates
to no entries, so pass ``--fundamentals-provider fmp`` (or openscreener).
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.plugins.nifty_momentum import _prepare_momentum
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_MOM_WEIGHT = 0.5
_VOL_WEIGHT = 0.5

ENTRY = "mom_z > 0 and roe_ttm >= 12 and debt_to_equity <= 2 and eps_growth_yoy > 0"


def _prepare_blend(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    # Start from the Nifty-momentum prepared frames (mom_z + rank_score).
    prepared = _prepare_momentum(ctx)

    vol_by_tv: dict[str, pd.Series] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        vol_by_tv[tv] = realized_volatility(bars["close"])

    if not vol_by_tv:
        return prepared

    invvol_df = -pd.DataFrame(vol_by_tv)
    invvol_pct = invvol_df.rank(axis=1, pct=True)

    out: dict[str, pd.DataFrame] = {}
    for tv, frame in prepared.items():
        if frame is None or frame.empty:
            out[tv] = frame
            continue
        mom_pct = frame["rank_score"]
        vol_pct = invvol_pct[tv].reindex(frame.index)
        frame["rank_score"] = _MOM_WEIGHT * mom_pct + _VOL_WEIGHT * vol_pct
        out[tv] = frame
    return out


def _lookback() -> int:
    return 253  # max(252 momentum, pct_change+252 low-vol)


register_expression_strategy(
    "quality_mom_lowvol",
    entry=ENTRY,
    exit=None,
    prepare_bars=_prepare_blend,
    required_lookback=_lookback,
)
