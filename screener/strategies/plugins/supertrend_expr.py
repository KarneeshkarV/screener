"""SuperTrend long-only expression strategy.

SuperTrend is the single most widely used trend indicator among Indian retail
traders (it is pre-installed on every Indian broker platform and popularised by
the "Supertrend" scanner community). This expression flavour makes the repo's
existing ``supertrend`` callable strategy usable with the rolling backtester:
``prepare_bars`` computes the indicator direction (``st_dir < 0`` = uptrend,
matching the repo's ``supertrend_dir`` convention) and the entry/exit flip on
its transitions.  --hold caps the maximum holding period.
"""

from __future__ import annotations

import pandas as pd

from screener.indicators.plugins.supertrend import supertrend_dir as _supertrend_dir
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_PERIOD = 10
_MULT = 3.0


def _prepare_supertrend(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        direction = _supertrend_dir(
            frame["high"].to_numpy(dtype=float),
            frame["low"].to_numpy(dtype=float),
            frame["close"].to_numpy(dtype=float),
            period=_PERIOD,
            mult=_MULT,
        )
        frame["st_dir"] = direction  # < 0 uptrend, > 0 downtrend (repo convention)
        out[tv] = frame
    return out


def _lookback() -> int:
    return 2 * _PERIOD + 1


register_expression_strategy(
    "supertrend_expr",
    entry="st_dir < 0",
    exit="st_dir > 0",
    prepare_bars=_prepare_supertrend,
    required_lookback=_lookback,
)
