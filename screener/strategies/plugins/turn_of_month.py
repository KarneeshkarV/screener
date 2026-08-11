"""Turn-of-month effect (Ariel 1987).

Paper: Ariel, "A Monthly Effect in Stock Returns", Journal of Financial
Economics 18(1), 1987. https://doi.org/10.1016/0304-405X(87)90066-3
India evidence: "Semi-monthly effect in stock returns: new evidence from Bombay
Stock Exchange" (2017). https://doi.org/10.21511/imfi.14(3-1).2017.01

Ariel showed that virtually the entire monthly equity premium accrues in the
days around the turn of the month (last ~4 trading days through the first ~3 of
the next month); Lakonishok & Smidt (1988) confirmed the same window. The
effect has persisted internationally, including the BSE.

Rule (implemented via ``prepare_bars`` adding ``day_of_month``):

    entry = day_of_month >= 28 or day_of_month <= 3    # into the TOM window
    exit  = day_of_month >= 4 and day_of_month <= 27   # out of the window

Long-only: hold liquid names through the turn-of-month window, sit flat the
rest of the month. Expected: much lower exposure (≈1/3 of days), modest
absolute return, high per-exposure return — an index-timing overlay rather than
a stock-selection strategy.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW_START = 28
_WINDOW_END = 3


def _prepare_tom(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        # ``Index.day`` is not typed on the generic Index; cast to DatetimeIndex.
        idx = frame.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
        frame["day_of_month"] = idx.day
        out[tv] = frame
    return out


register_expression_strategy(
    "turn_of_month",
    entry=f"day_of_month >= {_WINDOW_START} or day_of_month <= {_WINDOW_END}",
    exit=f"day_of_month >= {_WINDOW_END + 1} and day_of_month <= {_WINDOW_START - 1}",
    prepare_bars=_prepare_tom,
)
