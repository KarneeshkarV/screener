"""Mark Minervini Trend Template strategy."""

from __future__ import annotations

import pandas as pd

from screener.minervini import (
    MINERVINI_ENTRY_EXPR,
    MINERVINI_EXIT_EXPR,
    prepare_backtest_frames,
    required_history_bars,
)
from screener.strategies.spec import PrepareCtx, register_expression_strategy


def _prepare_mark_minervini(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return prepare_backtest_frames(ctx.bars_by_tv)


register_expression_strategy(
    "mark_minervini",
    entry=MINERVINI_ENTRY_EXPR,
    exit=MINERVINI_EXIT_EXPR,
    prepare_bars=_prepare_mark_minervini,
    required_lookback=required_history_bars,
)
