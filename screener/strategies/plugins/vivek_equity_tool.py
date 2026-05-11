"""Vivek Equity Tool: Pine entry/exit + frame prep with derived signal columns."""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, strategy


def _prepare_vivek(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    from screener.backtester.vivek_equity import prepare_vivek_equity_tool_frame

    return {
        symbol: prepare_vivek_equity_tool_frame(bars)
        for symbol, bars in ctx.bars_by_tv.items()
    }


def _vivek_lookback() -> int:
    from screener.backtester.vivek_equity import required_history_bars

    return required_history_bars()


@strategy(
    "vivek_equity_tool",
    entry="vivek_equity_entry > 0",
    exit="vivek_equity_exit > 0 or vivek_equity_close > 0",
    prepare_bars=_prepare_vivek,
    required_lookback=_vivek_lookback,
)
def _vivek_equity_tool() -> None:
    """Expression-only strategy. Body unused."""
