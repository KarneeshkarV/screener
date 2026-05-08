"""Custom strategy ports."""
from __future__ import annotations

import pandas as pd

from screener.strategies.trades import Trade, _walk


def strat_vivek_equity_tool(df: pd.DataFrame) -> list[Trade]:
    """Vivek Equity Tool: EMA10/EMA20 plus SMA40/ATR channel state machine."""
    from screener.backtester.vivek_equity import prepare_vivek_equity_tool_frame

    frame = df.copy()
    if "date" in frame.columns:
        frame = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame["date"])))
    prepared = prepare_vivek_equity_tool_frame(frame)
    close = prepared["close"].to_numpy(dtype=float)
    entries = prepared["vivek_equity_entry"].to_numpy(dtype=float) > 0
    exits = (
        (prepared["vivek_equity_exit"].to_numpy(dtype=float) > 0)
        | (prepared["vivek_equity_close"].to_numpy(dtype=float) > 0)
    )
    dates = prepared.index.to_numpy()
    return _walk(entries, exits, close, dates)
