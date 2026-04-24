"""Autoresearch sandbox — Claude Code edits ONLY this file during the loop.

Add new long-only strategies here as `strat_<name>(df) -> list[Trade]` and
register them in `NEW_STRATEGIES`. `df` is an OHLCV pandas DataFrame with
columns: date, open, high, low, close, volume, adj_close.

Helpers you can import from run_pinescript_strategies:
    _ema, _sma, _rma, _stdev, _rsi, _atr, _supertrend_dir, _walk, Trade

Rules (for the agent):
- Do NOT edit run_pinescript_strategies.py, engine.py, portfolio.py,
  slippage.py, or metrics.py — those define the evaluator and must stay
  fixed so comparisons remain fair.
- Long-only. Entries/exits must be decidable by bar close. No lookahead.
- Use _walk to turn entry/exit boolean arrays into round-trip Trade objects.
- Keep one strategy per function; register it in NEW_STRATEGIES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_pinescript_strategies import (
    Trade,
    _atr,
    _ema,
    _rma,
    _rsi,
    _sma,
    _stdev,
    _supertrend_dir,
    _walk,
)


# Example template — Claude should add new strat_* functions below this line.
def strat_example_rsi_mean_revert(df: pd.DataFrame) -> list[Trade]:
    """Placeholder example: enter when RSI(2) < 10, exit when close > SMA(5)."""
    close = df["close"].to_numpy(dtype=float)
    rsi2 = _rsi(close, 2)
    sma5 = _sma(close, 5)
    entries = rsi2 < 10
    exits = close > sma5
    return _walk(entries, exits, close, df["date"].values)


def strat_ibs_trend_filter(df: pd.DataFrame) -> list[Trade]:
    """Internal Bar Strength mean-reversion with a long-term trend filter.

    IBS = (close - low) / (high - low), measures where the close sits inside
    the day's range. Low IBS signals short-term oversold conditions. Combined
    with a 200-bar SMA trend filter, we only buy dips in established uptrends
    and exit when the bar closes above the previous day's high.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    rng = np.where((high - low) > 0, high - low, np.nan)
    ibs = (close - low) / rng
    ibs = np.nan_to_num(ibs, nan=0.5)

    sma200 = _sma(close, 200)

    # Use prior-bar signals to avoid any lookahead; decisions act on bar close.
    ibs_prev = np.concatenate(([0.5], ibs[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    high_prev = np.concatenate(([np.nan], high[:-1]))
    sma_prev = np.concatenate(([np.nan], sma200[:-1]))

    entries = (ibs_prev < 0.2) & (close_prev > sma_prev) & np.isfinite(sma_prev)
    exits = close > high_prev

    return _walk(entries, exits, close, df["date"].values)


def strat_donchian_20_10_trend(df: pd.DataFrame) -> list[Trade]:
    """Turtle-style 20/10 Donchian breakout, gated by SMA(100) trend filter.

    Entry: today's close breaks above the prior 20 bars' highest high AND
    close is above SMA(100) (i.e. established uptrend).
    Exit: today's close drops below the prior 10 bars' lowest low.

    The channel references use .shift(1) so the breakout level is fixed by
    bar close of the prior day — no lookahead. Distinct from bb_breakout
    (volatility-σ bands) and supertrend (ATR trailing stop) because it uses
    raw highest-high / lowest-low channels on a shorter window.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    upper_20 = (
        pd.Series(high).shift(1).rolling(20, min_periods=20).max().to_numpy()
    )
    lower_10 = (
        pd.Series(low).shift(1).rolling(10, min_periods=10).min().to_numpy()
    )
    sma100 = _sma(close, 100)

    valid_up = ~np.isnan(upper_20) & ~np.isnan(sma100)
    valid_dn = ~np.isnan(lower_10)

    entries = valid_up & (close > upper_20) & (close > sma100)
    exits = valid_dn & (close < lower_10)
    return _walk(entries, exits, close, df["date"].values)


NEW_STRATEGIES: dict = {
    "ibs_trend_filter": strat_ibs_trend_filter,
    "donchian_20_10_trend": strat_donchian_20_10_trend,
}
