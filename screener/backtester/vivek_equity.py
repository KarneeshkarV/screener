"""Derived signals for Vivek Singh's public "Vivek Equity Tool".

The calculation ports the publicly linked Pine v4/v5 indicator logic into
causal pandas columns that the existing Pine-like backtester can reference.
Source attribution: Vivek_AlfaTraders, MPL-2.0 notice in the published script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EMA_FAST_1 = 10
EMA_FAST_2 = 20
TREND_LEN = 40
RANGE_MULTIPLIER = 0.618


def required_history_bars() -> int:
    return TREND_LEN


def _atr_rma(bars: pd.DataFrame, length: int) -> pd.Series:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a < b) & (a.shift(1) >= b.shift(1))).fillna(False)


def prepare_vivek_equity_tool_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Return ``bars`` with Vivek Equity Tool signal columns appended.

    The TradingView script supports an alternate higher-timeframe SMA branch,
    but its default is current-timeframe mode. This port implements that default.
    """
    if bars is None or bars.empty:
        return pd.DataFrame()

    out = bars.copy()
    source = out["close"].astype(float)
    ema_fast_1 = source.ewm(span=EMA_FAST_1, adjust=False, min_periods=1).mean()
    ema_fast_2 = source.ewm(span=EMA_FAST_2, adjust=False, min_periods=1).mean()
    trend = source.rolling(TREND_LEN, min_periods=TREND_LEN).mean()
    channel_basis = _atr_rma(out, TREND_LEN) * RANGE_MULTIPLIER
    channel_top = trend + channel_basis
    channel_bottom = trend - channel_basis

    in_range = (
        ((out["open"] <= channel_top) | (out["close"] <= channel_top))
        & ((out["open"] >= channel_bottom) | (out["close"] >= channel_bottom))
    ).fillna(False)
    dir_trend = pd.Series(
        np.where(in_range, 0.0, np.where(source >= trend, 1.0, -1.0)),
        index=out.index,
    )

    buy_cond = (dir_trend == 1.0) & (ema_fast_1 > ema_fast_2)
    sell_cond = (dir_trend == -1.0) & (ema_fast_1 < ema_fast_2)
    buy_close_cond = (dir_trend == 1.0) & (ema_fast_1 < ema_fast_2)
    sell_close_cond = (dir_trend == -1.0) & (ema_fast_1 > ema_fast_2)
    close_cond = buy_close_cond | sell_close_cond

    condition_values: list[float] = []
    prev = 0.0
    for buy, sell, close_ in zip(buy_cond, sell_cond, close_cond):
        if prev != 1.0 and bool(buy):
            current = 1.0
        elif prev != -1.0 and bool(sell):
            current = -1.0
        elif prev != 0.0 and bool(close_):
            current = 0.0
        else:
            current = prev
        condition_values.append(current)
        prev = current

    condition = pd.Series(condition_values, index=out.index, dtype=float)
    prev_condition = condition.shift(1).fillna(0.0)

    out["vivek_equity_ema_fast_1"] = ema_fast_1
    out["vivek_equity_ema_fast_2"] = ema_fast_2
    out["vivek_equity_trend_sma"] = trend
    out["vivek_equity_channel_top"] = channel_top
    out["vivek_equity_channel_bottom"] = channel_bottom
    out["vivek_equity_in_range"] = in_range.astype(float)
    out["vivek_equity_direction"] = dir_trend
    out["vivek_equity_condition"] = condition
    out["vivek_equity_entry"] = ((condition == 1.0) & (prev_condition != 1.0)).astype(
        float
    )
    out["vivek_equity_exit"] = ((condition == -1.0) & (prev_condition != -1.0)).astype(
        float
    )
    out["vivek_equity_close"] = ((prev_condition != 0.0) & close_cond).astype(float)
    out["vivek_equity_golden_cross"] = _crossover(ema_fast_2, trend).astype(float)
    out["vivek_equity_death_cross"] = _crossunder(ema_fast_2, trend).astype(float)
    return out
