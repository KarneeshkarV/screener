"""Trend-following Pine strategy ports."""

from __future__ import annotations

import pandas as pd
import numpy as np

from screener.indicators.numpy import _ema, _rsi, _supertrend_dir
from screener.strategies.trades import Trade, _walk


def strat_supertrend(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    dp = np.concatenate(([d[0]], d[:-1]))
    entries = (d < 0) & (dp >= 0)
    exits = (d > 0) & (dp <= 0)
    return _walk(entries, exits, close, df["date"].values)


def strat_supertrend_rsi(df: pd.DataFrame) -> list[Trade]:
    """Supertrend bullish AND RSI crosses above 50; exit on RSI > 72 or flip."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    rsi = _rsi(close, 14)
    in_long = d < 0
    rsi_prev = np.concatenate(([np.nan], rsi[:-1]))
    entries = in_long & (rsi_prev < 50) & (rsi > 50)
    dp = np.concatenate(([d[0]], d[:-1]))
    flip_down = (d > 0) & (dp <= 0)
    exits = (rsi > 72) | flip_down
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross(df: pd.DataFrame) -> list[Trade]:
    """EMA10 crosses over EMA20; exit on bearish cross."""
    close = df["close"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    entries = (mfp <= msp) & (mf > ms)
    exits = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross_regime(df: pd.DataFrame) -> list[Trade]:
    """ma_cross entries gated by EMA150 > EMA600 bull regime."""
    close = df["close"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    regime = _ema(close, 150) > _ema(close, 600)
    entries = (mfp <= msp) & (mf > ms) & regime
    exits = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross_st_entry(df: pd.DataFrame) -> list[Trade]:
    """Entry = ma_cross AND supertrend bullish; exit = ma_cross bearish."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    entries = (mfp <= msp) & (mf > ms) & (d < 0)
    exits = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross_st_exit(df: pd.DataFrame) -> list[Trade]:
    """Entry = ma_cross bullish; exit = supertrend flips bearish."""
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    dp = np.concatenate(([d[0]], d[:-1]))
    entries = (mfp <= msp) & (mf > ms)
    exits = (d > 0) & (dp <= 0)
    return _walk(entries, exits, close, df["date"].values)
