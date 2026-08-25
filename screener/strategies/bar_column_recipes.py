"""Bar-local column builders shared by the converted expression strategies.

Each function here turns one ticker's OHLCV frame into one derived column, so
an entry/exit expression can name it like any other series. That is how a new
indicator reaches the strategies without the Pine grammar growing a function
(plan D10 in ``docs/plans/unify-screen-backtest.md``).

Every recipe delegates to ``screener/indicators/``; none reimplements an
indicator. They are pure and bar-local, seeing no panel, no market and no
fetcher, which is what lets the backtester and the pine_runner share them.
"""

from __future__ import annotations

import pandas as pd

from screener.indicators.plugins.bollinger_bands import bollinger_bands
from screener.indicators.plugins.rsi import rsi
from screener.indicators.plugins.sar import sar
from screener.indicators.plugins.supertrend import supertrend_dir

ST_PERIOD = 10
ST_MULT = 3.0


def supertrend_direction(bars: pd.DataFrame) -> pd.Series:
    """Supertrend direction: negative is an uptrend, positive a downtrend."""
    values = supertrend_dir(
        bars["high"].to_numpy(dtype=float),
        bars["low"].to_numpy(dtype=float),
        bars["close"].to_numpy(dtype=float),
        period=ST_PERIOD,
        mult=ST_MULT,
    )
    return pd.Series(values, index=bars.index, dtype=float)


def parabolic_sar(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(
        sar(
            bars["high"].to_numpy(dtype=float),
            bars["low"].to_numpy(dtype=float),
            bars["close"].to_numpy(dtype=float),
        ),
        index=bars.index,
        dtype=float,
    )


def _band(bars: pd.DataFrame, which: int, period: int, mult: float) -> pd.Series:
    bands = bollinger_bands(bars["close"].to_numpy(dtype=float), period, mult)
    return pd.Series(bands[which], index=bars.index, dtype=float)


def bb_upper_350(bars: pd.DataFrame) -> pd.Series:
    return _band(bars, 2, 350, 2.5)


def bb_mid_350(bars: pd.DataFrame) -> pd.Series:
    return _band(bars, 1, 350, 2.5)


def donchian_prior_high_20(bars: pd.DataFrame) -> pd.Series:
    """Highest high of the previous 20 bars, excluding the current one.

    A column rather than ``highest(high, 20)`` because Pine's window includes
    the current bar and the grammar has no shift operator.
    """
    return bars["high"].astype(float).rolling(20).max().shift(1)


def donchian_prior_low_10(bars: pd.DataFrame) -> pd.Series:
    return bars["low"].astype(float).rolling(10).min().shift(1)


def _rsi_series(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(
        rsi(bars["close"].to_numpy(dtype=float), 14), index=bars.index, dtype=float
    )


def rsi_prev5_min(bars: pd.DataFrame) -> pd.Series:
    """Lowest RSI over the previous 5 bars, excluding the current one."""
    return _rsi_series(bars).shift(1).rolling(5, min_periods=1).min()


def rsi_prev5_max(bars: pd.DataFrame) -> pd.Series:
    """Highest RSI over the previous 5 bars, excluding the current one."""
    return _rsi_series(bars).shift(1).rolling(5, min_periods=1).max()


def macd_line(bars: pd.DataFrame) -> pd.Series:
    """MACD(12,26) computed with the project's numpy EMA, not Pine's.

    The numpy EMA seeds from bar 0 where Pine's is NaN until it has n bars.
    Using the numpy one keeps the converted strategy faithful to the body it
    replaces instead of silently changing its warm-up.
    """
    from screener.indicators.plugins.ema import ema

    close = bars["close"].to_numpy(dtype=float)
    return pd.Series(ema(close, 12) - ema(close, 26), index=bars.index, dtype=float)


def macd_signal(bars: pd.DataFrame) -> pd.Series:
    from screener.indicators.plugins.ema import ema

    close = bars["close"].to_numpy(dtype=float)
    macd = ema(close, 12) - ema(close, 26)
    return pd.Series(ema(macd, 9), index=bars.index, dtype=float)


def _ao(bars: pd.DataFrame) -> pd.Series:
    mid = (bars["high"].astype(float) + bars["low"].astype(float)) / 2.0
    return mid.rolling(5, min_periods=5).mean() - mid.rolling(34, min_periods=34).mean()


def awesome_oscillator(bars: pd.DataFrame) -> pd.Series:
    """AO: SMA5 minus SMA34 of the bar midpoint."""
    return _ao(bars)


def ao_prev1(bars: pd.DataFrame) -> pd.Series:
    return _ao(bars).shift(1)


def ao_prev2(bars: pd.DataFrame) -> pd.Series:
    return _ao(bars).shift(2)


def _red(bars: pd.DataFrame) -> pd.Series:
    return (bars["open"].astype(float) > bars["close"].astype(float)).astype(float)


def _green(bars: pd.DataFrame) -> pd.Series:
    return (bars["open"].astype(float) < bars["close"].astype(float)).astype(float)


def red_prev1(bars: pd.DataFrame) -> pd.Series:
    """1.0 when the previous bar closed down.

    Red and green are not complements: a bar with open equal to close is
    neither, so both are carried rather than derived from one another.
    """
    return _red(bars).shift(1)


def red_prev2(bars: pd.DataFrame) -> pd.Series:
    return _red(bars).shift(2)


def green_prev1(bars: pd.DataFrame) -> pd.Series:
    return _green(bars).shift(1)


def green_prev2(bars: pd.DataFrame) -> pd.Series:
    return _green(bars).shift(2)
