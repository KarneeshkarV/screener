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


def strat_squeeze_breakout(df: pd.DataFrame) -> list[Trade]:
    """TTM-style squeeze: Bollinger Bands inside Keltner Channels → breakout.

    A squeeze fires when BB(20, 2σ) sits entirely inside KC(20, 1.5·ATR20) —
    a volatility contraction. Entry: the prior bar was in a squeeze AND
    today's close breaks above yesterday's upper Keltner band, with price
    above SMA(100) to gate direction. Exit: close falls back below the
    20-period middle (SMA20). This targets volatility expansion, not
    oversold dips (ibs_trend_filter) or σ-band breakouts (bb_breakout), and
    is distinct from Donchian high-low channels and ATR-trailing supertrend.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    sma20 = _sma(close, 20)
    std20 = _stdev(close, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20

    atr20 = _atr(high, low, close, 20)
    kc_upper = sma20 + 1.5 * atr20
    kc_lower = sma20 - 1.5 * atr20

    sma100 = _sma(close, 100)

    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Reference prior-bar indicator values so the decision is knowable at
    # bar close without lookahead.
    squeeze_prev = np.concatenate(([False], squeeze[:-1]))
    kc_upper_prev = np.concatenate(([np.nan], kc_upper[:-1]))

    valid = np.isfinite(kc_upper_prev) & np.isfinite(sma100) & np.isfinite(sma20)
    entries = valid & squeeze_prev & (close > kc_upper_prev) & (close > sma100)
    exits = np.isfinite(sma20) & (close < sma20)

    return _walk(entries, exits, close, df["date"].values)


def strat_cum_rsi2_pullback(df: pd.DataFrame) -> list[Trade]:
    """Connors-style cumulative RSI(2) pullback inside a long-term uptrend.

    Uses the *persistence* of an oversold reading rather than a single day:
    the sum of the last 3 RSI(2) readings falling below 45 (i.e. 3-day
    average RSI(2) < 15) identifies a multi-bar pullback, gated by price
    above SMA(200). Exit is a quick mean-reversion target (close > SMA(5)).
    Distinct from rsi_ema (EMA crossover), macd_rsi (MACD trigger), and
    ibs_trend_filter (single-bar range position) — this reads multi-day
    oversold persistence.
    """
    close = df["close"].to_numpy(dtype=float)
    rsi2 = _rsi(close, 2)
    sma200 = _sma(close, 200)
    sma5 = _sma(close, 5)

    cum3 = pd.Series(rsi2).rolling(3, min_periods=3).sum().to_numpy()

    # Reference prior-bar values so the signal is locked in at bar close.
    cum3_prev = np.concatenate(([np.nan], cum3[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    valid = np.isfinite(cum3_prev) & np.isfinite(sma200_prev)
    entries = valid & (cum3_prev < 45.0) & (close_prev > sma200_prev)
    exits = np.isfinite(sma5) & (close > sma5)

    return _walk(entries, exits, close, df["date"].values)


def strat_volume_capitulation_reclaim(df: pd.DataFrame) -> list[Trade]:
    """Heavy-volume down-day capitulation reclaim inside a long-term uptrend.

    Reads *volume* — a dimension none of the other sandbox strategies use.
    Thesis: a sharp, high-participation selloff within an uptrend often marks
    short-term capitulation; the earliest confirmation is a next-day bar that
    closes back above the capitulation bar's high.

    Entry (at today's close):
      - prior bar close dropped >= 1.5% vs the bar before (capitulation day)
      - prior bar volume > 1.3x its 20-bar avg volume (high participation)
      - prior bar close was above SMA(200) (long-term uptrend intact)
      - today's close > prior bar's high (reclaim confirmation)
    Exit: close falls below SMA(20) — momentum has failed.

    Distinct from ibs_trend_filter (single-bar range position), Donchian
    (channel breakout), squeeze_breakout (volatility expansion), and
    cum_rsi2_pullback (multi-day RSI persistence).
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    sma200 = _sma(close, 200)
    sma20 = _sma(close, 20)
    vol_avg20 = _sma(volume, 20)

    close_prev = np.concatenate(([np.nan], close[:-1]))
    close_prev2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    high_prev = np.concatenate(([np.nan], high[:-1]))
    vol_prev = np.concatenate(([np.nan], volume[:-1]))
    vol_avg_prev = np.concatenate(([np.nan], vol_avg20[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    with np.errstate(divide="ignore", invalid="ignore"):
        pullback_pct = (close_prev - close_prev2) / close_prev2

    valid = (
        np.isfinite(close_prev2)
        & np.isfinite(sma200_prev)
        & np.isfinite(vol_avg_prev)
        & (vol_avg_prev > 0)
    )

    entries = (
        valid
        & (pullback_pct <= -0.015)
        & (vol_prev > 1.3 * vol_avg_prev)
        & (close_prev > sma200_prev)
        & (close > high_prev)
    )
    exits = np.isfinite(sma20) & (close < sma20)

    return _walk(entries, exits, close, df["date"].values)


def strat_williams_vix_fix_spike(df: pd.DataFrame) -> list[Trade]:
    """Williams VIX Fix panic-low reversal inside a long-term uptrend.

    Larry Williams' VIX Fix synthesises a VIX-like fear oscillator from price
    only: WVF = 100 * (HighestClose_22 - Low) / HighestClose_22. A WVF spike
    beyond its rolling Bollinger upper band marks capitulation. Gated by
    price > SMA(200) this becomes a trend-following dip-buy using a fear
    reading rather than momentum, range position, volatility width, RSI
    persistence, or volume. Exit is a quick reversion trigger (close > SMA5).

    Entry (at today's close):
      - WVF on the prior bar > BB_upper(WVF, 20, 2.0) of the prior bar
      - prior bar close > SMA(200) of the prior bar
    Exit: close > SMA(5).
    """
    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    hh_close_22 = (
        pd.Series(close).rolling(22, min_periods=22).max().to_numpy()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        wvf = 100.0 * (hh_close_22 - low) / hh_close_22
    wvf = np.where(np.isfinite(wvf), wvf, np.nan)

    wvf_mid = _sma(np.nan_to_num(wvf, nan=0.0), 20)
    wvf_std = _stdev(np.nan_to_num(wvf, nan=0.0), 20)
    wvf_bb_up = wvf_mid + 2.0 * wvf_std

    sma200 = _sma(close, 200)
    sma5 = _sma(close, 5)

    wvf_prev = np.concatenate(([np.nan], wvf[:-1]))
    wvf_bb_up_prev = np.concatenate(([np.nan], wvf_bb_up[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(wvf_prev)
        & np.isfinite(wvf_bb_up_prev)
        & np.isfinite(sma200_prev)
    )
    entries = (
        valid
        & (wvf_prev > wvf_bb_up_prev)
        & (close_prev > sma200_prev)
    )
    exits = np.isfinite(sma5) & (close > sma5)

    return _walk(entries, exits, close, df["date"].values)


def strat_nr7_breakout_trend(df: pd.DataFrame) -> list[Trade]:
    """Toby Crabel NR7 (narrowest range of 7) volatility-contraction breakout.

    NR7 bar: the prior bar's high-low range is the smallest of the trailing
    seven bars — a sign of coiling / compression. Entry fires when today's
    close then breaks above the NR7 bar's high, confirming directional
    release. A SMA(100) trend filter keeps trades with the larger trend.
    Exit when close drops below SMA(20) (momentum fails).

    Distinct from:
      - squeeze_breakout (BB width vs KC width — relative volatility),
      - donchian_20_10_trend (20-bar absolute highest-high channel),
      - ibs_trend_filter (single-bar range position),
      - volume_capitulation_reclaim (volume spike reclaim),
      - cum_rsi2_pullback (multi-bar RSI persistence),
      - williams_vix_fix_spike (price-based fear oscillator).
    This reads absolute daily range compression, not band width or oscillator.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    day_range = high - low
    # Rolling min of day_range over 7 bars ending at each index.
    min_range_7 = (
        pd.Series(day_range).rolling(7, min_periods=7).min().to_numpy()
    )
    is_nr7 = np.isfinite(min_range_7) & (day_range <= min_range_7 + 1e-12)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    # Prior-bar references so entry is decidable at today's close; no lookahead.
    is_nr7_prev = np.concatenate(([False], is_nr7[:-1]))
    high_prev = np.concatenate(([np.nan], high[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(high_prev)
        & np.isfinite(sma100_prev)
        & np.isfinite(close_prev)
    )
    entries = (
        valid
        & is_nr7_prev
        & (close > high_prev)
        & (close_prev > sma100_prev)
    )
    exits = np.isfinite(sma20) & (close < sma20)

    return _walk(entries, exits, close, df["date"].values)


def strat_adx_dmi_trend_emergence(df: pd.DataFrame) -> list[Trade]:
    """Wilder DMI/ADX trend-emergence entry with +DI dominance.

    Classic Wilder directional-movement system (period 14): +DI / -DI
    measure directional pressure, ADX measures trend strength. Entry fires
    when the prior bar shows ADX > 20 and rising (trend is strengthening)
    AND +DI > -DI (bullish directional dominance), gated by close above
    SMA(100). Exit when +DI falls below -DI (directional flip) or close
    drops below SMA(20). This is the only sandbox strategy that uses
    Wilder's DMI family — distinct from RSI-based (rsi_ema, macd_rsi,
    cum_rsi2_pullback), band/range (ibs, donchian, squeeze, bb_breakout,
    nr7), volume (volume_capitulation_reclaim), and the WVF fear oscillator.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    high_prev_raw = np.concatenate(([np.nan], high[:-1]))
    low_prev_raw = np.concatenate(([np.nan], low[:-1]))
    close_prev_tr = np.concatenate(([np.nan], close[:-1]))

    up_move = high - high_prev_raw
    dn_move = low_prev_raw - low
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    plus_dm = np.nan_to_num(plus_dm, nan=0.0)
    minus_dm = np.nan_to_num(minus_dm, nan=0.0)

    tr_raw = np.maximum.reduce([
        high - low,
        np.abs(high - close_prev_tr),
        np.abs(low - close_prev_tr),
    ])
    tr_raw = np.nan_to_num(tr_raw, nan=high - low)

    n = 14
    atr_n = _rma(tr_raw, n)
    plus_di = 100.0 * _rma(plus_dm, n) / np.where(atr_n > 0, atr_n, np.nan)
    minus_di = 100.0 * _rma(minus_dm, n) / np.where(atr_n > 0, atr_n, np.nan)

    di_sum = plus_di + minus_di
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where(di_sum > 0, di_sum, np.nan)
    adx = _rma(np.nan_to_num(dx, nan=0.0), n)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    # Prior-bar references so the entry signal is decidable at bar close.
    adx_prev = np.concatenate(([np.nan], adx[:-1]))
    adx_prev2 = np.concatenate(([np.nan, np.nan], adx[:-2]))
    plus_di_prev = np.concatenate(([np.nan], plus_di[:-1]))
    minus_di_prev = np.concatenate(([np.nan], minus_di[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(adx_prev)
        & np.isfinite(adx_prev2)
        & np.isfinite(plus_di_prev)
        & np.isfinite(minus_di_prev)
        & np.isfinite(sma100_prev)
    )
    entries = (
        valid
        & (adx_prev > 20.0)
        & (adx_prev > adx_prev2)
        & (plus_di_prev > minus_di_prev)
        & (close_prev > sma100_prev)
    )
    exits = (
        (np.isfinite(plus_di) & np.isfinite(minus_di) & (plus_di < minus_di))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_pocket_pivot(df: pd.DataFrame) -> list[Trade]:
    """O'Neil/Morales/Kacher Pocket Pivot — up-close on volume exceeding the
    largest down-day volume of the trailing 10 bars, inside an SMA(50) uptrend.

    Thesis: a Pocket Pivot reveals institutional accumulation — the biggest
    volume bar of the last ~2 weeks is an up day, implying net buying is
    overwhelming net selling. Entry fires when today's close is above the
    prior close AND today's volume is strictly greater than every down-day
    volume in the prior 10 bars AND close sits above SMA(50) (uptrend gate).
    Exit on a close below SMA(20) — momentum has faded.

    Distinct from volume_capitulation_reclaim which reads capitulation
    *reversal* after heavy selling; this reads accumulation *continuation*
    where the biggest recent volume is bullish, not bearish.
    """
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    close_prev = np.concatenate(([np.nan], close[:-1]))
    up_day = close > close_prev
    down_day = close < close_prev
    # Down-day volume is kept; non-down bars get -1 so they don't dominate max.
    down_vol = np.where(down_day, volume, -1.0)

    # Largest down-day volume over the PRIOR 10 bars — shift(1) avoids lookahead.
    max_down_vol_10 = (
        pd.Series(down_vol).shift(1).rolling(10, min_periods=10).max().to_numpy()
    )

    sma50 = _sma(close, 50)
    sma20 = _sma(close, 20)

    valid = (
        np.isfinite(close_prev)
        & np.isfinite(sma50)
        & np.isfinite(max_down_vol_10)
    )
    entries = (
        valid
        & up_day
        & (volume > max_down_vol_10)
        & (close > sma50)
    )
    exits = np.isfinite(sma20) & (close < sma20)

    return _walk(entries, exits, close, df["date"].values)


NEW_STRATEGIES: dict = {
    "ibs_trend_filter": strat_ibs_trend_filter,
    "donchian_20_10_trend": strat_donchian_20_10_trend,
    "squeeze_breakout": strat_squeeze_breakout,
    "cum_rsi2_pullback": strat_cum_rsi2_pullback,
    "volume_capitulation_reclaim": strat_volume_capitulation_reclaim,
    "williams_vix_fix_spike": strat_williams_vix_fix_spike,
    "nr7_breakout_trend": strat_nr7_breakout_trend,
    "adx_dmi_trend_emergence": strat_adx_dmi_trend_emergence,
    "pocket_pivot": strat_pocket_pivot,
}
