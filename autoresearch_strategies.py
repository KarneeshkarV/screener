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


def strat_cmf_zero_reclaim(df: pd.DataFrame) -> list[Trade]:
    """Chaikin Money Flow (CMF) zero-line reclaim inside a long-term uptrend.

    CMF(N) = sum(MFV, N) / sum(volume, N), where
    MFV = volume * ((close - low) - (high - close)) / (high - low).
    It measures N-bar accumulation/distribution by weighting close-within-range
    by volume — a sustained money-flow reading rather than a single-bar event.
    A CMF flip from <= 0 to > 0 signals that net selling has turned to net
    buying; inside an established uptrend this is a high-conviction continuation.

    Entry (at today's close):
      - prior-bar CMF(20) <= 0 AND current-bar CMF(20) > 0 (zero-line reclaim)
      - close > SMA(100) (uptrend gate)
    Exit: CMF(20) falls back below 0 OR close < SMA(20).

    Distinct from:
      - volume_capitulation_reclaim (single-bar volume spike reclaim),
      - pocket_pivot (single-day up-volume > recent down-volume max),
      - ibs_trend_filter (single-bar range position, no volume),
      - cum_rsi2_pullback (multi-day RSI persistence, no volume).
    This is the only sandbox strategy that reads a multi-bar money-flow
    oscillator combining range position with volume.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    rng = high - low
    safe_rng = np.where(rng > 0, rng, np.nan)
    mf_mult = ((close - low) - (high - close)) / safe_rng
    mf_mult = np.nan_to_num(mf_mult, nan=0.0)
    mf_vol = mf_mult * volume

    n = 20
    sum_mfv = pd.Series(mf_vol).rolling(n, min_periods=n).sum().to_numpy()
    sum_vol = pd.Series(volume).rolling(n, min_periods=n).sum().to_numpy()
    cmf = np.where(sum_vol > 0, sum_mfv / sum_vol, np.nan)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    cmf_prev = np.concatenate(([np.nan], cmf[:-1]))

    valid = np.isfinite(cmf) & np.isfinite(cmf_prev) & np.isfinite(sma100)
    entries = valid & (cmf_prev <= 0.0) & (cmf > 0.0) & (close > sma100)
    exits = (
        (np.isfinite(cmf) & (cmf < 0.0))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_aroon_cross_trend(df: pd.DataFrame) -> list[Trade]:
    """Chande Aroon(25) bullish cross with Aroon Up >= 70, gated by SMA(100).

    Aroon measures *how recently* (in bars) the highest high / lowest low
    occurred within the last N bars — a time-since-extremes indicator rather
    than a price or volume one:
        AroonUp(N)   = 100 * (N - bars_since_HH_N) / N
        AroonDown(N) = 100 * (N - bars_since_LL_N) / N
    A fresh cross where AroonUp takes over AroonDown signals recent action is
    printing new highs faster than new lows — an early trend-emergence read.

    Entry (at today's close):
      - prior bar showed a fresh bullish cross (AroonUp2 <= AroonDown2 AND
        AroonUp1 > AroonDown1)
      - prior bar AroonUp >= 70 (HH occurred within the latest ~30% of window)
      - prior bar close > SMA(100)
    Exit: AroonDown > AroonUp OR close < SMA(20).

    Distinct from every other sandbox strategy: Donchian reads absolute
    high-low levels, NR7/squeeze read range compression, ADX reads trend
    strength magnitude, RSI/MACD/WVF are price-momentum oscillators, CMF /
    pocket_pivot / volume_capitulation use volume. Aroon reads *time since*
    the extreme — a geometry none of them capture.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    n = 25
    hh_arg = (
        pd.Series(high).rolling(n, min_periods=n).apply(np.argmax, raw=True).to_numpy()
    )
    ll_arg = (
        pd.Series(low).rolling(n, min_periods=n).apply(np.argmin, raw=True).to_numpy()
    )
    bars_since_hh = (n - 1) - hh_arg
    bars_since_ll = (n - 1) - ll_arg
    aroon_up = 100.0 * (n - bars_since_hh) / n
    aroon_dn = 100.0 * (n - bars_since_ll) / n

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    au_prev = np.concatenate(([np.nan], aroon_up[:-1]))
    ad_prev = np.concatenate(([np.nan], aroon_dn[:-1]))
    au_prev2 = np.concatenate(([np.nan, np.nan], aroon_up[:-2]))
    ad_prev2 = np.concatenate(([np.nan, np.nan], aroon_dn[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(au_prev)
        & np.isfinite(ad_prev)
        & np.isfinite(au_prev2)
        & np.isfinite(ad_prev2)
        & np.isfinite(sma100_prev)
    )
    fresh_cross = (au_prev2 <= ad_prev2) & (au_prev > ad_prev)
    entries = (
        valid
        & fresh_cross
        & (au_prev >= 70.0)
        & (close_prev > sma100_prev)
    )
    exits = (
        (np.isfinite(aroon_up) & np.isfinite(aroon_dn) & (aroon_dn > aroon_up))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_heikin_ashi_flip(df: pd.DataFrame) -> list[Trade]:
    """Heikin-Ashi bullish-body flip after a bearish run, gated by SMA(100).

    Heikin-Ashi transforms raw OHLC into smoothed synthetic candles:
        HA_close = (open + high + low + close) / 4
        HA_open  = (prev HA_open + prev HA_close) / 2
        HA_high  = max(high, HA_open, HA_close)
        HA_low   = min(low,  HA_open, HA_close)
    A bullish HA bar has HA_close > HA_open; a series of bullish HA bars marks
    a trend. The cleanest signal is the *flip* — the first bullish HA bar after
    a string of bearish ones — filtered for a meaningful body so noise doesn't
    trigger it.

    Entry (at today's close, using prior-bar HA values to avoid lookahead):
      - prior bar was HA bullish (HA_close > HA_open) with body > 30% of range
      - the bar before that (prev-prev) was HA bearish (HA_close <= HA_open)
      - prior-bar close > SMA(100) (uptrend gate)
    Exit: HA turns bearish (HA_close < HA_open) OR close < SMA(20).

    Distinct from every other sandbox strategy: this reads a synthetic
    smoothed candle transformation rather than price levels (Donchian), band
    width (squeeze/BB), range position (IBS), range compression (NR7),
    oscillators (RSI/MACD/WVF/Aroon/ADX), volume flows (CMF/pocket_pivot/
    volume_capitulation), or momentum magnitude. HA's recursive open makes
    it a genuinely different input space.
    """
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    n = close.shape[0]
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = np.empty(n, dtype=float)
    if n > 0:
        ha_open[0] = (open_[0] + close[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])

    ha_body = ha_close - ha_open
    ha_range = np.where((ha_high - ha_low) > 0, ha_high - ha_low, np.nan)
    body_frac = np.abs(ha_body) / ha_range
    body_frac = np.nan_to_num(body_frac, nan=0.0)

    bullish = ha_body > 0
    bearish = ha_body <= 0

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    bullish_prev = np.concatenate(([False], bullish[:-1]))
    bearish_prev2 = np.concatenate(([False, False], bearish[:-2]))
    body_frac_prev = np.concatenate(([0.0], body_frac[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = np.isfinite(close_prev) & np.isfinite(sma100_prev)
    entries = (
        valid
        & bullish_prev
        & bearish_prev2
        & (body_frac_prev > 0.3)
        & (close_prev > sma100_prev)
    )
    exits = (
        (ha_close < ha_open)
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_ichimoku_kumo_breakout(df: pd.DataFrame) -> list[Trade]:
    """Ichimoku Kinko Hyo cloud (Kumo) breakout with bullish Tenkan/Kijun state.

    Ichimoku builds its cloud from 26-bar-forward-displaced midpoints of two
    different Donchian windows:
        Tenkan(9)  = (HH9  + LL9)  / 2
        Kijun(26)  = (HH26 + LL26) / 2
        SenkouA    = (Tenkan + Kijun) / 2   (plotted 26 bars ahead)
        SenkouB(52)= (HH52 + LL52) / 2      (plotted 26 bars ahead)
    The cloud at time t is therefore computed from bars <= t-26 — knowable at
    today's close with no lookahead.

    Entry (at today's close):
      - prior-bar close is above prior-bar cloud upper (above the Kumo)
      - the bar before that was at or below its cloud upper (fresh breakout)
      - prior-bar Tenkan > Kijun (bullish TK-cross state)
    Exit: close falls below the Kijun line (standard Ichimoku trailing exit).

    Distinct from every other sandbox strategy: the 26-bar-displaced cloud is
    a unique support/resistance object that none of Donchian (same-bar H/L
    channel), supertrend (ATR trail), BB/KC (volatility bands), HA (synthetic
    candles), Aroon (time-since-extreme), NR7 (compression), ADX/DMI (trend
    magnitude), RSI/MACD/WVF/CMF (oscillators), or volume strategies capture.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    def donchian_mid(h: np.ndarray, l: np.ndarray, n: int) -> np.ndarray:
        hh = pd.Series(h).rolling(n, min_periods=n).max().to_numpy()
        ll = pd.Series(l).rolling(n, min_periods=n).min().to_numpy()
        return (hh + ll) / 2.0

    tenkan = donchian_mid(high, low, 9)
    kijun = donchian_mid(high, low, 26)
    senkou_a_raw = (tenkan + kijun) / 2.0
    senkou_b_raw = donchian_mid(high, low, 52)

    displacement = 26
    pad = np.full(displacement, np.nan)
    senkou_a = np.concatenate((pad, senkou_a_raw[:-displacement]))
    senkou_b = np.concatenate((pad, senkou_b_raw[:-displacement]))
    cloud_upper = np.maximum(senkou_a, senkou_b)

    close_prev = np.concatenate(([np.nan], close[:-1]))
    close_prev2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    cloud_upper_prev = np.concatenate(([np.nan], cloud_upper[:-1]))
    cloud_upper_prev2 = np.concatenate(([np.nan, np.nan], cloud_upper[:-2]))
    tenkan_prev = np.concatenate(([np.nan], tenkan[:-1]))
    kijun_prev = np.concatenate(([np.nan], kijun[:-1]))

    valid = (
        np.isfinite(cloud_upper_prev)
        & np.isfinite(cloud_upper_prev2)
        & np.isfinite(tenkan_prev)
        & np.isfinite(kijun_prev)
        & np.isfinite(close_prev)
        & np.isfinite(close_prev2)
    )
    fresh_breakout = (close_prev > cloud_upper_prev) & (
        close_prev2 <= cloud_upper_prev2
    )
    tk_bullish = tenkan_prev > kijun_prev
    entries = valid & fresh_breakout & tk_bullish
    exits = np.isfinite(kijun) & (close < kijun)

    return _walk(entries, exits, close, df["date"].values)


def strat_parabolic_sar_flip_trend(df: pd.DataFrame) -> list[Trade]:
    """Wilder Parabolic SAR bullish flip in SMA(100) uptrend.

    PSAR is an iterative trailing stop with an acceleration factor (AF) that
    ratchets up 0.02 each time a new extreme point (EP) is made, capped at
    0.20. A "flip" occurs when price penetrates the SAR, reversing the trend
    and resetting SAR to the prior EP with AF back to 0.02.

    Entry: the bar on which SAR flips from above price to below price (bear->
    bull reversal) AND close > SMA(100). This isolates the PSAR bullish
    reversal signal to established uptrends — filtering out whipsaws that
    occur in downtrends or chop.
    Exit: the next bull->bear SAR flip OR close < SMA(20) safety stop.

    Distinct from Supertrend (HL2 +/- ATR*mult, constant band width),
    Donchian (raw highest-high/lowest-low channels), Ichimoku (displaced
    midpoints), and every MA/oscillator/candle pattern strategy above,
    because PSAR's accelerating trailing-stop produces a different signal
    geometry — flips occur only after price violates an adaptive stop that
    tightens as the move extends.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)
    if n < 3:
        return []

    af_step = 0.02
    af_max = 0.20

    flip_up = np.zeros(n, dtype=bool)
    flip_down = np.zeros(n, dtype=bool)

    # Seed starting trend from the first two bars
    if close[1] >= close[0]:
        trend = 1
        sar_prev = low[0]
        ep = high[0]
    else:
        trend = -1
        sar_prev = high[0]
        ep = low[0]
    af = af_step

    for i in range(1, n):
        new_sar = sar_prev + af * (ep - sar_prev)
        if trend == 1:
            # In an uptrend SAR cannot penetrate low of prior two bars
            if i >= 2:
                new_sar = min(new_sar, low[i - 1], low[i - 2])
            else:
                new_sar = min(new_sar, low[i - 1])
            if low[i] < new_sar:
                trend = -1
                new_sar = ep
                ep = low[i]
                af = af_step
                flip_down[i] = True
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            if i >= 2:
                new_sar = max(new_sar, high[i - 1], high[i - 2])
            else:
                new_sar = max(new_sar, high[i - 1])
            if high[i] > new_sar:
                trend = 1
                new_sar = ep
                ep = high[i]
                af = af_step
                flip_up[i] = True
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        sar_prev = new_sar

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    entries = flip_up & np.isfinite(sma100) & (close > sma100)
    exits = flip_down | (np.isfinite(sma20) & (close < sma20))

    return _walk(entries, exits, close, df["date"].values)


def strat_vortex_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """Botes/Siepman Vortex Indicator (2010) bullish cross in SMA(100) uptrend.

    The Vortex Indicator measures *directional sweeps* relative to the prior
    bar's extremes — a different geometry than DMI's +DM/-DM (today vs
    yesterday's high-low delta, RMA-smoothed) or Aroon (time-since-extreme):
        VM+_t = |high_t - low_{t-1}|   (upward sweep from yesterday's low)
        VM-_t = |low_t  - high_{t-1}|  (downward sweep from yesterday's high)
        TR_t  = max(h-l, |h-c_prev|, |l-c_prev|)
        VI+(N) = sum(VM+, N) / sum(TR, N)
        VI-(N) = sum(VM-, N) / sum(TR, N)

    Entry (at today's close, using prior-bar values to avoid lookahead):
      - fresh bullish cross: VI+_{t-2} <= VI-_{t-2} AND VI+_{t-1} > VI-_{t-1}
      - prior-bar close > SMA(100) (uptrend gate)
    Exit: VI- > VI+ (directional flip) OR close < SMA(20).

    Distinct from adx_dmi_trend_emergence (Wilder DMI uses up_move/dn_move
    deltas with RMA smoothing, plus ADX trend-strength magnitude), from
    aroon_cross_trend (time-since-HH/LL), from donchian/ichimoku (level
    breakouts), from the RSI/MACD/WVF/CMF oscillators, and from every volume
    or candle strategy: VI reads the *reach* from yesterday's extremes, which
    picks up different pivots than any of the above.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    high_prev = np.concatenate(([np.nan], high[:-1]))
    low_prev = np.concatenate(([np.nan], low[:-1]))
    close_prev_tr = np.concatenate(([np.nan], close[:-1]))

    vm_plus = np.abs(high - low_prev)
    vm_minus = np.abs(low - high_prev)
    tr_raw = np.maximum.reduce([
        high - low,
        np.abs(high - close_prev_tr),
        np.abs(low - close_prev_tr),
    ])
    vm_plus = np.nan_to_num(vm_plus, nan=0.0)
    vm_minus = np.nan_to_num(vm_minus, nan=0.0)
    tr_raw = np.nan_to_num(tr_raw, nan=0.0)

    n = 14
    sum_vm_plus = pd.Series(vm_plus).rolling(n, min_periods=n).sum().to_numpy()
    sum_vm_minus = pd.Series(vm_minus).rolling(n, min_periods=n).sum().to_numpy()
    sum_tr = pd.Series(tr_raw).rolling(n, min_periods=n).sum().to_numpy()

    safe_tr = np.where(sum_tr > 0, sum_tr, np.nan)
    vi_plus = sum_vm_plus / safe_tr
    vi_minus = sum_vm_minus / safe_tr

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    vip_prev = np.concatenate(([np.nan], vi_plus[:-1]))
    vim_prev = np.concatenate(([np.nan], vi_minus[:-1]))
    vip_prev2 = np.concatenate(([np.nan, np.nan], vi_plus[:-2]))
    vim_prev2 = np.concatenate(([np.nan, np.nan], vi_minus[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(vip_prev)
        & np.isfinite(vim_prev)
        & np.isfinite(vip_prev2)
        & np.isfinite(vim_prev2)
        & np.isfinite(sma100_prev)
        & np.isfinite(close_prev)
    )
    fresh_cross = (vip_prev2 <= vim_prev2) & (vip_prev > vim_prev)
    entries = valid & fresh_cross & (close_prev > sma100_prev)
    exits = (
        (np.isfinite(vi_plus) & np.isfinite(vi_minus) & (vi_minus > vi_plus))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_rvi_signal_cross(df: pd.DataFrame) -> list[Trade]:
    """Donald Dorsey's Relative Vigor Index (RVI) bullish signal-line cross.

    RVI reads whether closes tend to finish near the top of each bar's range
    relative to the full range — a different dimension than any other
    sandbox strategy:
        CO_t = close_t - open_t
        HL_t = high_t - low_t
        SWMA1221(x)_t = (x_t + 2 x_{t-1} + 2 x_{t-2} + x_{t-3}) / 6
        Num = SMA(SWMA1221(CO), 10)
        Den = SMA(SWMA1221(HL), 10)
        RVI = Num / Den
        Signal = SWMA1221(RVI)
    A rising RVI means bars are finishing proportionally higher inside their
    ranges — a subtle vigor read. A signal-line cross-above is the standard
    trigger; gating by SMA(100) keeps trades aligned with the larger trend.

    Distinct from every existing sandbox strategy: the close-open body
    normalised by high-low range is a geometry none of the RSI/MACD/CMO/
    WVF/CMF/ADX/Aroon/Vortex oscillators capture. Heikin-Ashi rebuilds
    synthetic candles but does not expose a body-vs-range cross signal.

    Entry (at today's close, prior-bar values avoid lookahead):
      - prior-bar RVI > prior-bar Signal AND bar-before-that RVI <= Signal
        (fresh cross-above, no repeat firings)
      - prior-bar close > SMA(100)
    Exit: RVI < Signal OR close < SMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    co = close - open_
    hl = high - low

    def swma_1221(arr: np.ndarray) -> np.ndarray:
        s = pd.Series(arr)
        return (
            (s + 2.0 * s.shift(1) + 2.0 * s.shift(2) + s.shift(3)) / 6.0
        ).to_numpy()

    num = swma_1221(co)
    den = swma_1221(hl)

    n = 10
    num_avg = pd.Series(num).rolling(n, min_periods=n).mean().to_numpy()
    den_avg = pd.Series(den).rolling(n, min_periods=n).mean().to_numpy()

    rvi = np.where(
        np.isfinite(den_avg) & (den_avg > 0), num_avg / den_avg, np.nan
    )
    signal = swma_1221(rvi)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    rvi_prev = np.concatenate(([np.nan], rvi[:-1]))
    signal_prev = np.concatenate(([np.nan], signal[:-1]))
    rvi_prev2 = np.concatenate(([np.nan, np.nan], rvi[:-2]))
    signal_prev2 = np.concatenate(([np.nan, np.nan], signal[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(rvi_prev)
        & np.isfinite(signal_prev)
        & np.isfinite(rvi_prev2)
        & np.isfinite(signal_prev2)
        & np.isfinite(sma100_prev)
        & np.isfinite(close_prev)
    )
    fresh_cross = (rvi_prev2 <= signal_prev2) & (rvi_prev > signal_prev)
    entries = valid & fresh_cross & (close_prev > sma100_prev)
    exits = (
        (np.isfinite(rvi) & np.isfinite(signal) & (rvi < signal))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_fisher_transform_zero_cross(df: pd.DataFrame) -> list[Trade]:
    """Ehlers Fisher Transform zero-line bullish cross in SMA(100) uptrend.

    Ehlers' Fisher Transform reshapes normalised price into a near-Gaussian
    distribution, making turning points sharper than linear RSI-style
    oscillators. For window N=9:
        hl_mid = (high + low) / 2
        raw_t  = (hl_mid - rolling_min(hl_mid, N))
                 / (rolling_max(hl_mid, N) - rolling_min(hl_mid, N))
        x_t    = 0.66 * (raw_t - 0.5) + 0.67 * x_{t-1}, clipped to [-0.999, 0.999]
        Fisher_t = 0.5 * ln((1 + x_t) / (1 - x_t)) + 0.5 * Fisher_{t-1}
    The logarithmic reshape amplifies extremes, so Fisher crossing its zero
    line marks a decisive bullish regime flip.

    Entry (at today's close, prior-bar values for no lookahead):
      - Fisher_{t-2} <= 0 AND Fisher_{t-1} > 0 (fresh zero-cross)
      - close_{t-1} > SMA(100)
    Exit: Fisher < 0 OR close < SMA(20).

    Distinct from every existing sandbox oscillator: the log-Fisher reshape
    of a normalised high-low midpoint produces tail-amplified signals that
    none of RSI / MACD / CMO / WVF / CMF / ADX / Aroon / Vortex / RVI / HA /
    Ichimoku / PSAR / Donchian capture.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)

    hl_mid = (high + low) / 2.0
    N = 9
    maxh = pd.Series(hl_mid).rolling(N, min_periods=N).max().to_numpy()
    minl = pd.Series(hl_mid).rolling(N, min_periods=N).min().to_numpy()

    fisher = np.full(n, np.nan)
    x_prev = 0.0
    fish_prev = 0.0
    for i in range(n):
        if not (np.isfinite(maxh[i]) and np.isfinite(minl[i])):
            continue
        rng = maxh[i] - minl[i]
        if rng <= 0:
            raw = 0.5
        else:
            raw = (hl_mid[i] - minl[i]) / rng
        x = 0.66 * (raw - 0.5) + 0.67 * x_prev
        if x > 0.999:
            x = 0.999
        elif x < -0.999:
            x = -0.999
        fish = 0.5 * np.log((1.0 + x) / (1.0 - x)) + 0.5 * fish_prev
        fisher[i] = fish
        x_prev = x
        fish_prev = fish

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    fisher_prev = np.concatenate(([np.nan], fisher[:-1]))
    fisher_prev2 = np.concatenate(([np.nan, np.nan], fisher[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(fisher_prev)
        & np.isfinite(fisher_prev2)
        & np.isfinite(sma100_prev)
        & np.isfinite(close_prev)
    )
    fresh_cross = (fisher_prev2 <= 0.0) & (fisher_prev > 0.0)
    entries = valid & fresh_cross & (close_prev > sma100_prev)
    exits = (
        (np.isfinite(fisher) & (fisher < 0.0))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_trix_signal_cross(df: pd.DataFrame) -> list[Trade]:
    """Jack Hutson's TRIX signal-line bullish cross in SMA(100) uptrend.

    TRIX is the 1-bar rate-of-change of a triple-EMA-smoothed price series —
    a noise-filtered momentum measure that suppresses cycles shorter than the
    EMA window. For window N=15, signal window M=9:
        ema1 = EMA(close, N)
        ema2 = EMA(ema1,  N)
        ema3 = EMA(ema2,  N)
        TRIX = 10000 * (ema3_t - ema3_{t-1}) / ema3_{t-1}
        Signal = EMA(TRIX, M)
    Triple smoothing removes the leads and lags that RSI/MACD signals retain,
    so TRIX/Signal crosses filter out most whipsaws of faster oscillators.

    Entry (at today's close, prior-bar values for no lookahead):
      - fresh bullish cross: TRIX_{t-2} <= Signal_{t-2} AND TRIX_{t-1} > Signal_{t-1}
      - close_{t-1} > SMA(100) (uptrend gate)
    Exit: TRIX < Signal OR close < SMA(20).

    Distinct from every existing sandbox oscillator: MACD is a two-EMA diff
    (single smoothing), RVI is close-open over high-low with SWMA, RSI/CMO
    are up/down-move ratios, WVF is a fear proxy, Fisher is a log-reshape of
    normalised HL midpoint, Ichimoku/Vortex/Aroon/PSAR/Donchian use price
    geometry. TRIX's triple-EMA rate of change is a second-derivative
    momentum signal that none of them reproduce.
    """
    close = df["close"].to_numpy(dtype=float)

    n_ema = 15
    ema1 = _ema(close, n_ema)
    ema2 = _ema(ema1, n_ema)
    ema3 = _ema(ema2, n_ema)

    ema3_prev = np.concatenate(([np.nan], ema3[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        trix = 10000.0 * (ema3 - ema3_prev) / ema3_prev
    trix = np.where(np.isfinite(trix), trix, np.nan)

    signal = _ema(np.nan_to_num(trix, nan=0.0), 9)
    # Invalidate signal while trix itself is nan so the cross doesn't fire on
    # the zero-imputed warm-up region.
    signal = np.where(np.isfinite(trix), signal, np.nan)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    trix_prev = np.concatenate(([np.nan], trix[:-1]))
    signal_prev = np.concatenate(([np.nan], signal[:-1]))
    trix_prev2 = np.concatenate(([np.nan, np.nan], trix[:-2]))
    signal_prev2 = np.concatenate(([np.nan, np.nan], signal[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(trix_prev)
        & np.isfinite(signal_prev)
        & np.isfinite(trix_prev2)
        & np.isfinite(signal_prev2)
        & np.isfinite(sma100_prev)
        & np.isfinite(close_prev)
    )
    fresh_cross = (trix_prev2 <= signal_prev2) & (trix_prev > signal_prev)
    entries = valid & fresh_cross & (close_prev > sma100_prev)
    exits = (
        (np.isfinite(trix) & np.isfinite(signal) & (trix < signal))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_hma_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """Alan Hull's Hull Moving Average (HMA) bullish cross in SMA(100) uptrend.

    HMA substantially reduces moving-average lag by combining two WMAs:
        HMA(n) = WMA( 2 * WMA(close, n/2) - WMA(close, n), sqrt(n) )
    The nested WMA difference extrapolates the short-term trend, then a final
    WMA smooths the result over sqrt(n). The outcome tracks price much more
    responsively than SMA/EMA of comparable length, so HMA crosses are cleaner
    trend-flip signals with fewer whipsaws than SMA/EMA crosses.

    Entry (at today's close, prior-bar values for no lookahead):
      - fresh bullish HMA cross: HMA16_{t-2} <= HMA49_{t-2} AND HMA16_{t-1} > HMA49_{t-1}
      - close_{t-1} > SMA(100) (macro uptrend gate)
    Exit: HMA16 < HMA49 OR close < SMA(20).

    Distinct from every existing sandbox strategy: supertrend uses ATR-trailing
    bands (not MAs), TRIX uses a triple-EMA rate of change, MACD/RVI/RSI/CMO/
    Fisher/WVF/CMF are oscillators, Donchian/Ichimoku use raw H/L channels,
    squeeze/BB use σ bands, HA/NR7 use candle geometry, Aroon/PSAR/Vortex use
    pivots or adaptive stops. No existing strategy uses linearly-weighted
    moving averages — HMA's nested WMA with sqrt(n) outer smoothing is a
    genuinely different filter geometry.
    """
    close = df["close"].to_numpy(dtype=float)

    def _wma(arr: np.ndarray, n: int) -> np.ndarray:
        weights = np.arange(1, n + 1, dtype=float)
        wsum = weights.sum()
        return (
            pd.Series(arr)
            .rolling(n, min_periods=n)
            .apply(lambda x: np.dot(x, weights) / wsum, raw=True)
            .to_numpy()
        )

    def _hma(arr: np.ndarray, n: int) -> np.ndarray:
        half = max(2, n // 2)
        sqrt_n = max(2, int(round(np.sqrt(n))))
        w_half = _wma(arr, half)
        w_full = _wma(arr, n)
        raw = 2.0 * w_half - w_full
        # Fill warm-up NaNs with 0 for the outer WMA, then mask back to NaN
        # where the inner WMAs were still warming up.
        raw_mask = np.isfinite(raw)
        raw_clean = np.where(raw_mask, raw, 0.0)
        outer = _wma(raw_clean, sqrt_n)
        # Any outer window that included a warm-up NaN is invalid.
        mask_valid = (
            pd.Series(raw_mask.astype(float))
            .rolling(sqrt_n, min_periods=sqrt_n)
            .sum()
            .to_numpy()
            == sqrt_n
        )
        return np.where(mask_valid, outer, np.nan)

    hma_fast = _hma(close, 16)
    hma_slow = _hma(close, 49)
    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    fast_prev = np.concatenate(([np.nan], hma_fast[:-1]))
    slow_prev = np.concatenate(([np.nan], hma_slow[:-1]))
    fast_prev2 = np.concatenate(([np.nan, np.nan], hma_fast[:-2]))
    slow_prev2 = np.concatenate(([np.nan, np.nan], hma_slow[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(fast_prev)
        & np.isfinite(slow_prev)
        & np.isfinite(fast_prev2)
        & np.isfinite(slow_prev2)
        & np.isfinite(sma100_prev)
        & np.isfinite(close_prev)
    )
    fresh_cross = (fast_prev2 <= slow_prev2) & (fast_prev > slow_prev)
    entries = valid & fresh_cross & (close_prev > sma100_prev)
    exits = (
        (np.isfinite(hma_fast) & np.isfinite(hma_slow) & (hma_fast < hma_slow))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_kama_cross_trend(df: pd.DataFrame) -> list[Trade]:
    """Perry Kaufman's KAMA efficiency-ratio adaptive-MA bullish cross.

    KAMA is an adaptive moving average whose smoothing constant widens or
    tightens bar-by-bar based on the signal-to-noise ratio of recent price
    action. For window N=10, fast=2, slow=30:
        change_t = |close_t - close_{t-N}|
        volat_t  = sum_{i=0..N-1} |close_{t-i} - close_{t-i-1}|
        ER_t     = change_t / volat_t              (efficiency ratio, 0..1)
        fast_sc  = 2 / (2 + 1) = 0.6667
        slow_sc  = 2 / (30 + 1) ≈ 0.0645
        SC_t     = (ER_t * (fast_sc - slow_sc) + slow_sc)^2
        KAMA_t   = KAMA_{t-1} + SC_t * (close_t - KAMA_{t-1})
    When markets are trending cleanly (high ER) KAMA tracks close to price;
    when noisy (low ER) it flattens out — a self-adjusting filter that no
    fixed-window MA (SMA/EMA/WMA/HMA/TEMA) achieves.

    Entry (at today's close, prior-bar values for no lookahead):
      - fresh bullish cross: close_{t-2} <= KAMA_{t-2} AND close_{t-1} > KAMA_{t-1}
      - close_{t-1} > SMA(100) (macro uptrend gate)
    Exit: close < KAMA OR close < SMA(20).

    Distinct from every existing sandbox strategy: HMA is WMA-based and fixed
    window, TRIX is triple-EMA rate of change, supertrend is ATR-band trail,
    PSAR is an accelerating stop, Donchian/Ichimoku are H/L channels, the
    RSI/MACD/CMO/RVI/Fisher/WVF/CMF/ADX/Aroon/Vortex family are oscillators,
    HA/NR7 are candle-geometry, pocket_pivot/volume_capitulation/CMF are
    volume. KAMA's efficiency-ratio-driven smoothing constant produces a
    genuinely different filter geometry — it reshapes its own cutoff.
    """
    close = df["close"].to_numpy(dtype=float)
    n_bars = len(close)

    N = 10
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (30.0 + 1.0)

    abs_change = np.abs(np.diff(close, prepend=close[0]))
    volat = pd.Series(abs_change).rolling(N, min_periods=N).sum().to_numpy()
    close_n_ago = np.concatenate(
        (np.full(N, np.nan), close[:-N])
    )
    change = np.abs(close - close_n_ago)
    er = np.where(
        np.isfinite(volat) & (volat > 0), change / volat, np.nan
    )
    sc = np.where(
        np.isfinite(er),
        (er * (fast_sc - slow_sc) + slow_sc) ** 2,
        np.nan,
    )

    kama = np.full(n_bars, np.nan)
    # Seed KAMA at the first bar where SC is valid, using close as starting value.
    seeded = False
    for i in range(n_bars):
        if not seeded:
            if np.isfinite(sc[i]):
                kama[i] = close[i]
                seeded = True
            continue
        prev = kama[i - 1]
        if not np.isfinite(prev):
            kama[i] = close[i]
            continue
        if np.isfinite(sc[i]):
            kama[i] = prev + sc[i] * (close[i] - prev)
        else:
            kama[i] = prev

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    kama_prev = np.concatenate(([np.nan], kama[:-1]))
    kama_prev2 = np.concatenate(([np.nan, np.nan], kama[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    close_prev2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(kama_prev)
        & np.isfinite(kama_prev2)
        & np.isfinite(close_prev)
        & np.isfinite(close_prev2)
        & np.isfinite(sma100_prev)
    )
    fresh_cross = (close_prev2 <= kama_prev2) & (close_prev > kama_prev)
    entries = valid & fresh_cross & (close_prev > sma100_prev)
    exits = (
        (np.isfinite(kama) & (close < kama))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_schaff_trend_cycle(df: pd.DataFrame) -> list[Trade]:
    """Schaff Trend Cycle (STC) oversold-to-trend bullish cross.

    Doug Schaff's STC is a recursive double-smoothed stochastic applied to a
    MACD line — it oscillates 0..100 far faster than MACD itself while staying
    smoother than raw stochastic. Construction (length=10, fast=23, slow=50,
    smoothing factor f=0.5):

        macd_t  = EMA(close, 23) - EMA(close, 50)
        ll1     = min(macd, 10 bars);  hh1 = max(macd, 10 bars)
        %K1_t   = 100 * (macd_t - ll1) / (hh1 - ll1)         (guarded /0)
        %D1_t   = %D1_{t-1} + f * (%K1_t - %D1_{t-1})        (EMA-like via f)
        ll2     = min(%D1, 10 bars);  hh2 = max(%D1, 10 bars)
        %K2_t   = 100 * (%D1_t - ll2) / (hh2 - ll2)
        STC_t   = STC_{t-1} + f * (%K2_t - STC_{t-1})

    Intuition: first stochastic finds where MACD sits in its own recent range
    (normalises the trend signal); smoothing + second stochastic compresses
    out noise and yields a bounded cycle oscillator that turns up at
    oversold-to-trend transitions ~5 bars ahead of MACD and without the slow
    lag of a 9-period signal line. Schaff's own thresholds are 25/75.

    Entry (decisions at bar close, using prior-bar values — no lookahead):
      - fresh bullish cross: STC_{t-2} <= 25 AND STC_{t-1} > 25
      - close_{t-1} > SMA(100)  (macro uptrend gate)
    Exit: STC crosses down through 75 OR close < SMA(20).

    Distinct from every existing sandbox strategy: MACD_RSI uses raw MACD
    signal-line cross, TRIX is triple-EMA ROC with its own signal line, RVI
    is vigor-range, Fisher is a Gaussian-transformed price normalisation,
    KAMA/HMA/Donchian/Ichimoku are different filter/channel geometries. STC's
    recursive-double-stochastic-of-MACD construction produces a bounded cycle
    oscillator no other strategy approximates.
    """
    close = df["close"].to_numpy(dtype=float)
    n_bars = len(close)

    length = 10
    fast_len = 23
    slow_len = 50
    f = 0.5

    ema_fast = _ema(close, fast_len)
    ema_slow = _ema(close, slow_len)
    macd = ema_fast - ema_slow

    macd_s = pd.Series(macd)
    ll1 = macd_s.rolling(length, min_periods=length).min().to_numpy()
    hh1 = macd_s.rolling(length, min_periods=length).max().to_numpy()
    rng1 = hh1 - ll1
    k1 = np.where(
        np.isfinite(rng1) & (rng1 > 0),
        100.0 * (macd - ll1) / rng1,
        np.nan,
    )

    d1 = np.full(n_bars, np.nan)
    seeded = False
    for i in range(n_bars):
        if not seeded:
            if np.isfinite(k1[i]):
                d1[i] = k1[i]
                seeded = True
            continue
        prev = d1[i - 1]
        if np.isfinite(k1[i]):
            d1[i] = prev + f * (k1[i] - prev)
        else:
            d1[i] = prev

    d1_s = pd.Series(d1)
    ll2 = d1_s.rolling(length, min_periods=length).min().to_numpy()
    hh2 = d1_s.rolling(length, min_periods=length).max().to_numpy()
    rng2 = hh2 - ll2
    k2 = np.where(
        np.isfinite(rng2) & (rng2 > 0),
        100.0 * (d1 - ll2) / rng2,
        np.nan,
    )

    stc = np.full(n_bars, np.nan)
    seeded = False
    for i in range(n_bars):
        if not seeded:
            if np.isfinite(k2[i]):
                stc[i] = k2[i]
                seeded = True
            continue
        prev = stc[i - 1]
        if np.isfinite(k2[i]):
            stc[i] = prev + f * (k2[i] - prev)
        else:
            stc[i] = prev

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    stc_prev = np.concatenate(([np.nan], stc[:-1]))
    stc_prev2 = np.concatenate(([np.nan, np.nan], stc[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = (
        np.isfinite(stc_prev)
        & np.isfinite(stc_prev2)
        & np.isfinite(close_prev)
        & np.isfinite(sma100_prev)
    )
    fresh_cross = (stc_prev2 <= 25.0) & (stc_prev > 25.0)
    entries = valid & fresh_cross & (close_prev > sma100_prev)

    stc_cross_down_75 = (
        np.isfinite(stc_prev)
        & np.isfinite(stc)
        & (stc_prev >= 75.0)
        & (stc < 75.0)
    )
    exits = stc_cross_down_75 | (np.isfinite(sma20) & (close < sma20))

    return _walk(entries, exits, close, df["date"].values)


def strat_coppock_curve_zero_cross(df: pd.DataFrame) -> list[Trade]:
    """Coppock Curve zero-line bullish cross with SMA(100) trend filter.

    Coppock (Edwin Coppock, 1965) is a momentum composite originally used on
    monthly bars to spot long-term bottoms. Adapted to daily bars here:
        ROC_a = 14-bar rate of change (pct)
        ROC_b = 11-bar rate of change (pct)
        Coppock = WMA(10) of (ROC_a + ROC_b)
    A rising zero-line cross signals momentum flipping positive after a
    negative regime. Distinct from TRIX/KAMA/HMA/Fisher/RVI/Vortex in the
    journal: those are smoothed-MA or range-compression signals, this is a
    weighted-sum-of-two-ROCs composite.

    Entry: Coppock crosses above 0 on the prior bar (prev2 <= 0 < prev) and
    prior close > SMA(100).
    Exit: Coppock turns back below 0, or close drops below SMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    n_bars = close.size

    def _roc(arr: np.ndarray, length: int) -> np.ndarray:
        shifted = np.concatenate((np.full(length, np.nan), arr[:-length]))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(
                np.isfinite(shifted) & (shifted != 0.0),
                100.0 * (arr - shifted) / shifted,
                np.nan,
            )
        return out

    roc_a = _roc(close, 14)
    roc_b = _roc(close, 11)
    roc_sum = roc_a + roc_b

    # Weighted moving average (linear weights 1..10) of roc_sum.
    wma_len = 10
    weights = np.arange(1, wma_len + 1, dtype=float)
    w_sum = weights.sum()
    coppock = np.full(n_bars, np.nan)
    for i in range(wma_len - 1, n_bars):
        window = roc_sum[i - wma_len + 1 : i + 1]
        if np.all(np.isfinite(window)):
            coppock[i] = float(np.dot(window, weights) / w_sum)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    cop_prev = np.concatenate(([np.nan], coppock[:-1]))
    cop_prev2 = np.concatenate(([np.nan, np.nan], coppock[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid_entry = (
        np.isfinite(cop_prev)
        & np.isfinite(cop_prev2)
        & np.isfinite(close_prev)
        & np.isfinite(sma100_prev)
    )
    fresh_cross = (cop_prev2 <= 0.0) & (cop_prev > 0.0)
    entries = valid_entry & fresh_cross & (close_prev > sma100_prev)

    cop_below_zero = np.isfinite(coppock) & (coppock < 0.0)
    below_sma20 = np.isfinite(sma20) & (close < sma20)
    exits = cop_below_zero | below_sma20

    return _walk(entries, exits, close, df["date"].values)


def strat_connors_rsi_pullback(df: pd.DataFrame) -> list[Trade]:
    """Connors RSI (CRSI) deep-oversold pullback in a long-term uptrend.

    CRSI = average of three components, each scaled 0..100:
      1) RSI(close, 3)                — short-term momentum
      2) RSI(streak, 2)               — RSI of the consecutive up/down-day run
      3) PercentRank(1-day ROC, 100)  — today's return percentile vs last 100

    Classic Larry Connors pullback: enter when CRSI < 10 (exceptional dip) and
    close is above SMA(200) (only buy dips inside established uptrends). Exit
    when close > SMA(5), capturing the short mean-reversion bounce. Distinct
    from cum_rsi2_pullback (cumulative RSI2 sum), williams_vix_fix (range-
    based percentile of highs), and plain RSI mean-revert (single oscillator).
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    rsi3 = _rsi(close, 3)

    # Streak: signed count of consecutive up/down closes (0 on unchanged).
    streak = np.zeros(n, dtype=float)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = max(streak[i - 1], 0.0) + 1.0
        elif close[i] < close[i - 1]:
            streak[i] = min(streak[i - 1], 0.0) - 1.0
        else:
            streak[i] = 0.0
    rsi_streak = _rsi(streak, 2)

    roc1 = np.zeros(n, dtype=float)
    roc1[1:] = (close[1:] / np.where(close[:-1] != 0, close[:-1], 1e-12)) - 1.0

    # Percent rank of today's ROC1 within the trailing 100-bar window.
    pct_rank = (
        pd.Series(roc1)
        .rolling(100, min_periods=20)
        .rank(pct=True)
        .to_numpy()
        * 100.0
    )
    pct_rank = np.where(np.isfinite(pct_rank), pct_rank, 50.0)

    crsi = (rsi3 + rsi_streak + pct_rank) / 3.0

    sma200 = _sma(close, 200)
    sma5 = _sma(close, 5)

    crsi_prev = np.concatenate(([50.0], crsi[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    entries = (
        np.isfinite(crsi_prev)
        & (crsi_prev < 10.0)
        & np.isfinite(sma200_prev)
        & (close_prev > sma200_prev)
    )
    exits = np.isfinite(sma5) & (close > sma5)

    return _walk(entries, exits, close, df["date"].values)


def strat_elder_force_index_zero_cross(df: pd.DataFrame) -> list[Trade]:
    """Elder's Force Index (EFI) zero-line bullish cross in a long-term uptrend.

    Force Index combines direction, magnitude, and volume into one reading:
        EFI_raw(t) = (close(t) - close(t-1)) * volume(t)
        EFI_13    = EMA(EFI_raw, 13)     # Elder's "short-term" smoothing

    A smoothed EFI(13) crossing up through zero means volume-weighted
    momentum has flipped from net-distribution to net-accumulation. We gate
    the signal with an SMA(100) trend filter so we only act on this flip
    when the broader tape is already constructive (avoids the common EFI
    failure mode: positive crosses inside a downtrend that fade fast).

    Entry: EFI_13 crosses from <=0 to >0 on the prior bar AND prior close >
    SMA(100). Exit: EFI_13 < 0 (momentum rolled over) OR close < SMA(20)
    (short-term trend break). Distinct from CMF (range-normalised, slow),
    Chaikin Money Flow zero-reclaim (different construction), and from
    price-only oscillators TRIX/Coppock/Schaff — EFI uniquely weights the
    price change by traded volume on each bar.
    """
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n = len(close)

    prev_close = np.concatenate(([close[0]], close[:-1]))
    efi_raw = (close - prev_close) * volume
    efi13 = _ema(efi_raw, 13)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    efi_prev = np.concatenate(([0.0], efi13[:-1]))
    efi_prev2 = np.concatenate(([0.0], efi_prev[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    bullish_cross = (
        np.isfinite(efi_prev)
        & np.isfinite(efi_prev2)
        & (efi_prev2 <= 0.0)
        & (efi_prev > 0.0)
    )
    trend_ok = (
        np.isfinite(sma100_prev)
        & np.isfinite(close_prev)
        & (close_prev > sma100_prev)
    )

    entries = bullish_cross & trend_ok
    exits = (
        (np.isfinite(efi13) & (efi13 < 0.0))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_pring_kst_signal_cross(df: pd.DataFrame) -> list[Trade]:
    """Martin Pring's Know Sure Thing (KST) — signal-line bullish cross in a
    long-term uptrend.

    KST aggregates momentum across four timeframes using smoothed ROC:
        RCMA1 = SMA(10, ROC(10))
        RCMA2 = SMA(10, ROC(15))
        RCMA3 = SMA(10, ROC(20))
        RCMA4 = SMA(15, ROC(30))
        KST   = 1*RCMA1 + 2*RCMA2 + 3*RCMA3 + 4*RCMA4
        SIG   = SMA(9, KST)

    Entry: KST crosses above its 9-period signal line AND close > SMA(200)
    (established long-term uptrend). Exit: KST crosses back below SIG, OR
    close < SMA(20).

    Distinct from Coppock (WMA of ROC14+ROC11, zero-line cross only — no
    signal line), TRIX (triple-EMA of a single price series), Schaff Trend
    Cycle (double-smoothed stochastic of MACD), MACD (12/26 EMA diff).
    KST's edge is multi-timeframe ROC blending — it responds to intermediate
    momentum shifts that single-period oscillators miss.
    """
    close = df["close"].to_numpy(dtype=float)

    # ROC_n(t) = close[t]/close[t-n] - 1, NaN where unavailable.
    def _roc(arr: np.ndarray, n: int) -> np.ndarray:
        prev = np.concatenate((np.full(n, np.nan), arr[:-n])) if n > 0 else arr
        out = np.full_like(arr, np.nan, dtype=float)
        mask = np.isfinite(prev) & (prev != 0)
        out[mask] = arr[mask] / prev[mask] - 1.0
        return out

    rcma1 = _sma(_roc(close, 10), 10)
    rcma2 = _sma(_roc(close, 15), 10)
    rcma3 = _sma(_roc(close, 20), 10)
    rcma4 = _sma(_roc(close, 30), 15)
    kst = 1.0 * rcma1 + 2.0 * rcma2 + 3.0 * rcma3 + 4.0 * rcma4
    sig = _sma(kst, 9)

    sma200 = _sma(close, 200)
    sma20 = _sma(close, 20)

    kst_prev = np.concatenate(([np.nan], kst[:-1]))
    kst_prev2 = np.concatenate(([np.nan], kst_prev[:-1]))
    sig_prev = np.concatenate(([np.nan], sig[:-1]))
    sig_prev2 = np.concatenate(([np.nan], sig_prev[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    bullish_cross = (
        np.isfinite(kst_prev) & np.isfinite(kst_prev2)
        & np.isfinite(sig_prev) & np.isfinite(sig_prev2)
        & (kst_prev2 <= sig_prev2)
        & (kst_prev > sig_prev)
    )
    trend_ok = (
        np.isfinite(close_prev) & np.isfinite(sma200_prev)
        & (close_prev > sma200_prev)
    )

    entries = bullish_cross & trend_ok
    exits = (
        (np.isfinite(kst) & np.isfinite(sig) & (kst < sig))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_awesome_oscillator_saucer(df: pd.DataFrame) -> list[Trade]:
    """Bill Williams' Awesome Oscillator 'saucer' setup in a long-term uptrend.

    AO = SMA(5, median) - SMA(34, median), median = (high + low) / 2.
    The saucer is a momentum-V on the bullish side of zero: two consecutive
    declines in the AO histogram followed by an up-tick, all above zero. In
    the canonical four-bar window (using prior-bar values so decisions are
    bar-close knowable):
        AO[t-4] > AO[t-3] > AO[t-2]   # two successive declines (red bars)
        AO[t-1] > AO[t-2]             # turn-up bar (green)
        all four AO readings > 0

    This pattern captures a pause-and-resume in an established uptrend. We
    also require close > SMA(100) at t-1 so we only trade saucers inside a
    broader uptrend (avoids the classic failure of saucers inside downtrends).
    Exit: AO < 0 (momentum flipped) OR close < SMA(20). Distinct from MACD /
    TRIX / Coppock (EMA-of-price oscillators), RSI family, Aroon (extreme
    counters), Vortex (VM sums), and Heikin-Ashi / KAMA / HMA (price smoothers)
    because AO uses SMAs of the bar *midpoint* and the setup is a specific
    3-bar histogram shape rather than a single-point cross.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    median = (high + low) / 2.0
    ao = _sma(median, 5) - _sma(median, 34)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    ao1 = np.concatenate(([np.nan], ao[:-1]))
    ao2 = np.concatenate(([np.nan], ao1[:-1]))
    ao3 = np.concatenate(([np.nan], ao2[:-1]))
    ao4 = np.concatenate(([np.nan], ao3[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    saucer = (
        np.isfinite(ao1) & np.isfinite(ao2) & np.isfinite(ao3) & np.isfinite(ao4)
        & (ao4 > ao3) & (ao3 > ao2)
        & (ao1 > ao2)
        & (ao1 > 0) & (ao2 > 0) & (ao3 > 0) & (ao4 > 0)
    )
    trend_ok = (
        np.isfinite(sma100_prev) & np.isfinite(close_prev)
        & (close_prev > sma100_prev)
    )
    entries = saucer & trend_ok
    exits = (
        (np.isfinite(ao) & (ao < 0))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_mfi_oversold_recovery(df: pd.DataFrame) -> list[Trade]:
    """Money Flow Index (MFI) oversold-recovery in a long-term uptrend.

    Quong/Soudack MFI is a volume-weighted RSI built on the *typical price*
    instead of the close, and on the *direction of typical-price changes*
    (not range position or Δclose):
        typical_t   = (high_t + low_t + close_t) / 3
        raw_flow_t  = typical_t * volume_t
        pos_flow_t  = raw_flow_t if typical_t > typical_{t-1} else 0
        neg_flow_t  = raw_flow_t if typical_t < typical_{t-1} else 0
        MFI_t       = 100 - 100 / (1 + sum(pos,14) / sum(neg,14))

    An oversold MFI (<20) inside a long-term uptrend is a classic
    volume-confirmed dip. We fire on the *recovery* (fresh cross from
    <=20 to >20 of the prior bar) to avoid picking up falling knives, and
    require close > SMA(200) so we only buy dips in established uptrends.
    Exit: MFI > 80 (mean-reversion target) OR close < SMA(20) (trend break).

    Distinct from every existing sandbox volume/flow strategy:
      - CMF uses close-within-range multiplier × volume (range position).
      - EFI uses (Δclose × volume) — price change × volume, not typical.
      - pocket_pivot / volume_capitulation_reclaim are single-bar volume
        events with no RSI-style accumulation/ratio geometry.
    MFI is the only typical-price direction-weighted money-flow *ratio*
    oscillator in the sandbox.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    typical = (high + low + close) / 3.0
    raw_flow = typical * volume

    typical_prev = np.concatenate(([np.nan], typical[:-1]))
    direction = typical - typical_prev
    pos_flow = np.where(direction > 0, raw_flow, 0.0)
    neg_flow = np.where(direction < 0, raw_flow, 0.0)

    n = 14
    pos_sum = pd.Series(pos_flow).rolling(n, min_periods=n).sum().to_numpy()
    neg_sum = pd.Series(neg_flow).rolling(n, min_periods=n).sum().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(neg_sum > 0, pos_sum / neg_sum, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    # If neg_sum == 0 but pos_sum > 0, MFI saturates at 100.
    mfi = np.where(
        np.isfinite(mfi),
        mfi,
        np.where(np.isfinite(pos_sum) & (pos_sum > 0) & (neg_sum == 0), 100.0, np.nan),
    )

    sma200 = _sma(close, 200)
    sma20 = _sma(close, 20)

    mfi_prev = np.concatenate(([np.nan], mfi[:-1]))
    mfi_prev2 = np.concatenate(([np.nan], mfi_prev[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    fresh_recovery = (
        np.isfinite(mfi_prev) & np.isfinite(mfi_prev2)
        & (mfi_prev2 <= 20.0) & (mfi_prev > 20.0)
    )
    trend_ok = (
        np.isfinite(sma200_prev) & np.isfinite(close_prev)
        & (close_prev > sma200_prev)
    )
    entries = fresh_recovery & trend_ok
    exits = (
        (np.isfinite(mfi) & (mfi > 80.0))
        | (np.isfinite(sma20) & (close < sma20))
    )
    return _walk(entries, exits, close, df["date"].values)


def strat_tsi_signal_cross(df: pd.DataFrame) -> list[Trade]:
    """William Blau's True Strength Index (TSI) bullish signal-line cross.

    TSI double-smooths raw price momentum (Δclose) with two nested EMAs:
        mom       = close - close.shift(1)
        TSI       = 100 * EMA(EMA(mom, 25), 13) / EMA(EMA(|mom|, 25), 13)
        signal    = EMA(TSI, 7)
    Entry (prior-bar only, no lookahead):
        TSI[t-2] <= signal[t-2]  and  TSI[t-1] > signal[t-1]
        close[t-1] > SMA(100)[t-1]
    Exit: TSI < signal OR close < SMA(20).

    Distinct from every momentum oscillator already in the sandbox:
      - MACD / PPO: single-EMA difference on *price* (not momentum).
      - TRIX:       triple-EMA of price, then ROC of the smoothed price.
      - KST:        weighted sum of SMAs of ROC across 10/15/20/30.
      - Coppock:    WMA(10) over sum of two ROCs.
      - Schaff:     double-smoothed stochastic of MACD (bounded 0-100).
      - Fisher:     inverse-hyp transform of range-mid position.
      - RVI:        (close-open)/(high-low) style, not momentum.
      - EFI:        EMA of Δclose × volume (volume-weighted).
      - AO:         SMA(5) - SMA(34) of median price.
      - RSI/CRSI/CumRSI2/MFI: gains/(gains+losses) ratios.
    TSI is the only oscillator here that applies *two nested EMAs* to raw
    Δclose and normalizes by the same double-smoothing of |Δclose|, which
    suppresses whipsaw while preserving the sign and magnitude of momentum
    in a way no other indicator in the pool does.
    """
    close = df["close"].to_numpy(dtype=float)

    mom = np.concatenate(([np.nan], np.diff(close)))
    abs_mom = np.abs(mom)

    mom_f = np.nan_to_num(mom, nan=0.0)
    abs_f = np.nan_to_num(abs_mom, nan=0.0)

    ema1_mom = _ema(mom_f, 25)
    ema2_mom = _ema(ema1_mom, 13)
    ema1_abs = _ema(abs_f, 25)
    ema2_abs = _ema(ema1_abs, 13)

    with np.errstate(divide="ignore", invalid="ignore"):
        tsi = 100.0 * np.where(ema2_abs > 0, ema2_mom / ema2_abs, 0.0)
    signal = _ema(tsi, 7)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    tsi_prev1 = np.concatenate(([np.nan], tsi[:-1]))
    tsi_prev2 = np.concatenate(([np.nan], tsi_prev1[:-1]))
    sig_prev1 = np.concatenate(([np.nan], signal[:-1]))
    sig_prev2 = np.concatenate(([np.nan], sig_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    cross_up = (
        np.isfinite(tsi_prev1) & np.isfinite(tsi_prev2)
        & np.isfinite(sig_prev1) & np.isfinite(sig_prev2)
        & (tsi_prev2 <= sig_prev2)
        & (tsi_prev1 > sig_prev1)
    )
    trend_ok = (
        np.isfinite(sma100_prev) & np.isfinite(close_prev)
        & (close_prev > sma100_prev)
    )
    entries = cross_up & trend_ok

    exits = (
        (np.isfinite(tsi) & np.isfinite(signal) & (tsi < signal))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_stochastic_oversold_recovery(df: pd.DataFrame) -> list[Trade]:
    """Stochastic %K(14) oversold-recovery cross above 20 in an SMA(200) uptrend.

    %K = 100 * (close - LowestLow14) / (HighestHigh14 - LowestLow14) — locates
    today's close inside the trailing 14-bar high/low range. Below 20 is
    oversold. The entry trigger is the moment %K crosses up through 20 from
    below, gated by close > SMA(200) so dip-buys only fire in established
    uptrends. Exit is the symmetric overbought reading (%K > 80) OR a break
    of SMA(20) — capturing a quick reversion bounce while limiting downside.

    Mathematically distinct from the strategies already in the sandbox:
      - RSI variants (Connors, MACD-RSI, RSI-EMA, cum_rsi2): close-only
        momentum smoothed with Wilder RMA, no high/low range.
      - MFI: volume-weighted RSI on typical price.
      - Williams VIX Fix: HighestClose vs Low — measures fear, not range
        position.
      - Aroon: time since HH/LL, not magnitude.
      - Fisher / Schaff / Coppock: smoothed transforms / cycles.
      - IBS: single-bar (close-low)/(high-low) — no lookback range.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    hh14 = pd.Series(high).rolling(14, min_periods=14).max().to_numpy()
    ll14 = pd.Series(low).rolling(14, min_periods=14).min().to_numpy()

    rng = hh14 - ll14
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_k = 100.0 * np.where(rng > 0, (close - ll14) / rng, 50.0)
    pct_k = np.where(np.isfinite(pct_k), pct_k, 50.0)

    sma200 = _sma(close, 200)
    sma20 = _sma(close, 20)

    # Reference prior-bar values so the cross is locked in by bar close
    # (no lookahead). The entry fires the bar AFTER the cross completed.
    pct_k_prev1 = np.concatenate(([np.nan], pct_k[:-1]))
    pct_k_prev2 = np.concatenate(([np.nan], pct_k_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    cross_up_20 = (
        np.isfinite(pct_k_prev1) & np.isfinite(pct_k_prev2)
        & (pct_k_prev2 < 20.0)
        & (pct_k_prev1 >= 20.0)
    )
    trend_ok = (
        np.isfinite(sma200_prev) & np.isfinite(close_prev)
        & (close_prev > sma200_prev)
    )
    entries = cross_up_20 & trend_ok

    exits = (
        (np.isfinite(pct_k) & (pct_k > 80.0))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_cci_oversold_recovery(df: pd.DataFrame) -> list[Trade]:
    """Lambert's Commodity Channel Index(20) oversold-recovery cross above -100.

    CCI measures how far typical price has deviated from its moving average,
    normalized by the *mean absolute deviation* of TP from that average:
        TP      = (high + low + close) / 3
        SMA_TP  = SMA(TP, 20)
        MD      = mean( |TP - SMA_TP| ) over trailing 20 bars
        CCI     = (TP - SMA_TP) / (0.015 * MD)
    Readings below -100 mark an unusual (≈1-σ via 0.015 scaling) negative
    deviation — statistical oversold.

    Entry (prior-bar only — no lookahead):
        CCI[t-2] <= -100  and  CCI[t-1] > -100  (fresh upward recovery)
        close[t-1] > SMA(200)[t-1]              (long-term uptrend gate)
    Exit: CCI > +100 (overbought mean-reversion target) OR close < SMA(20).

    Distinct from every other oscillator in the sandbox because it is the
    only one that (a) operates on typical price (H+L+C)/3 rather than close
    or Δclose, AND (b) normalizes deviations by *mean absolute deviation*
    rather than:
      - stdev (Bollinger / squeeze / WVF)
      - RMA of gains-vs-losses (RSI family, Connors, MFI, CMO analogue)
      - EMA of |Δclose| (TSI)
      - high/low range (IBS, Stochastic, Aroon, Donchian, Williams %R)
      - volume flow (CMF, EFI, Pocket Pivot, Klinger)
      - EMA/SMA crossovers (MACD, TRIX, KST, Coppock, KAMA, HMA, AO)
      - ±DI directional comparisons (ADX, Vortex).
    MD-based normalization is mathematically unique here; the 0.015 constant
    further scales the indicator so ±100 approximates a 1-σ deviation for
    typical price distributions.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    tp = (high + low + close) / 3.0
    tp_s = pd.Series(tp)
    sma_tp = tp_s.rolling(20, min_periods=20).mean()
    mad = tp_s.rolling(20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = np.where(
            mad.to_numpy() > 0,
            (tp - sma_tp.to_numpy()) / (0.015 * mad.to_numpy()),
            0.0,
        )
    cci = np.where(np.isfinite(cci), cci, np.nan)

    sma200 = _sma(close, 200)
    sma20 = _sma(close, 20)

    cci_prev1 = np.concatenate(([np.nan], cci[:-1]))
    cci_prev2 = np.concatenate(([np.nan], cci_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    cross_up = (
        np.isfinite(cci_prev1) & np.isfinite(cci_prev2)
        & (cci_prev2 <= -100.0)
        & (cci_prev1 > -100.0)
    )
    trend_ok = (
        np.isfinite(sma200_prev) & np.isfinite(close_prev)
        & (close_prev > sma200_prev)
    )
    entries = cross_up & trend_ok

    exits = (
        (np.isfinite(cci) & (cci > 100.0))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_chaikin_oscillator_zero_cross(df: pd.DataFrame) -> list[Trade]:
    """Chaikin Oscillator bullish zero-line cross in SMA(100) uptrend.

    The Chaikin Oscillator is MACD applied to the Accumulation/Distribution
    Line (ADL), combining price-location-in-range with volume:
        MFM = ((close - low) - (high - close)) / (high - low)   # -1..+1
        MFV = MFM * volume                                       # signed flow
        ADL = cumulative sum of MFV
        ChaikinOsc = EMA(3, ADL) - EMA(10, ADL)

    Signal (prior-bar only — no lookahead):
        ChaikinOsc[t-2] <= 0  and  ChaikinOsc[t-1] > 0    (fresh bullish cross)
        close[t-1] > SMA(100)[t-1]                        (trend-up gate)
    Exit: ChaikinOsc < 0 (flow turns distribution-heavy) OR close < SMA(20).

    Distinct from every volume indicator already in the sandbox:
      - CMF sums MFV / sum(volume) over a fixed window → bounded -1..+1
      - EFI = EMA13(Δclose × volume) → uses price change, not range location
      - MFI is RSI of typical-price × volume → bounded 0..100 oscillator
      - Pocket Pivot compares today's up-volume to prior down-volumes
      - Klinger would integrate trend direction (not used here)
    Chaikin Osc is unique: it's MACD of a *cumulative* range-weighted volume
    series (ADL), so it measures the *acceleration* of accumulation rather
    than a level or ratio. Zero-line crosses mark regime transitions in the
    accumulation/distribution tug-of-war.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    rng = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = np.where(rng > 0, ((close - low) - (high - close)) / rng, 0.0)
    mfv = mfm * volume
    adl = np.cumsum(mfv)

    ema3_adl = _ema(adl, 3)
    ema10_adl = _ema(adl, 10)
    chaikin = ema3_adl - ema10_adl

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    chaikin_prev1 = np.concatenate(([np.nan], chaikin[:-1]))
    chaikin_prev2 = np.concatenate(([np.nan], chaikin_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    cross_up = (
        np.isfinite(chaikin_prev1) & np.isfinite(chaikin_prev2)
        & (chaikin_prev2 <= 0.0)
        & (chaikin_prev1 > 0.0)
    )
    trend_ok = (
        np.isfinite(sma100_prev) & np.isfinite(close_prev)
        & (close_prev > sma100_prev)
    )
    entries = cross_up & trend_ok

    exits = (
        (np.isfinite(chaikin) & (chaikin < 0.0))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_obv_ema_cross(df: pd.DataFrame) -> list[Trade]:
    """On-Balance Volume (Granville 1963) bullish cross of its own EMA(20).

    OBV accumulates volume signed by daily close change:
        OBV[t] = OBV[t-1] + volume[t] * sign(close[t] - close[t-1])
    It is a pure volume-momentum cumulative line, mathematically distinct
    from every other volume / flow indicator already in the sandbox:
      - CMF / ADL / Chaikin Osc weight volume by close's *position within
        the H-L range* ((C-L)-(H-C))/(H-L) — range-weighted.
      - Elder Force Index multiplies Δclose × volume (signed magnitude).
      - Pocket Pivot / Volume-capitulation compare raw volume percentiles.
      - MFI rescales TP×volume into an RSI bounded 0-100.
    OBV alone uses sign(Δclose) with NO magnitude — a step-wise ±volume
    accumulator. A bullish cross of OBV above its EMA(20) flags cumulative
    money-flow momentum turning up.

    Signal-line crosses have been the strongest family in the journal
    (TSI signal cross led OOS at 0.42). OBV's signal-line cross on
    cumulative sign-weighted volume has no mathematical overlap with TSI
    (price ΔΔ smoothed) or the Chaikin Osc (range-weighted ADL difference).

    Entry (prior-bar only — no lookahead):
        OBV[t-2] <= EMA20(OBV)[t-2]
        OBV[t-1] >  EMA20(OBV)[t-1]          (fresh bullish cross)
        close[t-1] > SMA(100)[t-1]           (uptrend gate)
    Exit: OBV < EMA20(OBV) OR close < SMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    dclose = np.concatenate(([0.0], np.diff(close)))
    signed_vol = np.where(
        dclose > 0, volume, np.where(dclose < 0, -volume, 0.0)
    )
    obv = np.cumsum(signed_vol)

    obv_ema = _ema(obv, 20)
    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    obv_prev1 = np.concatenate(([np.nan], obv[:-1]))
    obv_prev2 = np.concatenate(([np.nan], obv_prev1[:-1]))
    ema_prev1 = np.concatenate(([np.nan], obv_ema[:-1]))
    ema_prev2 = np.concatenate(([np.nan], ema_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    cross_up = (
        np.isfinite(obv_prev1) & np.isfinite(obv_prev2)
        & np.isfinite(ema_prev1) & np.isfinite(ema_prev2)
        & (obv_prev2 <= ema_prev2)
        & (obv_prev1 > ema_prev1)
    )
    trend_ok = (
        np.isfinite(sma100_prev) & np.isfinite(close_prev)
        & (close_prev > sma100_prev)
    )
    entries = cross_up & trend_ok

    exits = (
        (np.isfinite(obv_ema) & (obv < obv_ema))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_ultimate_oscillator_oversold(df: pd.DataFrame) -> list[Trade]:
    """Williams (1976) Ultimate Oscillator oversold→above-30 recovery in uptrend.

    UO weights BUYING PRESSURE over 3 timeframes (7/14/28) at 4:2:1, a
    multi-timeframe oscillator designed by Williams specifically to defeat
    the single-window whipsaws of RSI/Stochastic/CCI.

        BP[t]  = close[t] - min(low[t], close[t-1])     (buying pressure)
        TR[t]  = max(high[t], close[t-1]) - min(low[t], close[t-1])
        A7     = rollsum7(BP) / rollsum7(TR)
        A14    = rollsum14(BP) / rollsum14(TR)
        A28    = rollsum28(BP) / rollsum28(TR)
        UO     = 100 * (4*A7 + 2*A14 + A28) / 7

    Mathematically distinct from every existing oscillator in the sandbox:
      - RSI / Connors RSI: smoothed ratio of gains/losses, single window.
      - Stochastic: (close-lowestN) / (highestN-lowestN), single window.
      - MFI: RSI on TP×volume, single window, volume-weighted.
      - CCI: deviation of TP from SMA scaled by MAD, single window.
      - TSI: double-EMA of raw price changes (no range concept).
      - Awesome / MACD / PPO: difference of two moving averages.
    UO is the only oscillator that uses a *ratio of cumulative buying
    pressure to true range* across THREE overlapping timeframes, weighted
    so the fastest window dominates but the slowest provides context —
    Williams' explicit remedy for single-window mean-reversion false signals.

    Multi-TF weighting means UO is slower to reach extremes than RSI(2)
    or Stochastic(14), so when it finally prints oversold<30 the signal
    carries joint confirmation across 7/14/28 bars of pressure. Crossing
    back above 30 is Williams' classic 'buy when UO reclaims 30' rule.

    Entry (prior-bar only — no lookahead):
        UO[t-1]  >  30
        min(UO[t-28..t-2]) <= 30            (was recently oversold)
        close[t-1] > SMA(100)[t-1]           (established uptrend)
    Exit:
        UO > 70 (overbought) OR close < SMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)

    close_prev = np.concatenate(([np.nan], close[:-1]))
    true_low = np.minimum(low, np.where(np.isfinite(close_prev), close_prev, low))
    true_high = np.maximum(high, np.where(np.isfinite(close_prev), close_prev, high))

    bp = close - true_low
    tr = true_high - true_low

    bp_s = pd.Series(bp)
    tr_s = pd.Series(tr)

    sum7_bp = bp_s.rolling(7, min_periods=7).sum().to_numpy()
    sum7_tr = tr_s.rolling(7, min_periods=7).sum().to_numpy()
    sum14_bp = bp_s.rolling(14, min_periods=14).sum().to_numpy()
    sum14_tr = tr_s.rolling(14, min_periods=14).sum().to_numpy()
    sum28_bp = bp_s.rolling(28, min_periods=28).sum().to_numpy()
    sum28_tr = tr_s.rolling(28, min_periods=28).sum().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        avg7 = np.where(sum7_tr > 0, sum7_bp / sum7_tr, np.nan)
        avg14 = np.where(sum14_tr > 0, sum14_bp / sum14_tr, np.nan)
        avg28 = np.where(sum28_tr > 0, sum28_bp / sum28_tr, np.nan)

    uo = 100.0 * (4.0 * avg7 + 2.0 * avg14 + avg28) / 7.0

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    uo_prev1 = np.concatenate(([np.nan], uo[:-1]))
    # Recent oversold check: min of UO over the previous 28 bars (excl today) <= 30
    min_uo_prev = (
        pd.Series(uo).shift(1).rolling(28, min_periods=5).min().to_numpy()
    )
    close_prev_s = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    entries = (
        np.isfinite(uo_prev1) & (uo_prev1 > 30.0)
        & np.isfinite(min_uo_prev) & (min_uo_prev <= 30.0)
        & np.isfinite(sma100_prev) & np.isfinite(close_prev_s)
        & (close_prev_s > sma100_prev)
    )

    exits = (
        (np.isfinite(uo) & (uo > 70.0))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_nvi_fosback_trend(df: pd.DataFrame) -> list[Trade]:
    """Negative Volume Index (Dysart 1936 / Fosback 1976) above EMA(255).

    NVI is a cumulative price-drift index that ONLY updates on bars whose
    volume DECREASED vs. the prior bar; high-volume bars are ignored. The
    thesis (Dysart 1936, formalized by Fosback in "Stock Market Logic",
    1976): the crowd trades on high volume, smart money trades on quiet
    days, so price action on LOW-volume bars reflects informed accumulation.

        pct[t] = (close[t] - close[t-1]) / close[t-1]   if volume[t] < volume[t-1]
        pct[t] = 0                                       otherwise
        NVI[t] = 1000 * prod(1 + pct[0..t])

    Fosback's published rule: NVI above its EMA(255) correctly identified
    ~95% of historical bull markets on US data 1941-1975. We use the EMA-255
    cross as the signal (with a 100-bar trend gate for safety).

    Mathematically distinct from every other volume indicator in the sandbox:
      - OBV: accumulates FULL signed volume on every bar.
      - CMF / ADL / Chaikin Osc: money-flow weight = position in H-L range
        × volume, used on every bar.
      - Elder Force Index: Δprice × FULL volume every bar.
      - MFI: RSI of TP×volume every bar.
      - Pocket Pivot / Volume-capitulation: volume-SPIKE breakout patterns.
    NVI is the ONLY indicator here that UNWEIGHTS volume entirely by using
    it as a binary gate and isolates price drift on quiet bars — Dysart's
    smart-money hypothesis. No other strategy tests this regime.

    Entry (prior-bar only — no lookahead):
        NVI[t-2] <= EMA(NVI,255)[t-2]
        NVI[t-1] >  EMA(NVI,255)[t-1]          (fresh bullish cross)
        close[t-1] > SMA(100)[t-1]             (uptrend gate)
    Exit: NVI < EMA(NVI,255) OR close < SMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n = len(close)

    ret = np.zeros(n, dtype=float)
    if n > 1:
        pct = np.zeros(n, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct[1:] = np.where(
                close[:-1] > 0, (close[1:] - close[:-1]) / close[:-1], 0.0
            )
        low_vol = np.zeros(n, dtype=bool)
        low_vol[1:] = volume[1:] < volume[:-1]
        ret = np.where(low_vol, pct, 0.0)
    nvi = 1000.0 * np.cumprod(1.0 + ret)

    nvi_ema = _ema(nvi, 255)
    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    nvi_prev1 = np.concatenate(([np.nan], nvi[:-1]))
    nvi_prev2 = np.concatenate(([np.nan], nvi_prev1[:-1]))
    ema_prev1 = np.concatenate(([np.nan], nvi_ema[:-1]))
    ema_prev2 = np.concatenate(([np.nan], ema_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    cross_up = (
        np.isfinite(nvi_prev1) & np.isfinite(nvi_prev2)
        & np.isfinite(ema_prev1) & np.isfinite(ema_prev2)
        & (nvi_prev2 <= ema_prev2)
        & (nvi_prev1 > ema_prev1)
    )
    trend_ok = (
        np.isfinite(sma100_prev) & np.isfinite(close_prev)
        & (close_prev > sma100_prev)
    )
    entries = cross_up & trend_ok

    exits = (
        (np.isfinite(nvi_ema) & (nvi < nvi_ema))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_choppiness_regime_shift(df: pd.DataFrame) -> list[Trade]:
    """Choppiness Index regime transition: chop -> trend, gated by SMA(100) up.

    The Choppiness Index (E.W. Dreiss, 1990s) is a regime detector — it
    measures whether the market is *trending* or *consolidating*, not
    direction:

        TR[t]   = max(H-L, |H-prevC|, |L-prevC|)
        CI(n)   = 100 * log10( sum(TR, n) / (max(H,n) - min(L,n)) ) / log10(n)

    Range 0-100. CI > 61.8 = sideways/choppy (range fully filled by TR sum).
    CI < 38.2 = strongly trending (TR sum small relative to range, i.e. price
    moved decisively in one direction).

    This is mathematically distinct from EVERY indicator already tested:
      - It is NOT a momentum oscillator (RSI / Stoch / CCI / CMO / MFI /
        Williams%R / UO / Fisher / RVI / TSI / KST / DPO).
      - It is NOT a trend/cross indicator (MA cross / MACD / TRIX / Coppock
        / KAMA / HMA / Schaff / Aroon / Vortex / DMI / ParabolicSAR /
        Ichimoku / Donchian / NR7 / Squeeze / SuperTrend / BB).
      - It is NOT a volume indicator (OBV / CMF / ADL / Chaikin / NVI / EFI
        / Pocket-Pivot / Volume-capitulation).
      - It is NOT a candle-shape pattern (IBS / HeikinAshi / WilliamsVixFix).
      - It is NOT a centered momentum oscillator (Awesome / EFI / KST).
    Choppiness is a *range-fill ratio* — it ignores direction entirely and
    only measures how efficiently price has moved through its envelope. No
    other indicator in the sandbox tests this.

    Hypothesis: when CI was recently choppy (>=61.8 within last 10 bars) and
    has just collapsed into trending (<38.2), the new trend that's emerging
    is most likely UP if close > SMA(100) (uptrend filter — long-only).
    Trades in this regime tend to be sustained moves rather than mean-revert
    chop, so we ride them with a simple SMA(20) trailing exit plus a
    "chop returned" exit if CI re-enters >61.8.

    Entry (prior-bar values only, no lookahead):
        max(CI[t-10..t-1])  >= 61.8                (was choppy recently)
        CI[t-1]             <  38.2                (now strongly trending)
        close[t-1]          >  SMA(100)[t-1]       (long-only uptrend gate)
    Exit:
        CI > 61.8                                  (regime back to chop)
        OR close < SMA(20)                         (trend break)
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)

    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])

    period = 14
    sum_tr = (
        pd.Series(tr).rolling(period, min_periods=period).sum().to_numpy()
    )
    high_n = (
        pd.Series(high).rolling(period, min_periods=period).max().to_numpy()
    )
    low_n = (
        pd.Series(low).rolling(period, min_periods=period).min().to_numpy()
    )
    range_n = high_n - low_n

    log_n = np.log10(period)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(range_n > 0, sum_tr / range_n, np.nan)
        ci = 100.0 * np.log10(ratio) / log_n

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    ci_prev = np.concatenate(([np.nan], ci[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    chop_recent = (
        pd.Series(ci_prev).rolling(10, min_periods=1).max().to_numpy()
    )

    entries = (
        np.isfinite(ci_prev) & (ci_prev < 38.2)
        & np.isfinite(chop_recent) & (chop_recent >= 61.8)
        & np.isfinite(sma100_prev) & np.isfinite(close_prev)
        & (close_prev > sma100_prev)
    )

    exits = (
        (np.isfinite(ci) & (ci > 61.8))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_inverse_fisher_rsi(df: pd.DataFrame) -> list[Trade]:
    """Ehlers' Inverse Fisher Transform of RSI — saturated oversold reclaim.

    The Inverse Fisher Transform (Ehlers, "Cybernetic Analysis for Stocks
    and Futures", 2004) compresses any normally-distributed indicator into
    the bounded range [-1, +1] via a sigmoid-like S-curve, so threshold
    crossings of ±0.5 mark decisive momentum events with little midband
    noise. Applied to a rescaled, smoothed RSI:

        v[t]    = 0.1 * (RSI(14)[t] - 50)            # rescale RSI to ~[-5, 5]
        v_s[t]  = WMA(v, 9)[t]                       # linearly-weighted MA
        IFTR[t] = (exp(2 * v_s) - 1) / (exp(2 * v_s) + 1)

    Mathematically distinct from every other oscillator already tested:
      - Bare RSI / Stochastic / MFI / CCI / Williams %R / UO: linear
        oscillators with unbounded or 0..100 ranges and 20/80 thresholds —
        no sigmoid saturation.
      - Fisher Transform of price (already in sandbox): applies the
        FORWARD Fisher to a normalized PRICE range; the inverse Fisher
        applied to RSI is a different transform on a different input.
      - Connors RSI / Schaff Trend Cycle: composite oscillators built
        from RSI/stoch averages, no Fisher math.
      - TRIX / TSI / Coppock / KST / Awesome / RVI: smoothed momentum
        derivatives, not S-curve transforms.
    Ehlers' nonlinear S-curve flattens midband chop and makes the −0.5
    line a high-conviction reclaim level — a regime not exercised by any
    prior strategy here.

    Entry (prior-bar values only — no lookahead):
        IFTR[t-2] <= -0.5 AND IFTR[t-1] > -0.5      (oversold reclaim)
        close[t-1] > SMA(100)[t-1]                   (uptrend gate)
    Exit: IFTR > 0.5 (overbought saturation) OR close < SMA(20).
    """
    close = df["close"].to_numpy(dtype=float)

    rsi = _rsi(close, 14)
    v = 0.1 * (rsi - 50.0)

    period = 9
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    v_smooth = (
        pd.Series(v)
        .rolling(period, min_periods=period)
        .apply(lambda w: float(np.dot(w, weights)), raw=True)
        .to_numpy()
    )

    z = np.clip(2.0 * v_smooth, -50.0, 50.0)
    ez = np.exp(z)
    iftr = np.where(np.isfinite(z), (ez - 1.0) / (ez + 1.0), np.nan)

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    iftr_prev1 = np.concatenate(([np.nan], iftr[:-1]))
    iftr_prev2 = np.concatenate(([np.nan], iftr_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    cross_up = (
        np.isfinite(iftr_prev1) & np.isfinite(iftr_prev2)
        & (iftr_prev2 <= -0.5)
        & (iftr_prev1 > -0.5)
    )
    trend_ok = (
        np.isfinite(close_prev) & np.isfinite(sma100_prev)
        & (close_prev > sma100_prev)
    )
    entries = cross_up & trend_ok

    exits = (
        (np.isfinite(iftr) & (iftr > 0.5))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_minervini_vcp_breakout(df: pd.DataFrame) -> list[Trade]:
    """Minervini-style Volatility Contraction Pattern (VCP) breakout.

    Combines Mark Minervini's "Trend Template" stage-2 alignment with an
    explicit volatility contraction filter and 52-week-high proximity —
    a regime that none of the prior strategies in this sandbox encodes
    jointly:
      - nr7_breakout_trend : single-bar narrow-range, no multi-SMA stack.
      - donchian_20_10    : pure channel breakout, no contraction filter.
      - squeeze_breakout  : BB-inside-KC binary squeeze, ignores 52w high.
      - pocket_pivot      : O'Neil volume signature, not volatility-based.
    VCP captures the "tighter, smaller pullbacks" footprint of leadership
    stocks and demands a confirmed Stage-2 trend per Minervini ("Trade
    Like a Stock Market Wizard", 2013, ch. 6).

    Stage-2 trend template (subset of Minervini's 8 criteria):
        close > SMA50, SMA50 > SMA150, SMA150 > SMA200, SMA200 rising
        (vs. itself 21 bars ago), and close >= 0.85 * 252-bar high.

    Volatility contraction:
        ATR(20) / ATR(60) < 0.85    — recent range tighter than longer-range.

    Entry trigger (no lookahead — all references are prior-bar values
    except today's close, which is known at bar close):
        Stage-2 AND contraction AND close > prior 20-bar HIGH (.shift(1)).

    Exit: close < SMA(20) OR close < SMA(50) (broken trend structure).
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma150 = _sma(close, 150)
    sma200 = _sma(close, 200)

    atr20 = _atr(high, low, close, 20)
    atr60 = _atr(high, low, close, 60)
    vol_ratio = np.where(
        np.isfinite(atr60) & (atr60 > 0), atr20 / atr60, np.nan
    )

    high_252 = (
        pd.Series(high).shift(1).rolling(252, min_periods=126).max().to_numpy()
    )
    high_20_prev = (
        pd.Series(high).shift(1).rolling(20, min_periods=20).max().to_numpy()
    )

    sma200_lag = np.concatenate((np.full(21, np.nan), sma200[:-21]))
    sma200_rising = (
        np.isfinite(sma200) & np.isfinite(sma200_lag) & (sma200 > sma200_lag)
    )

    stage2 = (
        np.isfinite(sma50) & np.isfinite(sma150) & np.isfinite(sma200)
        & (close > sma50) & (sma50 > sma150) & (sma150 > sma200)
        & sma200_rising
    )

    near_high = np.isfinite(high_252) & (close >= 0.85 * high_252)
    contracted = np.isfinite(vol_ratio) & (vol_ratio < 0.85)
    breakout = np.isfinite(high_20_prev) & (close > high_20_prev)

    entries = stage2 & near_high & contracted & breakout

    exits = (
        (np.isfinite(sma20) & (close < sma20))
        | (np.isfinite(sma50) & (close < sma50))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_dpo_zero_cross(df: pd.DataFrame) -> list[Trade]:
    """Detrended Price Oscillator (Pring) zero-line cross in SMA(100) uptrend.

    DPO subtracts a *displaced* SMA from price to strip the long-term trend
    component and isolate short-cycle deviations. With period n=20:

        DPO[t] = close[t] - SMA(close, 20)[t - (n/2 + 1)]
               = close[t] - SMA(close, 20)[t - 11]

    The displacement is purely backward (we read the SMA at a past bar),
    so the indicator is decidable at bar close with no lookahead. A
    bullish zero-cross — DPO going from <=0 to >0 — marks the start of an
    upward cycle phase relative to the underlying drift.

    Mathematically distinct from every oscillator already in the sandbox:
      - MACD / TRIX / TSI / KST / Coppock / Schaff: derivatives of
        *smoothed* momentum (EMAs of EMAs). DPO does no momentum
        smoothing — it is raw price minus a centered SMA.
      - Aroon / Ichimoku / Donchian: built on highest-high / lowest-low
        windows, not deviation from a centered mean.
      - RSI / Stoch / MFI / CCI / Williams %R / Connors RSI: range-bounded
        oscillators on price action; DPO is unbounded and signed.
      - Awesome / Chaikin Osc: differences of two SMAs of median price /
        ADL. DPO is the difference between price and a single displaced
        SMA, with no second smoothing stage.
      - Fisher / Inverse Fisher: nonlinear S-curve transforms — DPO is
        purely linear.

    Entry (prior-bar values only — no lookahead):
        DPO[t-2] <= 0  AND  DPO[t-1] > 0       (bullish zero-cross)
        close[t-1] > SMA(100)[t-1]             (long-term uptrend gate)
    Exit: DPO < 0 (cycle turns down) OR close < SMA(20) (momentum break).
    """
    close = df["close"].to_numpy(dtype=float)

    n = 20
    shift_amt = n // 2 + 1  # 11
    sma_n = _sma(close, n)
    sma_displaced = pd.Series(sma_n).shift(shift_amt).to_numpy()
    dpo = close - sma_displaced

    sma100 = _sma(close, 100)
    sma20 = _sma(close, 20)

    dpo_prev1 = np.concatenate(([np.nan], dpo[:-1]))
    dpo_prev2 = np.concatenate(([np.nan], dpo_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    cross_up = (
        np.isfinite(dpo_prev1) & np.isfinite(dpo_prev2)
        & (dpo_prev2 <= 0.0)
        & (dpo_prev1 > 0.0)
    )
    trend_ok = (
        np.isfinite(close_prev) & np.isfinite(sma100_prev)
        & (close_prev > sma100_prev)
    )
    entries = cross_up & trend_ok

    exits = (
        (np.isfinite(dpo) & (dpo < 0.0))
        | (np.isfinite(sma20) & (close < sma20))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_clenow_momentum_score(df: pd.DataFrame) -> list[Trade]:
    """Clenow 'Stocks on the Move' OLS momentum-score crossover.

    Fits ordinary least squares of log(close) vs time on a 90-bar window
    to estimate a per-bar drift (slope) and the regression R²; the
    score is

        score = (e^{252·slope} − 1) · R²

    i.e., annualized exponential return rate weighted by trend
    cleanliness. High score = strong, low-noise uptrend; this is the
    Andreas Clenow (2015, *Stocks on the Move*) rotation criterion
    applied here as a single-asset entry filter.

    Mathematically distinct from every strategy in the sandbox:
      - All EMA/SMA momentum derivatives (MACD, TRIX, TSI, KST, Coppock,
        Schaff, KAMA, HMA, AwOsc, Chaikin, Vortex, RVI) are recursive
        smoothings; this one is a closed-form OLS fit.
      - Range-bounded oscillators (RSI, Stoch, MFI, CCI, Williams %R,
        Connors RSI, UltimateOsc) measure short-term overbought/oversold;
        score measures *trend quality*.
      - Aroon/Donchian/Ichimoku/Vortex use highest-high / lowest-low
        windows; this is purely a least-squares fit.
      - DPO is the only existing un-smoothed price-vs-mean deviation,
        but DPO does not estimate drift or trend cleanliness.
      - Choppiness Index is a log-ratio range measure, not a slope.

    Entry (prior-bar values only — no lookahead):
        score[t-2] <= 0.5 AND score[t-1] > 0.5      (cross up through 50%)
        close[t-1] > SMA(100)[t-1]                  (uptrend gate)
    Exit: score < 0 (trend gone) OR close < SMA(50).
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    window = 90

    positive = close > 0
    log_close = np.full(n, np.nan, dtype=float)
    log_close[positive] = np.log(close[positive])

    xs = np.arange(window, dtype=float)
    x_c = xs - xs.mean()
    x_var_fixed = float((x_c ** 2).sum())

    kernel = x_c[::-1].copy()
    cov_xy = np.full(n, np.nan, dtype=float)
    if n >= window:
        valid = np.convolve(np.nan_to_num(log_close, nan=0.0), kernel, mode="valid")
        cov_xy[window - 1:] = valid

    y_sum = pd.Series(log_close).rolling(window, min_periods=window).sum().to_numpy()
    y2_sum = pd.Series(log_close ** 2).rolling(window, min_periods=window).sum().to_numpy()
    var_y = y2_sum - (y_sum ** 2) / window

    with np.errstate(invalid="ignore", divide="ignore"):
        slope = cov_xy / x_var_fixed if x_var_fixed > 0 else np.full(n, np.nan)
        ss_res = var_y - (cov_xy ** 2) / x_var_fixed
        r2 = np.where(var_y > 1e-12, 1.0 - ss_res / var_y, 0.0)
    r2 = np.clip(r2, 0.0, 1.0)

    annualized = np.expm1(np.clip(slope * 252.0, -5.0, 5.0))
    score = annualized * r2

    sma100 = _sma(close, 100)
    sma50 = _sma(close, 50)

    score_prev1 = np.concatenate(([np.nan], score[:-1]))
    score_prev2 = np.concatenate(([np.nan], score_prev1[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    cross_up = (
        np.isfinite(score_prev1) & np.isfinite(score_prev2)
        & (score_prev2 <= 0.5)
        & (score_prev1 > 0.5)
    )
    trend_ok = (
        np.isfinite(close_prev) & np.isfinite(sma100_prev)
        & (close_prev > sma100_prev)
    )
    entries = cross_up & trend_ok

    exits = (
        (np.isfinite(score) & (score < 0.0))
        | (np.isfinite(sma50) & (close < sma50))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_bollinger_pctb_reversion(df: pd.DataFrame) -> list[Trade]:
    """Bollinger %B mean-reversion in long-term uptrend.

    %B = (close − lower) / (upper − lower) where bands are SMA(20) ± 2·σ(20).
    A %B < 0 reading means the bar closed below the lower Bollinger band — a
    short-term oversold extreme. We require the prior close to be above
    SMA(100) so we only fade dips inside an established uptrend, then exit
    on the snap-back to the middle band (%B > 0.5) OR a momentum-failure
    drop below SMA(20).

    Distinct from bb_breakout (which trades upper-band trend-follow
    breakouts), from ibs_trend_filter (single-bar range position rather than
    σ-bands), and from RSI/CCI oscillator variants (this measures distance
    from a volatility-scaled mean, not relative-strength counts).
    """
    close = df["close"].to_numpy(dtype=float)

    sma20 = _sma(close, 20)
    std20 = _stdev(close, 20)
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    width = upper - lower
    pctb = np.where(width > 0, (close - lower) / width, np.nan)

    sma100 = _sma(close, 100)

    pctb_prev = np.concatenate(([np.nan], pctb[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))

    valid = np.isfinite(pctb_prev) & np.isfinite(sma100_prev)
    entries = valid & (pctb_prev < 0.0) & (close_prev > sma100_prev)
    exits = (np.isfinite(pctb) & (pctb > 0.5)) | (
        np.isfinite(sma20) & (close < sma20)
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_anchored_vwap_reclaim(df: pd.DataFrame) -> list[Trade]:
    """Anchored-VWAP reclaim from the rolling 60-bar swing-low.

    For every bar t we anchor a volume-weighted average price (AVWAP) to the
    bar with the lowest close in the trailing 60 bars and accumulate
    typical-price·volume from that anchor through t. AVWAP is the institutional
    reference price paid by everyone who bought since the swing low; a
    cross-up reclaim from below means accumulation has now overwhelmed
    distribution since the most recent capitulation point.

    Entry (decided at bar close, prior bar references gate the trend):
        close[t-1] < AVWAP[t-1]               (was below the anchored mean)
        close[t]   >= AVWAP[t]                (reclaims it on this bar)
        close[t-1] > SMA(200)[t-1]            (broader bull regime)
    Exit:
        close < AVWAP  AND  close < SMA(20)   (volume-weighted ref AND
                                               short-term momentum both lost)

    Distinct from every strategy already in the sandbox:
      - Not a recursive EMA/SMA-derived oscillator (MACD, TRIX, TSI, KST,
        Coppock, Schaff, Chaikin, Vortex, RVI, AwOsc, KAMA, HMA, etc.).
      - Not a range/channel breakout (Donchian, Ichimoku, NR7, Aroon, BB,
        squeeze, Minervini VCP) — uses a volume-weighted price level.
      - Not a fixed-window OLS drift (Clenow) or detrended-mean (DPO,
        Bollinger %B) — anchor floats with the rolling swing low.
      - Not a generic OBV/CMF/EFI volume oscillator — AVWAP is a *price*
        the average buyer paid since the anchor, not a flow accumulator.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n = len(close)

    window = 60
    typ = (high + low + close) / 3.0
    pv = typ * volume

    cum_pv = np.concatenate(([0.0], np.cumsum(pv)))
    cum_v = np.concatenate(([0.0], np.cumsum(volume)))

    pos_in_win = (
        pd.Series(close)
        .rolling(window, min_periods=window)
        .apply(np.argmin, raw=True)
        .to_numpy()
    )

    idx_arr = np.arange(n)
    valid_anchor = np.isfinite(pos_in_win)
    pos_filled = np.where(valid_anchor, pos_in_win, 0.0).astype(np.int64)
    anchor_idx = np.clip(idx_arr - window + 1 + pos_filled, 0, n - 1)

    sum_pv = cum_pv[idx_arr + 1] - cum_pv[anchor_idx]
    sum_v = cum_v[idx_arr + 1] - cum_v[anchor_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        avwap = np.where(
            valid_anchor & (sum_v > 0.0), sum_pv / sum_v, np.nan
        )

    sma200 = _sma(close, 200)
    sma20 = _sma(close, 20)

    avwap_prev = np.concatenate(([np.nan], avwap[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    valid_entry = (
        np.isfinite(avwap_prev)
        & np.isfinite(avwap)
        & np.isfinite(sma200_prev)
        & np.isfinite(close_prev)
    )
    entries = (
        valid_entry
        & (close_prev < avwap_prev)
        & (close >= avwap)
        & (close_prev > sma200_prev)
    )
    exits = (
        np.isfinite(avwap)
        & np.isfinite(sma20)
        & (close < avwap)
        & (close < sma20)
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_connors_double_7s(df: pd.DataFrame) -> list[Trade]:
    """Larry Connors' Double 7s mean-reversion in a long-term uptrend.

    Classic asymmetric channel rule: in an established uptrend (close >
    SMA(200)), buy when the close prints at the lowest level of the prior 7
    closes (7-day closing low) — this is a shallow pullback, not a deep
    drawdown. Exit when the close prints at the highest level of the prior 7
    closes (7-day closing high). Uses raw closing-price extremes, so it is
    distinct from %B (volatility σ-band), Donchian (high/low channel),
    cum_rsi2 (oscillator persistence), and IBS (intra-bar range).
    """
    close = df["close"].to_numpy(dtype=float)
    sma200 = _sma(close, 200)

    close_s = pd.Series(close)
    # Rolling min/max of the prior 7 closes (excluding the current bar) so
    # the entry/exit signal is fully decidable at bar close with no leak.
    low7_prev = close_s.shift(1).rolling(7, min_periods=7).min().to_numpy()
    high7_prev = close_s.shift(1).rolling(7, min_periods=7).max().to_numpy()

    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))

    entries = (
        np.isfinite(low7_prev)
        & np.isfinite(sma200_prev)
        & (close <= low7_prev)
        & (close_prev > sma200_prev)
    )
    exits = np.isfinite(high7_prev) & (close >= high7_prev)

    return _walk(entries, exits, close, df["date"].values)


def strat_raschke_holy_grail(df: pd.DataFrame) -> list[Trade]:
    """Linda Raschke 'Holy Grail' — buy the first pullback to EMA(20) inside
    a strong ADX trend, on momentum recovery.

    Setup (all measured on the prior closed bar so the decision is fully
    bar-close decidable):
      1. ADX(14) > 30  — a strong trend exists.
      2. +DI > -DI     — directional bias is bullish.
      3. Prior bar's low touched/pierced EMA(20) — the pullback to the
         dynamic support has occurred.
      4. Today's close prints above the prior bar's high — momentum
         resumes upward through the pullback bar.
    Exit: close drops below EMA(20) for two consecutive bars (a clean break
    of the dynamic support that defined the setup).

    This differs from adx_dmi_trend_emergence (which fires on a fresh
    rising-ADX cross above 20 with no pullback requirement) by demanding a
    tactical pullback-and-recovery to EMA(20) inside an already-strong trend.
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

    ema20 = _ema(close, 20)

    # All "setup" features are read from the prior bar to keep decisions
    # decidable strictly at today's bar close.
    adx_prev = np.concatenate(([np.nan], adx[:-1]))
    plus_di_prev = np.concatenate(([np.nan], plus_di[:-1]))
    minus_di_prev = np.concatenate(([np.nan], minus_di[:-1]))
    ema20_prev = np.concatenate(([np.nan], ema20[:-1]))
    high_prev = np.concatenate(([np.nan], high[:-1]))
    low_prev = np.concatenate(([np.nan], low[:-1]))

    pullback_prev = low_prev <= ema20_prev  # pullback bar pierced EMA20
    entries = (
        np.isfinite(adx_prev)
        & np.isfinite(plus_di_prev)
        & np.isfinite(minus_di_prev)
        & np.isfinite(ema20_prev)
        & np.isfinite(high_prev)
        & (adx_prev > 30.0)
        & (plus_di_prev > minus_di_prev)
        & pullback_prev
        & (close > high_prev)
    )

    below_ema = np.isfinite(ema20) & (close < ema20)
    below_ema_prev = np.concatenate(([False], below_ema[:-1]))
    exits = below_ema & below_ema_prev

    return _walk(entries, exits, close, df["date"].values)


def strat_vwap_zscore_reversion(df: pd.DataFrame) -> list[Trade]:
    """Rolling 20-bar VWAP z-score mean reversion in a long-term uptrend.

    Compute a rolling 20-bar volume-weighted average price (VWAP_20). The
    residual = close - VWAP_20 captures short-term over/undershoot relative
    to where the recent flow-weighted price has been. We standardise the
    residual by its own 20-bar stdev so the entry threshold is regime
    adaptive (a noisy, high-vol stock needs a bigger absolute residual to
    cross -1.5σ than a quiet one). Long when price is meaningfully *below*
    the volume-weighted mean inside a long-term uptrend; exit on reversion
    to/above the VWAP or a regime stop.

    Decision is fully bar-close decidable — VWAP_20 and its residual stdev
    use only data up to and including bar t (close[t], volume[t]).

    Distinct from sandbox cousins:
      - anchored_vwap_reclaim: that anchors AVWAP to the rolling 60-bar
        swing-low and triggers on a one-shot cross-up reclaim. This one
        uses a *fixed-window* rolling VWAP and an adaptive *z-score*
        threshold, not a level-cross event.
      - bollinger_pctb_reversion: σ-bands around a price SMA. This uses
        residual-from-VWAP (volume-weighted), and its stdev is of the
        *residual itself*, not raw close.
      - connors_double_7s: raw 7-bar closing extrema. This is a continuous
        z-score with volume weighting, not a discrete N-bar low.
    """
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    n = 20
    pv = pd.Series(close * volume)
    v = pd.Series(volume)
    sum_pv = pv.rolling(n, min_periods=n).sum().to_numpy()
    sum_v = v.rolling(n, min_periods=n).sum().to_numpy()
    vwap_n = np.where(sum_v > 0.0, sum_pv / np.where(sum_v > 0.0, sum_v, np.nan), np.nan)

    residual = close - vwap_n
    res_std = (
        pd.Series(residual)
        .rolling(n, min_periods=n)
        .std(ddof=0)
        .to_numpy()
    )
    z = np.where(res_std > 0.0, residual / np.where(res_std > 0.0, res_std, np.nan), np.nan)

    sma100 = _sma(close, 100)
    sma50 = _sma(close, 50)

    z_prev = np.concatenate(([np.nan], z[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))
    sma100_prev = np.concatenate(([np.nan], sma100[:-1]))
    sma50_prev = np.concatenate(([np.nan], sma50[:-1]))

    entries = (
        np.isfinite(z_prev)
        & np.isfinite(sma100_prev)
        & np.isfinite(close_prev)
        & (z_prev < -1.5)
        & (close_prev > sma100_prev)
    )
    exits = (
        (np.isfinite(z_prev) & (z_prev >= 0.0))
        | (np.isfinite(sma50_prev) & np.isfinite(close_prev) & (close_prev < sma50_prev))
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_guppy_gmma_compression_release(df: pd.DataFrame) -> list[Trade]:
    """Guppy Multiple Moving Averages — compression-then-release long entry.

    Daryl Guppy's GMMA uses two ribbons of EMAs to read trader vs investor
    behaviour:
      short bundle (traders): EMA 3, 5, 8, 10, 12, 15
      long  bundle (investors): EMA 30, 35, 40, 45, 50, 60

    A high-quality bullish change-of-trend occurs when the short bundle —
    after being *compressed* (narrow spread, indicating indecision) — fans
    out and the *minimum* of the short bundle crosses above the *maximum* of
    the long bundle (clean bullish stack). The compression filter weeds out
    chop-driven crosses that the basic 'fast-EMA over slow-EMA' miss.

    Entry (decided on prior bar to avoid lookahead): yesterday's
      min(short_ribbon) > max(long_ribbon)  AND
      yesterday's min(short_ribbon) <= max(long_ribbon) two bars ago
        (i.e. a *fresh* cross-up, not an already-running trend), AND
      compression on the cross day: the short-ribbon spread on the bar
      before the cross was in its bottom 30 percent over the prior 60 bars
      (the 'rubber band' that snaps).
    Exit: yesterday's max(short_ribbon) < min(long_ribbon) (bundle
      penetration — short ribbon dips back into the long ribbon).

    Distinct from sandbox cousins:
      - hma_bullish_cross / kama_cross_trend / ma_cross: single fast-vs-slow
        crossover with no compression filter and no ribbon stack check.
      - squeeze_breakout (TTM): BB-inside-KC volatility compression, not
        EMA-ribbon compression — fires on price breakout, not stack flip.
      - clenow_momentum: regression slope on close, not ribbon geometry.
    """
    close = df["close"].to_numpy(dtype=float)

    short_periods = (3, 5, 8, 10, 12, 15)
    long_periods = (30, 35, 40, 45, 50, 60)

    short_ribbon = np.vstack([_ema(close, n) for n in short_periods])
    long_ribbon = np.vstack([_ema(close, n) for n in long_periods])

    short_min = short_ribbon.min(axis=0)
    short_max = short_ribbon.max(axis=0)
    long_min = long_ribbon.min(axis=0)
    long_max = long_ribbon.max(axis=0)

    # Ribbon spread normalised by close (so it scales with price).
    short_spread = (short_max - short_min) / np.where(close > 0.0, close, np.nan)

    spread_series = pd.Series(short_spread)
    spread_q30 = spread_series.rolling(60, min_periods=60).quantile(0.30).to_numpy()

    short_min_prev = np.concatenate(([np.nan], short_min[:-1]))
    short_min_prev2 = np.concatenate(([np.nan, np.nan], short_min[:-2]))
    short_max_prev = np.concatenate(([np.nan], short_max[:-1]))
    long_max_prev = np.concatenate(([np.nan], long_max[:-1]))
    long_max_prev2 = np.concatenate(([np.nan, np.nan], long_max[:-2]))
    long_min_prev = np.concatenate(([np.nan], long_min[:-1]))
    spread_prev2 = np.concatenate(([np.nan, np.nan], short_spread[:-2]))
    q30_prev2 = np.concatenate(([np.nan, np.nan], spread_q30[:-2]))

    fresh_cross = (
        np.isfinite(short_min_prev)
        & np.isfinite(long_max_prev)
        & np.isfinite(short_min_prev2)
        & np.isfinite(long_max_prev2)
        & (short_min_prev > long_max_prev)
        & (short_min_prev2 <= long_max_prev2)
    )
    compressed = (
        np.isfinite(spread_prev2)
        & np.isfinite(q30_prev2)
        & (spread_prev2 <= q30_prev2)
    )

    entries = fresh_cross & compressed
    exits = (
        np.isfinite(short_max_prev)
        & np.isfinite(long_min_prev)
        & (short_max_prev < long_min_prev)
    )

    return _walk(entries, exits, close, df["date"].values)


def strat_hammer_pin_bar_uptrend(df: pd.DataFrame) -> list[Trade]:
    """Bullish hammer (pin bar) at a 20-bar swing low in an SMA(50)>SMA(200)
    uptrend — long-only candle-pattern reversal.

    A hammer is a single-bar reversal pattern with a tiny body sitting at the
    top of the bar's range and a long lower wick that rejected a probe to new
    lows. Quant-textbook conditions used here (Bulkowski / Nison):
      - lower_wick >= 2.0 * body  (deep rejection from below)
      - upper_wick <= 0.5 * body  (close near the high)
      - body / range <= 0.35      (small real body relative to the day)
      - body > 0 and range > 0    (avoid degenerate / doji-like bars)

    Context filters that turn the pattern into a tradeable edge:
      - Today's low equals (or undercuts) the rolling 20-bar low — the pin
        must occur at a meaningful swing low, not mid-range.
      - SMA(50) > SMA(200) — only buy hammers in established uptrends. This
        is the classic 'pullback within uptrend' regime where mean-reversion
        candles work best.

    All entry conditions are evaluated on the bar's own close (open/high/low/
    close are all known by then), so no shift is needed — the agent enters at
    the close of the hammer day.

    Exit: bar close >= SMA(10) — a small target back at the short-term mean,
    consistent with the bounce thesis. If the bounce fails the next leg-down
    will eventually carry close < SMA(10) again, but in practice the
    walk-forward force-close caps the worst tail.

    Distinct from sandbox cousins:
      - ibs_trend_filter / cum_rsi2_pullback / connors_double_7s: oscillator
        / IBS / N-day-low *triggers*, no candle-shape requirement.
      - williams_vix_fix_spike: volatility-spike trigger, not bar anatomy.
      - heikin_ashi_flip: smoothed HA candle flip — does not require a long
        lower-wick rejection at a swing low.
    """
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    body = np.abs(close - open_)
    rng = high - low
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    safe_body = np.where(body > 0.0, body, np.nan)
    safe_rng = np.where(rng > 0.0, rng, np.nan)

    is_hammer = (
        (lower_wick >= 2.0 * body)
        & (upper_wick <= 0.5 * safe_body)
        & ((body / safe_rng) <= 0.35)
        & (body > 0.0)
        & (rng > 0.0)
    )
    is_hammer = np.where(np.isfinite(is_hammer), is_hammer, False).astype(bool)

    low_20 = pd.Series(low).rolling(20, min_periods=20).min().to_numpy()
    at_swing_low = np.isfinite(low_20) & (low <= low_20)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = is_hammer & at_swing_low & uptrend

    sma10 = _sma(close, 10)
    exits = np.isfinite(sma10) & (close >= sma10)

    return _walk(entries, exits, close, df["date"].values)


def strat_linreg_slope_signchange(df: pd.DataFrame) -> list[Trade]:
    """20-bar least-squares regression slope of close — long when the fitted
    slope crosses from non-positive to positive (drift turns bullish) inside
    an SMA(200)+SMA(50) uptrend; exit on slope flipping back negative or
    close < SMA(20).

    The OLS slope of the most recent N closes is a direct estimate of the
    stock's short-term price drift — the *rate of change of the fitted
    trend line*, not a moving-average comparison. A sign change from <= 0
    to > 0 marks the precise bar at which 20-bar drift becomes bullish, a
    classic Pring 'momentum re-emergence' tell that is structurally
    distinct from the EMA / Hull / KAMA / TRIX / Coppock / DPO crosses
    already in the sandbox (those compare price to a smoothed level, not
    to a regression line).

    Closed-form formula (efficient, no per-bar polyfit):
        slope_t = (n * Σ(x·y) - Σx · Σy) / (n · Σ(x²) - (Σx)²)
    with x = 0..n-1 inside the window.  Σx and Σ(x²) are constants; only
    Σy and Σ(x·y) need to roll.  All inputs at bar t are known by close
    of t — no shift / look-ahead.

    Regime filters (must hold AT entry bar):
      - close > SMA(200)  : long-only, regime ok.
      - close > SMA(50)   : avoid buying weak rallies still under SMA(50).
    Exit (either):
      - slope flips negative  : 20-bar drift turned down again.
      - close < SMA(20)       : short-term momentum failed.
    """
    close = df["close"].to_numpy(dtype=float)

    n = 20
    x = np.arange(n, dtype=float)
    sum_x = float(x.sum())
    sum_x2 = float((x * x).sum())
    denom = n * sum_x2 - sum_x * sum_x

    s = pd.Series(close)
    sum_y = s.rolling(n, min_periods=n).sum().to_numpy()
    sum_xy = s.rolling(n, min_periods=n).apply(
        lambda w: float(np.dot(x, w)), raw=True
    ).to_numpy()
    slope = (n * sum_xy - sum_x * sum_y) / denom

    slope_prev = pd.Series(slope).shift(1).to_numpy()
    sign_cross_up = (
        np.isfinite(slope)
        & np.isfinite(slope_prev)
        & (slope_prev <= 0.0)
        & (slope > 0.0)
    )

    sma200 = _sma(close, 200)
    sma50 = _sma(close, 50)
    sma20 = _sma(close, 20)
    uptrend = (
        np.isfinite(sma200)
        & np.isfinite(sma50)
        & (close > sma200)
        & (close > sma50)
    )

    entries = sign_cross_up & uptrend

    slope_neg = np.isfinite(slope) & (slope < 0.0)
    below_sma20 = np.isfinite(sma20) & (close < sma20)
    exits = slope_neg | below_sma20

    return _walk(entries, exits, close, df["date"].values)


def strat_td_sequential_buy_setup(df: pd.DataFrame) -> list[Trade]:
    """Tom DeMark TD Sequential — bullish setup completion (count of 9).

    DeMark's setup is a *count*, not a smoothed indicator. Bars where
    close[t] < close[t-4] increment a counter; any bar where close[t] >=
    close[t-4] resets it. When the counter reaches 9, the market has put
    in 9 consecutive bars of close-vs-4-bars-ago weakness — the canonical
    'sellers exhausted' signal. This is structurally different from every
    MA-cross / oscillator strategy already in the journal because it
    reasons about a discrete bar count, not a continuous indicator level.

    To avoid catching a true falling-knife, gate by SMA(200) regime: only
    take the buy-setup when the longer-term trend is up (close > SMA(200)).
    The 9-count then maps to a deep pullback in an established uptrend —
    exactly the textbook DeMark scenario.

    Entry  : setup_count == 9 AND close > SMA(200) AND SMA(50) > SMA(200)
    Exits  : close > SMA(10) (short-term momentum returns) OR
             close < lowest_low_5 — protects against breakdown.
    """
    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)

    setup = np.zeros(n, dtype=int)
    for i in range(4, n):
        if close[i] < close[i - 4]:
            setup[i] = setup[i - 1] + 1 if i > 0 else 1
        else:
            setup[i] = 0

    sma200 = _sma(close, 200)
    sma50 = _sma(close, 50)
    sma10 = _sma(close, 10)

    ll5 = pd.Series(low).shift(1).rolling(5, min_periods=5).min().to_numpy()

    regime = (
        np.isfinite(sma200)
        & np.isfinite(sma50)
        & (close > sma200)
        & (sma50 > sma200)
    )
    entries = (setup == 9) & regime

    momentum_back = np.isfinite(sma10) & (close > sma10)
    breakdown = np.isfinite(ll5) & (close < ll5)
    exits = momentum_back | breakdown

    return _walk(entries, exits, close, df["date"].values)


def strat_keltner_channel_breakout(df: pd.DataFrame) -> list[Trade]:
    """Keltner Channel upside breakout — long when close crosses above the
    EMA(20) + 2.0 * ATR(10) envelope inside an SMA(200) uptrend; exit when
    close drops back through the EMA(20) centerline.

    Chester Keltner's channel is built around a moving-average centerline
    with bands expanded by *true range* rather than standard deviation, so it
    reacts to gap-driven volatility (Bollinger ignores gaps) and stays
    smoother during sideways drift. An upside band penetration in an
    established uptrend marks a fresh expansion leg — distinct edge from:
      - bb_breakout / bollinger_pctb_reversion: σ-bands on close-only,
        no gap component.
      - donchian_20_10_trend: raw highest-high channel (price-based, no
        volatility scaling).
      - squeeze_breakout: Bollinger-inside-Keltner *compression* trigger,
        not a band penetration.
      - parabolic_sar / supertrend: ATR trailing stops, not channel bands.

    Crossover is computed at bar close from today vs prior bar values
    (both known by the close), so no lookahead.

    Entry : close > upper_kc AND prior close <= prior upper_kc
            AND close > SMA(200) regime filter.
    Exit  : close < EMA(20) — give back to the channel mean.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    ema20 = _ema(close, 20)
    atr10 = _atr(high, low, close, 10)
    upper_kc = ema20 + 2.0 * atr10

    sma200 = _sma(close, 200)

    close_prev = np.concatenate(([np.nan], close[:-1]))
    upper_prev = np.concatenate(([np.nan], upper_kc[:-1]))

    crossed_up = (
        np.isfinite(upper_kc)
        & np.isfinite(upper_prev)
        & (close > upper_kc)
        & (close_prev <= upper_prev)
    )
    regime = np.isfinite(sma200) & (close > sma200)
    entries = crossed_up & regime

    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_three_white_soldiers(df: pd.DataFrame) -> list[Trade]:
    """Three White Soldiers — three consecutive strong bullish candles, each
    closing higher than the previous, gated by an SMA(50)>SMA(200) uptrend.

    Classical Nison/Bulkowski reversal-continuation pattern: three back-to-back
    real-body bullish candles where each close is above the prior close, each
    body is decisive (body/range >= 0.55), and upper wicks are short (close
    near the bar high — upper_wick/range <= 0.25). The first candle must
    re-engage upward momentum (its body close > prior bar close) so the
    pattern is not a continuation of an already-running blow-off bar.

    Why this is a distinct edge from existing sandbox candle/structure plays:
      - hammer_pin_bar_uptrend: a *single* pin-bar with long lower wick at a
        20-bar swing low — opposite anatomy (tiny body, deep tail).
      - heikin_ashi_flip: smoothed HA candle colour flip — not a sequence of
        three raw bodies, and has no body-size requirement.
      - td_sequential_buy_setup: a 9-count of close < close[t-4] — duration
        rule on close-to-close differences, no body anatomy.
      - minervini_vcp_breakout: multi-week base-and-breakout structure, very
        different time horizon and uses pivot/52-wk highs.
      - linreg_slope_signchange / coppock / kst: smoothed momentum sign
        flips, not bar-by-bar anatomy.

    Entries/exits are decidable by bar close (open, high, low, close all
    known); the third soldier's own close is the trigger, no lookahead.

    Entry: three consecutive bars satisfy
              body > 0 & rng > 0
              body/rng >= 0.55
              upper_wick/rng <= 0.25
              close[t] > close[t-1]
            AND SMA(50) > SMA(200) regime.
    Exit : close < EMA(10) — short-term-mean give-back, mirrors the hammer
            strategy's exit so the bounce thesis times out cleanly.
    """
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    body = close - open_
    rng = high - low
    upper_wick = high - np.maximum(open_, close)

    safe_rng = np.where(rng > 0.0, rng, np.nan)
    body_pct = body / safe_rng
    upper_pct = upper_wick / safe_rng

    strong_bull = (
        (body > 0.0)
        & (rng > 0.0)
        & np.isfinite(body_pct)
        & (body_pct >= 0.55)
        & np.isfinite(upper_pct)
        & (upper_pct <= 0.25)
    )
    strong_bull = np.where(np.isfinite(strong_bull), strong_bull, False).astype(bool)

    close_1 = np.concatenate(([np.nan], close[:-1]))
    close_2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    close_3 = np.concatenate(([np.nan, np.nan, np.nan], close[:-3]))
    bull_1 = np.concatenate(([False], strong_bull[:-1]))
    bull_2 = np.concatenate(([False, False], strong_bull[:-2]))

    higher_closes = (
        np.isfinite(close_1)
        & np.isfinite(close_2)
        & np.isfinite(close_3)
        & (close > close_1)
        & (close_1 > close_2)
        & (close_2 > close_3)
    )

    soldiers = strong_bull & bull_1 & bull_2 & higher_closes

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = soldiers & uptrend

    ema10 = _ema(close, 10)
    exits = np.isfinite(ema10) & (close < ema10)

    return _walk(entries, exits, close, df["date"].values)


def strat_bullish_engulfing_pullback(df: pd.DataFrame) -> list[Trade]:
    """Bullish Engulfing candle on a short-term pullback inside SMA(50)>SMA(200)
    uptrend. Exit on close<EMA(10).

    A bullish engulfing requires a *bearish* prior bar whose body is fully
    contained inside today's *bullish* body:
        prev: close[t-1] < open[t-1]                  (down candle)
        now : close[t]   > open[t]                    (up candle)
              open[t]    <= close[t-1]                (today opens at/below
                                                       prior close)
              close[t]   >= open[t-1]                 (today closes at/above
                                                       prior open)
              body[t]    >  body[t-1]                 (strict engulfing)

    The pullback context is enforced by requiring yesterday's close to sit
    below SMA(10) — i.e. the bar being engulfed was a genuine retracement,
    not a flat consolidation bar at trend highs. Combined with the long-term
    SMA(50) > SMA(200) filter, the setup is a textbook trend-continuation
    reversal.

    Distinct from existing sandbox candle / pullback plays:
      - hammer_pin_bar_uptrend: single-bar pin (tiny body, deep lower wick),
        no requirement for an engulfed prior bar.
      - three_white_soldiers: three consecutive *same-direction* bull bodies
        with no engulfing relationship and no pullback context.
      - heikin_ashi_flip: smoothed-candle colour flip on HA-transformed
        prices, hides the raw engulfing relationship by construction.
      - raschke_holy_grail: requires ADX>30 strong-trend regime AND a wick
        touch of EMA(20); this strategy is purely about the two-bar body
        relationship and a soft-pullback (close<SMA(10)) context.
      - cum_rsi2_pullback / connors_rsi_pullback / connors_double_7s /
        ibs_trend_filter: oscillator-based pullback identifiers, no candle
        anatomy.
      - bullish_engulfing is a two-bar *body-relationship* pattern not
        captured by any close-only oscillator (RSI/MACD/CMO/TRIX/TSI/KST/
        Coppock/DPO/Schaff/AO/Fisher/Vortex/CMF/OBV/EFI/MFI/UO).

    Entries/exits are decidable by bar close (today's open and close are
    known at close); engulfing comparison uses prior bar values only.
    """
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    open_prev = np.concatenate(([np.nan], open_[:-1]))
    close_prev = np.concatenate(([np.nan], close[:-1]))

    body_now = close - open_
    body_prev = close_prev - open_prev

    bear_prev = np.isfinite(body_prev) & (body_prev < 0.0)
    bull_now = np.isfinite(body_now) & (body_now > 0.0)
    engulf_open = np.isfinite(close_prev) & (open_ <= close_prev)
    engulf_close = np.isfinite(open_prev) & (close >= open_prev)
    body_grew = (
        np.isfinite(body_now)
        & np.isfinite(body_prev)
        & (body_now > -body_prev)  # |body_now| > |body_prev| since body_prev<0
    )

    engulfing = bear_prev & bull_now & engulf_open & engulf_close & body_grew

    sma10 = _sma(close, 10)
    sma10_prev = np.concatenate(([np.nan], sma10[:-1]))
    pullback_ctx = np.isfinite(sma10_prev) & (close_prev < sma10_prev)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = engulfing & pullback_ctx & uptrend

    ema10 = _ema(close, 10)
    exits = np.isfinite(ema10) & (close < ema10)

    return _walk(entries, exits, close, df["date"].values)


def strat_qstick_zero_cross(df: pd.DataFrame) -> list[Trade]:
    """Q-Stick (Chande) zero-line bullish cross in an SMA(50)>SMA(200) uptrend.

    Q-Stick(n) = SMA_n(close - open). It smooths the *body* of each bar,
    measuring whether bullish bodies (close>open) or bearish bodies have
    dominated the recent window. A bullish zero-line cross — Q-Stick going
    from non-positive to positive at bar close — flags a regime shift in
    intraday close-vs-open pressure that is mathematically orthogonal to
    close-to-close momentum signals.

    Distinct from sandbox plays:
      - Single-/few-bar candle anatomy (hammer_pin_bar_uptrend,
        three_white_soldiers, heikin_ashi_flip) reads one or two bars;
        Q-Stick aggregates body bias over n bars and triggers on its sign.
      - Close-to-close momentum (DPO, Coppock, KST, TRIX, TSI, AO,
        linreg_slope, schaff) operates on close only; Q-Stick's numerator
        is (close-open) — uses the open, which the close-only set ignores.
      - RSI-family (RSI, Connors RSI, Stoch, Inverse Fisher, RVI) tracks
        up/down close *moves* and their magnitudes, not body magnitude.
      - CMF/Chaikin/MFI/EFI/OBV/NVI weight by volume; Q-Stick is unweighted
        and pure-price.

    Entry: Q-Stick(8) crosses up through 0 at bar close (prev<=0 & now>0)
           AND SMA(50) > SMA(200).
    Exit : close < EMA(20) — short-term-mean give-back.
    """
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    body = close - open_
    qstick = _sma(body, 8)

    qstick_prev = np.concatenate(([np.nan], qstick[:-1]))

    cross_up = (
        np.isfinite(qstick_prev)
        & np.isfinite(qstick)
        & (qstick_prev <= 0.0)
        & (qstick > 0.0)
    )

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = cross_up & uptrend

    ema20 = _ema(close, 20)
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_mass_index_reversal_bulge(df: pd.DataFrame) -> list[Trade]:
    """Donald Dorsey's Mass Index reversal-bulge — trend-resumption variant.

    Mass Index is a pure range-volatility indicator (no direction). It
    detects expansion-then-contraction in the high-low range, which Dorsey
    found precedes reversals.

    Construction:
      range_t   = high - low
      ema1_t    = EMA(range, 9)
      ema2_t    = EMA(ema1, 9)
      ratio_t   = ema1 / ema2
      MI_t      = sum_{i=0..24} ratio_{t-i}     (25-bar rolling sum)

    Dorsey's "reversal bulge": MI > 27 then crosses back below 26.5. The
    indicator is direction-agnostic, so we pair it with an uptrend filter
    to bias toward long entries on pullback reversals (trend resumptions).

    Distinct from sandbox plays:
      - Williams VIX Fix / NR7 use single-bar range collapse/expansion;
        Mass Index uses an *EMA ratio* of range, smoothed over 25 bars,
        and triggers on a multi-bar bulge-then-collapse, not single-bar.
      - Volatility-band strategies (squeeze, bb_breakout, keltner) compare
        price to volatility bands; Mass Index ignores price level entirely.
      - Choppiness Index measures trend/range regime; Mass Index measures
        range expansion-contraction acceleration via EMA-of-EMA ratio.
      - ADX/DMI/Aroon are directional trend strength; Mass Index is
        non-directional volatility-shape.

    Entry: MI bulge complete in uptrend.
      - prev MI > 27 at any point in the last `bulge_lookback` bars,
      - current MI crossed below 26.5 this bar (prev>=26.5 & now<26.5),
      - SMA(50) > SMA(200) (uptrend filter).
    Exit : close < EMA(20).
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    rng = high - low
    ema1 = _ema(rng, 9)
    ema2 = _ema(ema1, 9)

    ratio = np.where(
        np.isfinite(ema2) & (ema2 > 0.0),
        ema1 / ema2,
        np.nan,
    )

    # 25-bar rolling sum of the ratio = Mass Index.
    mi = (
        pd.Series(ratio).rolling(25, min_periods=25).sum().to_numpy()
    )

    mi_prev = np.concatenate(([np.nan], mi[:-1]))

    # Drop below 26.5 from at-or-above 26.5 (the "bulge collapse").
    cross_down = (
        np.isfinite(mi) & np.isfinite(mi_prev)
        & (mi_prev >= 26.5) & (mi < 26.5)
    )

    # Confirm a recent bulge: MI exceeded 27 within the last 25 bars
    # (use mi_prev so the lookback ends one bar before the trigger,
    # avoiding any same-bar coupling with the cross-down test).
    bulge = (
        pd.Series(mi_prev > 27.0)
        .rolling(25, min_periods=1)
        .max()
        .to_numpy()
        .astype(bool)
    )

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = cross_down & bulge & uptrend

    ema20 = _ema(close, 20)
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_stoch_rsi_oversold_cross(df: pd.DataFrame) -> list[Trade]:
    """Stochastic RSI %K/%D bullish cross out of oversold, in an SMA(50)>SMA(200)
    uptrend. Tushar Chande & Stanley Kroll (1994).

    StochRSI applies the Stochastic formula to RSI values, producing a 0..1
    oscillator that is more sensitive than either RSI or raw Stochastic alone:

        rsi_t        = RSI(close, 14)
        stoch_rsi_t  = (rsi_t - min(rsi, 14)) / (max(rsi, 14) - min(rsi, 14))
        %K_t         = SMA(stoch_rsi, 3)
        %D_t         = SMA(%K, 3)

    The bullish trigger is %K crossing above %D from below 0.20 — i.e. a
    momentum-of-momentum turn while the indicator was still oversold — gated
    by an SMA(50)>SMA(200) trend regime so we only buy oversold dips inside
    established uptrends. Exit when close drops below EMA(20) (short-term
    mean give-back) — same family of exits used by the other oscillator
    cross strategies in this sandbox.

    Distinct from sandbox plays:
      - stochastic_oversold_recovery: Stochastic on raw price (high/low
        range), no RSI compounding. StochRSI is mathematically a different
        function — it normalizes RSI, not price.
      - inverse_fisher_rsi: applies a sigmoid-style Fisher transform to RSI
        and trades zero-line crosses — no min/max stochastic scaling and no
        %K/%D smoothing pair.
      - connors_rsi_pullback: composite of RSI(3) + RSI of streaks +
        rolling rank of % change; not a stochastic of RSI.
      - rsi_ema (in stock strategies): a single RSI threshold, not a
        smoothed-cross-of-smoothed signal.
      - rvi_signal_cross / fisher_transform_zero_cross / awesome saucer /
        chaikin/macd-style zero-cross indicators all operate on price (or
        body/volume) directly — none apply a stochastic to a momentum
        oscillator's own range.

    All inputs are bar-close decidable and use _prev arrays for the cross
    test, so there is no lookahead.

    Entry: prev %K <= prev %D, current %K > current %D, prev %K < 0.20,
           SMA(50) > SMA(200).
    Exit : close < EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)

    rsi14 = _rsi(close, 14)

    rsi_series = pd.Series(rsi14)
    rsi_min = rsi_series.rolling(14, min_periods=14).min().to_numpy()
    rsi_max = rsi_series.rolling(14, min_periods=14).max().to_numpy()

    denom = rsi_max - rsi_min
    stoch_rsi = np.where(
        np.isfinite(denom) & (denom > 0.0),
        (rsi14 - rsi_min) / denom,
        np.nan,
    )

    k = _sma(stoch_rsi, 3)
    d = _sma(k, 3)

    k_prev = np.concatenate(([np.nan], k[:-1]))
    d_prev = np.concatenate(([np.nan], d[:-1]))

    cross_up = (
        np.isfinite(k_prev)
        & np.isfinite(d_prev)
        & np.isfinite(k)
        & np.isfinite(d)
        & (k_prev <= d_prev)
        & (k > d)
        & (k_prev < 0.20)
    )

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = cross_up & uptrend

    ema20 = _ema(close, 20)
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_cmo_oversold_recovery(df: pd.DataFrame) -> list[Trade]:
    """Chande Momentum Oscillator (CMO) recovery from oversold, in an
    SMA(50)>SMA(200) uptrend. Tushar Chande, *The New Technical Trader* (1994).

    CMO is a pure-price momentum oscillator that, unlike RSI, normalizes by
    the *sum* of up- and down-move magnitudes rather than the smoothed
    up/down ratio:

        diff_t   = close_t - close_{t-1}
        up_t     = max(diff_t,  0)
        down_t   = max(-diff_t, 0)
        Su_t     = sum_{i=0..n-1} up_{t-i}
        Sd_t     = sum_{i=0..n-1} down_{t-i}
        CMO_t    = 100 * (Su_t - Sd_t) / (Su_t + Sd_t)

    CMO ranges in [-100, +100], not [0, 100] like RSI, so its zero line and
    overbought/oversold lines are at -50 / 0 / +50. Because the denominator
    is the *total* directional energy (not just down-energy), CMO can flip
    sign faster than RSI when fresh up-moves enter the window.

    Entry: CMO(14) crosses up through -50 at bar close (prev<=-50 & now>-50)
           — i.e. the indicator is leaving the oversold zone — AND
           SMA(50) > SMA(200) (uptrend filter, only buy dips in trends).
    Exit : close < EMA(20) — short-term mean give-back, same exit family as
           the other sandbox oscillator-cross plays.

    Distinct from sandbox plays:
      - rsi_ema / connors_rsi_pullback / cum_rsi2_pullback / inverse_fisher
        all operate on RSI; CMO uses (up_sum - down_sum) / (up_sum + down_sum)
        with NO Wilder smoothing — the recursion structure is different.
      - stochastic_oversold_recovery / stoch_rsi_oversold_cross use a
        min/max range normalization (stochastic), not a directional sum
        ratio.
      - rvi_signal_cross uses high/low *range* energy partitioned by
        close-vs-open direction, not close-to-close differences.
      - awesome_oscillator_saucer / coppock / kst / trix / tsi / dpo /
        linreg_slope / qstick are momentum measures but none uses the
        symmetric ±100-bounded directional-sum formulation; they're
        unbounded or differently bounded.
      - mfi_oversold_recovery weights by typical-price * volume; CMO is
        unweighted pure close-difference.

    All inputs use prev-bar arrays for the cross test, so there is no
    lookahead — the trigger is decidable at bar close.
    """
    close = df["close"].to_numpy(dtype=float)

    diff = np.concatenate(([0.0], np.diff(close)))
    up = np.where(diff > 0.0, diff, 0.0)
    down = np.where(diff < 0.0, -diff, 0.0)

    n = 14
    su = pd.Series(up).rolling(n, min_periods=n).sum().to_numpy()
    sd = pd.Series(down).rolling(n, min_periods=n).sum().to_numpy()

    denom = su + sd
    cmo = np.where(
        np.isfinite(denom) & (denom > 0.0),
        100.0 * (su - sd) / denom,
        np.nan,
    )

    cmo_prev = np.concatenate(([np.nan], cmo[:-1]))

    cross_up = (
        np.isfinite(cmo) & np.isfinite(cmo_prev)
        & (cmo_prev <= -50.0) & (cmo > -50.0)
    )

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = cross_up & uptrend

    ema20 = _ema(close, 20)
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_wyckoff_spring_reclaim(df: pd.DataFrame) -> list[Trade]:
    """Wyckoff Spring (a.k.a. shakeout / terminal-shakeout) — a *false*
    breakdown below recent support followed by an immediate same-bar reclaim,
    inside a longer-term uptrend (SMA(50) > SMA(200)).

    Wyckoff describes the Spring as a stop-running probe below the prior
    trading-range support: smart money lets price dip beneath the obvious
    stops, absorbs the supply that comes out, and then closes the bar back
    *above* the support level. The fact that price could not stay below
    support is itself the signal — supply has been exhausted and the path of
    least resistance reverses up. Adam Grimes ("The Art and Science of
    Technical Analysis", 2012) calls the same structure a "failure test"
    pattern; Linda Raschke teaches it as "Turtle Soup Plus One"; in modern
    auction-market parlance it's a liquidity grab / sweep.

    Entry conditions, all decidable at bar close (prev-bar arrays prevent
    lookahead):
      1. support = min(low) over the prior N bars (N=20), excluding today —
         computed via low.shift(1).rolling(N).min() so today's low does not
         leak into the level.
      2. Today's low pierced support: low_t < support_t.
      3. Today's close reclaimed support: close_t > support_t.
      4. Today's close finished in the upper portion of today's range — IBS
         > 0.5 — confirming the reclaim is genuine, not a weak bounce that
         merely closed slightly above the wick low.
      5. Trend filter: SMA(50) > SMA(200) at bar close — only fade
         shakeouts in established uptrends, where Wyckoff accumulation
         schematics make the most sense.

    Exit: close < EMA(20) — short-term mean give-back. Same exit family as
    the other sandbox cross/recovery plays so results are comparable.

    Distinct from sandbox plays:
      - donchian_20_10_trend / keltner_channel_breakout / squeeze_breakout
        all enter on UPSIDE breakouts of a channel; this enters on a
        DOWNSIDE breakout that *fails* — opposite directional signature.
      - bollinger_pctb_reversion / vwap_zscore_reversion / cum_rsi2_pullback
        / connors_rsi_pullback / connors_double_7s use oscillator/statistical
        oversold readings, not a structural support-pierce-and-reclaim.
      - williams_vix_fix_spike triggers on a volatility-of-low spike, not a
        single-bar undercut of a horizontal price level.
      - volume_capitulation_reclaim requires a high-volume capitulation bar
        below SMA(20) and reclaim of the *moving average*; the Spring uses
        a horizontal *swing-low* support and is volume-agnostic — the price
        action itself proves absorption.
      - hammer_pin_bar_uptrend looks at single-bar wick geometry near a
        rising MA but does not require piercing a horizontal support level.
      - td_sequential_buy_setup is a 9-bar count of consecutive lower closes;
        the Spring is a one-bar event.

    Hypothesis: stop-runs below visible swing lows that are immediately
    reabsorbed should mark the end of short-term distribution and offer a
    favourable risk/reward long entry inside an ongoing trend.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    n = 20
    # Prior-N-bar low, excluding today — shift(1) so today's low cannot leak.
    support = (
        pd.Series(low).shift(1).rolling(n, min_periods=n).min().to_numpy()
    )

    pierced = np.isfinite(support) & (low < support)
    reclaimed = np.isfinite(support) & (close > support)

    rng = high - low
    ibs = np.where(rng > 0.0, (close - low) / rng, np.nan)
    strong_close = np.isfinite(ibs) & (ibs > 0.5)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = pierced & reclaimed & strong_close & uptrend

    ema20 = _ema(close, 20)
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_williams_fractal_breakout(df: pd.DataFrame) -> list[Trade]:
    """Bill Williams 5-bar up-fractal swing-high breakout in an SMA50>SMA200 uptrend.

    An up-fractal pivot at bar p satisfies: high[p] > high[p-1], high[p-2],
    high[p+1], high[p+2]. The pattern is confirmed only at bar p+2 (when the
    last two right-side bars are known), so the level is shifted by +2 to
    avoid lookahead. We forward-fill the most-recently confirmed up-fractal
    high; entry fires when close pierces that level while the SMA50>SMA200
    trend filter is on. Exit when close drops below EMA20. This is a
    discrete pivot-based breakout — distinct from Donchian rolling-window
    channels, σ-band Bollinger breakouts, and ATR-trail supertrend.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)

    high_s = pd.Series(high)
    is_pivot = (
        (high_s > high_s.shift(1))
        & (high_s > high_s.shift(2))
        & (high_s > high_s.shift(-1))
        & (high_s > high_s.shift(-2))
    ).fillna(False).to_numpy()

    pivot_high = np.where(is_pivot, high, np.nan)
    # Shift by +2 so the level becomes usable only at the confirmation bar.
    last_fractal = pd.Series(pivot_high).shift(2).ffill().to_numpy()

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    valid = (
        ~np.isnan(last_fractal)
        & ~np.isnan(sma50)
        & ~np.isnan(sma200)
        & ~np.isnan(ema20)
    )
    uptrend = sma50 > sma200

    entries = valid & uptrend & (close > last_fractal)
    exits = ~np.isnan(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_weinstein_stage2_breakout(df: pd.DataFrame) -> list[Trade]:
    """Stan Weinstein Stage 2 ("advancing phase") breakout from a Stage 1 base.

    Weinstein's stage analysis (*Secrets For Profiting in Bull and Bear
    Markets*, 1988) classifies long-term price action into four stages around
    the 30-week (≈150-bar) moving average:

        Stage 1: basing   — MA flat, price oscillating around it.
        Stage 2: advance  — MA turning up, price breaks out above MA + base
                            high; this is the only stage Weinstein buys.
        Stage 3: top      — MA flattening, price churning above it.
        Stage 4: decline  — MA falling, price below.

    The Stage 2 entry is structurally different from a vanilla MA cross:
      - Vanilla MA cross (already tested as stock:ma_cross) just needs a
        fast MA to cross a slow MA — it fires constantly in choppy markets
        and has no requirement that the long-term trend be turning.
      - Stage 2 requires (a) the long MA to be RISING (slope > 0 measured
        as MA[t-1] > MA[t-21], i.e. one month of upward slope), AND (b) a
        fresh breakout above the prior 50-bar high. The MA-rising condition
        filters out Stage 1 chop and Stage 4 declines; the high-breakout
        confirms the base resolution.

    This is also distinct from:
      - Donchian 20/10 (no MA-slope requirement; uses 20-bar channel).
      - Minervini VCP (requires volatility contraction — different gate).
      - Clenow momentum (regression-based ranking — no breakout).
      - Pocket pivot (volume-based — no MA-slope structure).

    Entry (decided at bar close, all signals shifted to prior-bar values to
    eliminate lookahead):
        SMA(close, 150)[t-1]      > SMA(close, 150)[t-21]   (MA rising 1mo)
        close[t-1]                > SMA(close, 150)[t-1]    (above 30wk MA)
        close[t-1]                > prior 50-bar highest close
                                                            (base breakout)
    Exit:
        close < EMA(close, 20)                               (trend break)

    Using close-of-prior-bar high in the breakout test (rather than today's
    high) avoids any same-bar lookahead — entry is decidable at today's
    close with strictly historical inputs.
    """
    close = df["close"].to_numpy(dtype=float)

    sma150 = _sma(close, 150)
    ema20 = _ema(close, 20)

    sma150_prev = np.concatenate(([np.nan], sma150[:-1]))
    sma150_lag = np.concatenate(
        ([np.nan] * 21, sma150[:-21])
    ) if len(sma150) > 21 else np.full_like(sma150, np.nan)

    close_prev = np.concatenate(([np.nan], close[:-1]))

    # 50-bar highest CLOSE excluding today (shift(1)).
    high_close_50 = (
        pd.Series(close).shift(1).rolling(50, min_periods=50).max().to_numpy()
    )

    valid = (
        np.isfinite(sma150_prev)
        & np.isfinite(sma150_lag)
        & np.isfinite(close_prev)
        & np.isfinite(high_close_50)
        & np.isfinite(ema20)
    )

    ma_rising = sma150_prev > sma150_lag
    above_ma = close_prev > sma150_prev
    base_breakout = close_prev > high_close_50

    entries = valid & ma_rising & above_ma & base_breakout
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_klinger_volume_oscillator_signal_cross(df: pd.DataFrame) -> list[Trade]:
    """Klinger Volume Oscillator (KVO) bullish signal-line cross inside an
    SMA(50)>SMA(200) uptrend. Exit when close < EMA(20).

    Stephen J. Klinger (Technical Analysis of Stocks & Commodities, Dec 1997)
    designed the KVO to track long-term money-flow trends while remaining
    sensitive to short-term reversals. Construction (all bar-close values):

        trend_t      = +1 if (high+low+close)_t > (h+l+c)_{t-1}, else -1
        DM_t         = high_t - low_t                          (daily range)
        CM_t         = CM_{t-1} + DM_t          if trend_t == trend_{t-1}
                     = DM_{t-1} + DM_t          otherwise (reset on flip)
        VF_t (Force) = volume_t * |2*(DM_t/CM_t) - 1| * trend_t * 100
        KVO_t        = EMA(VF, 34) - EMA(VF, 55)
        Signal_t     = EMA(KVO, 13)

    Buy on the bar where KVO crosses up through Signal (KVO_{t-1} <= Sig_{t-1}
    and KVO_t > Sig_t), provided SMA(50) > SMA(200) at that bar. The CM
    "cumulative-measure" term resets every time the daily H+L+C trend flips,
    which makes the |2*DM/CM - 1| ratio a *relative* measure of how today's
    range compares to the running streak — a unique structural feature.

    This is structurally distinct from every other volume-flow play in the
    sandbox:
      - obv_ema_cross uses cumulative signed volume only (no range component)
      - chaikin_oscillator_zero_cross uses A/D close-position-in-range
        accumulation, EMA(3)-EMA(10), zero-line cross
      - cmf_zero_reclaim is the 21-bar Chaikin Money Flow zero-reclaim
      - elder_force_index_zero_cross is (close - close_prev) * volume EMA-13
      - mfi_oversold_recovery is the 14-bar typical-price money-flow ratio
      - nvi_fosback_trend conditions on negative-volume-index vs its 255-EMA
      - pocket_pivot is a single-bar volume-spike pattern, not an oscillator
    KVO is the only one combining {trend direction, daily range, streak-
    reset cumulative range, volume} into a dual-EMA oscillator with its
    own EMA(13) signal line.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n = len(close)

    if n == 0:
        return []

    hlc = high + low + close
    hlc_prev = np.concatenate(([np.nan], hlc[:-1]))
    trend = np.where(hlc > hlc_prev, 1.0, -1.0)
    trend[0] = 0.0  # undefined on first bar

    dm = high - low

    cm = np.zeros(n, dtype=float)
    for i in range(1, n):
        if trend[i] == trend[i - 1]:
            cm[i] = cm[i - 1] + dm[i]
        else:
            cm[i] = dm[i - 1] + dm[i]

    safe_cm = np.where(cm > 0.0, cm, np.nan)
    vf_ratio = np.where(np.isfinite(safe_cm), 2.0 * (dm / safe_cm) - 1.0, 0.0)
    vf = volume * np.abs(vf_ratio) * trend * 100.0
    vf = np.nan_to_num(vf, nan=0.0, posinf=0.0, neginf=0.0)

    kvo = _ema(vf, 34) - _ema(vf, 55)
    signal = _ema(kvo, 13)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    kvo_prev = np.concatenate(([np.nan], kvo[:-1]))
    signal_prev = np.concatenate(([np.nan], signal[:-1]))

    cross_up = (
        np.isfinite(kvo_prev)
        & np.isfinite(signal_prev)
        & (kvo_prev <= signal_prev)
        & (kvo > signal)
    )
    uptrend = (
        np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)
    )

    entries = cross_up & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_demarker_oversold_reclaim(df: pd.DataFrame) -> list[Trade]:
    """Tom DeMark's DeMarker (DeM) oversold-reclaim cross inside an
    SMA(50)>SMA(200) uptrend; exit when close < EMA(20).

    DeMarker is a *range-extreme* oscillator — unlike RSI (close-to-close
    deltas), CCI (typical-price vs SMA), Stoch (close inside H/L window),
    or MFI (typical-price * volume), DeM measures whether each bar is
    extending the prior bar's high/low *envelope* and how that compares
    over a window. Construction (Tom DeMark, "The New Science of Technical
    Analysis", 1994):

        DeMax_t  = max(high_t  - high_{t-1}, 0)     # only count up-extension
        DeMin_t  = max(low_{t-1} - low_t, 0)        # only count down-extension
        DeM_t    = SMA(DeMax, n) / (SMA(DeMax, n) + SMA(DeMin, n))

    Bounded 0..1; <0.30 = oversold (downside-extension dominates window),
    >0.70 = overbought. The signal is the cross UP through 0.30 from below
    — "downside extension exhausted, range-extension flipping back to the
    upside." Long-only filter: SMA(50) > SMA(200).

    Distinct from every oscillator already tested:
      - RSI / CMO / TSI / RVI / Coppock / KST / Schaff / Inverse-Fisher all
        use *close-to-close* changes (Δclose). DeM uses Δhigh and Δlow
        independently, capturing range-envelope extension rather than
        directional close drift.
      - Stochastic / Stoch-RSI / Williams %R / Ultimate Oscillator place
        close inside the H/L window. DeM compares each bar's H to PRIOR
        bar's H (and L to prior L) — a *bar-over-bar extension* measure.
      - CCI / MFI / Awesome / Fisher / DPO / Chaikin osc / EFI all use
        typical price or volume-weighted variants. DeM ignores typical
        price and volume entirely, using only H/L extensions.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)

    if n == 0:
        return []

    high_prev = np.concatenate(([np.nan], high[:-1]))
    low_prev = np.concatenate(([np.nan], low[:-1]))

    demax = np.where(np.isfinite(high_prev), np.maximum(high - high_prev, 0.0), 0.0)
    demin = np.where(np.isfinite(low_prev), np.maximum(low_prev - low, 0.0), 0.0)

    period = 14
    sma_demax = _sma(demax, period)
    sma_demin = _sma(demin, period)
    denom = sma_demax + sma_demin
    with np.errstate(divide="ignore", invalid="ignore"):
        dem = np.where(denom > 0, sma_demax / denom, np.nan)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    dem_prev = np.concatenate(([np.nan], dem[:-1]))
    dem_prev2 = np.concatenate(([np.nan, np.nan], dem[:-2]))

    cross_up = (
        np.isfinite(dem_prev2)
        & np.isfinite(dem_prev)
        & (dem_prev2 < 0.30)
        & (dem_prev >= 0.30)
    )
    uptrend_prev = np.concatenate(
        ([False], (np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200))[:-1])
    )

    entries = cross_up & uptrend_prev
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_range_filter_buy(df: pd.DataFrame) -> list[Trade]:
    """Donovan Wall's Range Filter — recursive trailing line locked above
    or below price by ``mult * EMA(|Δclose|, n)`` (smoothed). Entry: close
    crosses UP through the prior-bar filter while the filter itself is
    rising and SMA(50) > SMA(200). Exit: close < EMA(20).

    Construction (per the original Pine v4 publication, Donovan Wall 2019):

        Δ_t          = |close_t - close_{t-1}|
        avg_range_t  = EMA(Δ, n)                       # n = 14
        sr_t         = EMA(avg_range, 2n-1) * mult     # mult = 2.618 ≈ φ²
        rf_t = ┌ max(rf_{t-1}, close_t - sr_t)   if close_t > rf_{t-1}
               │ min(rf_{t-1}, close_t + sr_t)   if close_t < rf_{t-1}
               └ rf_{t-1}                         otherwise

    The filter is *recursive* — each value depends on the prior filter
    value — so it cannot be expressed as a moving average, KAMA, HMA, or
    supertrend. It locks in monotonic levels (only steps up while price
    rises, only steps down while price falls) and ignores moves smaller
    than the smoothed-range envelope, producing a piecewise-flat trail.

    Distinct from every prior strategy:
      - Supertrend (ATR-based) flips around (high+low)/2 with ATR bands;
        Range Filter trails *close* with EMA-of-|Δclose|.
      - KAMA / Schaff / TRIX / HMA are smooth moving averages; the Range
        Filter is non-monotonic locked steps.
      - Donchian / Keltner / squeeze / Bollinger use H/L envelopes of past
        N bars; the Range Filter uses recursive close vs. close-change EMA.
      - Parabolic SAR accelerates with each bar in trend; Range Filter
        does not — its lock-step is range-based, not time-based.

    The cross-up + rising-filter conjunction targets fresh expansion out
    of consolidation while the filter has just begun trending; the
    SMA(50)>SMA(200) gate keeps it in established uptrends and the EMA20
    exit (consistent with iter 11–13 strategies) cuts the trade once
    short-term momentum fails.
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    if n == 0:
        return []

    period = 14
    mult = 2.618

    diff = np.zeros(n, dtype=float)
    diff[1:] = np.abs(close[1:] - close[:-1])
    avg_range = _ema(diff, period)
    smooth_range = _ema(avg_range, period * 2 - 1) * mult

    rf = np.full(n, np.nan, dtype=float)
    start = 0
    while start < n and not np.isfinite(smooth_range[start]):
        start += 1
    if start >= n:
        return []
    rf[start] = float(close[start])
    for i in range(start + 1, n):
        sr = smooth_range[i]
        prev_rf = rf[i - 1]
        if not np.isfinite(sr) or not np.isfinite(prev_rf):
            rf[i] = prev_rf
            continue
        c = close[i]
        if c > prev_rf:
            rf[i] = max(prev_rf, c - sr)
        elif c < prev_rf:
            rf[i] = min(prev_rf, c + sr)
        else:
            rf[i] = prev_rf

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    rf_prev = np.concatenate(([np.nan], rf[:-1]))
    rf_prev2 = np.concatenate(([np.nan, np.nan], rf[:-2]))
    close_prev = np.concatenate(([np.nan], close[:-1]))

    cross_up = (
        np.isfinite(rf_prev)
        & np.isfinite(rf_prev2)
        & np.isfinite(close_prev)
        & (close_prev <= rf_prev)
        & (close > rf)
        & (rf >= rf_prev)
        & (rf_prev >= rf_prev2)
    )
    uptrend = (
        np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)
    )

    entries = cross_up & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_rsi_brown_range_shift(df: pd.DataFrame) -> list[Trade]:
    """Constance Brown's RSI range-shift trend trigger.

    Brown's observation: in uptrends RSI(14) tends to oscillate roughly within
    40-80, while in downtrends within 20-60. A move from the lower (downtrend)
    range up into the upper (uptrend) range signals a regime change to bullish.

    Trigger: RSI(14) was <=35 within the last 20 bars (i.e. recently visited
    downtrend territory) AND now crosses up through 60 (entering uptrend
    range) AND price > SMA(50) (broad uptrend gate).
    Exit: bar close drops below EMA(20).

    Distinct from the existing rsi_ema (RSI vs EMA cross), stoch_rsi cross
    (different oscillator), connors_rsi (3-component composite), and inverse
    fisher rsi (transform-based) strategies — this one keys off Brown's
    *range-shift* concept rather than overbought/oversold thresholds.
    """
    close = df["close"].to_numpy(dtype=float)
    rsi14 = _rsi(close, 14)
    sma50 = _sma(close, 50)
    ema20 = _ema(close, 20)

    rsi_series = pd.Series(rsi14)
    # Was RSI <=35 within the prior 20 bars (exclude current bar via shift).
    was_oversold = (
        rsi_series.shift(1).rolling(20, min_periods=5).min().to_numpy() <= 35.0
    )

    rsi_prev = np.concatenate(([np.nan], rsi14[:-1]))
    cross_up_60 = (
        np.isfinite(rsi_prev) & (rsi_prev <= 60.0) & (rsi14 > 60.0)
    )
    uptrend = np.isfinite(sma50) & (close > sma50)

    entries = cross_up_60 & was_oversold & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_vidya_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """Chande VIDYA (Variable Index Dynamic Average) bullish cross.

    VIDYA is an adaptive EMA where the smoothing constant is scaled by
    |CMO|/100 — the Chande Momentum Oscillator's absolute value. When price
    is moving strongly (high |CMO|), VIDYA tracks closely; when price chops
    (|CMO| near 0), VIDYA effectively freezes. This adaptive frozen-in-chop
    behavior is distinct from KAMA (efficiency-ratio based) and Hull MA
    (weighted-length based).

    Entry: prior bar's close was at or below VIDYA(14, cmo=9) and current
    close crosses above it, VIDYA itself is rising vs 1 bar ago, and the
    SMA50 > SMA200 long-term uptrend regime holds.
    Exit: close < EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    cmo_period = 9
    vidya_period = 14

    # Chande Momentum Oscillator on close diffs.
    diff = np.diff(close, prepend=close[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    sum_up = pd.Series(up).rolling(cmo_period, min_periods=cmo_period).sum().to_numpy()
    sum_dn = pd.Series(dn).rolling(cmo_period, min_periods=cmo_period).sum().to_numpy()
    denom = sum_up + sum_dn
    cmo = np.where(denom > 0, (sum_up - sum_dn) / denom, 0.0)
    abs_cmo = np.abs(cmo)

    alpha = 2.0 / (vidya_period + 1)
    vidya = np.full(n, np.nan)
    seeded = False
    for i in range(n):
        if not np.isfinite(abs_cmo[i]):
            continue
        if not seeded:
            vidya[i] = close[i]
            seeded = True
            continue
        k = alpha * abs_cmo[i]
        vidya[i] = k * close[i] + (1.0 - k) * vidya[i - 1]

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    close_prev = np.concatenate(([np.nan], close[:-1]))
    vidya_prev = np.concatenate(([np.nan], vidya[:-1]))
    vidya_prev2 = np.concatenate(([np.nan, np.nan], vidya[:-2]))

    cross_up = (
        np.isfinite(vidya)
        & np.isfinite(vidya_prev)
        & (close_prev <= vidya_prev)
        & (close > vidya)
    )
    rising = np.isfinite(vidya_prev2) & (vidya > vidya_prev2)
    uptrend = np.isfinite(sma200) & (sma50 > sma200)

    entries = cross_up & rising & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_acceleration_bands_breakout(df: pd.DataFrame) -> list[Trade]:
    """Price Headley Acceleration Bands upside breakout.

    Headley's bands scale around each bar by (H-L)/(H+L) — a normalized,
    *price-relative* range factor — rather than ATR (Keltner) or σ
    (Bollinger). They expand sharply on wide-range bars and contract on
    inside bars, giving a fundamentally different envelope shape than
    other volatility bands already in the sandbox.

    Upper raw  = high * (1 + 4 * (high - low) / (high + low))
    Upper band = SMA20(upper raw)

    Entry: today's close crosses above the prior 20-bar SMA of the upper
    raw band, today's close > SMA200, and prior close was at/below band.
    Exit : close < EMA(20).

    Uses prior-bar values for the cross test so decision is bar-close
    safe and free of lookahead. Distinct from:
      - keltner_channel_breakout (ATR-scaled),
      - bollinger / pctb (stdev-scaled),
      - donchian_20_10_trend (raw high channel, no scaling).
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    hl_sum = high + low
    factor = np.where(hl_sum > 0, (high - low) / hl_sum, 0.0)
    upper_raw = high * (1.0 + 4.0 * factor)

    upper_band = (
        pd.Series(upper_raw).rolling(20, min_periods=20).mean().to_numpy()
    )
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    close_prev = np.concatenate(([np.nan], close[:-1]))
    upper_prev = np.concatenate(([np.nan], upper_band[:-1]))

    crossed_up = (
        np.isfinite(upper_band)
        & np.isfinite(upper_prev)
        & (close > upper_band)
        & (close_prev <= upper_prev)
    )
    regime = np.isfinite(sma200) & (close > sma200)

    entries = crossed_up & regime
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_qqe_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """QQE (Quantitative Qualitative Estimation) bullish trailing-line cross.

    QQE smooths RSI(14) with EMA(5) into RsiMa, then builds a volatility-
    adaptive trailing band from a doubly Wilder-smoothed |ΔRsiMa| (×4.236).
    The trailing line ratchets monotonically up under RsiMa while RsiMa
    holds above it, and resets to RsiMa+DAR/RsiMa-DAR on regime flips.
    A bullish QQE cross fires when RsiMa crosses up through the trailing
    line — a *smoothed* RSI breakout with adaptive volatility cushion.

    Distinct from:
      - rsi_ema / rsi_brown_range_shift (raw RSI, no volatility band),
      - inverse_fisher_rsi (Fisher transform of RSI, no trailing line),
      - schaff_trend_cycle (double-smoothed stochastic, not RSI),
      - tsi_signal_cross (true-strength index of momentum, not RSI ATR).

    Entry: prior bar RsiMa was at or below trailing line, current RsiMa
    crosses above it, and SMA50>SMA200 long-term uptrend gates direction.
    Exit: close < EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    rsi_period = 14
    smoothing = 5
    wilder_period = 27
    qqe_factor = 4.236

    rsi = _rsi(close, rsi_period)
    rsi_ma = _ema(rsi, smoothing)

    rsi_ma_prev = np.concatenate(([np.nan], rsi_ma[:-1]))
    delta = np.abs(rsi_ma - rsi_ma_prev)
    delta_safe = np.where(np.isfinite(delta), delta, 0.0)

    atr_rsi = _rma(delta_safe, wilder_period)
    dar = _rma(atr_rsi, wilder_period) * qqe_factor

    newlong = rsi_ma - dar
    newshort = rsi_ma + dar

    tr_level = np.full(n, np.nan)
    seeded = False
    for i in range(n):
        if not (np.isfinite(rsi_ma[i]) and np.isfinite(dar[i])):
            continue
        if not seeded:
            tr_level[i] = newlong[i]
            seeded = True
            continue
        prev = tr_level[i - 1]
        if not np.isfinite(prev):
            tr_level[i] = newlong[i]
            continue
        prev_rsi = rsi_ma[i - 1]
        cur_rsi = rsi_ma[i]
        if np.isfinite(prev_rsi) and prev_rsi > prev and cur_rsi > prev:
            tr_level[i] = max(prev, newlong[i])
        elif np.isfinite(prev_rsi) and prev_rsi < prev and cur_rsi < prev:
            tr_level[i] = min(prev, newshort[i])
        elif cur_rsi > prev:
            tr_level[i] = newlong[i]
        else:
            tr_level[i] = newshort[i]

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    tr_prev = np.concatenate(([np.nan], tr_level[:-1]))

    cross_up = (
        np.isfinite(rsi_ma)
        & np.isfinite(rsi_ma_prev)
        & np.isfinite(tr_level)
        & np.isfinite(tr_prev)
        & (rsi_ma_prev <= tr_prev)
        & (rsi_ma > tr_level)
    )
    uptrend = np.isfinite(sma200) & (sma50 > sma200)

    entries = cross_up & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_elder_ray_bear_reclaim(df: pd.DataFrame) -> list[Trade]:
    """Elder Ray dip-buy — Bear Power turning up from below zero while EMA13 rises.

    Dr. Alexander Elder's Bull/Bear Power: BullPower = high - EMA(13),
    BearPower = low - EMA(13). The classic long setup: a rising EMA(13)
    confirms uptrend, Bear Power negative (dip in progress), but Bear Power
    higher than the prior bar (selling pressure waning) — i.e. the dip is
    being absorbed. Entry adds a long-term trend gate (close > SMA(50)
    > SMA(200)) so we only buy bears in established uptrends, and a
    Bull Power > 0 filter so buyers are still in control. Exit: close
    falls below EMA(13). Distinct from RSI/Stoch/MACD/Williams setups in
    the journal because the signal is two-sided power decomposition off
    a 13-bar EMA, not an oscillator threshold or signal-line cross.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    ema13 = _ema(close, 13)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)

    bull_power = high - ema13
    bear_power = low - ema13

    bear_prev = np.concatenate(([np.nan], bear_power[:-1]))
    ema13_prev = np.concatenate(([np.nan], ema13[:-1]))

    valid = (
        np.isfinite(ema13)
        & np.isfinite(ema13_prev)
        & np.isfinite(sma50)
        & np.isfinite(sma200)
        & np.isfinite(bear_prev)
    )

    rising_ema = ema13 > ema13_prev
    uptrend = (close > sma50) & (sma50 > sma200)
    bear_negative_rising = (bear_power < 0.0) & (bear_power > bear_prev)
    bull_positive = bull_power > 0.0

    entries = (
        valid
        & rising_ema
        & uptrend
        & bear_negative_rising
        & bull_positive
    )
    exits = np.isfinite(ema13) & (close < ema13)

    return _walk(entries, exits, close, df["date"].values)


def strat_morning_star_pullback(df: pd.DataFrame) -> list[Trade]:
    """Morning Star — classical 3-bar bullish reversal candle pattern after a
    short-term pullback inside an SMA50>SMA200 uptrend.

    The pattern (Nison) is a *reversal* signature, not a continuation:
      Bar t-2 ("body 1"): a decisive bearish candle —
            close[t-2] < open[t-2], (open[t-2]-close[t-2])/range[t-2] >= 0.55.
      Bar t-1 ("star"):   a small-bodied indecision bar that gaps/opens at or
            below the prior close —
            |close[t-1]-open[t-1]| / range[t-1] <= 0.30
            AND max(open[t-1], close[t-1]) <= close[t-2]   (body sits at or
                                                            below the bear's
                                                            close).
      Bar t   ("body 3"): a decisive bullish candle that *reclaims* well into
            body 1 —
            close[t] > open[t], (close[t]-open[t])/range[t] >= 0.55,
            close[t] > (open[t-2] + close[t-2]) / 2  (above bar1 midpoint).

    Why this is distinct from existing sandbox candle/structure plays:
      - hammer_pin_bar_uptrend: a *single* pin-bar with long lower wick at a
        20-bar swing low — single-bar anatomy, no preceding bear+star sequence.
      - three_white_soldiers: three *consecutive bullish* candles closing
        higher — a continuation sequence, not a bear→indecision→bull flip.
      - bullish_engulfing_pullback: a 2-bar engulfing — body 2 fully covers
        body 1 with no indecision-star bar in the middle.
      - heikin_ashi_flip / td_sequential_buy_setup: smoothed colour flips and
        9-count duration rules — no body anatomy, no gap-down star middle.
      - wyckoff_spring_reclaim: swing-low spike-and-reclaim across days, not
        a 3-bar candle anatomy with a small middle body.

    Regime: SMA50 > SMA200 (the bear+star+bull anatomy marks the resumption
    of an uptrend after a pullback). Entry decision uses bar t close (no
    lookahead — open/high/low/close of bar t are all known by close).

    Exit: close < EMA20 — the resumption thesis times out cleanly when the
    bounce loses the short-term mean.
    """
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    if n < 3:
        return []

    rng = high - low
    safe_rng = np.where(rng > 0.0, rng, np.nan)
    body = close - open_
    body_pct = np.abs(body) / safe_rng

    open_1 = np.concatenate(([np.nan], open_[:-1]))
    close_1 = np.concatenate(([np.nan], close[:-1]))
    body_pct_1 = np.concatenate(([np.nan], body_pct[:-1]))

    open_2 = np.concatenate(([np.nan, np.nan], open_[:-2]))
    close_2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    body_pct_2 = np.concatenate(([np.nan, np.nan], body_pct[:-2]))

    bear_body1 = (
        np.isfinite(close_2)
        & np.isfinite(open_2)
        & (close_2 < open_2)
        & np.isfinite(body_pct_2)
        & (body_pct_2 >= 0.55)
    )

    star_max = np.maximum(open_1, close_1)
    star_body = (
        np.isfinite(body_pct_1)
        & (body_pct_1 <= 0.30)
        & np.isfinite(star_max)
        & np.isfinite(close_2)
        & (star_max <= close_2)
    )

    bull_body3 = (
        np.isfinite(close)
        & np.isfinite(open_)
        & (close > open_)
        & np.isfinite(body_pct)
        & (body_pct >= 0.55)
    )

    body1_mid = (open_2 + close_2) / 2.0
    reclaim_mid = np.isfinite(body1_mid) & (close > body1_mid)

    pattern = bear_body1 & star_body & bull_body3 & reclaim_mid

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = pattern & uptrend

    ema20 = _ema(close, 20)
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_polarized_fractal_efficiency(df: pd.DataFrame) -> list[Trade]:
    """Polarized Fractal Efficiency (Hannula, 1994) bullish zero-line cross.

    PFE measures how *efficient* price travel is over an n-bar window by
    comparing the straight-line distance between close[t] and close[t-n] to
    the sum of bar-to-bar diagonal path lengths. A perfectly straight-up
    move produces PFE → +100; choppy noise drives |PFE| → 0.

      diff_n      = close[t] - close[t-n]
      straight    = sqrt(diff_n^2 + n^2)             # diagonal of net move
      bar_step    = sqrt((close[i]-close[i-1])^2 + 1) # per-bar diagonal
      path_sum    = Σ bar_step over the last n bars
      raw_PFE     = 100 * sign(diff_n) * straight / path_sum
      PFE         = EMA(raw_PFE, 5)

    Why this is distinct from existing sandbox indicators:
      - choppiness_regime_shift: range-vs-ATR entropy gauge, *unsigned*
        (no direction). PFE is signed — it flips polarity at zero.
      - linreg_slope_signchange: signed regression slope; PFE measures
        path efficiency, not slope.
      - aroon_cross_trend / vortex_bullish_cross: time-since-extreme and
        directional movement crosses, not bar-to-bar fractal path length.
      - fisher_transform_zero_cross / coppock_curve / kst: oscillator
        cross-zeros derived from price levels, not geometric path length.
      - inverse_fisher_rsi: a Fisher transform of RSI — momentum, not
        fractal efficiency.

    Setup: PFE crosses up through zero (signed efficiency flips bullish)
    inside an SMA50 > SMA200 uptrend. Exit on close < EMA20 — the same
    short-term mean exit used by other zero-cross momentum strategies in
    this sandbox so the comparison isolates the *signal*, not the exit.
    """
    close = df["close"].to_numpy(dtype=float)
    n_window = 10
    n = len(close)
    if n < n_window + 5:
        return []

    diff_n = np.full(n, np.nan)
    diff_n[n_window:] = close[n_window:] - close[:-n_window]
    straight = np.sqrt(diff_n * diff_n + float(n_window) ** 2)

    bar_step = np.full(n, np.nan)
    bar_step[1:] = np.sqrt((close[1:] - close[:-1]) ** 2 + 1.0)
    path_sum = (
        pd.Series(bar_step).rolling(n_window, min_periods=n_window).sum().to_numpy()
    )

    sign_n = np.sign(diff_n)
    raw_pfe = np.where(
        np.isfinite(path_sum) & (path_sum > 0.0) & np.isfinite(straight),
        100.0 * sign_n * straight / path_sum,
        np.nan,
    )

    pfe = _ema(np.nan_to_num(raw_pfe, nan=0.0), 5)
    pfe[: n_window + 4] = np.nan
    pfe_prev = np.concatenate(([np.nan], pfe[:-1]))

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    cross_up = (
        np.isfinite(pfe)
        & np.isfinite(pfe_prev)
        & (pfe_prev <= 0.0)
        & (pfe > 0.0)
    )

    entries = cross_up & uptrend

    ema20 = _ema(close, 20)
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_wavetrend_lb_oversold_cross(df: pd.DataFrame) -> list[Trade]:
    """LazyBear WaveTrend Oscillator (WT) — bullish crossover from the
    oversold zone, gated by SMA(50) > SMA(200) trend filter.

    The WaveTrend Oscillator (popularised by "LazyBear" on TradingView, but
    derived from older centred-momentum work) measures how far the *typical
    price* (HLC3) sits from its own EMA, normalised by an EMA of the
    *absolute* deviation — i.e. a self-scaled momentum z-score. Computation:

        ap   = (high + low + close) / 3              # average / typical price
        esa  = EMA(ap, n1=10)                        # smoothed centre line
        d    = EMA(|ap - esa|, n1=10)                # smoothed mean abs dev
        ci   = (ap - esa) / (0.015 * d)              # channel index, ~CCI core
        wt1  = EMA(ci, n2=21)                        # the WaveTrend line
        wt2  = SMA(wt1, 4)                           # signal line

    Standard LazyBear setup: a long signal fires when wt1 crosses *up*
    through wt2 while wt1 is sitting in the oversold zone (< -60).

    Why this is genuinely distinct from prior sandbox indicators:
      - MACD / TSI / KST / Coppock / TRIX / DPO: all derive from *price* or
        *price differences*, not from the absolute-deviation normalisation
        WT applies. WT's denominator (EMA of |ap-esa|) makes it scale-free
        in a way EMA-difference oscillators are not.
      - CCI (cci_oversold_recovery): single-pass typical-price deviation
        normalised by mean-deviation, *not* double-EMA-smoothed and without
        a separate signal-line cross.
      - StochRSI / Stoch / Ult Osc / Inverse Fisher RSI / Fisher Transform:
        all built on RSI or stochastic of close, not HLC3 deviation.
      - QQE / RVI / Schaff Trend Cycle: smoothed-RSI or smoothed-stoch
        machinery, again close-based, not deviation-z-scored HLC3.
      - Awesome Oscillator (saucer): SMA(median, 5) - SMA(median, 34) — no
        normalisation by absolute deviation.

    Hypothesis: a fresh wt1↑wt2 cross while wt1 is still below -60 marks the
    moment a typical-price pullback has *just* exhausted, and the trend
    filter (50>200) ensures we are dip-buying inside the prevailing uptrend
    rather than catching a downtrend bounce.

    Exit: close < EMA(20) — same short-term mean give-back used across the
    other oscillator-cross sandbox plays so the *signal* is the only
    variable.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    ap = (high + low + close) / 3.0
    n1 = 10
    n2 = 21

    esa = _ema(ap, n1)
    abs_dev = np.abs(ap - esa)
    d = _ema(abs_dev, n1)

    denom = 0.015 * d
    ci = np.where(np.isfinite(denom) & (denom > 0.0), (ap - esa) / denom, np.nan)

    wt1 = _ema(np.nan_to_num(ci, nan=0.0), n2)
    # mask early bars where the inputs were not yet defined.
    warmup = n1 + n2
    wt1[:warmup] = np.nan
    wt2 = _sma(wt1, 4)

    wt1_prev = np.concatenate(([np.nan], wt1[:-1]))
    wt2_prev = np.concatenate(([np.nan], wt2[:-1]))

    cross_up = (
        np.isfinite(wt1)
        & np.isfinite(wt2)
        & np.isfinite(wt1_prev)
        & np.isfinite(wt2_prev)
        & (wt1_prev <= wt2_prev)
        & (wt1 > wt2)
    )
    oversold = np.isfinite(wt1) & (wt1 < -60.0)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    uptrend = np.isfinite(sma50) & np.isfinite(sma200) & (sma50 > sma200)

    entries = cross_up & oversold & uptrend

    ema20 = _ema(close, 20)
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_alma_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """Arnaud Legoux Moving Average (ALMA) bullish cross in SMA50>SMA200 uptrend.

    ALMA applies a Gaussian-weighted kernel that is offset toward the most
    recent bar so it tracks turns faster than EMA/SMA while still suppressing
    high-frequency noise more aggressively than a WMA.

        m = floor(offset * (N - 1));   s = N / sigma
        w[i] = exp(-((i - m)^2) / (2 * s^2)),    i = 0..N-1
        ALMA[t] = sum_i w[i] * close[t - N + 1 + i] / sum_i w[i]

    The Gaussian peak shifted toward the right edge (offset≈0.85) gives a
    response curve unlike the linear-WMA basis of HMA, the efficiency-ratio
    basis of KAMA, or the CMO-volatility basis of VIDYA — all of which already
    appear in the sandbox.

    Entry (decided at bar close, prior-bar values to avoid lookahead):
      - fresh bullish ALMA cross of close: close_{t-2} <= ALMA_{t-2} AND
        close_{t-1} > ALMA_{t-1}
      - SMA(50)_{t-1} > SMA(200)_{t-1} (macro uptrend gate)
    Exit: close < EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)

    def _alma(arr: np.ndarray, n: int, offset: float, sigma: float) -> np.ndarray:
        m = int(np.floor(offset * (n - 1)))
        s = n / float(sigma)
        i = np.arange(n, dtype=float)
        w = np.exp(-((i - m) ** 2) / (2.0 * s * s))
        wsum = w.sum()
        return (
            pd.Series(arr)
            .rolling(n, min_periods=n)
            .apply(lambda x: np.dot(x, w) / wsum, raw=True)
            .to_numpy()
        )

    alma = _alma(close, 21, 0.85, 6.0)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    close_prev = np.concatenate(([np.nan], close[:-1]))
    close_prev2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    alma_prev = np.concatenate(([np.nan], alma[:-1]))
    alma_prev2 = np.concatenate(([np.nan, np.nan], alma[:-2]))
    sma50_prev = np.concatenate(([np.nan], sma50[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(alma_prev)
        & np.isfinite(alma_prev2)
        & np.isfinite(close_prev)
        & np.isfinite(close_prev2)
        & np.isfinite(sma50_prev)
        & np.isfinite(sma200_prev)
    )
    fresh_cross = (close_prev2 <= alma_prev2) & (close_prev > alma_prev)
    uptrend = sma50_prev > sma200_prev

    entries = valid & fresh_cross & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_pretty_good_oscillator_zero_cross(df: pd.DataFrame) -> list[Trade]:
    """Mark Johnson's Pretty Good Oscillator (PGO) — fresh bullish zero cross.

    PGO normalizes the close's distance from its moving-average basis by an
    EMA of True Range, expressing today's deviation in ATR-style units. From
    Mark Johnson's TASC 1995 piece:

        PGO[t] = (close[t] - SMA(close, n)[t]) / EMA(TR, n)[t]

    A fresh upward zero cross signals that the close has just reclaimed its
    n-bar mean, scaled to the prevailing volatility regime so the threshold
    is meaningful across different ATR environments. Distinct from:
      - donchian / supertrend / keltner: those use price-channel crossovers,
        not a centered oscillator around a moving-average mean.
      - bollinger_pctb_reversion: %B uses stdev for width, PGO uses TR.
      - vwap_zscore_reversion: VWAP is volume-weighted intraday-anchored,
        PGO is plain SMA.
      - linreg_slope_signchange: PGO tracks deviation from a flat mean,
        linreg tracks the slope itself.

    Entry (decided at bar close, prior-bar values to avoid lookahead):
      - PGO_{t-2} <= 0 AND PGO_{t-1} > 0 (fresh upward zero cross)
      - SMA(50)_{t-1} > SMA(200)_{t-1} (macro uptrend gate)
    Exit: close < EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    n = 30
    sma_n = _sma(close, n)

    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr_candidates = np.stack(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ],
        axis=0,
    )
    # First bar has NaN prev_close, so fall back to high - low there.
    tr = np.where(
        np.isfinite(prev_close),
        np.nanmax(tr_candidates, axis=0),
        high - low,
    )

    ema_tr = _ema(tr, n)
    pgo = np.where(
        np.isfinite(ema_tr) & (ema_tr > 0),
        (close - sma_n) / np.where(ema_tr > 0, ema_tr, np.nan),
        np.nan,
    )

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    pgo_prev = np.concatenate(([np.nan], pgo[:-1]))
    pgo_prev2 = np.concatenate(([np.nan, np.nan], pgo[:-2]))
    sma50_prev = np.concatenate(([np.nan], sma50[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(pgo_prev)
        & np.isfinite(pgo_prev2)
        & np.isfinite(sma50_prev)
        & np.isfinite(sma200_prev)
    )
    fresh_cross = (pgo_prev2 <= 0.0) & (pgo_prev > 0.0)
    uptrend = sma50_prev > sma200_prev

    entries = valid & fresh_cross & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_ehlers_cog_signal_cross(df: pd.DataFrame) -> list[Trade]:
    """John Ehlers' Center of Gravity (COG) oscillator — bullish signal cross.

    From Ehlers' 2002 article "The Center of Gravity Oscillator." The COG is a
    near-zero-lag oscillator built from a weighted sum of the last N closes:

        COG[t] = - sum_{i=0..N-1} ((i+1) * close[t-i])
                 / sum_{i=0..N-1}        close[t-i]

    The negation flips orientation so rising COG corresponds to rising price.
    Because the heaviest weight sits on the *oldest* bar in the window, the
    oscillator turns over with very little lag relative to a same-length SMA,
    making the COG-vs-its-own-3-bar-SMA crossover a low-lag momentum signal.

    Distinct from indicators already in this file:
      - rsi / stoch_rsi / cmo / cci: ratio-of-gains oscillators with their
        own bounded ranges; COG is a weighted-mean centroid, not a momentum
        ratio.
      - fisher_transform / inverse_fisher_rsi: Gaussian-mapped oscillators on
        normalized price; COG uses raw weighted sums, no transform.
      - polarized_fractal_efficiency / linreg_slope_signchange: measure path
        efficiency / regression slope, not a centroid.
      - pretty_good_oscillator: PGO is a price-deviation-in-ATR-units; COG
        is a dimensionless weighted-mean position indicator.
      - kama / vidya / alma / hma: those are adaptive/weighted *moving
        averages*; COG is an *oscillator* derived from a weighted centroid.

    Entry (decided at bar close, prior-bar values to avoid lookahead):
      - COG_{t-2} <= signal_{t-2} AND COG_{t-1} > signal_{t-1}
        (fresh upward signal-line cross, signal = SMA(COG, 3))
      - SMA(50)_{t-1} > SMA(200)_{t-1} (macro uptrend gate)
    Exit: close < EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    n = 10

    s = pd.Series(close)
    num = sum((i + 1) * s.shift(i) for i in range(n))
    den = sum(s.shift(i) for i in range(n))
    cog = -(num / den.replace(0.0, np.nan)).to_numpy(dtype=float)

    sig = pd.Series(cog).rolling(3, min_periods=3).mean().to_numpy()

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    cog_prev = np.concatenate(([np.nan], cog[:-1]))
    cog_prev2 = np.concatenate(([np.nan, np.nan], cog[:-2]))
    sig_prev = np.concatenate(([np.nan], sig[:-1]))
    sig_prev2 = np.concatenate(([np.nan, np.nan], sig[:-2]))
    sma50_prev = np.concatenate(([np.nan], sma50[:-1]))
    sma200_prev = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(cog_prev)
        & np.isfinite(cog_prev2)
        & np.isfinite(sig_prev)
        & np.isfinite(sig_prev2)
        & np.isfinite(sma50_prev)
        & np.isfinite(sma200_prev)
    )
    fresh_cross = (cog_prev2 <= sig_prev2) & (cog_prev > sig_prev)
    uptrend = sma50_prev > sma200_prev

    entries = valid & fresh_cross & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_smi_blau_oversold_cross(df: pd.DataFrame) -> list[Trade]:
    """William Blau's Stochastic Momentum Index (SMI) — oversold signal cross.

    Blau (1993, "Momentum, Direction, and Divergence") centered the classic
    Stochastic so it ranges -100..+100 instead of 0..100. Where %K measures
    today's close as a fraction of the trailing high-low range, SMI measures
    today's close *relative to the midpoint* of that range and double-smooths
    both numerator and denominator with nested EMAs:

        HH       = rolling max(high, N)
        LL       = rolling min(low,  N)
        midpt    = (HH + LL) / 2
        D        = close - midpt           (signed distance from mid)
        HLR      = HH - LL                 (range)
        SMI      = 100 * EMA(EMA(D,   q), q) / (0.5 * EMA(EMA(HLR, q), q))
        signal   = EMA(SMI, m)

    With N=10, q=3, m=3 (Blau's classic settings) the indicator turns over
    several bars before %K and %D, and oscillates symmetrically around zero.

    Entry (prior-bar values only, no lookahead):
      - SMI_{t-2} <= signal_{t-2} AND SMI_{t-1} > signal_{t-1}
        (fresh upward signal-line cross)
      - SMI_{t-1} < -40 (signal originates from oversold territory, the
        regime where the cross has the strongest forward edge per Blau)
      - SMA(50)_{t-1} > SMA(200)_{t-1} (macro uptrend filter)
    Exit: close < EMA(20).

    Distinct from every existing sandbox indicator:
      - stochastic_oversold_recovery: raw %K crossing 20 from below — single
        ratio of close-to-range, no double smoothing, no centering.
      - stoch_rsi_oversold_cross: stochastic of RSI, not of price midpoint.
      - tsi_signal_cross: double-EMA of Δclose / |Δclose| — momentum-based,
        SMI is *position-in-range* based.
      - schaff_trend_cycle: double stochastic of MACD — uses MACD as input
        and is bounded 0..100; SMI uses raw range geometry, ±100.
      - fisher / inverse_fisher_rsi: Gaussian transforms of normalized price.
      - awesome_oscillator_saucer: SMA(5)-SMA(34) of median, not centered
        in a high-low range and not double-smoothed.
    SMI is uniquely the *signed, double-smoothed midpoint distance* in the
    pool — no other strategy combines (a) range-centered geometry, (b) a
    nested EMA on both numerator and denominator, and (c) ±100 bounds.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    n = 10
    q = 3
    m = 3

    h_s = pd.Series(high)
    l_s = pd.Series(low)
    hh = h_s.rolling(n, min_periods=n).max().to_numpy()
    ll = l_s.rolling(n, min_periods=n).min().to_numpy()
    midpt = (hh + ll) / 2.0
    d = close - midpt
    hlr = hh - ll

    d_f = np.nan_to_num(d, nan=0.0)
    hlr_f = np.nan_to_num(hlr, nan=0.0)

    ema1_d = _ema(d_f, q)
    ema2_d = _ema(ema1_d, q)
    ema1_r = _ema(hlr_f, q)
    ema2_r = _ema(ema1_r, q)

    half_r = 0.5 * ema2_r
    with np.errstate(divide="ignore", invalid="ignore"):
        smi = 100.0 * np.where(half_r > 0, ema2_d / half_r, 0.0)
    # Mask the bars before the rolling window is fully formed so EMA warmup
    # noise never produces a stale cross.
    warmup_mask = np.isfinite(hh) & np.isfinite(ll)
    smi = np.where(warmup_mask, smi, np.nan)
    signal = _ema(np.nan_to_num(smi, nan=0.0), m)
    signal = np.where(warmup_mask, signal, np.nan)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    smi_p1 = np.concatenate(([np.nan], smi[:-1]))
    smi_p2 = np.concatenate(([np.nan], smi_p1[:-1]))
    sig_p1 = np.concatenate(([np.nan], signal[:-1]))
    sig_p2 = np.concatenate(([np.nan], sig_p1[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(smi_p1) & np.isfinite(smi_p2)
        & np.isfinite(sig_p1) & np.isfinite(sig_p2)
        & np.isfinite(sma50_p1) & np.isfinite(sma200_p1)
    )
    fresh_cross = (smi_p2 <= sig_p2) & (smi_p1 > sig_p1)
    oversold = smi_p1 < -40.0
    uptrend = sma50_p1 > sma200_p1

    entries = valid & fresh_cross & oversold & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_gann_hilo_activator_flip(df: pd.DataFrame) -> list[Trade]:
    """Gann HiLo Activator (GHLA) — trend-state flip from down to up.

    The Gann HiLo Activator (popularised by Robert Krausz from Gann's swing
    methods) is a binary-state trend follower built from two simple moving
    averages: SMA(high, N) and SMA(low, N). It carries two pieces of state:

      - state ∈ {+1, -1}: +1 when close last broke ABOVE the prior bar's
        SMA(high, N); -1 when close last broke BELOW the prior bar's
        SMA(low, N). Otherwise the state is carried forward.
      - the active "activator" line: SMA(low, N) while state == +1,
        SMA(high, N) while state == -1.

    With N=10 the activator hugs price loosely from below in uptrends and
    from above in downtrends, only flipping when price decisively breaches
    the *opposite* SMA. The flip itself is the signal: the strategy enters
    when state changes from -1 to +1 — i.e. the close has just punched
    above the prior bar's SMA(high,10) after a stretch of being capped by
    the SMA(high) line in a down-state.

    Distinct from indicators already in the file:
      - aroon_cross_trend / vortex_bullish_cross: built from positions of
        rolling-window highs/lows but produce continuous oscillators; GHLA
        outputs a discrete +1/-1 state from comparison of close to *MAs of*
        highs/lows, not to the raw highs/lows themselves.
      - parabolic_sar_flip_trend: PSAR's stop accelerates with each new
        extreme; GHLA uses fixed-window SMAs of H and L, no acceleration.
      - donchian_20_10_trend / range_filter_buy: Donchian/Range Filter
        compare close to raw rolling highs/lows or a smoothed-deviation
        envelope; GHLA compares close to the *means* of recent highs/lows,
        which sits inside the raw range and reacts sooner.
      - supertrend / keltner / acceleration_bands: ATR- or stdev-scaled
        envelopes around price; GHLA has no volatility scaling.
      - ma_cross / hma / kama / vidya / alma: those compare close to a
        single MA of close; GHLA's two MAs are of high and of low (not of
        close), and the active line switches sides based on a state
        machine — that two-line, one-active hand-off is the unique part.
      - heikin_ashi_flip: HA flips on smoothed open/close colour change;
        GHLA flips on close vs SMA-of-extremes thresholds.

    Entry (prior-bar values only, no lookahead):
      - state_{t-2} == -1 AND state_{t-1} == +1 (fresh down→up flip)
      - SMA(50)_{t-1} > SMA(200)_{t-1} (macro uptrend filter)
    Exit: close < EMA(20).
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    n = 10
    high_sma = _sma(high, n)
    low_sma = _sma(low, n)

    high_sma_p1 = np.concatenate(([np.nan], high_sma[:-1]))
    low_sma_p1 = np.concatenate(([np.nan], low_sma[:-1]))

    state = np.zeros(len(close), dtype=np.int8)
    cur = 0
    for i in range(len(close)):
        hp = high_sma_p1[i]
        lp = low_sma_p1[i]
        if not (np.isfinite(hp) and np.isfinite(lp)):
            state[i] = 0
            continue
        if close[i] > hp:
            cur = 1
        elif close[i] < lp:
            cur = -1
        state[i] = cur

    state_p1 = np.concatenate(([0], state[:-1]))
    state_p2 = np.concatenate(([0], state_p1[:-1]))

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_flip = (state_p2 == -1) & (state_p1 == 1)
    uptrend = sma50_p1 > sma200_p1
    valid = np.isfinite(sma50_p1) & np.isfinite(sma200_p1)

    entries = valid & fresh_flip & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_random_walk_index_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """Random Walk Index (Poulos) — RWI_high crosses up through RWI_low in uptrend.

    RWI compares actual price movement to what a random walk would produce
    over the same window. RWI_high(n) = (high - low_{t-n}) / (ATR(n) * sqrt(n)),
    RWI_low(n) = (high_{t-n} - low) / (ATR(n) * sqrt(n)). Values >1.0 mean
    movement exceeds the noise band. Entry on a fresh RWI_high > RWI_low cross
    with RWI_high > 1.0 inside SMA50>SMA200; exit close<EMA20. Distinct from
    ADX/DMI (Wilder smoothing of directional movement) and Aroon (time-since
    extremes) — RWI is a per-bar trend-vs-noise ratio.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    n = 8
    atr_n = _atr(high, low, close, n)
    sqrt_n = float(np.sqrt(n))

    high_n = pd.Series(high).shift(n).to_numpy()
    low_n = pd.Series(low).shift(n).to_numpy()

    denom = atr_n * sqrt_n
    safe = denom > 0
    rwi_up = np.where(safe, (high - low_n) / np.where(safe, denom, 1.0), 0.0)
    rwi_dn = np.where(safe, (high_n - low) / np.where(safe, denom, 1.0), 0.0)
    rwi_up = np.nan_to_num(rwi_up, nan=0.0, posinf=0.0, neginf=0.0)
    rwi_dn = np.nan_to_num(rwi_dn, nan=0.0, posinf=0.0, neginf=0.0)

    rwi_up_p1 = np.concatenate(([0.0], rwi_up[:-1]))
    rwi_dn_p1 = np.concatenate(([0.0], rwi_dn[:-1]))
    rwi_up_p2 = np.concatenate(([0.0], rwi_up_p1[:-1]))
    rwi_dn_p2 = np.concatenate(([0.0], rwi_dn_p1[:-1]))

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_cross = (rwi_up_p2 <= rwi_dn_p2) & (rwi_up_p1 > rwi_dn_p1)
    strong_up = rwi_up_p1 > 1.0
    uptrend = sma50_p1 > sma200_p1
    valid = np.isfinite(sma50_p1) & np.isfinite(sma200_p1)

    entries = valid & fresh_cross & strong_up & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_premier_stochastic_oscillator(df: pd.DataFrame) -> list[Trade]:
    """Lee Leibfarth's Premier Stochastic Oscillator (PSO).

    PSO normalizes Stochastic %K to NSK = 0.1*(%K-50), double-EMA-smooths it,
    then applies a Fisher transform: PSO = (e^SS - 1)/(e^SS + 1), producing a
    bounded oscillator in (-1,+1) with reduced lag and clean cross signals.
    Distinct from Stoch / StochRSI (raw bounded values), Fisher Transform of
    price (uses normalized price, not stoch), and Inverse Fisher RSI (operates
    on RSI). Entry: PSO crosses up through 0 inside SMA50>SMA200; exit
    close<EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    k_len = 8
    smooth_len = 5

    high_k = pd.Series(high).rolling(k_len, min_periods=k_len).max().to_numpy()
    low_k = pd.Series(low).rolling(k_len, min_periods=k_len).min().to_numpy()
    rng = high_k - low_k
    pct_k = np.where(rng > 0, (close - low_k) / np.where(rng > 0, rng, 1.0) * 100.0, 50.0)
    pct_k = np.nan_to_num(pct_k, nan=50.0, posinf=100.0, neginf=0.0)

    nsk = 0.1 * (pct_k - 50.0)
    ema1 = _ema(nsk, smooth_len)
    ema2 = _ema(ema1, smooth_len)
    ss = np.clip(ema2, -50.0, 50.0)
    expv = np.exp(ss)
    pso = (expv - 1.0) / (expv + 1.0)
    pso = np.nan_to_num(pso, nan=0.0, posinf=1.0, neginf=-1.0)

    pso_p1 = np.concatenate(([0.0], pso[:-1]))
    pso_p2 = np.concatenate(([0.0], pso_p1[:-1]))

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_cross = (pso_p2 <= 0.0) & (pso_p1 > 0.0)
    uptrend = sma50_p1 > sma200_p1
    valid = np.isfinite(sma50_p1) & np.isfinite(sma200_p1)

    entries = valid & fresh_cross & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_mcginley_dynamic_cross(df: pd.DataFrame) -> list[Trade]:
    """John R. McGinley's Dynamic — an adaptive smoother that self-adjusts to
    market velocity via a fourth-power ratio term.

    Recursion (N=14):
        MD_t = MD_{t-1} + (close_t - MD_{t-1}) / (N * (close_t / MD_{t-1})^4)

    The (close/MD)^4 factor accelerates the line in fast trends (price diverges
    above MD => ratio>1, denominator grows, BUT note: a larger denominator
    SLOWS adjustment, while a smaller denominator (ratio<1) speeds it). The
    asymmetric response cushions whipsaws while still tracking sustained moves.

    Distinct from every adaptive-MA already in the sandbox:
      - KAMA: efficiency-ratio (signal/noise) smoothing constant.
      - VIDYA: Chande-CMO-driven smoothing constant.
      - HMA: WMA-of-WMA cascade with √N final WMA, fixed window.
      - ALMA: Gaussian-weighted offset-MA.
    McGinley's geometry is the only one where the SC is a power-law function
    of price/MD ratio itself — no other entry uses this dynamic.

    Entry (prior-bar arrays, no lookahead):
      - close_{t-2} <= MD_{t-2} AND close_{t-1} > MD_{t-1}  (fresh bullish cross)
      - SMA50_{t-1} > SMA200_{t-1}  (macro uptrend gate)
    Exit: close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    n_bars = len(close)

    N = 14
    md = np.full(n_bars, np.nan)
    if n_bars >= N:
        seed_idx = N - 1
        md[seed_idx] = float(np.mean(close[:N]))
        for i in range(N, n_bars):
            prev = md[i - 1]
            if not np.isfinite(prev) or prev <= 0.0:
                md[i] = close[i]
                continue
            ratio = close[i] / prev
            # Clamp ratio to keep ratio**4 numerically stable on extreme bars.
            ratio = float(np.clip(ratio, 0.5, 2.0))
            md[i] = prev + (close[i] - prev) / (N * (ratio ** 4))

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    md_p1 = np.concatenate(([np.nan], md[:-1]))
    md_p2 = np.concatenate(([np.nan, np.nan], md[:-2]))
    close_p1 = np.concatenate(([np.nan], close[:-1]))
    close_p2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_cross = (close_p2 <= md_p2) & (close_p1 > md_p1)
    uptrend = sma50_p1 > sma200_p1
    valid = (
        np.isfinite(md_p1)
        & np.isfinite(md_p2)
        & np.isfinite(close_p1)
        & np.isfinite(close_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid & fresh_cross & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_frama_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """John Ehlers' Fractal Adaptive Moving Average (FRAMA, N=16) — bullish cross.

    Reference: Ehlers, "FRAMA — Fractal Adaptive Moving Average" (Stocks &
    Commodities, 2005). The smoothing constant adapts via the fractal dimension
    of the high-low range over the lookback window:

        Split the N-bar window into two halves of length N/2.
        N1 = (max(high[first half]) - min(low[first half])) / (N/2)
        N2 = (max(high[second half]) - min(low[second half])) / (N/2)
        N3 = (max(high[full N]) - min(low[full N])) / N
        D  = (log(N1 + N2) - log(N3)) / log(2)        # Hurst fractal dim
        alpha = exp(-4.6 * (D - 1))                    # clamped to [0.01, 1.0]
        FRAMA_t = alpha * close_t + (1 - alpha) * FRAMA_{t-1}

    When the per-bar range over the halves equals the per-bar range over the
    whole window the price moves cleanly (D~1 -> alpha~1, fast tracking). When
    the halves contain twice the per-bar range of the whole (zig-zag/noise),
    D~2 -> alpha~0.01 (heavy smoothing). This range-geometry adaptation is
    distinct from every other adaptive smoother already in the sandbox:
      - KAMA: Kaufman efficiency-ratio (close/abs-noise) drives SC.
      - VIDYA: Chande Momentum Oscillator drives SC.
      - McGinley: (close/MD)^4 power-law factor in denominator.
      - HMA / ALMA: fixed-weight WMA / Gaussian kernels (no adaptation).

    Entry (prior-bar arrays, no lookahead):
      - close_{t-2} <= FRAMA_{t-2} AND close_{t-1} > FRAMA_{t-1}  (fresh cross up)
      - SMA50_{t-1} > SMA200_{t-1}  (macro uptrend gate)
    Exit: close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n_bars = len(close)

    N = 16
    half = N // 2

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    hh_full = high_s.rolling(N, min_periods=N).max().to_numpy()
    ll_full = low_s.rolling(N, min_periods=N).min().to_numpy()
    hh_recent = high_s.rolling(half, min_periods=half).max().to_numpy()
    ll_recent = low_s.rolling(half, min_periods=half).min().to_numpy()
    hh_older = (
        high_s.rolling(half, min_periods=half).max().shift(half).to_numpy()
    )
    ll_older = (
        low_s.rolling(half, min_periods=half).min().shift(half).to_numpy()
    )

    frama = np.full(n_bars, np.nan)
    log2 = np.log(2.0)
    for i in range(n_bars):
        if i < N - 1:
            continue
        n1 = (hh_older[i] - ll_older[i]) / half
        n2 = (hh_recent[i] - ll_recent[i]) / half
        n3 = (hh_full[i] - ll_full[i]) / N
        if (
            not (np.isfinite(n1) and np.isfinite(n2) and np.isfinite(n3))
            or (n1 + n2) <= 0.0
            or n3 <= 0.0
        ):
            d = 1.0
        else:
            d = (np.log(n1 + n2) - np.log(n3)) / log2
        d = float(np.clip(d, 1.0, 2.0))
        alpha = float(np.exp(-4.6 * (d - 1.0)))
        alpha = float(np.clip(alpha, 0.01, 1.0))
        prev = frama[i - 1]
        if not np.isfinite(prev):
            prev = float(np.mean(close[i - N + 1 : i + 1]))
        frama[i] = alpha * close[i] + (1.0 - alpha) * prev

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    frama_p1 = np.concatenate(([np.nan], frama[:-1]))
    frama_p2 = np.concatenate(([np.nan, np.nan], frama[:-2]))
    close_p1 = np.concatenate(([np.nan], close[:-1]))
    close_p2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_cross = (close_p2 <= frama_p2) & (close_p1 > frama_p1)
    uptrend = sma50_p1 > sma200_p1
    valid = (
        np.isfinite(frama_p1)
        & np.isfinite(frama_p2)
        & np.isfinite(close_p1)
        & np.isfinite(close_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid & fresh_cross & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_macd_v_oversold_reclaim(df: pd.DataFrame) -> list[Trade]:
    """MACD-V (volatility-normalized MACD, Spiroglou 2022) — oversold-zone reclaim.

    Reference: Alex Spiroglou, "MACD-V: Volatility Normalised Momentum"
    (2022 NAAIM Wagner Award winner). The core insight is that the raw MACD
    line scales with absolute price and asset volatility, which makes its
    thresholds asset-specific and unstable across regimes. Dividing by ATR(26)
    and rescaling produces a momentum oscillator with universal, stationary
    bounds:

        MACD       = EMA(close, 12) - EMA(close, 26)
        ATR_26     = Wilder ATR over 26 bars
        MACD-V_t   = (MACD_t / ATR_26_t) * 100

    Spiroglou's published zones (validated across SPX, sector ETFs, single
    names, FX, crypto):
        Overbought  > +150
        Bullish      +50 to +150
        Neutral      -50 to +50
        Bearish      -150 to -50
        Oversold     < -150

    The strategy fires on a *fresh reclaim of -50 from the bearish zone*: this
    catches the moment momentum exits a downswing and re-enters neutral while
    the macro uptrend gate remains intact. This is structurally distinct from
    every existing oscillator in the sandbox:
      - Raw MACD/TSI/TRIX cross zero (no vol normalization, asset-dependent
        thresholds).
      - Stochastic/RSI/CCI cross fixed bounds but use price-only inputs (no ATR).
      - Premier Stochastic/Fisher transform reshape distribution but again no
        explicit volatility normalization.
    MACD-V is the only oscillator here whose threshold is interpretable as a
    multiple of average daily range.

    Entry (prior-bar arrays, no lookahead):
      - MACD-V_{t-2} <= -50  AND  MACD-V_{t-1} > -50  (fresh upcross of -50)
      - SMA50_{t-1} > SMA200_{t-1}                    (macro uptrend gate)
    Exit: close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    atr26 = _atr(high, low, close, 26)

    safe_atr = np.where(np.isfinite(atr26) & (atr26 > 0.0), atr26, np.nan)
    macd_v = (macd / safe_atr) * 100.0

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    macd_v_p1 = np.concatenate(([np.nan], macd_v[:-1]))
    macd_v_p2 = np.concatenate(([np.nan, np.nan], macd_v[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_reclaim = (macd_v_p2 <= -50.0) & (macd_v_p1 > -50.0)
    uptrend = sma50_p1 > sma200_p1
    valid = (
        np.isfinite(macd_v_p1)
        & np.isfinite(macd_v_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid & fresh_reclaim & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_mama_fama_cross(df: pd.DataFrame) -> list[Trade]:
    """John Ehlers' MAMA/FAMA cross — Hilbert-Transform adaptive MAs.

    Reference: John F. Ehlers, "MESA Adaptive Moving Average" (Stocks &
    Commodities, 2001). Ehlers applies the Hilbert Transform to the (H+L)/2
    series to construct an analytic signal, recovering the in-phase (I) and
    quadrature (Q) components. From these he derives:
      - the dominant cycle period (via arctan(Im/Re) on the analytic-signal
        phase rotation), and
      - the instantaneous phase angle (atan(Q1/I1)).
    The smoothing constant alpha is then set proportional to the phase
    rotation rate per bar:

        alpha = clip(FastLimit / DeltaPhase, SlowLimit, FastLimit)
        MAMA_t = alpha * price_t + (1 - alpha) * MAMA_{t-1}
        FAMA_t = 0.5*alpha * MAMA_t + (1 - 0.5*alpha) * FAMA_{t-1}

    With FastLimit=0.5, SlowLimit=0.05, MAMA accelerates aggressively when
    the analytic-signal phase is rotating quickly (price trending) and damps
    heavily when the phase rotation stalls (price cycling/ranging). FAMA
    follows MAMA with half the alpha, producing a slower trailing line.

    Distinct from every adaptive-MA already in the sandbox:
      - KAMA: Kaufman efficiency ratio (signal/noise) drives SC.
      - VIDYA: Chande CMO drives SC.
      - McGinley: (close/MD)^4 power-law factor in denominator.
      - FRAMA: range-based fractal-dimension drives SC.
      - HMA / ALMA: fixed weights, no adaptation.
    MAMA/FAMA is the only entry deriving its smoothing constant from the
    rotational rate of the analytic-signal phase via the Hilbert Transform.

    Entry (prior-bar arrays, no lookahead):
      - MAMA_{t-2} <= FAMA_{t-2} AND MAMA_{t-1} > FAMA_{t-1}  (fresh cross up)
      - SMA50_{t-1} > SMA200_{t-1}                            (macro uptrend)
    Exit: close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n_bars = len(close)

    fast_limit = 0.5
    slow_limit = 0.05

    price = (high + low) / 2.0
    smooth = np.zeros(n_bars)
    detrender = np.zeros(n_bars)
    I1 = np.zeros(n_bars)
    Q1 = np.zeros(n_bars)
    jI = np.zeros(n_bars)
    jQ = np.zeros(n_bars)
    I2 = np.zeros(n_bars)
    Q2 = np.zeros(n_bars)
    Re = np.zeros(n_bars)
    Im = np.zeros(n_bars)
    period = np.zeros(n_bars)
    smooth_period = np.zeros(n_bars)
    phase = np.zeros(n_bars)
    mama = np.full(n_bars, np.nan)
    fama = np.full(n_bars, np.nan)

    for i in range(n_bars):
        if i < 6:
            mama[i] = price[i]
            fama[i] = price[i]
            continue
        smooth[i] = (
            4.0 * price[i]
            + 3.0 * price[i - 1]
            + 2.0 * price[i - 2]
            + price[i - 3]
        ) / 10.0
        adj = 0.075 * period[i - 1] + 0.54
        detrender[i] = (
            0.0962 * smooth[i]
            + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6]
        ) * adj
        Q1[i] = (
            0.0962 * detrender[i]
            + 0.5769 * detrender[i - 2]
            - 0.5769 * detrender[i - 4]
            - 0.0962 * detrender[i - 6]
        ) * adj
        I1[i] = detrender[i - 3]
        jI[i] = (
            0.0962 * I1[i]
            + 0.5769 * I1[i - 2]
            - 0.5769 * I1[i - 4]
            - 0.0962 * I1[i - 6]
        ) * adj
        jQ[i] = (
            0.0962 * Q1[i]
            + 0.5769 * Q1[i - 2]
            - 0.5769 * Q1[i - 4]
            - 0.0962 * Q1[i - 6]
        ) * adj
        i2_raw = I1[i] - jQ[i]
        q2_raw = Q1[i] + jI[i]
        I2[i] = 0.2 * i2_raw + 0.8 * I2[i - 1]
        Q2[i] = 0.2 * q2_raw + 0.8 * Q2[i - 1]
        re_raw = I2[i] * I2[i - 1] + Q2[i] * Q2[i - 1]
        im_raw = I2[i] * Q2[i - 1] - Q2[i] * I2[i - 1]
        Re[i] = 0.2 * re_raw + 0.8 * Re[i - 1]
        Im[i] = 0.2 * im_raw + 0.8 * Im[i - 1]
        if Im[i] != 0.0 and Re[i] != 0.0:
            new_period = 360.0 / np.degrees(np.arctan(Im[i] / Re[i]))
        else:
            new_period = period[i - 1]
        prev_p = period[i - 1]
        if prev_p > 0.0:
            if new_period > 1.5 * prev_p:
                new_period = 1.5 * prev_p
            if new_period < 0.67 * prev_p:
                new_period = 0.67 * prev_p
        new_period = float(np.clip(new_period, 6.0, 50.0))
        period[i] = 0.2 * new_period + 0.8 * prev_p
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]
        if I1[i] != 0.0:
            phase[i] = np.degrees(np.arctan(Q1[i] / I1[i]))
        else:
            phase[i] = phase[i - 1]
        delta_phase = phase[i - 1] - phase[i]
        if delta_phase < 1.0:
            delta_phase = 1.0
        alpha = fast_limit / delta_phase
        if alpha < slow_limit:
            alpha = slow_limit
        if alpha > fast_limit:
            alpha = fast_limit
        prev_mama = mama[i - 1] if np.isfinite(mama[i - 1]) else price[i]
        prev_fama = fama[i - 1] if np.isfinite(fama[i - 1]) else price[i]
        mama[i] = alpha * price[i] + (1.0 - alpha) * prev_mama
        fama[i] = 0.5 * alpha * mama[i] + (1.0 - 0.5 * alpha) * prev_fama

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    mama_p1 = np.concatenate(([np.nan], mama[:-1]))
    mama_p2 = np.concatenate(([np.nan, np.nan], mama[:-2]))
    fama_p1 = np.concatenate(([np.nan], fama[:-1]))
    fama_p2 = np.concatenate(([np.nan, np.nan], fama[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_cross = (mama_p2 <= fama_p2) & (mama_p1 > fama_p1)
    uptrend = sma50_p1 > sma200_p1
    valid = (
        np.isfinite(mama_p1)
        & np.isfinite(mama_p2)
        & np.isfinite(fama_p1)
        & np.isfinite(fama_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid & fresh_cross & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_andean_oscillator_bull_cross(df: pd.DataFrame) -> list[Trade]:
    """Andean Oscillator (Alex Orekhov, 2022) — one-sided EMA-envelope dispersion.

    Reference: alexgrover, "Andean Oscillator", TradingView (2022). Two pairs
    of asymmetric exponential envelopes are tracked on close and close^2 (and
    similarly using open as a damping anchor). The upper envelopes ratchet up
    on new highs and decay back toward the open; the lower envelopes ratchet
    down on new lows and decay back up. Letting alpha=2/(N+1):

        up1_t = max(C_t, O_t, up1_{t-1} - alpha*(up1_{t-1} - O_t))
        up2_t = max(C_t^2, O_t^2, up2_{t-1} - alpha*(up2_{t-1} - O_t^2))
        dn1_t = min(C_t, O_t, dn1_{t-1} + alpha*(O_t - dn1_{t-1}))
        dn2_t = min(C_t^2, O_t^2, dn2_{t-1} + alpha*(O_t^2 - dn2_{t-1}))

    Treating up1/dn1 as one-sided E[X] estimators and up2/dn2 as one-sided
    E[X^2] estimators, the second-moment formula sigma^2 = E[X^2] - E[X]^2
    yields two separate dispersion components:

        bull_t = sqrt(max(0, dn2_t - dn1_t^2))   # range above lower envelope
        bear_t = sqrt(max(0, up2_t - up1_t^2))   # range below upper envelope
        signal_t = EMA(max(bull_t, bear_t), 9)

    Intuition: bull rises when prices stretch ABOVE the (slowly rising) min
    envelope — i.e. bullish thrust; bear rises when prices stretch BELOW the
    (slowly falling) max envelope — i.e. bearish thrust. A fresh cross of
    bull above the smoothed signal, with bull > bear, marks an emergent
    bullish regime. This dispersion-of-extrema construction is fundamentally
    different from every oscillator already in the sandbox (RSI/Stoch/MFI/CCI
    rank-based, MACD/TRIX/TSI EMA-difference, RVI/Klinger/Chaikin volume,
    Ehlers Hilbert-Transform, Aroon time-since-extreme).

    Entry (prior-bar arrays, no lookahead):
      - bull_{t-2} <= signal_{t-2} AND bull_{t-1} > signal_{t-1}  (fresh cross)
      - bull_{t-1} > bear_{t-1}                                   (bullish regime)
      - SMA50_{t-1} > SMA200_{t-1}                                (macro uptrend)
    Exit: close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    n_bars = len(close)

    length = 50
    sig_len = 9
    alpha = 2.0 / (length + 1.0)

    up1 = np.zeros(n_bars)
    up2 = np.zeros(n_bars)
    dn1 = np.zeros(n_bars)
    dn2 = np.zeros(n_bars)

    if n_bars > 0:
        up1[0] = max(close[0], open_[0])
        up2[0] = max(close[0] * close[0], open_[0] * open_[0])
        dn1[0] = min(close[0], open_[0])
        dn2[0] = min(close[0] * close[0], open_[0] * open_[0])

    for i in range(1, n_bars):
        c = close[i]
        o = open_[i]
        c2 = c * c
        o2 = o * o
        up1[i] = max(c, o, up1[i - 1] - alpha * (up1[i - 1] - o))
        up2[i] = max(c2, o2, up2[i - 1] - alpha * (up2[i - 1] - o2))
        dn1[i] = min(c, o, dn1[i - 1] + alpha * (o - dn1[i - 1]))
        dn2[i] = min(c2, o2, dn2[i - 1] + alpha * (o2 - dn2[i - 1]))

    bull = np.sqrt(np.maximum(0.0, dn2 - dn1 * dn1))
    bear = np.sqrt(np.maximum(0.0, up2 - up1 * up1))
    signal = _ema(np.maximum(bull, bear), sig_len)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    bull_p1 = np.concatenate(([np.nan], bull[:-1]))
    bull_p2 = np.concatenate(([np.nan, np.nan], bull[:-2]))
    bear_p1 = np.concatenate(([np.nan], bear[:-1]))
    sig_p1 = np.concatenate(([np.nan], signal[:-1]))
    sig_p2 = np.concatenate(([np.nan, np.nan], signal[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_cross = (bull_p2 <= sig_p2) & (bull_p1 > sig_p1)
    bullish_regime = bull_p1 > bear_p1
    uptrend = sma50_p1 > sma200_p1
    valid = (
        np.isfinite(bull_p1)
        & np.isfinite(bull_p2)
        & np.isfinite(bear_p1)
        & np.isfinite(sig_p1)
        & np.isfinite(sig_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid & fresh_cross & bullish_regime & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_tillson_t3_cross(df: pd.DataFrame) -> list[Trade]:
    """Tillson T3 Moving Average (Tim Tillson, 1998) — close-vs-T3 fresh upcross.

    Reference: Tillson, T. (1998), "Smoothing Techniques for More Accurate
    Signals", Stocks & Commodities Magazine. T3 is constructed as a triple
    application of a generalized-DEMA operator GD(p,b) = (1+b)*EMA(p) - b*EMA(EMA(p)).
    The closed form via a 6-EMA chain (each EMA applied to the previous output
    with the same length n) is:

        e1 = EMA(close, n);   e2 = EMA(e1, n);   e3 = EMA(e2, n)
        e4 = EMA(e3, n);      e5 = EMA(e4, n);   e6 = EMA(e5, n)
        T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
        c1 = -b^3
        c2 = 3*b^2 + 3*b^3
        c3 = -6*b^2 - 3*b - 3*b^3
        c4 = 1 + 3*b + 3*b^2 + b^3        (coeffs sum to 1 → unit-DC gain)

    The volume factor b ∈ (0,1) (typically 0.7) trades responsiveness against
    smoothness: b→0 collapses T3 onto a 3-stage cascaded EMA, b→1 sharpens it
    toward a triple-DEMA. Tillson's design goal was a smoother that responds
    quickly to genuine trend changes while heavily attenuating bar-to-bar
    noise — i.e. less lag than equivalent-length single EMA, less overshoot
    than DEMA/TEMA. This is mathematically distinct from every adaptive
    smoother already in the sandbox: McGinley (denominator damping), FRAMA
    (fractal-dim-adaptive alpha), MAMA/FAMA (Hilbert-Transform phase-adaptive),
    KAMA (efficiency-ratio adaptive), VIDYA (CMO-adaptive), ALMA (Gaussian
    window), HMA (sqrt-length WMA chain), Heikin-Ashi (OHLC averaging).

    Entry (prior-bar arrays, no lookahead):
      - close_{t-2} <= T3_{t-2} AND close_{t-1} > T3_{t-1}   (fresh upcross)
      - SMA50_{t-1} > SMA200_{t-1}                            (macro uptrend)
    Exit: close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)

    n = 14
    b = 0.7
    b2 = b * b
    b3 = b2 * b
    c1 = -b3
    c2 = 3.0 * b2 + 3.0 * b3
    c3 = -6.0 * b2 - 3.0 * b - 3.0 * b3
    c4 = 1.0 + 3.0 * b + 3.0 * b2 + b3

    e1 = _ema(close, n)
    e2 = _ema(e1, n)
    e3 = _ema(e2, n)
    e4 = _ema(e3, n)
    e5 = _ema(e4, n)
    e6 = _ema(e5, n)
    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    close_p1 = np.concatenate(([np.nan], close[:-1]))
    close_p2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    t3_p1 = np.concatenate(([np.nan], t3[:-1]))
    t3_p2 = np.concatenate(([np.nan, np.nan], t3[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_cross = (close_p2 <= t3_p2) & (close_p1 > t3_p1)
    uptrend = sma50_p1 > sma200_p1
    valid = (
        np.isfinite(close_p1)
        & np.isfinite(close_p2)
        & np.isfinite(t3_p1)
        & np.isfinite(t3_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid & fresh_cross & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_ehlers_laguerre_rsi(df: pd.DataFrame) -> list[Trade]:
    """Ehlers Laguerre RSI (Cybernetic Analysis for Stocks and Futures, 2004).

    Ehlers' Laguerre filter is a four-stage all-pass cascade controlled by a
    single damping factor γ ∈ (0, 1). Unlike a simple EMA chain (which is
    just successive low-pass smoothing), the Laguerre filter encodes both
    amplitude and *phase* response so that for the same effective lag it
    yields a much smoother envelope than an N-stage EMA. The recurrences are:

        L0_t = (1 - γ)·price_t + γ·L0_{t-1}
        L1_t = -γ·L0_t + L0_{t-1} + γ·L1_{t-1}
        L2_t = -γ·L1_t + L1_{t-1} + γ·L2_{t-1}
        L3_t = -γ·L2_t + L2_{t-1} + γ·L3_{t-1}

    The Laguerre RSI then accumulates pairwise differences across the four
    stages: at each bar t define three pair-deltas Δi = Li - L(i+1) for
    i ∈ {0,1,2}; let CU = Σ max(Δi, 0) and CD = Σ max(-Δi, 0). Then

        LRSI_t = CU / (CU + CD)              ∈ [0, 1]

    Conventional Ehlers thresholds: LRSI < 0.15 oversold, LRSI > 0.85
    overbought. Because the filter has a sharp roll-off, LRSI tends to
    "stick" at extremes during persistent moves, so a *fresh* upcross out of
    the oversold zone (rather than a level read) is the trade-relevant
    event — it marks the moment damping releases and price rotates back up.

    This is mathematically distinct from every smoother already in the
    sandbox (EMA/SMA/RMA, McGinley denominator-damped, FRAMA fractal-dim
    α, MAMA/FAMA Hilbert-phase, KAMA efficiency-ratio, VIDYA CMO-adaptive,
    ALMA Gaussian-window, HMA sqrt-WMA chain, Heikin-Ashi OHLC averaging,
    T3 6-EMA Tillson, Coppock ROC sum) and from RSI variants already
    registered (Wilder RSI, Connors RSI, Stoch RSI, Inverse Fisher RSI,
    Brown range-shift RSI). It is a *phase-aware* oscillator built on
    Laguerre polynomial impulse responses, not on momentum or smoothed
    price differences.

    Entry (prior-bar arrays only — no lookahead):
      - LRSI_{t-2} <= 0.15 AND LRSI_{t-1} > 0.15   (fresh oversold release)
      - SMA50_{t-1} > SMA200_{t-1}                  (macro uptrend filter)
    Exit: close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size

    gamma = 0.5
    one_m_g = 1.0 - gamma

    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)
    lrsi = np.full(n, np.nan)

    for i in range(n):
        if not np.isfinite(close[i]):
            if i > 0:
                L0[i] = L0[i - 1]
                L1[i] = L1[i - 1]
                L2[i] = L2[i - 1]
                L3[i] = L3[i - 1]
                lrsi[i] = lrsi[i - 1]
            continue
        if i == 0:
            L0[i] = close[i]
            L1[i] = close[i]
            L2[i] = close[i]
            L3[i] = close[i]
            continue
        L0[i] = one_m_g * close[i] + gamma * L0[i - 1]
        L1[i] = -gamma * L0[i] + L0[i - 1] + gamma * L1[i - 1]
        L2[i] = -gamma * L1[i] + L1[i - 1] + gamma * L2[i - 1]
        L3[i] = -gamma * L2[i] + L2[i - 1] + gamma * L3[i - 1]
        d01 = L0[i] - L1[i]
        d12 = L1[i] - L2[i]
        d23 = L2[i] - L3[i]
        cu = (d01 if d01 > 0 else 0.0) + (d12 if d12 > 0 else 0.0) + (d23 if d23 > 0 else 0.0)
        cd = (-d01 if d01 < 0 else 0.0) + (-d12 if d12 < 0 else 0.0) + (-d23 if d23 < 0 else 0.0)
        denom = cu + cd
        if denom > 0:
            lrsi[i] = cu / denom

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    lrsi_p1 = np.concatenate(([np.nan], lrsi[:-1]))
    lrsi_p2 = np.concatenate(([np.nan, np.nan], lrsi[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    threshold = 0.15
    fresh_release = (lrsi_p2 <= threshold) & (lrsi_p1 > threshold)
    uptrend = sma50_p1 > sma200_p1
    valid = (
        np.isfinite(lrsi_p1)
        & np.isfinite(lrsi_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid & fresh_release & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_rmi_oversold_cross(df: pd.DataFrame) -> list[Trade]:
    """Relative Momentum Index (Altman, Stocks & Commodities, Feb 1993).

    The Relative Momentum Index generalises Wilder's RSI by replacing the
    one-bar price change with an m-bar momentum. Where RSI(n) computes
    Wilder-smoothed averages of |close_t - close_{t-1}| split into up/down
    sides, RMI(n, m) does the same on |close_t - close_{t-m}|. The closed
    form is

        mom_t = close_t - close_{t-m}
        up_t  = max(mom_t, 0)
        dn_t  = max(-mom_t, 0)
        AvgUp_t = RMA_n(up_t),   AvgDn_t = RMA_n(dn_t)        (Wilder smoother)
        RMI_t = 100 · AvgUp_t / (AvgUp_t + AvgDn_t)            ∈ [0, 100]

    With m=1 RMI collapses to Wilder RSI; with m>1 the indicator measures
    sustained directional drift instead of bar-by-bar tugs, so it spends
    less time at neutral 50 during trends and the oversold readings are
    less noise-driven. Altman's original parameters were n=20, m=5; the
    later commonly-cited combo is n=14, m=5 — used here.

    This is mathematically distinct from every RSI-family member already
    registered: Wilder RSI (Connors_RSI_pullback, RSI_brown_range_shift,
    Connors_double_7s, cum_RSI2_pullback all use Wilder RSI internally),
    Stochastic RSI (stoch_rsi_oversold_cross), Inverse Fisher RSI
    (inverse_fisher_rsi), Connors composite RSI (connors_rsi_pullback),
    Laguerre RSI (ehlers_laguerre_rsi). The momentum-period generalisation
    is what makes RMI a separate object, not a reparameterisation.

    Entry (prior-bar arrays — no lookahead):
      - RMI_{t-2} <= 30 AND RMI_{t-1} > 30   (fresh oversold release)
      - SMA50_{t-1} > SMA200_{t-1}           (macro uptrend filter)
    Exit: close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size

    momentum_period = 5
    smooth_period = 14
    threshold = 30.0

    if n <= momentum_period:
        return []

    mom = np.full(n, np.nan)
    mom[momentum_period:] = close[momentum_period:] - close[:-momentum_period]
    up = np.where(np.isfinite(mom) & (mom > 0), mom, 0.0)
    dn = np.where(np.isfinite(mom) & (mom < 0), -mom, 0.0)
    # mask leading bars where momentum is undefined so RMA doesn't include zeros
    up[:momentum_period] = np.nan
    dn[:momentum_period] = np.nan

    avg_up = _rma(up, smooth_period)
    avg_dn = _rma(dn, smooth_period)

    rmi = np.full(n, np.nan)
    denom = avg_up + avg_dn
    valid = np.isfinite(avg_up) & np.isfinite(avg_dn) & (denom > 0)
    rmi[valid] = 100.0 * avg_up[valid] / denom[valid]

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    rmi_p1 = np.concatenate(([np.nan], rmi[:-1]))
    rmi_p2 = np.concatenate(([np.nan, np.nan], rmi[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    fresh_release = (rmi_p2 <= threshold) & (rmi_p1 > threshold)
    uptrend = sma50_p1 > sma200_p1
    valid_arr = (
        np.isfinite(rmi_p1)
        & np.isfinite(rmi_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid_arr & fresh_release & uptrend
    exits = np.isfinite(ema20) & (close < ema20)

    return _walk(entries, exits, close, df["date"].values)


def strat_bill_williams_alligator_awake(df: pd.DataFrame) -> list[Trade]:
    """Bill Williams Alligator (Trading Chaos, 1995) — bullish "awakening".

    The Alligator is a trio of Wilder-smoothed (SMMA = RMA) moving averages
    of the *median* price m_t = (high_t + low_t)/2, each *displaced forward
    in time* so the lines visually anticipate price:

        Jaw   = SMMA(m, 13)  shifted +8 bars   (slow base, "blue")
        Teeth = SMMA(m, 8)   shifted +5 bars   (medium, "red")
        Lips  = SMMA(m, 5)   shifted +3 bars   (fast,   "green")

    A forward shift of k means the value plotted at bar t is the SMMA value
    computed using only data through bar t-k — strictly causal at decision
    time t. The geometric idea (Williams' "fractal market" framing) is that
    when the three lines lie flat and intertwined, the alligator is "asleep"
    — price is in chop with no exploitable trend. When the lines fan out and
    order themselves with Lips > Teeth > Jaw, the alligator has "woken up
    with its mouth open" pointing upward — a regime where directional moves
    persist long enough for trend-following to pay. The diagnostic event is
    the *transition* from non-bullish ordering to bullish ordering, not a
    continuous read on the gap.

    This is mathematically distinct from every MA system already registered:
    none use the (a) Wilder-smoothed median price, (b) three-line ordering
    constraint Lips > Teeth > Jaw simultaneously, *and* (c) the forward-
    displacement geometry that makes the lines act as time-shifted support
    references. KAMA / VIDYA / FRAMA / MAMA-FAMA / McGinley vary the alpha
    of a single line; HMA / ALMA / Tillson T3 / Heikin-Ashi reshape a single
    smoother's impulse response; Donchian / Keltner / Bollinger build
    channels. The Alligator's contribution is regime detection through
    multi-timeframe MA *alignment* on shifted SMMAs of the median — Williams'
    Profitunity-system anchor.

    Entry (prior-bar arrays — strictly causal):
      - Lips_{t-1} > Teeth_{t-1} > Jaw_{t-1}   (alligator awake & bullish)
      - NOT (Lips_{t-2} > Teeth_{t-2} > Jaw_{t-2})  (fresh awakening, not
        a sustained-trend re-entry)
      - SMA50_{t-1} > SMA200_{t-1}              (macro uptrend filter)
    Exit: Lips < Teeth (mouth begins closing) OR close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = close.size

    median = (high + low) / 2.0

    raw_jaw = _rma(median, 13)
    raw_teeth = _rma(median, 8)
    raw_lips = _rma(median, 5)

    def _shift_forward(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if k < n:
            out[k:] = arr[: n - k]
        return out

    jaw = _shift_forward(raw_jaw, 8)
    teeth = _shift_forward(raw_teeth, 5)
    lips = _shift_forward(raw_lips, 3)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    lips_p1 = np.concatenate(([np.nan], lips[:-1]))
    teeth_p1 = np.concatenate(([np.nan], teeth[:-1]))
    jaw_p1 = np.concatenate(([np.nan], jaw[:-1]))
    lips_p2 = np.concatenate(([np.nan, np.nan], lips[:-2]))
    teeth_p2 = np.concatenate(([np.nan, np.nan], teeth[:-2]))
    jaw_p2 = np.concatenate(([np.nan, np.nan], jaw[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    bullish_now = (lips_p1 > teeth_p1) & (teeth_p1 > jaw_p1)
    bullish_prev = (lips_p2 > teeth_p2) & (teeth_p2 > jaw_p2)
    fresh_awake = bullish_now & ~bullish_prev

    uptrend = sma50_p1 > sma200_p1
    valid = (
        np.isfinite(lips_p1)
        & np.isfinite(teeth_p1)
        & np.isfinite(jaw_p1)
        & np.isfinite(lips_p2)
        & np.isfinite(teeth_p2)
        & np.isfinite(jaw_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    entries = valid & fresh_awake & uptrend
    mouth_closing = np.isfinite(lips) & np.isfinite(teeth) & (lips < teeth)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = mouth_closing | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_williams_ac_zero_acceleration(df: pd.DataFrame) -> list[Trade]:
    """Bill Williams Acceleration/Deceleration (AC) — fresh upside acceleration.

    AC is the *second derivative* indicator from Williams' Profitunity system
    in "New Trading Dimensions" (1998). It is built on top of the Awesome
    Oscillator (AO), but instead of reading AO levels it reads how fast AO
    itself is changing:

        median   = (high + low) / 2
        AO_t     = SMA(median, 5)_t  -  SMA(median, 34)_t
        AC_t     = AO_t  -  SMA(AO, 5)_t

    Williams' theoretical claim is that price changes direction *after*
    momentum changes direction, and momentum changes direction *after*
    acceleration changes direction. So among AO-family signals, AC is the
    earliest leading indicator in his hierarchy — it fires before AO crosses
    zero, before MACD turns, before MA crosses. Geometrically, AO is the
    (smoothed) first derivative of price; AC = AO − SMA(AO,5) is the
    deviation of AO from its own running mean, which behaves like the
    discrete second derivative (acceleration) of price.

    This is mathematically distinct from every AO/MACD-family strategy
    already registered. The Awesome Oscillator Saucer setup
    (`awesome_oscillator_saucer`) trades a *3-bar pause-and-resume shape in
    AO itself while AO stays positive*, i.e. a level/stall pattern in the
    first-derivative oscillator. AC, by contrast, fires when the second
    derivative (AO minus its own moving average) flips sign from negative to
    positive — the exact moment deceleration ends and acceleration begins.
    The two signals do not overlap: a saucer can occur with strongly
    positive AC throughout (no fresh acceleration), and a fresh AC zero-up
    cross typically occurs when AO is still well below its recent average
    (the saucer pattern is impossible there). MACD-V / TRIX / Coppock are
    derivatives of *EMA-of-close*, not SMA-of-median; they have different
    noise profiles, lag structure, and trigger geometry.

    Williams' canonical rule: "When AC is below zero, you need two
    consecutive green bars (rising AC) to buy. When AC is above zero, just
    one green bar is enough." We use a strict, testable form — fresh upside
    zero-cross in AC (strongest version of the same logic):

    Entry (prior-bar arrays — strictly causal, no lookahead):
      - AC_{t-2} < 0  AND  AC_{t-1} >= 0           (fresh zero up-cross)
      - AC_{t-1} > AC_{t-2}                         (rising acceleration)
      - SMA50_{t-1} > SMA200_{t-1}                  (macro uptrend filter)
    Exit: AC < 0 (acceleration flips back negative) OR close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    median = (high + low) / 2.0
    ao = _sma(median, 5) - _sma(median, 34)
    ac = ao - _sma(ao, 5)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    ac_p1 = np.concatenate(([np.nan], ac[:-1]))
    ac_p2 = np.concatenate(([np.nan, np.nan], ac[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(ac_p1)
        & np.isfinite(ac_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_zero_up = (ac_p2 < 0.0) & (ac_p1 >= 0.0) & (ac_p1 > ac_p2)
    uptrend = sma50_p1 > sma200_p1

    entries = valid & fresh_zero_up & uptrend
    accel_negative = np.isfinite(ac) & (ac < 0.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = accel_negative | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_ehlers_roofing_filter(df: pd.DataFrame) -> list[Trade]:
    """Ehlers Roofing Filter zero-up-cross — bandpass momentum (Cycle Analytics 2013).

    The Roofing Filter is John F. Ehlers' canonical bandpass detrender for
    isolating the dominant tradable cycle component of price. It is the
    cascade of two IIR filters:
      (1) a 2-pole HIGH-PASS filter — removes trend and cycles longer than HP
      (2) a 2-pole SUPER SMOOTHER   — removes noise and cycles shorter than LP
    With HP=48 and LP=10 the Roofing Filter passes only the cycle band that
    Ehlers argues is the cleanest swing-momentum signal on daily bars.
    Crosses of zero from below mark the start of a fresh up-cycle.

    Math (Ehlers, Cycle Analytics for Traders, 2013, ch. 3 & 4):
        a1 = (cos(0.707·2π/HP) + sin(0.707·2π/HP) − 1) / cos(0.707·2π/HP)
        HP_t = (1 − a1/2)² · (c_t − 2 c_{t−1} + c_{t−2})
             + 2 (1 − a1) · HP_{t−1}  −  (1 − a1)² · HP_{t−2}
        b1 = exp(−1.414·π / LP)
        cc = 2 b1 · cos(1.414·π / LP)
        c2 = cc,  c3 = −b1²,  c1 = 1 − c2 − c3
        Filt_t = c1 · (HP_t + HP_{t−1}) / 2 + c2 · Filt_{t−1} + c3 · Filt_{t−2}

    No other sandbox strategy uses a 2-pole-HP-into-2-pole-SS bandpass IIR.
    Schaff Trend Cycle is a stochastic-of-stochastic, MACD-V is volatility-
    normalised EMA-difference, MAMA/FAMA uses a Hilbert discriminator to drive
    adaptive EMA, Coppock is weighted rate-of-change, DPO detrends by shift,
    Pretty-Good-Oscillator is an ATR-normalised range z-score, TRIX is a
    triple-EMA derivative — different filters with different impulse
    responses and different geometric triggers. The Roofing Filter is unique
    here as a literal bandpass IIR.

    Entry (prior-bar arrays — strictly causal, no lookahead):
      - Filt_{t−2} < 0  AND  Filt_{t−1} >= 0       (fresh zero up-cross)
      - SMA50_{t−1} > SMA200_{t−1}                  (macro uptrend filter)
    Exit: Filt < 0  OR  close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size

    HP_PER = 48
    LP_PER = 10

    cos_hp = np.cos(0.707 * 2.0 * np.pi / HP_PER)
    sin_hp = np.sin(0.707 * 2.0 * np.pi / HP_PER)
    a1 = (cos_hp + sin_hp - 1.0) / cos_hp
    one_minus_a1 = 1.0 - a1
    one_minus_half_a1_sq = (1.0 - a1 / 2.0) ** 2
    one_minus_a1_sq = one_minus_a1 ** 2

    b1 = np.exp(-1.414 * np.pi / LP_PER)
    cc = 2.0 * b1 * np.cos(1.414 * np.pi / LP_PER)
    c2 = cc
    c3 = -(b1 * b1)
    c1 = 1.0 - c2 - c3

    hp = np.zeros(n)
    filt = np.zeros(n)
    for i in range(2, n):
        hp[i] = (
            one_minus_half_a1_sq * (close[i] - 2.0 * close[i - 1] + close[i - 2])
            + 2.0 * one_minus_a1 * hp[i - 1]
            - one_minus_a1_sq * hp[i - 2]
        )
        filt[i] = (
            c1 * (hp[i] + hp[i - 1]) / 2.0
            + c2 * filt[i - 1]
            + c3 * filt[i - 2]
        )

    warmup = max(HP_PER, LP_PER) * 2 + 5
    filt_masked = filt.astype(float).copy()
    filt_masked[:warmup] = np.nan

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    filt_p1 = np.concatenate(([np.nan], filt_masked[:-1]))
    filt_p2 = np.concatenate(([np.nan, np.nan], filt_masked[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(filt_p1)
        & np.isfinite(filt_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_zero_up = (filt_p2 < 0.0) & (filt_p1 >= 0.0)
    uptrend = sma50_p1 > sma200_p1
    entries = valid & fresh_zero_up & uptrend

    below_zero = np.isfinite(filt_masked) & (filt_masked < 0.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_zero | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_ehlers_trendflex(df: pd.DataFrame) -> list[Trade]:
    """Ehlers Trendflex zero up-cross — slope-sum normalised by adaptive RMS (S&C 2020).

    Trendflex is John F. Ehlers' "is this a real trend or noise?" indicator
    from Stocks & Commodities, January 2020 ("Reflex — A New Zero-Lag
    Indicator"). It first denoises price with a 2-pole SuperSmoother, then
    sums slopes of the smoothed series over the lookback window, and finally
    normalises by an exponentially-averaged RMS of those slopes. The result
    is a stationary signal that oscillates around zero in roughly [-2, +2].

    Math (Ehlers, S&C 2020):
        a1 = exp(-1.414·π / N)
        cc = 2 a1 · cos(1.414·π / N)
        SS_t = (1 − cc + a1²) · (c_t + c_{t−1})/2  +  cc · SS_{t−1}  −  a1² · SS_{t−2}

        slope_t = (1/N) · Σ_{j=1..N} (SS_t − SS_{t−j})
                = SS_t − mean(SS_{t−1}, …, SS_{t−N})

        ms_t      = 0.04 · slope_t²  +  0.96 · ms_{t−1}
        Trendflex = slope_t / sqrt(ms_t)        (else 0)

    Distinct from existing sandbox indicators: Roofing Filter is a 2-pole HP
    cascaded into a 2-pole SS bandpass; MAMA/FAMA uses a Hilbert-derived
    discriminator to drive an adaptive EMA; Laguerre RSI uses a 4-stage
    Laguerre filter; CoG is a windowed centroid; MACD-V is volatility-
    normalised EMA difference; DPO is a shift-detrended SMA; Schaff Trend
    Cycle is a stochastic-of-stochastic. Trendflex is unique here as a
    SUPERSMOOTHER-DRIVEN SLOPE-SUM with adaptive-RMS normalisation.

    Entry (prior-bar arrays — strictly causal, no lookahead):
      - TF_{t−2} < 0  AND  TF_{t−1} >= 0          (fresh zero up-cross)
      - SMA50_{t−1} > SMA200_{t−1}                 (macro uptrend filter)
    Exit: TF < 0  OR  close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size

    PERIOD = 20

    a1 = np.exp(-1.414 * np.pi / PERIOD)
    cc = 2.0 * a1 * np.cos(1.414 * np.pi / PERIOD)
    c2 = cc
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3

    ss = np.zeros(n)
    for i in range(2, n):
        ss[i] = (
            c1 * (close[i] + close[i - 1]) / 2.0
            + c2 * ss[i - 1]
            + c3 * ss[i - 2]
        )

    ss_shift = np.concatenate(([np.nan], ss[:-1]))
    mean_lag = _sma(ss_shift, PERIOD)
    slope = ss - mean_lag

    ms = np.zeros(n)
    trendflex = np.full(n, np.nan)
    warmup = 2 * PERIOD + 5
    for i in range(1, n):
        s = slope[i] if np.isfinite(slope[i]) else 0.0
        ms[i] = 0.04 * s * s + 0.96 * ms[i - 1]
        if i >= warmup and ms[i] > 0.0 and np.isfinite(slope[i]):
            trendflex[i] = slope[i] / np.sqrt(ms[i])

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    tf_p1 = np.concatenate(([np.nan], trendflex[:-1]))
    tf_p2 = np.concatenate(([np.nan, np.nan], trendflex[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(tf_p1)
        & np.isfinite(tf_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_zero_up = (tf_p2 < 0.0) & (tf_p1 >= 0.0)
    uptrend = sma50_p1 > sma200_p1
    entries = valid & fresh_zero_up & uptrend

    below_zero = np.isfinite(trendflex) & (trendflex < 0.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_zero | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_twiggs_money_flow(df: pd.DataFrame) -> list[Trade]:
    """Twiggs Money Flow zero up-cross — Colin Twiggs' true-range CMF variant.

    Twiggs Money Flow (Colin Twiggs, IncredibleCharts ~1999) refines Chaikin
    Money Flow with two structural changes that matter on real equity data:

      1. TRUE range (gap-aware) is used in place of the current-bar high-low
         spread. The accumulation factor becomes
              ((close - TR_low) - (TR_high - close)) / TR
         where TR_high = max(high, prev_close), TR_low = min(low, prev_close),
         TR = TR_high - TR_low. Volume on overnight gap days is therefore
         attributed to the side of the gap rather than dropped or distorted
         by an unrepresentative intraday range.
      2. Both the accumulation/distribution numerator and the volume
         denominator are smoothed by EMA(21) instead of the rolling SMA(20)
         used by Chaikin. EMA's exponential weighting gives a faster, less
         noisy line that flips around zero on cleaner accumulation regime
         changes.

    Math:
        TR_high_t = max(high_t, close_{t-1})
        TR_low_t  = min(low_t,  close_{t-1})
        TR_t      = TR_high_t - TR_low_t
        ADV_t     = ((close_t - TR_low_t) - (TR_high_t - close_t)) / TR_t · vol_t
        TMF_t     = EMA(ADV, 21)_t / EMA(volume, 21)_t

    Distinct from existing sandbox strategies:
      - cmf_zero_reclaim:   high-low range, SMA(20) numerator and denominator,
                            no gap correction.
      - klinger_volume_oscillator_signal_cross: cumulative volume force
                            signed by trend (high+low+close direction).
      - chaikin_oscillator_zero_cross: MACD(3,10) of the A/D line, not a
                            money-flow ratio.
      - obv_ema_cross:      sign-of-close-change × volume, no range weighting.
      - elder_force_index_zero_cross: price-change × volume, no accumulation
                            factor.

    Entry (prior-bar arrays — strictly causal, no lookahead):
      - TMF_{t-2} < 0  AND  TMF_{t-1} >= 0          (fresh zero up-cross)
      - SMA50_{t-1} > SMA200_{t-1}                   (macro uptrend filter)
    Exit: TMF < 0  OR  close < EMA20.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n = close.size

    PERIOD = 21

    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr_high = np.where(np.isfinite(prev_close), np.maximum(high, prev_close), high)
    tr_low = np.where(np.isfinite(prev_close), np.minimum(low, prev_close), low)
    tr = tr_high - tr_low

    safe_tr = np.where(tr > 0.0, tr, np.nan)
    accum_factor = ((close - tr_low) - (tr_high - close)) / safe_tr
    adv = accum_factor * volume
    adv = np.where(np.isfinite(adv), adv, 0.0)

    ema_adv = _ema(adv, PERIOD)
    ema_vol = _ema(volume, PERIOD)

    tmf = np.full(n, np.nan)
    valid_ratio = (
        np.isfinite(ema_adv)
        & np.isfinite(ema_vol)
        & (ema_vol > 0.0)
    )
    tmf[valid_ratio] = ema_adv[valid_ratio] / ema_vol[valid_ratio]

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    tmf_p1 = np.concatenate(([np.nan], tmf[:-1]))
    tmf_p2 = np.concatenate(([np.nan, np.nan], tmf[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(tmf_p1)
        & np.isfinite(tmf_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_zero_up = (tmf_p2 < 0.0) & (tmf_p1 >= 0.0)
    uptrend = sma50_p1 > sma200_p1
    entries = valid & fresh_zero_up & uptrend

    below_zero = np.isfinite(tmf) & (tmf < 0.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_zero | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_elder_impulse_bull(df: pd.DataFrame) -> list[Trade]:
    """Elder Impulse System (Alexander Elder, 'Come Into My Trading Room' 2002).

    Elder colour-codes each bar by combining trend and momentum: GREEN when
    BOTH the 13-EMA is rising AND the 12,26,9 MACD histogram is rising
    (trend + momentum aligned bullish), RED when both are falling, and BLUE
    otherwise. The system says: don't fight green/red — only take longs in a
    fresh green-bar transition.

    Entry: fresh transition into a green impulse bar (green on prior bar but
    not on the bar before that), with SMA50 > SMA200 trend filter to keep us
    in established uptrends only.
    Exit: impulse turns RED (EMA13 falling AND MACD histogram falling) OR
    close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size

    ema13 = _ema(close, 13)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    macd_hist = macd - macd_signal

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    ema13_prev = np.concatenate(([np.nan], ema13[:-1]))
    hist_prev = np.concatenate(([np.nan], macd_hist[:-1]))

    valid_slope = (
        np.isfinite(ema13)
        & np.isfinite(ema13_prev)
        & np.isfinite(macd_hist)
        & np.isfinite(hist_prev)
    )
    impulse_green = valid_slope & (ema13 > ema13_prev) & (macd_hist > hist_prev)
    impulse_red = valid_slope & (ema13 < ema13_prev) & (macd_hist < hist_prev)

    green_p1 = np.concatenate(([False], impulse_green[:-1]))
    green_p2 = np.concatenate(([False, False], impulse_green[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = np.isfinite(sma50_p1) & np.isfinite(sma200_p1)
    fresh_green = green_p1 & (~green_p2)
    uptrend = sma50_p1 > sma200_p1
    entries = valid & fresh_green & uptrend

    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = impulse_red | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_vw_macd_signal_cross(df: pd.DataFrame) -> list[Trade]:
    """Volume-Weighted MACD (Buff Dormeier, 'Investing with Volume Analysis' 2011).

    Standard MACD substitutes a Volume-Weighted MA for the EMA, so trend shifts
    are confirmed by participation. VWMA(n) = Σ(close*vol, n) / Σ(vol, n). The
    VW-MACD line is VWMA12 − VWMA26 and the signal is EMA9 of that line.
    Dormeier argues volume-confirmed crosses cut whipsaw versus price-only MACD.

    Entry: fresh signal-line up-cross (line crosses above signal this bar) with
    line above zero for trend confirmation, inside SMA50 > SMA200.
    Exit: line crosses below signal OR close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n = close.size

    pv = close * volume

    def _vwma(period: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if n < period:
            return out
        pv_cum = np.cumsum(pv)
        vol_cum = np.cumsum(volume)
        for i in range(period - 1, n):
            if i == period - 1:
                num = pv_cum[i]
                den = vol_cum[i]
            else:
                num = pv_cum[i] - pv_cum[i - period]
                den = vol_cum[i] - vol_cum[i - period]
            if den > 0:
                out[i] = num / den
        return out

    vwma12 = _vwma(12)
    vwma26 = _vwma(26)
    vwmacd = vwma12 - vwma26
    signal = _ema(vwmacd, 9)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    line_prev = np.concatenate(([np.nan], vwmacd[:-1]))
    sig_prev = np.concatenate(([np.nan], signal[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(line_prev)
        & np.isfinite(sig_prev)
        & np.isfinite(vwmacd)
        & np.isfinite(signal)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    fresh_up_cross = valid & (line_prev <= sig_prev) & (vwmacd > signal)
    above_zero = vwmacd > 0
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_up_cross & above_zero & uptrend

    cross_down = np.isfinite(vwmacd) & np.isfinite(signal) & (vwmacd < signal)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = cross_down | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_chande_forecast_oscillator(df: pd.DataFrame) -> list[Trade]:
    """Chande Forecast Oscillator (Tushar Chande, 'Beyond Technical Analysis' 1997).

    CFO is the residual between price and its n-bar linear-regression
    forecast, expressed as a percent of price:
        CFO_i = 100 * (close_i − LR_forecast_i) / close_i,
    where LR_forecast_i is the OLS regression line over the last n closes
    evaluated at bar i (intercept + slope*(n−1)). Positive CFO means price
    is running ahead of its statistical trend; negative means it lags.

    Distinct from strat_linreg_slope_signchange (which keys on the *sign*
    of the fitted slope). Two stocks can share the same upward slope yet
    sit on opposite sides of their regression lines — CFO is a *level*
    signal, capturing the moment a stalled price snaps back above its
    own OLS fit while the longer-term trend is intact.

    Entry: CFO crosses up through zero (CFO_{i-1} <= 0 < CFO_i) inside an
        SMA50 > SMA200 long-term uptrend.
    Exit: CFO < 0 OR close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    period = 14

    x = np.arange(period, dtype=float)
    sum_x = float(x.sum())
    sum_x2 = float((x * x).sum())
    denom = period * sum_x2 - sum_x * sum_x

    s = pd.Series(close)
    sum_y = s.rolling(period, min_periods=period).sum().to_numpy()
    sum_xy = (
        s.rolling(period, min_periods=period)
        .apply(lambda w: float(np.dot(x, w)), raw=True)
        .to_numpy()
    )

    slope = (period * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / period
    forecast = intercept + slope * (period - 1)

    cfo = np.where(
        (close != 0) & np.isfinite(forecast),
        100.0 * (close - forecast) / close,
        np.nan,
    )

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    cfo_prev = np.concatenate(([np.nan], cfo[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(cfo_prev)
        & np.isfinite(cfo)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_up_cross = valid & (cfo_prev <= 0.0) & (cfo > 0.0)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_up_cross & uptrend

    cfo_neg = np.isfinite(cfo) & (cfo < 0.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = cfo_neg | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_tema_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """Triple Exponential Moving Average (Patrick Mulloy, Stocks & Commodities
    Jan/Feb 1994 — 'Smoothing Data With Faster Moving Averages').

    TEMA compensates for the lag inherent in a single EMA by combining three
    cascaded EMAs:
        TEMA(n) = 3*EMA(n) - 3*EMA(EMA(n)) + EMA(EMA(EMA(n))).
    Mulloy's identity removes the second-order lag term so the curve hugs
    price without the over-shoot of plain EMA chains. Fast/slow TEMA crosses
    therefore react sooner than EMA crosses while remaining smoother than
    raw price, in principle giving cleaner trend-onset timing than the
    EMA, DEMA, KAMA, HMA, ALMA, VIDYA, McGinley, T3 and FRAMA variants
    already on the bench (none of those decompose into the 3·EMA − 3·EMA²
    + EMA³ identity).

    Entry: fresh up-cross of TEMA(10) above TEMA(30) inside SMA50 > SMA200.
    Exit: TEMA(10) < TEMA(30) OR close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)

    def _tema(x: np.ndarray, n: int) -> np.ndarray:
        e1 = _ema(x, n)
        e2 = _ema(e1, n)
        e3 = _ema(e2, n)
        return 3.0 * e1 - 3.0 * e2 + e3

    tema_fast = _tema(close, 10)
    tema_slow = _tema(close, 30)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    fast_prev = np.concatenate(([np.nan], tema_fast[:-1]))
    slow_prev = np.concatenate(([np.nan], tema_slow[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    n = close.size
    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 90)
    warmup[warmup_start:] = True

    valid = (
        warmup
        & np.isfinite(fast_prev)
        & np.isfinite(slow_prev)
        & np.isfinite(tema_fast)
        & np.isfinite(tema_slow)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    fresh_up_cross = valid & (fast_prev <= slow_prev) & (tema_fast > tema_slow)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_up_cross & uptrend

    cross_down = (
        np.isfinite(tema_fast) & np.isfinite(tema_slow) & (tema_fast < tema_slow)
    )
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = cross_down | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_disparity_index_zero_cross(df: pd.DataFrame) -> list[Trade]:
    """Disparity Index (Steve Nison, 'Beyond Candlesticks' 1994) — zero up-cross.

    Disparity Index (DI) is a Japanese momentum gauge popularised in the
    West by Nison: DI(n) = (Close - SMA(n)) / SMA(n) * 100. It expresses
    how far price has stretched from its mean as a percentage, so a zero
    up-cross marks the moment a stock reclaims its trailing average from
    below — a different topology than fast/slow MA crosses (which compare
    two smoothings of price) and different from oscillator zero crosses
    like CMO, TSI, TRIX, CFO or DPO that operate on derivatives of price
    rather than raw close-vs-mean residual.

    Entry: fresh DI(14) up-cross above 0 inside SMA50 > SMA200 trend.
    Exit: DI(14) < 0 OR close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)

    sma14 = _sma(close, 14)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    with np.errstate(divide="ignore", invalid="ignore"):
        di = np.where(
            np.isfinite(sma14) & (sma14 > 0),
            (close - sma14) / sma14 * 100.0,
            np.nan,
        )

    di_prev = np.concatenate(([np.nan], di[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    n = close.size
    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 210)
    warmup[warmup_start:] = True

    valid = (
        warmup
        & np.isfinite(di_prev)
        & np.isfinite(di)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )

    fresh_up_cross = valid & (di_prev <= 0.0) & (di > 0.0)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_up_cross & uptrend

    di_below = np.isfinite(di) & (di < 0.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = di_below | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_bressert_dss_oversold_cross(df: pd.DataFrame) -> list[Trade]:
    """Walter Bressert's Double Smoothed Stochastic (DSS) — fresh cross above 30.

    Bressert ("The Power of Oscillator/Cycle Combinations", 1991) defined the
    DSS as a stochastic-of-a-stochastic with EMA smoothing on each leg, which
    eliminates the saw-tooth chop of raw %K while preserving its 0..100 bounds:

        raw_K   = 100 * (close - LLV(low, N)) / (HHV(high, N) - LLV(low, N))
        emaK    = EMA(raw_K, s1)                       # first smoothing
        stoch2  = 100 * (emaK - LLV(emaK, N))
                  / (HHV(emaK, N) - LLV(emaK, N))      # restochasticize
        DSS     = EMA(stoch2, s1)                      # second smoothing

    Settings N=13, s1=8 are Bressert's published values. Because the smoothing
    is applied *inside* the stochastic envelope (not on its output), DSS is
    structurally distinct from every existing oscillator in the sandbox:
      - stochastic_oversold_recovery: raw %K, no smoothing.
      - stoch_rsi_oversold_cross: stochastic of RSI (a momentum input).
      - smi_blau_oversold_cross: signed midpoint distance, ±100 range with
        double EMA on numerator/denominator separately.
      - schaff_trend_cycle: double stochastic of MACD, not of price range.
      - premier_stochastic_oscillator: 5-period stoch with Fisher transform.

    Entry: fresh DSS up-cross above 30 (oversold lift) inside SMA50 > SMA200.
    Exit: DSS crosses below 70 (lose momentum from overbought) OR close < EMA20.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    n = close.size
    N = 13
    s1 = 8

    h_s = pd.Series(high)
    l_s = pd.Series(low)
    hhv1 = h_s.rolling(N, min_periods=N).max().to_numpy()
    llv1 = l_s.rolling(N, min_periods=N).min().to_numpy()
    rng1 = hhv1 - llv1

    raw_k = np.full(n, np.nan)
    valid1 = np.isfinite(rng1) & (rng1 > 0)
    raw_k[valid1] = 100.0 * (close[valid1] - llv1[valid1]) / rng1[valid1]

    raw_k_filled = np.nan_to_num(raw_k, nan=0.0)
    ema_k = _ema(raw_k_filled, s1)
    ema_k = np.where(np.isfinite(raw_k), ema_k, np.nan)

    ema_k_s = pd.Series(ema_k)
    hhv2 = ema_k_s.rolling(N, min_periods=N).max().to_numpy()
    llv2 = ema_k_s.rolling(N, min_periods=N).min().to_numpy()
    rng2 = hhv2 - llv2

    stoch2 = np.full(n, np.nan)
    valid2 = np.isfinite(rng2) & (rng2 > 0) & np.isfinite(ema_k)
    stoch2[valid2] = 100.0 * (ema_k[valid2] - llv2[valid2]) / rng2[valid2]

    stoch2_filled = np.nan_to_num(stoch2, nan=0.0)
    dss = _ema(stoch2_filled, s1)
    dss = np.where(np.isfinite(stoch2), dss, np.nan)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    dss_p1 = np.concatenate(([np.nan], dss[:-1]))
    dss_p2 = np.concatenate(([np.nan], dss_p1[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 220)
    warmup[warmup_start:] = True

    valid = (
        warmup
        & np.isfinite(dss_p1)
        & np.isfinite(dss_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_up_cross_30 = valid & (dss_p2 <= 30.0) & (dss_p1 > 30.0)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_up_cross_30 & uptrend

    fresh_dn_cross_70 = (
        np.isfinite(dss_p1) & np.isfinite(dss_p2)
        & (dss_p2 >= 70.0) & (dss_p1 < 70.0)
    )
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = fresh_dn_cross_70 | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_chandelier_exit_reclaim(df: pd.DataFrame) -> list[Trade]:
    """Chuck LeBeau Chandelier Exit reclaim — fresh up-cross of the trailing line.

    LeBeau introduced the Chandelier Exit in S&C (1992-95) as a volatility-
    adaptive trailing stop "hung from the ceiling" of recent highs:

        chandelier = HHV(close, 22) - 3 * ATR(22)

    Conventional use is as a long stop. Here we invert it as a trend-restart
    signal: when price has been BELOW the chandelier line and reclaims it on a
    bar close, the same volatility envelope that would have stopped a long now
    confirms a fresh resumption of the up-leg. Combined with SMA50>SMA200 the
    setup is a textbook "weak hand shake-out / strong hand re-entry" filter.

    Structurally distinct from existing strategies in the sandbox:
      - Supertrend variants use HL2 ± k*ATR with alternating upper/lower bands
        and a flip-direction state — Chandelier is one-sided, anchored to the
        rolling HHV(close), not to HL2.
      - Donchian / Keltner / Acceleration-Bands fire on price piercing a
        channel rail; Chandelier reclaim fires on a *recovery* of an
        ATR-discounted high, which is a different geometric event.
      - Parabolic SAR is acceleration-driven, not HHV-anchored.

    Entry: prev close <= prev chandelier AND today's close > today's chandelier
           inside SMA50 > SMA200 (fresh reclaim, no lookahead).
    Exit:  close < chandelier (lose the line again) OR close < EMA20.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    n = close.size
    N = 22
    K = 3.0

    atr = _atr(high, low, close, N)
    hhv_close = pd.Series(close).rolling(N, min_periods=N).max().to_numpy()
    chand = hhv_close - K * atr

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    close_p1 = np.concatenate(([np.nan], close[:-1]))
    chand_p1 = np.concatenate(([np.nan], chand[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 220)
    warmup[warmup_start:] = True

    valid = (
        warmup
        & np.isfinite(close_p1)
        & np.isfinite(chand_p1)
        & np.isfinite(chand)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_reclaim = valid & (close_p1 <= chand_p1) & (close > chand)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_reclaim & uptrend

    lose_line = np.isfinite(chand) & (close < chand)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = lose_line | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_trend_intensity_index(df: pd.DataFrame) -> list[Trade]:
    """Trend Intensity Index (M.H. Pee, S&C 2002) — fresh up-cross of TII above 50.

    TII measures directional dominance over the second half of an N-bar window.
    For N=60, compute SMA(close, N). Over the last M=N//2=30 bars:

        SDpos = sum(close - SMA) for bars where close > SMA
        SDneg = sum(SMA - close) for bars where close < SMA
        TII   = 100 * SDpos / (SDpos + SDneg)

    TII > 50 means the recent half-window has accumulated more above-mean
    deviation than below — an uptrend is dominant in magnitude, not just in
    bar count. A fresh up-cross of 50 inside SMA50>SMA200 captures regime
    emergence with price now sitting in the upper half of its 60-bar mean.

    Distinct from existing sandbox strategies:
      - Aroon counts bar position of HHV/LLV — TII sums signed close-vs-SMA
        deviations (magnitude-weighted, not order-statistic).
      - DPO is a single-bar detrended print — TII aggregates 30 bars of
        sign-bucketed deviation.
      - CFO is one-bar regression residual; TII spans a rolling window.
      - Choppiness Index uses ATR vs range — TII uses deviation imbalance.

    Entry: prev TII <= 50 AND today TII > 50 inside SMA50 > SMA200.
    Exit:  TII < 50 OR close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size
    N = 60
    M = N // 2

    sma = _sma(close, N)
    dev = close - sma
    pos_dev = np.where(np.isfinite(dev) & (dev > 0), dev, 0.0)
    neg_dev = np.where(np.isfinite(dev) & (dev < 0), -dev, 0.0)

    sd_pos = pd.Series(pos_dev).rolling(M, min_periods=M).sum().to_numpy()
    sd_neg = pd.Series(neg_dev).rolling(M, min_periods=M).sum().to_numpy()

    denom = sd_pos + sd_neg
    tii = np.where(denom > 0, 100.0 * sd_pos / denom, np.nan)
    # Mask values where SMA wasn't defined yet
    tii = np.where(np.isfinite(sma), tii, np.nan)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    tii_p1 = np.concatenate(([np.nan], tii[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 220)
    warmup[warmup_start:] = True

    valid = (
        warmup
        & np.isfinite(tii_p1)
        & np.isfinite(tii)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_cross = valid & (tii_p1 <= 50.0) & (tii > 50.0)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_cross & uptrend

    below_50 = np.isfinite(tii) & (tii < 50.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_50 | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_volume_flow_indicator(df: pd.DataFrame) -> list[Trade]:
    """Volume Flow Indicator (Markos Katsanos, TASC June 2004) — VFI(130) bullish zero-cross.

    VFI is a published normalized money-flow oscillator distinct from OBV,
    CMF, TMF, KVO, and Force Index. Two filters tame OBV's flaws:
        - noise cutoff: ignore typical-price moves smaller than
          0.2 · stdev(log_typical, 30) · close (small bars don't count)
        - volume cap: cap each bar's volume at 2.5 · SMA(volume, 50)
          (extreme spikes don't dominate)
    VFI = SMA( signed_capped_volume, 130 ) / SMA(volume, 50), then EMA(3) smooth.

    Long: VFI smoothed crosses up through 0 from below inside SMA50>SMA200.
    Exit: VFI<0 or close<EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n = close.size

    typical = (high + low + close) / 3.0
    safe_tp = np.where(typical > 0, typical, np.nan)
    log_tp = np.log(safe_tp)
    inter = np.full(n, np.nan)
    inter[1:] = log_tp[1:] - log_tp[:-1]

    vinter = pd.Series(inter).rolling(30, min_periods=30).std(ddof=0).to_numpy()
    cutoff = 0.2 * vinter * close

    vave = pd.Series(volume).rolling(50, min_periods=50).mean().to_numpy()
    vmax = vave * 2.5
    vc = np.where(np.isfinite(vmax), np.minimum(volume, vmax), np.nan)

    mf = np.full(n, np.nan)
    mf[1:] = typical[1:] - typical[:-1]

    vcp_signed = np.where(
        np.isfinite(mf) & np.isfinite(cutoff) & np.isfinite(vc),
        np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0.0)),
        np.nan,
    )
    vcp_for_sum = np.where(np.isfinite(vcp_signed), vcp_signed, 0.0)

    period = 130
    vcp_sma = (
        pd.Series(vcp_for_sum)
        .rolling(period, min_periods=period)
        .mean()
        .to_numpy()
    )
    vfi_raw = np.where(
        np.isfinite(vave) & (vave > 0) & np.isfinite(vcp_sma),
        vcp_sma / vave,
        np.nan,
    )

    # EMA(3) of vfi_raw, starting once vfi_raw is finite, to avoid NaN propagation.
    vfi = np.full(n, np.nan)
    alpha = 2.0 / (3 + 1)
    started = False
    prev = np.nan
    for i in range(n):
        v = vfi_raw[i]
        if not np.isfinite(v):
            continue
        if not started:
            prev = v
            started = True
        else:
            prev = alpha * v + (1 - alpha) * prev
        vfi[i] = prev

    ema20 = _ema(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)

    vfi_p1 = np.concatenate(([np.nan], vfi[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    valid = (
        np.isfinite(vfi_p1)
        & np.isfinite(vfi)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_cross = valid & (vfi_p1 <= 0.0) & (vfi > 0.0)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_cross & uptrend

    below_zero = np.isfinite(vfi) & (vfi < 0.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_zero | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_pmo_signal_cross(df: pd.DataFrame) -> list[Trade]:
    """Carl Swenlin DecisionPoint Price Momentum Oscillator (StockCharts) — PMO/signal cross.

    PMO = 20-EMA of [10 × 35-EMA of 1-bar percent ROC]. Signal = 10-EMA of PMO.
    Long on a fresh PMO bullish cross above its signal line inside an
    SMA(50)>SMA(200) uptrend. Exits on PMO<signal or close<EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    roc = np.zeros(n, dtype=float)
    if n > 1:
        prev = close[:-1]
        roc[1:] = np.where(prev > 0, 100.0 * (close[1:] / prev - 1.0), 0.0)

    smooth1 = _ema(roc, 35) * 10.0
    pmo = _ema(smooth1, 20)
    signal = _ema(pmo, 10)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    pmo_p1 = np.concatenate(([np.nan], pmo[:-1]))
    pmo_p2 = np.concatenate(([np.nan, np.nan], pmo[:-2]))
    sig_p1 = np.concatenate(([np.nan], signal[:-1]))
    sig_p2 = np.concatenate(([np.nan, np.nan], signal[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    # Warm-up so the chained EMAs (35 → 20 → 10) have time to settle.
    idx = np.arange(n)
    warm = idx >= 80

    valid = (
        warm
        & np.isfinite(pmo_p1)
        & np.isfinite(pmo_p2)
        & np.isfinite(sig_p1)
        & np.isfinite(sig_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_cross = valid & (pmo_p2 <= sig_p2) & (pmo_p1 > sig_p1)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_cross & uptrend

    pmo_below = np.isfinite(pmo) & np.isfinite(signal) & (pmo < signal)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = pmo_below | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_adaptive_price_zone_breakout(df: pd.DataFrame) -> list[Trade]:
    """Adaptive Price Zone (APZ) breakout — Lee Leibfarth, TASC Sep 2006.

    APZ is a volatility channel built around Patrick Mulloy's DEMA so the
    centerline tracks price faster than a single-EMA Keltner basis:

        DEMA(x, N) = 2*EMA(x, N) - EMA(EMA(x, N), N)
        center     = DEMA(close, N)
        rangeBand  = DEMA(high-low, N)
        upper/lower = center ± k * rangeBand

    Long on a fresh bar-close breakout above the upper APZ band (prior bar
    closed at/below the band, current close above) inside an SMA(50)>SMA(200)
    regime. Exit when close falls below the DEMA centerline or below EMA(20).

    Distinct from Keltner (EMA + ATR), Bollinger (SMA + stdev), Acceleration
    Bands (SMA × HL%), and ALMA (Gaussian-weighted MA cross).
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(close)

    N = 20
    BAND_K = 2.0

    ema_close = _ema(close, N)
    dema_close = 2.0 * ema_close - _ema(ema_close, N)

    rng = high - low
    ema_rng = _ema(rng, N)
    dema_rng = 2.0 * ema_rng - _ema(ema_rng, N)

    upper = dema_close + BAND_K * dema_rng

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    close_p1 = np.concatenate(([np.nan], close[:-1]))
    close_p2 = np.concatenate(([np.nan, np.nan], close[:-2]))
    upper_p1 = np.concatenate(([np.nan], upper[:-1]))
    upper_p2 = np.concatenate(([np.nan, np.nan], upper[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    idx = np.arange(n)
    warm = idx >= 60

    valid = (
        warm
        & np.isfinite(upper_p1)
        & np.isfinite(upper_p2)
        & np.isfinite(close_p1)
        & np.isfinite(close_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_breakout = valid & (close_p2 <= upper_p2) & (close_p1 > upper_p1)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_breakout & uptrend

    below_center = np.isfinite(dema_close) & (close < dema_close)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_center | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_stiffness_indicator(df: pd.DataFrame) -> list[Trade]:
    """Stiffness Indicator (Joe Sharp, Active Trader Magazine 2003) — fresh cross above 90.

    Sharp's published rule measures the "stiffness" of an uptrend as the
    fraction of the last 60 bars on which the close held above
    ma100 - 0.2 * stdev(close, 60). A reading near 100 means dips have been
    rare — the trend is uninterrupted and "stiff". Long when stiffness rises
    above 90 (sustained-strength regime begins) inside SMA50>SMA200; exit
    when stiffness drops below 50 (trend losing its hold) or close falls
    below SMA(50).

    Distinct from every sandbox strategy: this is a state-occupancy count
    of price vs a single dynamic threshold over a rolling window — not a
    moving-average crossover, not an oscillator zero-cross, not an
    ADX/RSI/CCI mean-cross, not a candlestick pattern. The construction
    (counting bars above a single mean-minus-σ floor) is unique here.

    Entry: prev stiffness <= 90 AND today stiffness > 90 in SMA50>SMA200.
    Exit:  stiffness < 50 OR close < SMA(50).
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size

    ma100 = _sma(close, 100)
    sd60 = _stdev(close, 60)
    threshold = ma100 - 0.2 * sd60

    above_raw = (np.isfinite(threshold) & (close > threshold)).astype(float)
    above_marked = np.where(np.isfinite(threshold), above_raw, np.nan)
    stiff_raw = (
        pd.Series(above_marked).rolling(60, min_periods=60).sum().to_numpy()
    )
    stiff = stiff_raw / 60.0 * 100.0

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)

    stiff_p1 = np.concatenate(([np.nan], stiff[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 220)
    warmup[warmup_start:] = True

    valid = (
        warmup
        & np.isfinite(stiff)
        & np.isfinite(stiff_p1)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_cross_up = valid & (stiff_p1 <= 90.0) & (stiff > 90.0)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_cross_up & uptrend

    below_50 = np.isfinite(stiff) & (stiff < 50.0)
    below_sma50 = np.isfinite(sma50) & (close < sma50)
    exits = below_50 | below_sma50

    return _walk(entries, exits, close, df["date"].values)


def strat_dual_thrust_breakout(df: pd.DataFrame) -> list[Trade]:
    """Dual Thrust (Michael Chalek 1995) — open-anchored prior-range breakout.

    Chalek's classic CTA system projects the next bar's actionable level from
    the dispersion of the prior N bars and the current bar's open. With
    HH/LL/HC/LC the rolling high-of-high, low-of-low, high-of-close and
    low-of-close over the last N=4 bars (all referenced via .shift(1) so the
    range is fixed at yesterday's close — no lookahead), define

        Range    = max(HH - LC, HC - LL)
        BuyLine  = today_open + K1 * Range     (K1 = 0.5)
        SellLine = today_open - K2 * Range     (K2 = 0.5)

    Long-entry when the bar closes at or above the BuyLine while SMA50>SMA200
    (regime gate makes it asymmetric / long-only despite the symmetric
    construction). Exit when close falls below SellLine (Chalek's stop-and-
    reverse line) OR close < EMA(20) trend break.

    Distinct lineage: Dual Thrust uses *both* range-of-highs and range-of-
    closes simultaneously (max of two) — neither Donchian (highs only),
    Keltner/ATR (range of true range), nor Bollinger (σ of close) replicates
    this combined dispersion measure, and the open-anchored projection is
    unique to Chalek among sandbox strategies.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)

    N = 4
    K1 = 0.5
    K2 = 0.5

    hh = pd.Series(high).shift(1).rolling(N, min_periods=N).max().to_numpy()
    ll = pd.Series(low).shift(1).rolling(N, min_periods=N).min().to_numpy()
    hc = pd.Series(close).shift(1).rolling(N, min_periods=N).max().to_numpy()
    lc = pd.Series(close).shift(1).rolling(N, min_periods=N).min().to_numpy()

    rng = np.maximum(hh - lc, hc - ll)

    buy_line = open_ + K1 * rng
    sell_line = open_ - K2 * rng

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    valid = (
        np.isfinite(rng)
        & np.isfinite(sma50)
        & np.isfinite(sma200)
        & np.isfinite(ema20)
    )

    entries = valid & (close >= buy_line) & (sma50 > sma200)

    below_sell = np.isfinite(sell_line) & (close < sell_line)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_sell | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_chande_dynamic_momentum_index(df: pd.DataFrame) -> list[Trade]:
    """Chande/Kroll Dynamic Momentum Index — variable-period RSI fresh up-cross of 50.

    From Tushar Chande & Stanley Kroll, "The New Technical Trader" (Wiley 1994),
    chapter on adaptive indicators. Standard RSI uses a fixed 14-bar lookback.
    The DMI varies the lookback inversely with recent realized volatility:
    when volatility rises the lookback shortens (more responsive); when
    volatility settles the lookback lengthens (smoother).

        SD5  = rolling 5-bar stdev of close
        ASD  = SMA(SD5, 10)                     (avg of recent stdev)
        VI   = SD5 / ASD                         (Chande's volatility index)
        TD   = clip(round(14 / VI), 5, 30)       per-bar adaptive lookback
        DMI  = RSI(close) computed with per-bar lookback TD

    Distinct lineage among tried sandbox strategies:
      - rmi_oversold_cross: RMI uses a fixed momentum lag and fixed N — no
        volatility adaptation of the lookback itself.
      - rsi_brown_range_shift: bull/bear regime ranges of fixed-N RSI.
      - inverse_fisher_rsi: Fisher transform of fixed-N RSI.
      - connors_rsi_pullback: composite of three fixed-N components.
      - vidya_bullish_cross: variable-α EMA, not a momentum oscillator.
    None of them shrink the RSI window itself when volatility expands.

    Entry: prev DMI <= 50 AND today DMI > 50, inside SMA50 > SMA200 uptrend.
    Exit:  DMI < 50 OR close < EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size

    sd5 = _stdev(close, 5)
    asd = _sma(sd5, 10)
    vi = np.where((asd > 0) & np.isfinite(asd) & np.isfinite(sd5), sd5 / asd, np.nan)

    td_raw = np.where(vi > 0, 14.0 / vi, np.nan)
    td = np.where(np.isfinite(td_raw), np.clip(np.round(td_raw), 5, 30), np.nan)

    # Pre-compute RSI for each candidate lookback then select per bar.
    rsi_table = np.full((26, n), np.nan, dtype=float)
    for k, period in enumerate(range(5, 31)):
        rsi_table[k] = _rsi(close, period)

    dmi = np.full(n, np.nan, dtype=float)
    valid_td = np.isfinite(td)
    td_int = np.where(valid_td, td.astype(int), 0)
    idx = np.where(valid_td, td_int - 5, 0)
    rows = np.clip(idx, 0, 25)
    cols = np.arange(n)
    selected = rsi_table[rows, cols]
    dmi = np.where(valid_td, selected, np.nan)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    dmi_p1 = np.concatenate(([np.nan], dmi[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 220)
    warmup[warmup_start:] = True

    valid = (
        warmup
        & np.isfinite(dmi_p1)
        & np.isfinite(dmi)
        & np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
    )
    fresh_cross = valid & (dmi_p1 <= 50.0) & (dmi > 50.0)
    uptrend = sma50_p1 > sma200_p1
    entries = fresh_cross & uptrend

    below_50 = np.isfinite(dmi) & (dmi < 50.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_50 | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_accumulative_swing_index_cross(df: pd.DataFrame) -> list[Trade]:
    """Wilder's Accumulative Swing Index — fresh bullish cross above its 9-EMA signal.

    From J. Welles Wilder Jr., "New Concepts in Technical Trading Systems"
    (Trend Research, Greensboro NC, 1978), Chapter 8 ("Swing Index System"),
    pp. 87-96. Wilder's stated motivation: the daily close alone or daily
    range alone do not capture a security's true directional change. The
    Swing Index combines intra-bar (close - open) and inter-bar
    (close[t] - close[t-1]) impulses with a Wilder-defined volatility range
    factor R and the K-extreme (the larger distance from prior close to
    today's high or low) to produce a bounded [-100, +100] per-bar swing
    reading. Cumulating SI yields the Accumulative Swing Index (ASI), a
    price-impulse equivalent of the OBV line — but for OHLC-derived
    information rather than volume.

        N  = (close - close[1]) + 0.5·(close - open) + 0.25·(close[1] - open[1])
        K  = max(|high - close[1]|, |low - close[1]|)
        Three-case R per Wilder:
           if |H-C[1]| largest: R =  (H-C[1])   - 0.5·(L-C[1]) + 0.25·(C[1]-O[1])
           if |L-C[1]| largest: R =  (L-C[1])   - 0.5·(H-C[1]) + 0.25·(C[1]-O[1])
           else (H-L largest):  R =  (H-L)                      + 0.25·(C[1]-O[1])
        T  = SMA(TR, 20)  (per-symbol scale, replaces Wilder's futures "limit move")
        SI = clamp(50 · N / |R| · K / T, [-100, +100])
        ASI = cumsum(SI)

    Distinct lineage among the 112 tried sandbox strategies:
      - obv_ema_cross: cumulative SIGNED VOLUME line (Granville 1963), no OHLC mix.
      - chaikin_oscillator_zero_cross / twiggs_money_flow / cmf_zero_reclaim:
        Accumulation/Distribution-derived volume-weighted oscillators.
      - elder_force_index_zero_cross: |Δclose × volume|, no intra-bar OC term.
      - fisher_transform_zero_cross: hyperbolic transform of price extremes.
      - heikin_ashi_flip: synthetic candle direction, not normalized swing magnitude.
    None of these implement Wilder's K/R/T-normalized swing index nor cumulate it.

    Entry: prev ASI <= signal AND today ASI > signal, inside SMA50 > SMA200 uptrend.
    Exit:  ASI < signal OR close < EMA(20).
    """
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = close.size

    close_prev = np.concatenate(([np.nan], close[:-1]))
    open_prev = np.concatenate(([np.nan], open_[:-1]))

    abs_hc = np.abs(high - close_prev)
    abs_lc = np.abs(low - close_prev)
    hl = high - low

    case_a = (abs_hc >= abs_lc) & (abs_hc >= hl)
    case_b = (abs_lc > abs_hc) & (abs_lc >= hl)

    r_a = (high - close_prev) - 0.5 * (low - close_prev) + 0.25 * (close_prev - open_prev)
    r_b = (low - close_prev) - 0.5 * (high - close_prev) + 0.25 * (close_prev - open_prev)
    r_c = (high - low) + 0.25 * (close_prev - open_prev)

    r_raw = np.where(case_a, r_a, np.where(case_b, r_b, r_c))
    r_abs = np.abs(r_raw)

    n_term = (close - close_prev) + 0.5 * (close - open_) + 0.25 * (close_prev - open_prev)
    k = np.maximum(abs_hc, abs_lc)

    tr = np.maximum(np.maximum(hl, abs_hc), abs_lc)
    t_param = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()

    valid_si = (
        np.isfinite(r_abs)
        & (r_abs > 0)
        & np.isfinite(t_param)
        & (t_param > 0)
        & np.isfinite(n_term)
        & np.isfinite(k)
    )
    si = np.where(
        valid_si,
        50.0 * n_term / np.where(r_abs > 0, r_abs, 1.0) * (k / np.where(t_param > 0, t_param, 1.0)),
        0.0,
    )
    si = np.clip(si, -100.0, 100.0)
    si = np.where(np.isfinite(si), si, 0.0)

    asi = np.cumsum(si)
    signal = _ema(asi, 9)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    asi_p1 = np.concatenate(([np.nan], asi[:-1]))
    sig_p1 = np.concatenate(([np.nan], signal[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 220)
    warmup[warmup_start:] = True

    fresh_cross = (
        np.isfinite(asi_p1)
        & np.isfinite(sig_p1)
        & np.isfinite(asi)
        & np.isfinite(signal)
        & (asi_p1 <= sig_p1)
        & (asi > signal)
    )
    uptrend = (
        np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
        & (sma50_p1 > sma200_p1)
    )
    entries = warmup & fresh_cross & uptrend

    below_signal = np.isfinite(asi) & np.isfinite(signal) & (asi < signal)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_signal | below_ema20

    return _walk(entries, exits, close, df["date"].values)


def strat_trend_trigger_factor(df: pd.DataFrame) -> list[Trade]:
    """Trend Trigger Factor (M.H. Pee, S&C Dec 2004) — fresh up-cross above +100.

    TTF compares the buying-power range of the current N-bar window against
    the selling-power range of the prior N-bar window. With N=15:

        BuyPower  = HighestHigh(0..N-1)  - LowestLow(N..2N-1)
        SellPower = HighestHigh(N..2N-1) - LowestLow(0..N-1)
        TTF       = 100 * (BuyPower - SellPower) / (0.5 * (BuyPower + SellPower))

    Pee's published interpretation: TTF > +100 marks an established uptrend,
    TTF < -100 marks a downtrend. The fresh cross up through +100 captures
    the moment the rolling high-range of the most recent N bars decisively
    overtakes the comparable window N bars ago.

    Distinct from existing sandbox strategies:
      - Trend Intensity Index (Pee 2002) sums magnitude-weighted deviations
        from an SMA — TTF compares HHV/LLV ranges across windows.
      - Aroon ranks bar position of HHV/LLV — TTF works with raw range arithmetic.
      - Random Walk Index measures range vs sqrt(N)·ATR — TTF subtracts
        windowed BP and SP scaled by their average.
      - Donchian breakout uses a single rolling high/low — TTF differences
        two adjacent N-windows of HHV and LLV.

    Entry: prev TTF <= 100 AND today TTF > 100, gated by SMA50 > SMA200.
    Exit:  TTF < -100 OR close < EMA20.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = close.size
    N = 15

    hh = pd.Series(high).rolling(N, min_periods=N).max().to_numpy()
    ll = pd.Series(low).rolling(N, min_periods=N).min().to_numpy()

    if n > N:
        hh_prev = np.concatenate((np.full(N, np.nan), hh[:-N]))
        ll_prev = np.concatenate((np.full(N, np.nan), ll[:-N]))
    else:
        hh_prev = np.full(n, np.nan)
        ll_prev = np.full(n, np.nan)

    bp = hh - ll_prev
    sp = hh_prev - ll
    denom = 0.5 * (bp + sp)

    valid_ttf = (
        np.isfinite(bp)
        & np.isfinite(sp)
        & np.isfinite(denom)
        & (np.abs(denom) > 1e-12)
    )
    ttf = np.where(valid_ttf, 100.0 * (bp - sp) / np.where(np.abs(denom) > 1e-12, denom, 1.0), np.nan)

    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)
    ema20 = _ema(close, 20)

    ttf_p1 = np.concatenate(([np.nan], ttf[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 220)
    warmup[warmup_start:] = True

    fresh_cross_up = (
        np.isfinite(ttf_p1)
        & np.isfinite(ttf)
        & (ttf_p1 <= 100.0)
        & (ttf > 100.0)
    )
    uptrend = (
        np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
        & (sma50_p1 > sma200_p1)
    )
    entries = warmup & fresh_cross_up & uptrend

    below_neg100 = np.isfinite(ttf) & (ttf < -100.0)
    below_ema20 = np.isfinite(ema20) & (close < ema20)
    exits = below_neg100 | below_ema20

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
    "cmf_zero_reclaim": strat_cmf_zero_reclaim,
    "aroon_cross_trend": strat_aroon_cross_trend,
    "heikin_ashi_flip": strat_heikin_ashi_flip,
    "ichimoku_kumo_breakout": strat_ichimoku_kumo_breakout,
    "parabolic_sar_flip_trend": strat_parabolic_sar_flip_trend,
    "vortex_bullish_cross": strat_vortex_bullish_cross,
    "rvi_signal_cross": strat_rvi_signal_cross,
    "fisher_transform_zero_cross": strat_fisher_transform_zero_cross,
    "trix_signal_cross": strat_trix_signal_cross,
    "hma_bullish_cross": strat_hma_bullish_cross,
    "kama_cross_trend": strat_kama_cross_trend,
    "schaff_trend_cycle": strat_schaff_trend_cycle,
    "coppock_curve_zero_cross": strat_coppock_curve_zero_cross,
    "connors_rsi_pullback": strat_connors_rsi_pullback,
    "elder_force_index_zero_cross": strat_elder_force_index_zero_cross,
    "awesome_oscillator_saucer": strat_awesome_oscillator_saucer,
    "pring_kst_signal_cross": strat_pring_kst_signal_cross,
    "mfi_oversold_recovery": strat_mfi_oversold_recovery,
    "tsi_signal_cross": strat_tsi_signal_cross,
    "stochastic_oversold_recovery": strat_stochastic_oversold_recovery,
    "cci_oversold_recovery": strat_cci_oversold_recovery,
    "chaikin_oscillator_zero_cross": strat_chaikin_oscillator_zero_cross,
    "obv_ema_cross": strat_obv_ema_cross,
    "ultimate_oscillator_oversold": strat_ultimate_oscillator_oversold,
    "nvi_fosback_trend": strat_nvi_fosback_trend,
    "choppiness_regime_shift": strat_choppiness_regime_shift,
    "inverse_fisher_rsi": strat_inverse_fisher_rsi,
    "minervini_vcp_breakout": strat_minervini_vcp_breakout,
    "dpo_zero_cross": strat_dpo_zero_cross,
    "clenow_momentum_score": strat_clenow_momentum_score,
    "bollinger_pctb_reversion": strat_bollinger_pctb_reversion,
    "anchored_vwap_reclaim": strat_anchored_vwap_reclaim,
    "connors_double_7s": strat_connors_double_7s,
    "raschke_holy_grail": strat_raschke_holy_grail,
    "vwap_zscore_reversion": strat_vwap_zscore_reversion,
    "guppy_gmma_compression_release": strat_guppy_gmma_compression_release,
    "hammer_pin_bar_uptrend": strat_hammer_pin_bar_uptrend,
    "linreg_slope_signchange": strat_linreg_slope_signchange,
    "td_sequential_buy_setup": strat_td_sequential_buy_setup,
    "keltner_channel_breakout": strat_keltner_channel_breakout,
    "three_white_soldiers": strat_three_white_soldiers,
    "qstick_zero_cross": strat_qstick_zero_cross,
    "bullish_engulfing_pullback": strat_bullish_engulfing_pullback,
    "mass_index_reversal_bulge": strat_mass_index_reversal_bulge,
    "stoch_rsi_oversold_cross": strat_stoch_rsi_oversold_cross,
    "cmo_oversold_recovery": strat_cmo_oversold_recovery,
    "wyckoff_spring_reclaim": strat_wyckoff_spring_reclaim,
    "williams_fractal_breakout": strat_williams_fractal_breakout,
    "weinstein_stage2_breakout": strat_weinstein_stage2_breakout,
    "klinger_volume_oscillator_signal_cross": strat_klinger_volume_oscillator_signal_cross,
    "demarker_oversold_reclaim": strat_demarker_oversold_reclaim,
    "range_filter_buy": strat_range_filter_buy,
    "rsi_brown_range_shift": strat_rsi_brown_range_shift,
    "vidya_bullish_cross": strat_vidya_bullish_cross,
    "acceleration_bands_breakout": strat_acceleration_bands_breakout,
    "qqe_bullish_cross": strat_qqe_bullish_cross,
    "elder_ray_bear_reclaim": strat_elder_ray_bear_reclaim,
    "morning_star_pullback": strat_morning_star_pullback,
    "polarized_fractal_efficiency": strat_polarized_fractal_efficiency,
    "wavetrend_lb_oversold_cross": strat_wavetrend_lb_oversold_cross,
    "alma_bullish_cross": strat_alma_bullish_cross,
    "pretty_good_oscillator_zero_cross": strat_pretty_good_oscillator_zero_cross,
    "ehlers_cog_signal_cross": strat_ehlers_cog_signal_cross,
    "smi_blau_oversold_cross": strat_smi_blau_oversold_cross,
    "gann_hilo_activator_flip": strat_gann_hilo_activator_flip,
    "random_walk_index_bullish_cross": strat_random_walk_index_bullish_cross,
    "premier_stochastic_oscillator": strat_premier_stochastic_oscillator,
    "mcginley_dynamic_cross": strat_mcginley_dynamic_cross,
    "frama_bullish_cross": strat_frama_bullish_cross,
    "macd_v_oversold_reclaim": strat_macd_v_oversold_reclaim,
    "mama_fama_cross": strat_mama_fama_cross,
    "andean_oscillator_bull_cross": strat_andean_oscillator_bull_cross,
    "tillson_t3_cross": strat_tillson_t3_cross,
    "ehlers_laguerre_rsi": strat_ehlers_laguerre_rsi,
    "rmi_oversold_cross": strat_rmi_oversold_cross,
    "bill_williams_alligator_awake": strat_bill_williams_alligator_awake,
    "williams_ac_zero_acceleration": strat_williams_ac_zero_acceleration,
    "ehlers_roofing_filter": strat_ehlers_roofing_filter,
    "ehlers_trendflex": strat_ehlers_trendflex,
    "twiggs_money_flow": strat_twiggs_money_flow,
    "elder_impulse_bull": strat_elder_impulse_bull,
    "vw_macd_signal_cross": strat_vw_macd_signal_cross,
    "chande_forecast_oscillator": strat_chande_forecast_oscillator,
    "tema_bullish_cross": strat_tema_bullish_cross,
    "disparity_index_zero_cross": strat_disparity_index_zero_cross,
    "bressert_dss_oversold_cross": strat_bressert_dss_oversold_cross,
    "chandelier_exit_reclaim": strat_chandelier_exit_reclaim,
    "trend_intensity_index": strat_trend_intensity_index,
    "volume_flow_indicator": strat_volume_flow_indicator,
    "pmo_signal_cross": strat_pmo_signal_cross,
    "adaptive_price_zone_breakout": strat_adaptive_price_zone_breakout,
    "stiffness_indicator": strat_stiffness_indicator,
    "dual_thrust_breakout": strat_dual_thrust_breakout,
    "chande_dynamic_momentum_index": strat_chande_dynamic_momentum_index,
    "accumulative_swing_index_cross": strat_accumulative_swing_index_cross,
    "trend_trigger_factor": strat_trend_trigger_factor,
}
