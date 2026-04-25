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
}
