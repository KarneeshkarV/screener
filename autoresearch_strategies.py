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
}
