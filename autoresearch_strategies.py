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



def strat_composite_index_brown(df: pd.DataFrame) -> list[Trade]:
    """Constance Brown Composite Index (Brown, "Technical Analysis for the
    Trading Professional", 1999). Designed to fix RSI's failure to print
    divergences at major reversals. CI = RSI_Momentum(9) + SMA3(RSI14), where
    RSI_Momentum = RSI14 - RSI14[9]. Long on a fresh bullish cross of CI above
    its 13-bar SMA signal line, gated by SMA50>SMA200 trend filter. Exit on
    bearish CI/signal cross-down.
    """
    close = df["close"].to_numpy(dtype=float)
    n = close.size
    if n < 50:
        return []
    rsi14 = _rsi(close, 14)
    rsi14_lag9 = np.concatenate((np.full(9, np.nan), rsi14[:-9]))
    rsi_mom9 = rsi14 - rsi14_lag9
    sma3_rsi = _sma(rsi14, 3)
    ci = rsi_mom9 + sma3_rsi
    ci_signal = _sma(ci, 13)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)

    ci_p1 = np.concatenate(([np.nan], ci[:-1]))
    sig_p1 = np.concatenate(([np.nan], ci_signal[:-1]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    sma200_p1 = np.concatenate(([np.nan], sma200[:-1]))

    cross_up = (
        np.isfinite(ci_p1)
        & np.isfinite(sig_p1)
        & np.isfinite(ci)
        & np.isfinite(ci_signal)
        & (ci_p1 <= sig_p1)
        & (ci > ci_signal)
    )
    uptrend = (
        np.isfinite(sma50_p1)
        & np.isfinite(sma200_p1)
        & (sma50_p1 > sma200_p1)
    )
    warmup = np.zeros(n, dtype=bool)
    warmup_start = min(n, 220)
    warmup[warmup_start:] = True

    entries = warmup & cross_up & uptrend

    cross_dn = (
        np.isfinite(ci_p1)
        & np.isfinite(sig_p1)
        & np.isfinite(ci)
        & np.isfinite(ci_signal)
        & (ci_p1 >= sig_p1)
        & (ci < ci_signal)
    )
    exits = cross_dn

    return _walk(entries, exits, close, df["date"].values)


def strat_sushi_roll_reversal(df: pd.DataFrame) -> list[Trade]:
    """Mark Fisher Sushi Roll Reversal (Fisher 'The Logical Trader' 2002).

    Two consecutive 5-bar windows form the pattern:
      old = bars t-9..t-5 (oldest 5)
      new = bars t-4..t   (most recent 5 ending at current bar)
    Bullish 5-bar outside-up reversal triggers when the new window engulfs
    the old on BOTH sides AND closes above the old high:
      max(high[new]) > max(high[old])  -- higher high
      min(low[new])  < min(low[old])   -- lower low
      close[t] > max(high[old])        -- reclaim above old window high

    Filters: prior 5-bar window was actually trending down
    (close[t-5] < close[t-10]) and long-term uptrend close > SMA(100), so
    we buy genuine reversals inside structural uptrends.
    Exit: close drops below the prior 5-bar lowest low (5-bar trailing floor).
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = close.size
    if n < 120:
        return []

    high_s = pd.Series(high)
    low_s = pd.Series(low)

    # New 5-bar window ending at t (inclusive)
    high_new = high_s.rolling(5, min_periods=5).max().to_numpy()
    low_new = low_s.rolling(5, min_periods=5).min().to_numpy()
    # Old 5-bar window ending at t-5 (i.e. shifted by 5)
    high_old = high_s.shift(5).rolling(5, min_periods=5).max().to_numpy()
    low_old = low_s.shift(5).rolling(5, min_periods=5).min().to_numpy()

    sma100 = _sma(close, 100)

    close_t5 = np.concatenate((np.full(5, np.nan), close[:-5]))
    close_t10 = np.concatenate((np.full(10, np.nan), close[:-10]))

    engulf = (
        np.isfinite(high_new)
        & np.isfinite(low_new)
        & np.isfinite(high_old)
        & np.isfinite(low_old)
        & (high_new > high_old)
        & (low_new < low_old)
        & (close > high_old)
    )
    prior_down = (
        np.isfinite(close_t5)
        & np.isfinite(close_t10)
        & (close_t5 < close_t10)
    )
    uptrend = np.isfinite(sma100) & (close > sma100)

    entries = engulf & prior_down & uptrend

    # Exit when close drops below prior bar's 5-bar trailing low
    low_new_prev = pd.Series(low_new).shift(1).to_numpy()
    exits = np.isfinite(low_new_prev) & (close < low_new_prev)

    return _walk(entries, exits, close, df["date"].values)




def strat_vpci_bullish_cross(df: pd.DataFrame) -> list[Trade]:
    """Volume Price Confirmation Indicator (Buff Dormeier 2007, NAAIM Wagner).

    VPCI is a multiplicative composite of three sub-components designed to
    confirm trend conviction by reconciling price action with volume flow:

        VWMA_n = sum(close*vol, n) / sum(vol, n)
        VPC    = VWMA_long  - SMA(close, long)        (price-volume confirm)
        VPR    = VWMA_short / SMA(close, short)        (price-volume ratio)
        VM     = SMA(vol, short) / SMA(vol, long)       (volume multiplier)
        VPCI   = VPC * VPR * VM

    With short=5, long=25 (Dormeier defaults). When VPCI is rising and crosses
    above its own SMA(8) signal line in an established uptrend, price moves
    are being confirmed by volume — a high-conviction long signal. We additionally
    require VPCI > 0 to avoid bear-market head-fakes. Exit on bearish cross of
    signal line or close < SMA(50) trend break.

    Reference: Buff P. Dormeier, "Investing with Volume Analysis" (2011 / 2007
    NAAIM Wagner Award paper). Distinct from vw_macd_signal_cross (additive
    EMA-difference of VWMA) — VPCI is a multiplicative product of three terms.
    """
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    n_short = 5
    n_long = 25

    pv = close * volume
    pv_s = pd.Series(pv)
    vol_s = pd.Series(volume)
    close_s = pd.Series(close)

    vwma_short = (
        pv_s.rolling(n_short, min_periods=n_short).sum()
        / vol_s.rolling(n_short, min_periods=n_short).sum()
    ).to_numpy()
    vwma_long = (
        pv_s.rolling(n_long, min_periods=n_long).sum()
        / vol_s.rolling(n_long, min_periods=n_long).sum()
    ).to_numpy()
    sma_short = close_s.rolling(n_short, min_periods=n_short).mean().to_numpy()
    sma_long = close_s.rolling(n_long, min_periods=n_long).mean().to_numpy()
    vsma_short = vol_s.rolling(n_short, min_periods=n_short).mean().to_numpy()
    vsma_long = vol_s.rolling(n_long, min_periods=n_long).mean().to_numpy()

    vpc = vwma_long - sma_long
    with np.errstate(divide="ignore", invalid="ignore"):
        vpr = np.where(sma_short > 0, vwma_short / sma_short, np.nan)
        vm = np.where(vsma_long > 0, vsma_short / vsma_long, np.nan)
    vpci = vpc * vpr * vm

    sig = pd.Series(vpci).rolling(8, min_periods=8).mean().to_numpy()
    sma50 = _sma(close, 50)

    vpci_p1 = np.concatenate(([np.nan], vpci[:-1]))
    vpci_p2 = np.concatenate(([np.nan, np.nan], vpci[:-2]))
    sig_p1 = np.concatenate(([np.nan], sig[:-1]))
    sig_p2 = np.concatenate(([np.nan, np.nan], sig[:-2]))
    sma50_p1 = np.concatenate(([np.nan], sma50[:-1]))
    close_p1 = np.concatenate(([np.nan], close[:-1]))

    valid_entry = (
        np.isfinite(vpci_p1)
        & np.isfinite(vpci_p2)
        & np.isfinite(sig_p1)
        & np.isfinite(sig_p2)
        & np.isfinite(sma50_p1)
        & np.isfinite(close_p1)
    )

    fresh_up = (vpci_p1 > sig_p1) & (vpci_p2 <= sig_p2)
    above_zero = vpci_p1 > 0
    uptrend = close_p1 > sma50_p1

    entries = valid_entry & fresh_up & above_zero & uptrend

    fresh_down = (
        np.isfinite(vpci_p1)
        & np.isfinite(sig_p1)
        & np.isfinite(vpci_p2)
        & np.isfinite(sig_p2)
        & (vpci_p1 < sig_p1)
        & (vpci_p2 >= sig_p2)
    )
    below_sma50 = np.isfinite(sma50) & (close < sma50)
    exits = fresh_down | below_sma50

    return _walk(entries, exits, close, df["date"].values)



NEW_STRATEGIES: dict = {
    "example_rsi_mean_revert": strat_example_rsi_mean_revert,
    "donchian_20_10_trend": strat_donchian_20_10_trend,
    "squeeze_breakout": strat_squeeze_breakout,
    "pocket_pivot": strat_pocket_pivot,
    "parabolic_sar_flip_trend": strat_parabolic_sar_flip_trend,
    "kama_cross_trend": strat_kama_cross_trend,
    "pring_kst_signal_cross": strat_pring_kst_signal_cross,
    "chaikin_oscillator_zero_cross": strat_chaikin_oscillator_zero_cross,
    "choppiness_regime_shift": strat_choppiness_regime_shift,
    "dpo_zero_cross": strat_dpo_zero_cross,
    "linreg_slope_signchange": strat_linreg_slope_signchange,
    "keltner_channel_breakout": strat_keltner_channel_breakout,
    "qstick_zero_cross": strat_qstick_zero_cross,
    "klinger_volume_oscillator_signal_cross": strat_klinger_volume_oscillator_signal_cross,
    "demarker_oversold_reclaim": strat_demarker_oversold_reclaim,
    "range_filter_buy": strat_range_filter_buy,
    "vidya_bullish_cross": strat_vidya_bullish_cross,
    "acceleration_bands_breakout": strat_acceleration_bands_breakout,
    "qqe_bullish_cross": strat_qqe_bullish_cross,
    "alma_bullish_cross": strat_alma_bullish_cross,
    "pretty_good_oscillator_zero_cross": strat_pretty_good_oscillator_zero_cross,
    "ehlers_cog_signal_cross": strat_ehlers_cog_signal_cross,
    "gann_hilo_activator_flip": strat_gann_hilo_activator_flip,
    "premier_stochastic_oscillator": strat_premier_stochastic_oscillator,
    "mcginley_dynamic_cross": strat_mcginley_dynamic_cross,
    "frama_bullish_cross": strat_frama_bullish_cross,
    "mama_fama_cross": strat_mama_fama_cross,
    "tillson_t3_cross": strat_tillson_t3_cross,
    "ehlers_laguerre_rsi": strat_ehlers_laguerre_rsi,
    "bill_williams_alligator_awake": strat_bill_williams_alligator_awake,
    "williams_ac_zero_acceleration": strat_williams_ac_zero_acceleration,
    "twiggs_money_flow": strat_twiggs_money_flow,
    "elder_impulse_bull": strat_elder_impulse_bull,
    "chande_forecast_oscillator": strat_chande_forecast_oscillator,
    "disparity_index_zero_cross": strat_disparity_index_zero_cross,
    "adaptive_price_zone_breakout": strat_adaptive_price_zone_breakout,
    "chande_dynamic_momentum_index": strat_chande_dynamic_momentum_index,
    "accumulative_swing_index_cross": strat_accumulative_swing_index_cross,
    "trend_trigger_factor": strat_trend_trigger_factor,
    "composite_index_brown": strat_composite_index_brown,
    "sushi_roll_reversal": strat_sushi_roll_reversal,
    "vpci_bullish_cross": strat_vpci_bullish_cross,
}
