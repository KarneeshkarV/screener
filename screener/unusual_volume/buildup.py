"""Build-up detector — pre-breakout accumulation over a multi-week window.

A "build-up" is a 2–8 week phase where smart money quietly accumulates a
stock before the breakout. Five fingerprints, each scored 0..1:

  1. Range compression          — ATR is contracting (now / window-max < 0.6)
                                  and/or Bollinger Band width is squeezed
                                  below half its own 20-bar SMA.
  2. Up/down volume asymmetry   — volume on up days outweighs volume on down
                                  days over the window (ratio >= 1.5).
  3. Higher lows                — linear-regression slope of `low` over the
                                  window is positive AND the last three
                                  swing-lows are ascending.
  4. Sustained delivery (India) — rolling mean of DELIV_PER >= 45 AND >=60%
                                  of bars hit 50%. Skipped (sub-score=None)
                                  when no delivery data is supplied.
  5. Close-near-high            — mean intraday "absorption" score
                                  (close-low)/(high-low) >= 0.65.

The composite score is the sum of fingerprint scores divided by the count
of fingerprints actually evaluated, so it is always 0..1 regardless of
whether delivery is in play.

This module operates on the same OHLCV panel the rest of unusual_volume/
already loads — it adds no new data dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_WINDOW = 20
DEFAULT_MIN_SCORE = 0.6

ATR_LEN = 14
BB_LEN = 20
BB_MULT = 2.0

# Fingerprint thresholds — picked from the build-up literature (NR7 / TTM
# squeeze / volatility-compression breakout playbooks). All thresholds are
# soft: the sub-score scales linearly past the threshold, so a marginal
# pass shouldn't dominate the composite.
ATR_RATIO_THRESHOLD = 0.60      # atr_now / max(atr, window) — lower = tighter
BB_SQUEEZE_THRESHOLD = 0.50     # bb_width_now / SMA(bb_width, 20)
UPDOWN_VOL_THRESHOLD = 1.5      # up_vol / down_vol
SLOPE_NORM_FLOOR = 0.0          # higher-lows slope must be > 0
DELIVERY_MEAN_THRESHOLD = 45.0  # rolling mean DELIV_PER
DELIVERY_HIT_THRESHOLD = 0.60   # fraction of bars with DELIV_PER >= 50
ABSORPTION_THRESHOLD = 0.65     # mean (close-low)/(high-low)


@dataclass
class BuildupScore:
    symbol: str
    as_of: date
    window: int
    range_compression: Optional[float]
    updown_volume: Optional[float]
    higher_lows: Optional[float]
    sustained_delivery: Optional[float]
    close_near_high: Optional[float]
    composite: float
    flags: list[str] = field(default_factory=list)
    # Diagnostic raw values — handy when triaging a score in the journal.
    atr_ratio: Optional[float] = None
    bb_squeeze_ratio: Optional[float] = None
    updown_ratio: Optional[float] = None
    low_slope_norm: Optional[float] = None
    delivery_mean: Optional[float] = None
    delivery_hit_rate: Optional[float] = None
    absorption_mean: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat() if isinstance(self.as_of, date) else str(self.as_of)
        return d


def _atr(df: pd.DataFrame, length: int = ATR_LEN) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing — the canonical ATR; matches Pine `ta.atr`.
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _bb_width(close: pd.Series, length: int = BB_LEN, mult: float = BB_MULT) -> pd.Series:
    basis = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    upper = basis + mult * sd
    lower = basis - mult * sd
    # Normalise by basis so the ratio is comparable across price levels.
    return (upper - lower) / basis.replace(0, np.nan)


def _score_range_compression(df: pd.DataFrame, window: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if len(df) < max(BB_LEN, ATR_LEN) + window:
        return None, None, None
    atr = _atr(df)
    win = atr.iloc[-window:]
    if win.isna().any() or win.max() <= 0:
        return None, None, None
    atr_ratio = float(atr.iloc[-1] / win.max())

    bbw = _bb_width(df["close"].astype(float))
    bbw_sma = bbw.rolling(BB_LEN, min_periods=BB_LEN).mean()
    if pd.isna(bbw.iloc[-1]) or pd.isna(bbw_sma.iloc[-1]) or bbw_sma.iloc[-1] <= 0:
        bb_ratio = None
    else:
        bb_ratio = float(bbw.iloc[-1] / bbw_sma.iloc[-1])

    # Sub-score: lower ratios = tighter compression. Map [0..threshold] -> [1..0]
    # then take the better (tighter) of the two engines.
    atr_sub = max(0.0, min(1.0, (ATR_RATIO_THRESHOLD - atr_ratio) / ATR_RATIO_THRESHOLD + 0.5))
    if bb_ratio is None:
        bb_sub = 0.0
    else:
        bb_sub = max(0.0, min(1.0, (BB_SQUEEZE_THRESHOLD - bb_ratio) / BB_SQUEEZE_THRESHOLD + 0.5))
    sub = max(atr_sub, bb_sub)
    return float(sub), atr_ratio, bb_ratio


def _score_updown_volume(df: pd.DataFrame, window: int) -> tuple[Optional[float], Optional[float]]:
    win = df.iloc[-window:]
    if len(win) < window:
        return None, None
    close = win["close"].astype(float).values
    open_ = win["open"].astype(float).values
    vol = win["volume"].astype(float).values
    up_mask = close > open_
    down_mask = close < open_
    up_vol = vol[up_mask].sum()
    down_vol = vol[down_mask].sum()
    if down_vol <= 0 and up_vol <= 0:
        return None, None
    if down_vol <= 0:
        ratio = float("inf")
    else:
        ratio = float(up_vol / down_vol)
    # Map ratio to 0..1: 1.0 (parity) -> 0, threshold (1.5) -> 0.5, 3.0+ -> 1.
    if not np.isfinite(ratio):
        sub = 1.0
    else:
        sub = max(0.0, min(1.0, (ratio - 1.0) / 2.0))
    return sub, (None if not np.isfinite(ratio) else ratio)


def _score_higher_lows(df: pd.DataFrame, window: int) -> tuple[Optional[float], Optional[float]]:
    win = df.iloc[-window:]
    if len(win) < window:
        return None, None
    lows = win["low"].astype(float).values
    if (lows <= 0).any():
        return None, None
    # Slope of log(low) so the score is scale-free; normalise to per-bar %.
    x = np.arange(len(lows), dtype=float)
    y = np.log(lows)
    if np.allclose(y, y[0]):
        return 0.0, 0.0
    slope, _ = np.polyfit(x, y, 1)
    slope_pct = float(slope * 100.0)  # rough %/bar
    # Confirm with last three swing-lows being ascending.
    swing_lows = _swing_lows(lows)
    last3_ok = len(swing_lows) >= 3 and swing_lows[-1] > swing_lows[-2] > swing_lows[-3]
    if slope_pct <= SLOPE_NORM_FLOOR:
        return 0.0, slope_pct
    # 0.3%/bar over a 20-bar window ≈ 6% rise — call that a full score.
    base = max(0.0, min(1.0, slope_pct / 0.3))
    sub = base * (1.0 if last3_ok else 0.7)
    return float(sub), slope_pct


def _swing_lows(lows: np.ndarray, k: int = 2) -> list[float]:
    """Return values that are local minima with k bars on each side."""
    out: list[float] = []
    for i in range(k, len(lows) - k):
        seg = lows[i - k : i + k + 1]
        if lows[i] == seg.min() and (seg == lows[i]).sum() == 1:
            out.append(float(lows[i]))
    return out


def _score_sustained_delivery(
    delivery_panel: Optional[pd.DataFrame],
    symbol: str,
    as_of: date,
    window: int,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if delivery_panel is None or delivery_panel.empty:
        return None, None, None
    sym = symbol.upper()
    df = delivery_panel[delivery_panel["SYMBOL"].str.upper() == sym].copy()
    if df.empty:
        return None, None, None
    df = df.sort_values("date")
    df = df[df["date"] <= as_of]
    win = df.tail(window)
    if len(win) < max(5, window // 2):
        return None, None, None
    pct = win["DELIV_PER"].astype(float).dropna()
    if pct.empty:
        return None, None, None
    mean_pct = float(pct.mean())
    hit_rate = float((pct >= 50.0).sum() / len(pct))
    # Sub-score: blended mean + hit rate.
    mean_sub = max(0.0, min(1.0, (mean_pct - 30.0) / 40.0))  # 30 -> 0, 70 -> 1
    hit_sub = max(0.0, min(1.0, (hit_rate - 0.3) / 0.6))     # 30% -> 0, 90% -> 1
    sub = (mean_sub + hit_sub) / 2.0
    return float(sub), mean_pct, hit_rate


def _score_close_near_high(df: pd.DataFrame, window: int) -> tuple[Optional[float], Optional[float]]:
    win = df.iloc[-window:]
    if len(win) < window:
        return None, None
    high = win["high"].astype(float)
    low = win["low"].astype(float)
    close = win["close"].astype(float)
    rng = (high - low).replace(0, np.nan)
    absorption = (close - low) / rng
    absorption = absorption.dropna()
    if absorption.empty:
        return None, None
    mean_abs = float(absorption.mean())
    # 0.5 (mid-range) -> 0; 0.85 -> 1.
    sub = max(0.0, min(1.0, (mean_abs - 0.5) / 0.35))
    return float(sub), mean_abs


def compute_buildup_score(
    symbol: str,
    bars: pd.DataFrame,
    as_of: date,
    delivery_panel: Optional[pd.DataFrame] = None,
    window: int = DEFAULT_WINDOW,
) -> Optional[BuildupScore]:
    """Score one ticker. Returns None when the bar history is too short."""
    if bars is None or bars.empty:
        return None
    df = bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"]).values))
        else:
            return None
    df = df.sort_index()
    as_of_ts = pd.Timestamp(as_of).normalize()
    df = df[df.index <= as_of_ts]
    if len(df) < max(BB_LEN, ATR_LEN) + window:
        return None

    rng_sub, atr_ratio, bb_ratio = _score_range_compression(df, window)
    upd_sub, upd_ratio = _score_updown_volume(df, window)
    hl_sub, slope_pct = _score_higher_lows(df, window)
    del_sub, del_mean, del_hit = _score_sustained_delivery(
        delivery_panel, symbol, as_of, window
    )
    cnh_sub, abs_mean = _score_close_near_high(df, window)

    subs = [rng_sub, upd_sub, hl_sub, del_sub, cnh_sub]
    populated = [s for s in subs if s is not None]
    if not populated:
        return None
    composite = sum(populated) / len(populated)

    flags: list[str] = []
    if rng_sub is not None and rng_sub >= 0.6:
        flags.append("compression")
    if upd_sub is not None and upd_sub >= 0.5:
        flags.append("up_vol_dominant")
    if hl_sub is not None and hl_sub >= 0.5:
        flags.append("higher_lows")
    if del_sub is not None and del_sub >= 0.5:
        flags.append("sustained_delivery")
    if cnh_sub is not None and cnh_sub >= 0.5:
        flags.append("close_near_high")

    return BuildupScore(
        symbol=symbol.upper(),
        as_of=as_of,
        window=window,
        range_compression=rng_sub,
        updown_volume=upd_sub,
        higher_lows=hl_sub,
        sustained_delivery=del_sub,
        close_near_high=cnh_sub,
        composite=round(composite, 4),
        flags=flags,
        atr_ratio=None if atr_ratio is None else round(atr_ratio, 4),
        bb_squeeze_ratio=None if bb_ratio is None else round(bb_ratio, 4),
        updown_ratio=None if upd_ratio is None else round(upd_ratio, 4),
        low_slope_norm=None if slope_pct is None else round(slope_pct, 4),
        delivery_mean=None if del_mean is None else round(del_mean, 2),
        delivery_hit_rate=None if del_hit is None else round(del_hit, 4),
        absorption_mean=None if abs_mean is None else round(abs_mean, 4),
    )


def scan_buildups(
    bars_by_symbol: dict[str, pd.DataFrame],
    as_of: date,
    delivery_panel: Optional[pd.DataFrame] = None,
    window: int = DEFAULT_WINDOW,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[BuildupScore]:
    """Score every ticker; return those above ``min_score`` sorted desc."""
    out: list[BuildupScore] = []
    for sym, bars in bars_by_symbol.items():
        score = compute_buildup_score(
            sym, bars, as_of, delivery_panel=delivery_panel, window=window
        )
        if score is None:
            continue
        if score.composite >= min_score:
            out.append(score)
    out.sort(key=lambda s: s.composite, reverse=True)
    return out
