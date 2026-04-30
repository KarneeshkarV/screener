"""Per-bar volume-anomaly detector.

For a given as-of date, computes RVOL across multiple windows, a 90-day
volume Z-score, and the 252-day percentile rank, then emits one ``Event``
per ticker that crosses the configured thresholds. Direction (BUYING /
SELLING / CHURN / REVERSAL) and strength (MODERATE / HIGH / EXTREME) are
attached via ``classify``.

The detector is deliberately decoupled from data loading and India delivery
overlays — those live in ``cli`` / ``delivery``. This module only sees a
``pd.DataFrame`` of OHLCV per ticker.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .classify import Direction, Strength, classify_direction, classify_strength


# Threshold tiers — moderate is the entry filter; the others stamp strength.
DEFAULT_MIN_RVOL = 2.0
DEFAULT_MIN_Z = 2.0
HIGH_RVOL = 3.0
HIGH_Z = 2.5
EXTREME_RVOL = 5.0
EXTREME_Z = 3.5


@dataclass
class Event:
    symbol: str
    date: date
    close: float
    pct_change: float
    volume: float
    avg_volume_20d: float
    rvol: float
    rvol_5d: float
    rvol_50d: float
    rvol_90d: float
    z_score: float
    pct_rank_252d: float
    direction: Direction
    strength: Strength
    # India-only — populated by the delivery overlay, default None elsewhere.
    delivery_qty: Optional[float] = None
    delivery_pct: Optional[float] = None
    delivery_rvol: Optional[float] = None
    conviction_score: Optional[float] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    notes: str = ""
    # Build-up overlay — populated by buildup.scan_buildups, default None.
    buildup_score: Optional[float] = None
    buildup_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.isoformat() if isinstance(self.date, date) else str(self.date)
        return d


def _rolling_pct_rank(series: pd.Series, window: int) -> pd.Series:
    """For each row, return the percentile rank of the row's value within
    its trailing ``window`` of observations (last value inclusive)."""
    s = series.astype(float)
    return s.rolling(window, min_periods=window).apply(
        lambda x: (x <= x.iloc[-1]).sum() / len(x), raw=False
    )


def _safe_ratio(num: float, denom: float) -> float:
    if denom is None or denom == 0 or pd.isna(denom):
        return float("nan")
    return float(num) / float(denom)


def detect_ticker(
    symbol: str,
    bars: pd.DataFrame,
    as_of: date,
    min_rvol: float = DEFAULT_MIN_RVOL,
    min_z: float = DEFAULT_MIN_Z,
) -> Optional[Event]:
    """Return an ``Event`` for ``symbol`` on ``as_of`` if it crosses thresholds.

    ``bars`` must be a DataFrame with a DatetimeIndex (or a ``date`` column)
    and lower-case OHLCV columns. Thresholds are evaluated on the bar at
    ``as_of`` (or the last bar on/before it if ``as_of`` is a non-trading
    day).
    """
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
    if df.empty:
        return None

    vol = df["volume"].astype(float)
    sma_5 = vol.rolling(5, min_periods=5).mean()
    sma_20 = vol.rolling(20, min_periods=20).mean()
    sma_50 = vol.rolling(50, min_periods=50).mean()
    sma_90 = vol.rolling(90, min_periods=90).mean()
    std_90 = vol.rolling(90, min_periods=90).std(ddof=0)
    pct_rank_252 = _rolling_pct_rank(vol, 252)

    # Use prior-bar averages so the spike day isn't included in its own mean —
    # this matches how RVOL is conventionally read.
    sma_5_prev = sma_5.shift(1)
    sma_20_prev = sma_20.shift(1)
    sma_50_prev = sma_50.shift(1)
    sma_90_prev = sma_90.shift(1)
    mean_90_prev = sma_90.shift(1)
    std_90_prev = std_90.shift(1)

    last = df.iloc[-1]
    last_ts = df.index[-1]
    if last_ts != as_of_ts and (as_of_ts - last_ts).days > 7:
        # No bar within the last week of the as-of date — skip.
        return None

    v = float(last["volume"])
    avg20 = sma_20_prev.iloc[-1]
    if pd.isna(avg20) or avg20 <= 0:
        return None

    rvol_5 = _safe_ratio(v, sma_5_prev.iloc[-1])
    rvol_20 = _safe_ratio(v, avg20)
    rvol_50 = _safe_ratio(v, sma_50_prev.iloc[-1])
    rvol_90 = _safe_ratio(v, sma_90_prev.iloc[-1])
    mean_90 = mean_90_prev.iloc[-1]
    std_90_v = std_90_prev.iloc[-1]
    z = (
        float("nan")
        if pd.isna(mean_90) or pd.isna(std_90_v) or std_90_v == 0
        else (v - float(mean_90)) / float(std_90_v)
    )
    pct_rank = (
        float(pct_rank_252.iloc[-1])
        if not pd.isna(pct_rank_252.iloc[-1])
        else float("nan")
    )

    # Threshold check: the strongest of (rvol_20, rvol_5, z) decides whether
    # we emit. Strength is computed from the (rvol_20, z) pair so the
    # short-term burst alone can't tag EXTREME.
    rvol_for_emit = max(
        rvol_20 if not np.isnan(rvol_20) else 0.0,
        rvol_5 if not np.isnan(rvol_5) else 0.0,
    )
    z_for_emit = z if not np.isnan(z) else 0.0
    if rvol_for_emit < min_rvol and z_for_emit < min_z:
        return None

    open_px = float(last["open"])
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])
    prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else close
    pct_change = (close - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0

    direction = classify_direction(open_px, high, low, close, prev_close)
    rvol_for_strength = rvol_20 if not np.isnan(rvol_20) else 0.0
    z_for_strength = z if not np.isnan(z) else 0.0
    strength = classify_strength(rvol_for_strength, z_for_strength)

    return Event(
        symbol=symbol,
        date=last_ts.date(),
        close=close,
        pct_change=round(pct_change, 4),
        volume=v,
        avg_volume_20d=float(avg20),
        rvol=round(rvol_20, 4) if not np.isnan(rvol_20) else float("nan"),
        rvol_5d=round(rvol_5, 4) if not np.isnan(rvol_5) else float("nan"),
        rvol_50d=round(rvol_50, 4) if not np.isnan(rvol_50) else float("nan"),
        rvol_90d=round(rvol_90, 4) if not np.isnan(rvol_90) else float("nan"),
        z_score=round(z, 4) if not np.isnan(z) else float("nan"),
        pct_rank_252d=round(pct_rank, 4) if not np.isnan(pct_rank) else float("nan"),
        direction=direction,
        strength=strength,
    )


def detect_market(
    bars_by_symbol: dict[str, pd.DataFrame],
    as_of: date,
    min_rvol: float = DEFAULT_MIN_RVOL,
    min_z: float = DEFAULT_MIN_Z,
) -> list[Event]:
    """Run ``detect_ticker`` across every symbol and return the surviving events.

    Caller is expected to have already applied universe filters; this function
    just runs the math.
    """
    out: list[Event] = []
    for sym, df in bars_by_symbol.items():
        if df is None or df.empty:
            continue
        ev = detect_ticker(sym, df, as_of, min_rvol=min_rvol, min_z=min_z)
        if ev is not None:
            out.append(ev)
    return out
