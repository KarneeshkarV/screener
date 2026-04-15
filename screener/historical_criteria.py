"""Local criterion evaluators for historical backtesting.

OHLCV-only criteria (no external data needed):
    ema          — EMA5 > EMA20 > EMA100 > EMA200 bullish stack
    breakout     — close within 10% of 52-week high + above-average volume
    ema_breakout — ema AND breakout combined

Fundamentals-augmented criteria (yfinance quarterly data required):
    value         — P/E in (0, 20]
    quality       — ROE > 15%, D/E < 1
    cheap_quality — value + quality + EMA20 > EMA200
    undervalued   — P/E in (0, 12], above-average volume
    dividend      — div_yield > 3%, P/E in (0, 25], D/E < 1.5
    momentum_value— P/E in (0, 25], RSI 50–70, EMA5 > EMA20 > EMA200

For fundamental criteria the evaluator returns None when fundamental data
is unavailable, and the ticker is silently skipped.

All evaluators have the same signature::

    eval_fn(ohlcv: DataFrame, as_of: Timestamp,
            fundamentals: dict | None = None) -> dict | None

Returned dict::

    {"passes": bool, "score_inputs": {...indicator snapshot...}}
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

_MIN_BARS_EMA = 260
_MIN_BARS_BASIC = 30

# Criteria that need yfinance fundamentals in addition to OHLCV.
FUND_CRITERIA: frozenset[str] = frozenset({
    "value", "quality", "cheap_quality",
    "undervalued", "dividend", "momentum_value",
})


# ── internal helpers ─────────────────────────────────────────────────────────

def _slice_to_asof(ohlcv: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[df["date"] <= as_of].reset_index(drop=True)


def _indicator_snapshot(df: pd.DataFrame) -> dict:
    close = pd.to_numeric(df["close"], errors="coerce").ffill()

    ema5   = close.ewm(span=5,   adjust=False).mean().iloc[-1]
    ema20  = close.ewm(span=20,  adjust=False).mean().iloc[-1]
    ema100 = close.ewm(span=100, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
    rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100.0

    lookback = min(252, len(df))
    high_52w = float(close.tail(lookback).max())
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    momentum = (last_close / prev_close - 1.0) if prev_close > 0 else 0.0

    has_vol = "volume" in df.columns and pd.to_numeric(df["volume"], errors="coerce").notna().any()
    if has_vol:
        volume = pd.to_numeric(df["volume"], errors="coerce").ffill()
        last_vol     = float(volume.iloc[-1])
        vol_10d_avg  = float(volume.tail(10).mean())
        dollar_vol_20d = float((close * volume).tail(20).mean())
    else:
        last_vol = vol_10d_avg = dollar_vol_20d = None

    return {
        "close": last_close, "ema5": float(ema5), "ema20": float(ema20),
        "ema100": float(ema100), "ema200": float(ema200),
        "rsi": float(rsi), "high_52w": high_52w, "momentum": momentum,
        "volume": last_vol, "vol_10d_avg": vol_10d_avg,
        "dollar_vol_20d": dollar_vol_20d,
    }


def _merge_snap(snap: dict, fundamentals: Optional[dict]) -> dict:
    if fundamentals:
        return {**snap, **{k: v for k, v in fundamentals.items()}}
    return snap


# ── OHLCV-only criteria ──────────────────────────────────────────────────────

def eval_ema_stack(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """EMA5 > EMA20 > EMA100 > EMA200 bullish stack."""
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_EMA:
        return None
    snap = _indicator_snapshot(df)
    passes = snap["ema5"] > snap["ema20"] > snap["ema100"] > snap["ema200"] > 0
    return {"passes": passes, "score_inputs": _merge_snap(snap, fundamentals)}


def eval_breakout(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """Close within 10% of 52-week high AND volume > 10-bar average."""
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_BASIC:
        return None
    snap = _indicator_snapshot(df)
    passes_high = snap["close"] >= 0.9 * snap["high_52w"]
    passes_vol = (
        snap["volume"] > snap["vol_10d_avg"]
        if snap["vol_10d_avg"] and snap["vol_10d_avg"] > 0
        else True
    )
    return {"passes": passes_high and passes_vol, "score_inputs": _merge_snap(snap, fundamentals)}


def eval_ema_breakout(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """EMA bullish stack AND near-52w-high breakout combined."""
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_EMA:
        return None
    snap = _indicator_snapshot(df)
    passes_ema  = snap["ema5"] > snap["ema20"] > snap["ema100"] > snap["ema200"] > 0
    passes_high = snap["close"] >= 0.9 * snap["high_52w"]
    passes_vol  = (
        snap["volume"] > snap["vol_10d_avg"]
        if snap["vol_10d_avg"] and snap["vol_10d_avg"] > 0
        else True
    )
    return {"passes": passes_ema and passes_high and passes_vol,
            "score_inputs": _merge_snap(snap, fundamentals)}


# ── Fundamentals-augmented criteria ─────────────────────────────────────────

def eval_value(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """P/E in (0, 20] — low valuation."""
    if fundamentals is None:
        return None
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_BASIC:
        return None
    snap = _indicator_snapshot(df)
    pe = fundamentals.get("pe")
    if pe is None:
        return None
    passes = 0 < pe <= 20
    return {"passes": passes, "score_inputs": _merge_snap(snap, fundamentals)}


def eval_quality(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """ROE > 15% and D/E < 1."""
    if fundamentals is None:
        return None
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_BASIC:
        return None
    snap = _indicator_snapshot(df)
    roe = fundamentals.get("roe")
    de  = fundamentals.get("de")
    if roe is None or de is None:
        return None
    passes = roe > 15 and de < 1
    return {"passes": passes, "score_inputs": _merge_snap(snap, fundamentals)}


def eval_cheap_quality(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """P/E in (0, 20], ROE > 15%, D/E < 1, EMA20 > EMA200."""
    if fundamentals is None:
        return None
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_EMA:
        return None
    snap = _indicator_snapshot(df)
    pe  = fundamentals.get("pe")
    roe = fundamentals.get("roe")
    de  = fundamentals.get("de")
    if any(x is None for x in (pe, roe, de)):
        return None
    passes = (0 < pe <= 20) and (roe > 15) and (de < 1) and (snap["ema20"] > snap["ema200"])
    return {"passes": passes, "score_inputs": _merge_snap(snap, fundamentals)}


def eval_undervalued(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """P/E in (0, 12] and above-average volume — deep value."""
    if fundamentals is None:
        return None
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_BASIC:
        return None
    snap = _indicator_snapshot(df)
    pe = fundamentals.get("pe")
    if pe is None:
        return None
    passes_pe  = 0 < pe <= 12
    passes_vol = (
        snap["volume"] > snap["vol_10d_avg"]
        if snap["vol_10d_avg"] and snap["vol_10d_avg"] > 0
        else True
    )
    passes = passes_pe and passes_vol
    return {"passes": passes, "score_inputs": _merge_snap(snap, fundamentals)}


def eval_dividend(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """Dividend yield > 3%, P/E in (0, 25], D/E < 1.5."""
    if fundamentals is None:
        return None
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_BASIC:
        return None
    snap = _indicator_snapshot(df)
    pe        = fundamentals.get("pe")
    de        = fundamentals.get("de")
    div_yield = fundamentals.get("div_yield", 0.0)
    if pe is None or de is None:
        return None
    passes = (div_yield > 3) and (0 < pe <= 25) and (de < 1.5)
    return {"passes": passes, "score_inputs": _merge_snap(snap, fundamentals)}


def eval_momentum_value(
    ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    fundamentals: Optional[dict] = None,
) -> Optional[dict]:
    """P/E in (0, 25], RSI 50–70, EMA5 > EMA20 > EMA200."""
    if fundamentals is None:
        return None
    df = _slice_to_asof(ohlcv, as_of)
    if len(df) < _MIN_BARS_EMA:
        return None
    snap = _indicator_snapshot(df)
    pe = fundamentals.get("pe")
    if pe is None:
        return None
    passes = (
        (0 < pe <= 25)
        and (50 <= snap["rsi"] <= 70)
        and (snap["ema5"] > snap["ema20"] > snap["ema200"])
    )
    return {"passes": passes, "score_inputs": _merge_snap(snap, fundamentals)}


# ── registry ─────────────────────────────────────────────────────────────────

HIST_CRITERIA: dict[str, Callable] = {
    "ema":            eval_ema_stack,
    "breakout":       eval_breakout,
    "ema_breakout":   eval_ema_breakout,
    "value":          eval_value,
    "quality":        eval_quality,
    "cheap_quality":  eval_cheap_quality,
    "undervalued":    eval_undervalued,
    "dividend":       eval_dividend,
    "momentum_value": eval_momentum_value,
}


# ── cohort scoring ────────────────────────────────────────────────────────────

def compute_scores(snapshots: list[dict]) -> list[float]:
    """Compute a local setup_score for each matched ticker's snapshot.

    Weights (OHLCV-based; fundamentals determine pass/fail, not rank):
        liquidity   35%  (log dollar-volume 20-day, percentile)
        trend       35%  (EMA spacing, percentile)
        rsi_quality 15%  (closeness to RSI=60)
        momentum    10%  (1-day return, percentile)
        extension  -15%  (penalty when close > ema20 by >12%)
    """
    import math

    if not snapshots:
        return []

    dvols = [s.get("dollar_vol_20d") or 0.0 for s in snapshots]
    log_dvols = [math.log1p(max(0.0, v)) for v in dvols]
    dvol_pct = pd.Series(log_dvols).rank(pct=True).tolist()

    trend_raw = []
    for s in snapshots:
        c = s.get("close", 0)
        if c <= 0:
            trend_raw.append(0.0)
            continue
        e5, e20, e100, e200 = s.get("ema5",0), s.get("ema20",0), s.get("ema100",0), s.get("ema200",0)
        spread = (e5-e20)/c + (e20-e100)/c + (e100-e200)/c
        trend_raw.append(max(0.0, spread))
    trend_pct = pd.Series(trend_raw).rank(pct=True).tolist()

    mom_pct = pd.Series([s.get("momentum", 0.0) for s in snapshots]).rank(pct=True).tolist()

    scores: list[float] = []
    for i, s in enumerate(snapshots):
        rsi = s.get("rsi") or 50.0
        rsi_q = max(0.0, min(1.0, 1.0 - abs(rsi - 60.0) / 40.0))

        c, e20 = s.get("close", 0), s.get("ema20", 0)
        penalty = (
            max(0.0, min(0.15, ((c - e20) / e20 - 0.12) / 0.25 * 0.15))
            if e20 > 0 else 0.0
        )

        scores.append(round(
            0.35 * dvol_pct[i]
            + 0.35 * trend_pct[i]
            + 0.15 * rsi_q
            + 0.10 * mom_pct[i]
            - penalty,
            6,
        ))

    return scores
