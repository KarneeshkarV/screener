"""Sentiment strategies for earnings-drift backtest.

Each strategy returns a score in [0, 1] for a given earnings event.
A score ≥ threshold (default 0.5) means "bullish entry signal".

Strategies:
  1. price_momentum  — 5d & 20d return positive going into earnings
  2. volume_surge    — volume >1.5× 20d avg on E-1/E-2
  3. analyst_sentiment — yfinance upgrades > downgrades
  4. iv_sentiment    — P/C ratio <0.7 + IV percentile (US: yfinance, India: NSE)
  5. combined_score  — weighted average of above
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalResult:
    """Outcome of a strategy evaluation on one earnings event."""

    ticker: str
    earnings_date: pd.Timestamp
    strategy: str
    score: float
    passed: bool
    details: dict


# ── Strategy implementations ────────────────────────────────────────────


def price_momentum(
    ticker: str,
    earnings_date: pd.Timestamp,
    bars: pd.DataFrame,
    threshold: float = 0.5,
) -> SignalResult:
    """Long if 5d & 20d returns are both positive going into earnings.

    Score = (positive_short + positive_long) / 2, each term is 0 or 1.
    """
    if bars is None or bars.empty or len(bars) < 21:
        return SignalResult(
            ticker,
            earnings_date,
            "price_momentum",
            0.0,
            False,
            {"reason": "insufficient_data"},
        )

    e_minus_1 = bars[bars.index <= earnings_date]
    if len(e_minus_1) < 21:
        return SignalResult(
            ticker,
            earnings_date,
            "price_momentum",
            0.0,
            False,
            {"reason": "insufficient_data"},
        )

    close = e_minus_1["close"].astype(float)
    ret_5d = (close.iloc[-1] / close.iloc[-6]) - 1.0 if len(close) >= 6 else 0.0
    ret_20d = (close.iloc[-1] / close.iloc[-21]) - 1.0 if len(close) >= 21 else 0.0

    score_short = 1.0 if ret_5d > 0 else 0.0
    score_long = 1.0 if ret_20d > 0 else 0.0
    score = (score_short + score_long) / 2.0
    passed = score >= threshold

    return SignalResult(
        ticker=ticker,
        earnings_date=earnings_date,
        strategy="price_momentum",
        score=round(score, 4),
        passed=passed,
        details={"ret_5d": round(ret_5d, 6), "ret_20d": round(ret_20d, 6)},
    )


def volume_surge(
    ticker: str,
    earnings_date: pd.Timestamp,
    bars: pd.DataFrame,
    threshold: float = 0.5,
    surge_factor: float = 1.5,
) -> SignalResult:
    """Long if volume > surge_factor × 20d avg on E-1 or E-2.

    Score = min(volume / (surge_factor × avg), 1.0) clipped, then
    1.0 if volume > surge_factor × avg, else 0.0.
    """
    if bars is None or bars.empty or len(bars) < 21:
        return SignalResult(
            ticker,
            earnings_date,
            "volume_surge",
            0.0,
            False,
            {"reason": "insufficient_data"},
        )

    e_minus_1 = bars[bars.index <= earnings_date]
    if len(e_minus_1) < 21:
        return SignalResult(
            ticker,
            earnings_date,
            "volume_surge",
            0.0,
            False,
            {"reason": "insufficient_data"},
        )

    vol = e_minus_1["volume"].astype(float)
    avg_20d = vol.iloc[-21:-1].mean()  # 20 bars before E-1
    if avg_20d <= 0:
        return SignalResult(
            ticker,
            earnings_date,
            "volume_surge",
            0.0,
            False,
            {"reason": "zero_avg_volume"},
        )

    # Check volume on E-1 and E-2
    recent_vols = vol.iloc[-2:] if len(vol) >= 2 else vol.iloc[-1:]
    max_recent = recent_vols.max()
    ratio = max_recent / avg_20d

    # Score: 1.0 if ratio > surge_factor, scaled otherwise
    if ratio >= surge_factor:
        score = min(ratio / surge_factor, 1.0)
    else:
        score = max(0.0, ratio / surge_factor) * 0.5  # partial credit

    passed = score >= threshold
    return SignalResult(
        ticker=ticker,
        earnings_date=earnings_date,
        strategy="volume_surge",
        score=round(score, 4),
        passed=passed,
        details={"volume_ratio": round(ratio, 4), "avg_20d_volume": round(avg_20d, 2)},
    )


def analyst_sentiment(
    ticker: str,
    earnings_date: pd.Timestamp,
    sentiment_data: Optional[dict],
    threshold: float = 0.5,
) -> SignalResult:
    """Long if yfinance net upgrades > 0.

    sentiment_data comes from fetch_analyst_sentiment().
    Score = sigmoid-like mapping of net upgrades to [0, 1].
    """
    if sentiment_data is None:
        return SignalResult(
            ticker,
            earnings_date,
            "analyst_sentiment",
            0.0,
            False,
            {"reason": "no_data"},
        )

    net = sentiment_data["net"]
    # Map net upgrades to [0, 1] using sigmoid
    # net=0 → 0.5, net>0 → >0.5, net<0 → <0.5
    score = 1.0 / (1.0 + np.exp(-net * 0.3))
    passed = score >= threshold

    return SignalResult(
        ticker=ticker,
        earnings_date=earnings_date,
        strategy="analyst_sentiment",
        score=round(score, 4),
        passed=passed,
        details={
            "upgrades": sentiment_data["upgrades"],
            "downgrades": sentiment_data["downgrades"],
            "net": net,
        },
    )


def iv_sentiment(
    ticker: str,
    earnings_date: pd.Timestamp,
    iv_data: Optional[dict],
    threshold: float = 0.5,
) -> SignalResult:
    """P/C ratio < 0.7 is bullish; median IV as confidence.

    US: uses yfinance options (full P/C + IV data).
    India: uses NSE option chain via jugaad_data (P/C + IV from strikes).

    When no options data is available, returns neutral score 0.5
    which is excluded from combined weight.
    """
    if iv_data is None:
        # No data at all — return neutral
        return SignalResult(
            ticker,
            earnings_date,
            "iv_sentiment",
            0.5,
            threshold <= 0.5,
            {"reason": "no_options_data"},
        )

    pc_ratio = iv_data["pc_ratio"]
    # median_iv is in % terms (e.g. 40.11 for 40.11% IV, 22.23 for 22.23% IV)
    median_iv = iv_data.get("median_iv", float("nan"))

    # P/C ratio signal: < 0.7 → bullish (score > 0.5), > 1.0 → bearish
    if pc_ratio < 0.7:
        pc_score = 1.0
    elif pc_ratio < 1.0:
        pc_score = 1.0 - (pc_ratio - 0.7) / 0.3 * 0.5
    else:
        pc_score = max(0.0, 1.0 - (pc_ratio - 1.0) * 0.5)

    # Median IV score: higher IV → more drift potential.
    # Scale: IV < 20% → low (0.3), 20-50% → moderate (0.5), > 50% → high (0.7-1.0).
    # If NaN (no IV data), use neutral 0.35.
    if not np.isnan(median_iv):
        if median_iv < 20:
            iv_score = 0.3
        elif median_iv < 50:
            iv_score = 0.3 + (median_iv - 20) / 30 * 0.4  # 0.3 → 0.7
        else:
            iv_score = min(0.7 + (median_iv - 50) / 50 * 0.3, 1.0)
    else:
        iv_score = 0.35

    # Combined: P/C ratio dominates (70%), IV adds (30%)
    score = pc_score * 0.7 + iv_score * 0.3
    passed = score >= threshold

    return SignalResult(
        ticker=ticker,
        earnings_date=earnings_date,
        strategy="iv_sentiment",
        score=round(score, 4),
        passed=passed,
        details={
            "pc_ratio": round(pc_ratio, 4),
            "median_iv_pct": round(median_iv, 2) if not np.isnan(median_iv) else None,
        },
    )


# ── Combined scoring ────────────────────────────────────────────────────

STRATEGY_WEIGHTS = {
    "price_momentum": 0.30,
    "volume_surge": 0.25,
    "analyst_sentiment": 0.25,
    "iv_sentiment": 0.20,
}


def combined_score(
    scores: dict[str, float],
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Weighted average of individual strategy scores.

    Strategies with score == 0.5 (neutral skip, e.g. iv_sentiment for India)
    are excluded from the weight sum so they don't dilute the result.
    """
    w = weights or STRATEGY_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0
    for strat, score in scores.items():
        if strat not in w:
            continue
        # Skip neutral/missing signals (0.5 from iv_sentiment skip)
        if score == 0.5 and strat == "iv_sentiment":
            continue
        total_weight += w[strat]
        weighted_sum += w[strat] * score
    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 4)


STRATEGY_FUNCS = {
    "price_momentum": price_momentum,
    "volume_surge": volume_surge,
    "analyst_sentiment": analyst_sentiment,
    "iv_sentiment": iv_sentiment,
}
