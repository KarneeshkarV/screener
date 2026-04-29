"""Direction + strength classification for unusual-volume events."""
from __future__ import annotations

from typing import Literal


Direction = Literal[
    "BUYING", "SELLING", "CHURN", "REVERSAL", "QUIET_ACCUMULATION", "BUILDUP"
]
Strength = Literal["MODERATE", "HIGH", "EXTREME"]


def classify_direction(
    open_px: float,
    high: float,
    low: float,
    close: float,
    prev_close: float,
) -> Direction:
    """Tag a high-volume bar as BUYING / SELLING / CHURN / REVERSAL.

    Rules (in priority order):
      • REVERSAL — gap > 2% but bar closes opposite the gap direction.
      • CHURN    — |daily change| < 1%; volume spike without resolution.
      • BUYING   — close > open AND close in upper third of day's range.
      • SELLING  — close < open AND close in lower third of day's range.
      • Otherwise CHURN (mid-range close on a directionless bar).
    """
    rng = max(high - low, 1e-9)
    if prev_close > 0:
        gap = (open_px - prev_close) / prev_close
        change = (close - prev_close) / prev_close
        if abs(gap) > 0.02 and (gap * change) < 0:
            return "REVERSAL"
        if abs(change) < 0.01:
            return "CHURN"
    upper_third = low + rng * (2.0 / 3.0)
    lower_third = low + rng * (1.0 / 3.0)
    if close > open_px and close >= upper_third:
        return "BUYING"
    if close < open_px and close <= lower_third:
        return "SELLING"
    return "CHURN"


def classify_strength(rvol: float, z: float) -> Strength:
    """Return the strongest tier the event qualifies for."""
    if rvol >= 5.0 or z >= 3.5:
        return "EXTREME"
    if rvol >= 3.0 or z >= 2.5:
        return "HIGH"
    return "MODERATE"
