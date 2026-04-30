"""Unusual-volume detection module.

Flags abnormal trading volume on a per-day, per-ticker basis for both US and
Indian equities. The India path additionally overlays NSE delivery quantity
(via jugaad-data) so volume spikes without delivery (intraday churn) are
separated from real institutional footprints.
"""
from __future__ import annotations

from .detector import (
    Event,
    Strength,
    Direction,
    DEFAULT_MIN_RVOL,
    DEFAULT_MIN_Z,
    HIGH_RVOL,
    HIGH_Z,
    EXTREME_RVOL,
    EXTREME_Z,
    detect_ticker,
    detect_market,
)

__all__ = [
    "Event",
    "Strength",
    "Direction",
    "DEFAULT_MIN_RVOL",
    "DEFAULT_MIN_Z",
    "HIGH_RVOL",
    "HIGH_Z",
    "EXTREME_RVOL",
    "EXTREME_Z",
    "detect_ticker",
    "detect_market",
]
