"""Shared leaf number formatters used across the display modules.

These are the genuinely-duplicated formatting primitives (NaN/None guard,
percent, market-cap, volume, fixed-decimal float). The per-module table
*layouts* deliberately stay in their own modules — only these leaves are
shared so number formats remain consistent and tested in one place.
"""

from __future__ import annotations

import math


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt_float(v, ndp: int = 2) -> str:
    """Fixed-decimal float, ``-`` for missing values."""
    if _is_missing(v):
        return "-"
    return f"{v:.{ndp}f}"


def fmt_pct(v) -> str:
    """Signed percent with two decimals (e.g. ``+1.23%``)."""
    if _is_missing(v):
        return "-"
    return f"{v:+.2f}%"


def fmt_volume(v) -> str:
    """Compact volume: B / M / K tiers, ``-`` for missing values."""
    if _is_missing(v):
        return "-"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


def fmt_mcap(v) -> str:
    """Compact market cap: T / B / M tiers, ``-`` for missing values."""
    if _is_missing(v):
        return "-"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    return f"{v:,.0f}"
