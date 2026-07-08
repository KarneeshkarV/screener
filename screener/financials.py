"""Shared helpers for normalizing provider financial values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def to_number(value: Any) -> float | None:
    """Coerce comma/percent formatted provider values to ``float | None``."""
    if value is None:
        return None
    try:
        out = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def first_number(
    mapping: Mapping[Any, Any],
    *keys: str,
    case_insensitive: bool = False,
) -> float | None:
    """Return the first parseable numeric value found for ``keys``."""
    if case_insensitive:
        lowered = {str(k).lower(): v for k, v in mapping.items()}
        for key in keys:
            parsed = to_number(lowered.get(key.lower()))
            if parsed is not None:
                return parsed
        return None

    for key in keys:
        parsed = to_number(mapping.get(key))
        if parsed is not None:
            return parsed
    return None


def pct_change(new: float | None, old: float | None) -> float | None:
    """Percentage change using absolute old value as the denominator."""
    if new is None or old is None or old == 0:
        return None
    return ((new - old) / abs(old)) * 100.0


def as_percent(value: float | None) -> float | None:
    """Normalize decimal ratios to percent values, leaving percent inputs alone."""
    if value is None:
        return None
    return value * 100.0 if abs(value) <= 2.0 else value


__all__ = ["as_percent", "first_number", "pct_change", "to_number"]
