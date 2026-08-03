"""Shared coercion helpers for venue option-chain payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def number(value: object, *, nonnegative: bool = False) -> float | None:
    """Coerce a venue value to a finite float, optionally rejecting negatives."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or (nonnegative and result < 0):
        return None
    return result


def positive(value: object) -> float | None:
    """Return a finite positive float or ``None``."""
    result = number(value)
    return result if result is not None and result > 0 else None


def nonnegative_or_zero(value: object) -> float:
    """Return a finite nonnegative float, defaulting invalid values to zero."""
    return number(value, nonnegative=True) or 0.0


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> object:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def quote_pair(
    row: Mapping[str, Any],
    *,
    bid_keys: Sequence[str] = ("bid",),
    ask_keys: Sequence[str] = ("ask",),
) -> tuple[float | None, float | None]:
    """Normalize a bid/ask pair and discard both sides when it is crossed."""
    bid = number(_first_value(row, bid_keys), nonnegative=True)
    ask = number(_first_value(row, ask_keys), nonnegative=True)
    if bid is not None and ask is not None and ask < bid:
        return None, None
    return bid, ask


__all__ = ["nonnegative_or_zero", "number", "positive", "quote_pair"]
