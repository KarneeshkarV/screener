"""Shared constants and serialization helpers for earnings data providers."""

from __future__ import annotations

from typing import Any

import numpy as np


EARNINGS_CACHE_DAYS = 30
SENTIMENT_CACHE_DAYS = 1
MAX_WORKERS = 12


def jsonable(value: Any) -> Any:
    """Recursively coerce provider values into JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return jsonable(value.item())
    return str(value)


__all__ = ["EARNINGS_CACHE_DAYS", "MAX_WORKERS", "SENTIMENT_CACHE_DAYS", "jsonable"]
