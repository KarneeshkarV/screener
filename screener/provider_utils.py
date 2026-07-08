"""Shared utilities for fundamental data providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from screener import fmp
from screener.financials import first_number
from screener.financials import pct_change as pct_change
from screener.financials import to_number as to_number

# Numeric aliases commonly used across provider adapters
_num = to_number
_pct_change = pct_change


def _first_num(
    mapping: Mapping[Any, Any], *keys: str, case_insensitive: bool = True
) -> float | None:
    """Extract the first parseable numeric value from mapping."""
    return first_number(mapping, *keys, case_insensitive=case_insensitive)


def fmp_get(
    path: str,
    params: Mapping[str, Any],
    api_key: str,
    *,
    session: Any = None,
    timeout: float = fmp.DEFAULT_TIMEOUT,
    headers: Mapping[str, str] | None = None,
) -> Any:
    """Shared FMP client request."""
    resolved_headers = fmp.DEFAULT_HEADERS if headers is None else headers
    client = fmp.FmpClient(
        api_key,
        base_url=fmp.FMP_V3_BASE_URL,
        headers=resolved_headers,
        timeout=timeout,
        session=session,
    )
    return client.get(path, params)
