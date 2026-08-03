"""Earnings-event extensions of neutral trade contracts."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import Field

from screener.ledger import EventTradeSummary, Trade


class ExecutedEventTrade(Trade):
    """Completed earnings-event extension of the neutral trade lifecycle."""

    ticker: str
    earnings_date: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    return_pct: float


class EarningsTrade(ExecutedEventTrade):
    strategy: str
    score: float
    passed_filter: bool
    details: dict[str, Any] = Field(default_factory=dict)


class PeadTrade(ExecutedEventTrade):
    surprise_pct: float
    holding_days: int
    passed_filter: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "EarningsTrade",
    "EventTradeSummary",
    "ExecutedEventTrade",
    "PeadTrade",
]
