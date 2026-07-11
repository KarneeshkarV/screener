"""Typed trade contracts shared by the earnings event-study engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecutedEventTrade:
    """Fields common to every completed earnings-event trade."""

    ticker: str
    earnings_date: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    return_pct: float


@runtime_checkable
class EventTradeSummary(Protocol):
    """Minimal contract consumed by event-study metric aggregation."""

    @property
    def entry_date(self) -> date: ...

    @property
    def exit_date(self) -> date: ...

    @property
    def return_pct(self) -> float: ...

    @property
    def passed_filter(self) -> bool: ...


@dataclass(frozen=True)
class EarningsTrade(ExecutedEventTrade):
    strategy: str
    score: float
    passed_filter: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeadTrade(ExecutedEventTrade):
    surprise_pct: float
    holding_days: int
    passed_filter: bool = True
    details: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "EarningsTrade",
    "EventTradeSummary",
    "ExecutedEventTrade",
    "PeadTrade",
]
