"""Typed contracts for the options position-P&L backtester.

``OptionPositionTrade`` satisfies the neutral ``EventTradeSummary`` protocol
used by ``screener.ledger.compute_event_trade_summary``. ``return_pct`` is
premium-relative (``pnl / gross_entry_premium``), not margin-relative.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from screener.ledger import ExitReason, Trade
from screener.options.models import OptionRight

StrikeRule = str  # "atm" | "moneyness:<±pct>" | "delta:<abs>"
ExpiryRule = str  # "front" | "next" | "dte:<n>"


@dataclass(frozen=True)
class LegSpec:
    """One leg of a multi-leg structure template."""

    right: OptionRight
    side: int  # +1 long, -1 short
    lots: int = 1
    strike_rule: StrikeRule = "atm"
    expiry_rule: ExpiryRule = "front"


@dataclass(frozen=True)
class StructureSpec:
    """Named multi-leg option structure."""

    name: str
    legs: tuple[LegSpec, ...]


@dataclass(frozen=True)
class LegFill:
    """Resolved fill for one leg of a completed (or open) position."""

    right: OptionRight
    strike: float
    expiry: date
    side: int
    lots: int
    lot_size: float
    entry_price: float
    exit_price: float | None = None
    entry_iv: float | None = None


class OptionPositionTrade(Trade):
    """One completed multi-leg option extension of the neutral lifecycle.

    ``return_pct`` is premium-relative (``pnl / gross entry premium * 100``),
    not margin-relative. ``passed_filter`` is always True for taken trades so
    event-trade aggregation includes them.
    """

    symbol: str
    structure: str
    entry_date: date
    exit_date: date
    legs: tuple[LegFill, ...]
    entry_premium: float  # signed: debit > 0, credit < 0
    exit_premium: float  # signed, same convention
    pnl: float  # rupees after costs
    return_pct: float
    exit_reason: ExitReason
    passed_filter: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class OptionsBacktestConfig(BaseModel):
    """Knobs for the India EOD options position backtester."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    market: Literal["india"] = "india"
    tickers: tuple[str, ...]
    start: date
    end: date
    structure: str = "long_call"
    strike_rule: str = "atm"
    expiry_rule: str = "front"
    width_pct: float = 0.05
    lots: int = 1
    entry_expr: str = "true"
    exit_expr: str | None = None
    screen_criterion: str | None = None
    target_pct: float | None = None
    stop_pct: float | None = None
    exit_dte: int = 1
    max_hold: int | None = None
    slippage_pct: float = 0.0
    commission_per_order: float = 0.0
    min_oi: float = 0.0
    min_volume: float = 0.0
    refresh: bool = False
    initial_capital: float = Field(default=100_000.0, gt=0)


__all__ = [
    "ExitReason",
    "ExpiryRule",
    "LegFill",
    "LegSpec",
    "OptionPositionTrade",
    "OptionsBacktestConfig",
    "StrikeRule",
    "StructureSpec",
]
