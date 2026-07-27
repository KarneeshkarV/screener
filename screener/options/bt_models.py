"""Typed contracts for the options position-P&L backtester.

``OptionPositionTrade`` satisfies the ``EventTradeSummary`` protocol used by
``screener.earnings_backtest.metrics.compute_backtest_summary`` so summary
stats come for free. ``return_pct`` is premium-relative
(``pnl / gross_entry_premium``), not margin-relative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from screener.options.models import OptionRight

StrikeRule = str  # "atm" | "moneyness:<±pct>" | "delta:<abs>"
ExpiryRule = str  # "front" | "next" | "dte:<n>"
ExitReason = Literal[
    "expiry",
    "target",
    "stop",
    "dte",
    "exit_expr",
    "time",
    "end",
    "roll",
    "session_end",  # Phase 4.2: intraday flatten at the session's last snapshot
]
# Fill model for entry/exit pricing. ``legacy`` reproduces the historical
# mid-or-last-with-settle-fallback + percent-slippage behaviour byte-for-byte.
FillModel = Literal["legacy", "mid", "cross"]
# Short-option margin approximation. ``none`` disables margin tracking (the
# historical default). ``span`` is an India SPAN-like worst-of scenario grid;
# ``regt`` is a US Reg-T approximation.
MarginModel = Literal["none", "span", "regt"]
# Expiry settlement mark. ``intrinsic`` (default) settles legs at intrinsic
# value against the underlying close; ``settle`` prefers the official
# per-contract settlement price when present.
Settlement = Literal["intrinsic", "settle"]


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


@dataclass(frozen=True)
class OptionPositionTrade:
    """One completed multi-leg option trade.

    ``return_pct`` is premium-relative (``pnl / gross entry premium * 100``),
    not margin-relative. ``passed_filter`` is always True for taken trades so
    ``compute_backtest_summary`` includes them.
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
    details: dict[str, Any] = field(default_factory=dict)


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
    # Exit on the Nth trading day AFTER entry. Entry day is day 0: no
    # stop/target/roll/dte/time/exit_expr is evaluated that session, and
    # ``hold_days`` is not incremented until the first post-entry session.
    # ``None`` disables the time stop.
    max_hold: int | None = None
    slippage_pct: float = 0.0
    commission_per_order: float = 0.0
    min_oi: float = 0.0
    min_volume: float = 0.0
    refresh: bool = False
    initial_capital: float = Field(default=100_000.0, gt=0)

    # --- Phase 4.1: fill realism -------------------------------------------
    # ``legacy`` (default) keeps the historical fill exactly. ``mid`` fills at
    # the bid/ask mid; ``cross`` crosses the spread (buy the ask, sell the bid).
    fill_model: FillModel = "legacy"
    # Extra slippage applied on top of the fill model, in basis points of the
    # fill price and/or whole ticks. Both default to 0 (no-op).
    slippage_bps: float = Field(default=0.0, ge=0)
    slippage_ticks: float = Field(default=0.0, ge=0)
    tick_size: float = Field(default=0.05, gt=0)
    # When a leg has no quotes, proxy a full spread of this fraction of the mark
    # (widened by observed settle/close dispersion). 0 disables proxying.
    illiquid_spread_pct: float = Field(default=0.0, ge=0)

    # --- Phase 4.1: margin model -------------------------------------------
    margin_model: MarginModel = "none"
    # Cap on portfolio margin as a fraction of ``initial_capital``. ``None``
    # (default) is unlimited; entries that would breach the cap are skipped.
    margin_cap_pct: float | None = Field(default=None, gt=0)
    # SPAN-like scenario grid parameters (India). Underlying is shocked by
    # ``span_price_scan_pct`` and IV by ``span_vol_scan`` (absolute vol points).
    span_price_scan_pct: float = Field(default=0.05, gt=0)
    span_vol_scan: float = Field(default=0.10, ge=0)
    span_exposure_pct: float = Field(default=0.03, ge=0)
    # Reg-T naked-option parameters (US): margin = current_premium +
    # max(regt_pct*underlying - OTM, regt_min_pct*spot for calls /
    # regt_min_pct*strike for puts). Premium uses the day's mark, not the
    # frozen entry fill.
    regt_pct: float = Field(default=0.20, ge=0)
    regt_min_pct: float = Field(default=0.10, ge=0)

    # --- Phase 4.1: expiry mechanics ---------------------------------------
    settlement: Settlement = "intrinsic"
    # Physical (stock-option) vs cash (index) settlement. Only affects recorded
    # assignment metadata; P&L is intrinsic either way in an EOD model.
    physical_settlement: bool = False

    # --- Phase 4.1: roll rules ---------------------------------------------
    # Roll (exit + same-session re-enter) when days-to-expiry drops to
    # ``roll_dte`` or the position's |net delta| reaches ``roll_delta``.
    roll_dte: int | None = Field(default=None, ge=0)
    roll_delta: float | None = Field(default=None, gt=0)
    # Expiry rule used for the re-entered structure when rolling.
    roll_expiry_rule: str = "next"


__all__ = [
    "ExitReason",
    "ExpiryRule",
    "FillModel",
    "LegFill",
    "LegSpec",
    "MarginModel",
    "OptionPositionTrade",
    "OptionsBacktestConfig",
    "Settlement",
    "StrikeRule",
    "StructureSpec",
]
