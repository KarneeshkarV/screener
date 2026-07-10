"""Source-neutral, immutable option-chain models.

Implied volatility is normalized to a decimal throughout this package
(``0.25`` means 25%). Prices and greeks retain the units supplied by the
underlying venue after numeric normalization.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OptionRight = Literal["call", "put"]
OptionsMarket = Literal["us", "india"]


def _as_utc_datetime(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


class OptionContract(BaseModel):
    """One normalized listed option contract."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    underlying: str
    expiry: date
    strike: float = Field(ge=0)
    right: OptionRight
    oi: float = Field(default=0.0, ge=0)
    oi_change: float | None = None
    volume: float = Field(default=0.0, ge=0)
    iv: float | None = Field(default=None, ge=0)
    bid: float | None = Field(default=None, ge=0)
    ask: float | None = Field(default=None, ge=0)
    last: float | None = Field(default=None, ge=0)
    previous_close: float | None = Field(default=None, ge=0)
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    lot_size: float | None = Field(default=None, gt=0)
    as_of: datetime
    source: str

    @field_validator("symbol", "underlying", "source")
    @classmethod
    def _normalize_nonempty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("underlying")
    @classmethod
    def _normalize_underlying(cls, value: str) -> str:
        return value.upper()

    @field_validator("as_of", mode="before")
    @classmethod
    def _normalize_as_of(cls, value: datetime | date) -> datetime:
        return _as_utc_datetime(value)

    @model_validator(mode="after")
    def _validate_quote(self) -> "OptionContract":
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError("ask must be greater than or equal to bid")
        return self


class OptionChain(BaseModel):
    """A normalized chain, potentially containing multiple expiries."""

    model_config = ConfigDict(frozen=True)

    underlying: str
    market: OptionsMarket
    spot: float | None = Field(default=None, gt=0)
    as_of: datetime
    source: str
    contracts: tuple[OptionContract, ...]

    @field_validator("source")
    @classmethod
    def _normalize_nonempty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("underlying")
    @classmethod
    def _normalize_underlying(cls, value: str) -> str:
        return value.upper()

    @field_validator("as_of", mode="before")
    @classmethod
    def _normalize_as_of(cls, value: datetime | date) -> datetime:
        return _as_utc_datetime(value)

    @model_validator(mode="after")
    def _validate_contracts(self) -> "OptionChain":
        mismatched = {
            contract.underlying
            for contract in self.contracts
            if contract.underlying != self.underlying
        }
        if mismatched:
            raise ValueError(
                f"contract underlying mismatch: expected {self.underlying}, "
                f"got {sorted(mismatched)}"
            )
        return self

    @property
    def expiries(self) -> tuple[date, ...]:
        return tuple(sorted({contract.expiry for contract in self.contracts}))


class ChainMetrics(BaseModel):
    """Flat, panel-friendly metrics derived from a normalized chain."""

    model_config = ConfigDict(frozen=True)

    underlying: str
    market: OptionsMarket
    as_of: datetime
    source: str
    spot: float | None = None
    contract_count: int = Field(ge=0)
    expiry_count: int = Field(ge=0)
    front_expiry: date | None = None
    next_expiry: date | None = None
    call_oi: float = 0.0
    put_oi: float = 0.0
    call_volume: float = 0.0
    put_volume: float = 0.0
    pcr: float | None = None
    pcr_volume: float | None = None
    call_put_oi_ratio: float | None = None
    max_pain_strike: float | None = None
    max_pain_distance_pct: float | None = None
    support_strikes: tuple[float, ...] = ()
    resistance_strikes: tuple[float, ...] = ()
    atm_iv: float | None = None
    median_iv: float | None = None
    put_call_iv_skew: float | None = None
    term_structure_slope: float | None = None
    implied_move_pct: float | None = None
    call_oi_change: float | None = None
    put_oi_change: float | None = None
    oi_chg_ratio: float | None = None
    call_writing_near_spot: float | None = None
    put_writing_near_spot: float | None = None
    notional_oi: float | None = None
    notional_volume: float | None = None

    @field_validator("underlying", "source")
    @classmethod
    def _normalize_nonempty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


__all__ = [
    "ChainMetrics",
    "OptionChain",
    "OptionContract",
    "OptionRight",
    "OptionsMarket",
]
