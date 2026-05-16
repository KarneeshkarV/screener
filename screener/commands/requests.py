"""Pydantic request models for Click command handlers (stable CLI boundaries)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MarketId = Literal["us", "india"]


class ScreenRequest(BaseModel):
    """Inputs for ``screener screen``."""

    market: MarketId
    criteria_names: tuple[str, ...]
    limit: int
    order_by: str
    output_csv: bool
    detail: bool
    refresh: bool
    cache_ttl: str

    model_config = ConfigDict(frozen=True)

    @field_validator("criteria_names")
    @classmethod
    def _non_empty_criteria(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("criteria_names must not be empty")
        return value

    @field_validator("order_by", "cache_ttl")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class GarpRequest(BaseModel):
    """Inputs for ``screener garp``."""

    market: MarketId
    universe_size: int = Field(default=200)
    limit: int = Field(default=30)
    workers: int = Field(default=8)
    output_csv: bool = False
    refresh: bool = False
    cache_ttl: str = Field(default="1d")

    model_config = ConfigDict(frozen=True)

    @field_validator("cache_ttl")
    @classmethod
    def _strip_cache_ttl(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("cache_ttl must not be empty")
        return normalized


class PromoterBuysRequest(BaseModel):
    """Inputs for ``screener promoter-buys``."""

    market: MarketId
    universe_size: int = Field(default=200)
    limit: int = Field(default=30)
    min_change_pct: float = Field(default=0.0)
    min_yf_net_pct: float | None = None
    require_both: bool = False
    min_market_cap: float | None = None
    workers: int = Field(default=10)
    output_csv: bool = False
    refresh: bool = False
    cache_ttl: str = Field(default="15m")

    model_config = ConfigDict(frozen=True)

    @field_validator("cache_ttl")
    @classmethod
    def _strip_cache_ttl(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("cache_ttl must not be empty")
        return normalized
