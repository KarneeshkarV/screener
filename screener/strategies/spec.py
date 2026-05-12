"""Strategy descriptor and decorator used by every plugin file.

A strategy comes in one of two flavors:

- **callable** (`fn(df) -> list[Trade]`) — the pine-port style used by
  `screener.research.pine_runner`. Register with ``@strategy("name") def fn(df)``.
- **expression** (entry/exit Pine strings) — used by the historical/rolling
  backtester. Register with ``@strategy("name", entry="...", exit="...")``.

Strategies that need bar prep before the backtester evaluates signals attach a
``prepare_bars`` hook and an optional ``required_lookback``. This replaces the
``if cfg.strategy_name == ...`` branches that used to live in the core.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional, TypeVar, cast

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    SkipValidation,
    field_validator,
    model_validator,
)

from screener._registry import Registry, autodiscover
from screener.backtester.data import PriceFetcher
from screener.backtester.models import BacktestConfig
from screener.strategies.trades import Trade


StrategyFn = Callable[[pd.DataFrame], list[Trade]]
F = TypeVar("F", bound=Callable[..., Any])


class PrepareCtx(BaseModel):
    """Inputs handed to a strategy's ``prepare_bars`` hook."""

    cfg: BacktestConfig
    bars_by_tv: dict[str, pd.DataFrame]
    price_panel: dict[str, pd.DataFrame]
    tv_symbols: list[str]
    start: date
    end: date
    fetcher: SkipValidation[PriceFetcher]
    warnings: list[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)


PrepareBarsFn = Callable[[PrepareCtx], dict[str, pd.DataFrame]]
LookbackFn = Callable[[], int]


class StrategySpec(BaseModel):
    """One strategy in the registry. Has callable OR expression form (or both)."""

    name: str
    callable_fn: Optional[StrategyFn] = None
    entry: Optional[str] = None
    exit: Optional[str] = None
    prepare_bars: Optional[PrepareBarsFn] = None
    required_lookback: Optional[LookbackFn] = None

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("strategy name must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_sources(self) -> "StrategySpec":
        if self.callable_fn is None and self.entry is None:
            raise ValueError(
                f"strategy {self.name!r}: either callable_fn or entry must be set"
            )
        return self


registry: Registry[StrategySpec] = Registry("strategy")


def strategy(
    name: str,
    *,
    entry: Optional[str] = None,
    exit: Optional[str] = None,
    prepare_bars: Optional[PrepareBarsFn] = None,
    required_lookback: Optional[LookbackFn] = None,
    **meta: Any,
) -> Callable[[F], F]:
    """Decorator. Two shapes:

    Callable strategy (decorates a fn ``(df) -> list[Trade]``):
        ``@strategy("supertrend") def strat_supertrend(df): ...``

    Expression-only strategy (decorates a placeholder; body is ignored):
        ``@strategy("ema_trend", entry="...", exit="...") def _ema_trend(): pass``
    """

    def _wrap(value: F) -> F:
        spec = StrategySpec(
            name=name,
            callable_fn=cast(StrategyFn, value) if entry is None else None,
            entry=entry,
            exit=exit,
            prepare_bars=prepare_bars,
            required_lookback=required_lookback,
        )
        registry.add(name, spec, **meta)
        return value

    return _wrap


def discover_plugins() -> None:
    """Import every plugin module so its ``@strategy`` decorators fire."""
    from screener.strategies import plugins

    autodiscover(plugins)
