"""Strategy descriptor and decorator used by every plugin file.

A strategy comes in one of two flavors:

- **callable** (`fn(df) -> list[Trade]`) — the pine-port style used by
  `screener.research.pine_runner`. Register with ``@strategy("name") def fn(df)``.
- **expression** (entry/exit Pine strings) — used by the historical/rolling
  backtester. Register with ``register_expression_strategy(...)``.

Strategies that need bar prep before the backtester evaluates signals attach a
``prepare_bars`` hook and an optional ``required_lookback``. This replaces the
``if cfg.strategy_name == ...`` branches that used to live in the core.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any, Callable, Literal, Optional, TypeVar, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, SkipValidation, field_validator

from screener._registry import Registry, autodiscover
from screener.backtester.data import PriceFetcher
from screener.backtester.models import BacktestConfig
from screener.strategies.trades import Trade


StrategyFn = Callable[[pd.DataFrame], list[Trade]]
F = TypeVar("F", bound=Callable[..., Any])
V = TypeVar("V")


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
    """Shared identity for the registry's two explicit strategy shapes."""

    kind: Literal["callable", "expression"]
    name: str

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("strategy name must not be empty")
        return normalized


class CallableStrategySpec(StrategySpec):
    """A strategy executed directly against an OHLCV frame."""

    kind: Literal["callable"] = "callable"
    callable_fn: StrategyFn


class ExpressionStrategySpec(StrategySpec):
    """A backtest strategy expressed as entry/exit rules and optional bar prep."""

    kind: Literal["expression"] = "expression"
    entry: str
    exit: Optional[str] = None
    prepare_bars: Optional[PrepareBarsFn] = None
    required_lookback: Optional[LookbackFn] = None

    @field_validator("entry")
    @classmethod
    def _normalize_entry(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("strategy entry must not be empty")
        return normalized


registry: Registry[StrategySpec] = Registry("strategy")


class DerivedView(Mapping[str, V]):
    """Read-only, live ``name -> value`` projection of :data:`registry`.

    This is *not* a stored dict: every lookup and iteration re-reads the
    underlying :data:`registry`, so there is no second copy of the strategy
    table that can drift out of sync (e.g. if a plugin registers late). The
    ``project`` callback maps a :class:`StrategySpec` to a value, or to ``None``
    to exclude that spec from the view.

    Exists so the historical import sites — ``registry.STRATEGIES`` (callable
    strategies for the pine runner) and ``expressions.NAMED_STRATEGIES``
    (entry/exit expression strategies for the backtester) — keep working as
    thin derived accessors of the one registry.
    """

    def __init__(self, project: Callable[[StrategySpec], Optional[V]]) -> None:
        self._project = project

    def __getitem__(self, key: str) -> V:
        value = self._project(registry.get(key))
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return (
            name for name, spec in registry.items() if self._project(spec) is not None
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)


def strategy(
    name: str,
    **meta: Any,
) -> Callable[[F], F]:
    """Register a callable ``(frame) -> trades`` strategy."""

    def _wrap(value: F) -> F:
        spec = CallableStrategySpec(
            name=name,
            callable_fn=cast(StrategyFn, value),
        )
        registry.add(name, spec, **meta)
        return value

    return _wrap


def register_expression_strategy(
    name: str,
    *,
    entry: str,
    exit: Optional[str] = None,
    prepare_bars: Optional[PrepareBarsFn] = None,
    required_lookback: Optional[LookbackFn] = None,
    **meta: Any,
) -> ExpressionStrategySpec:
    """Register an expression strategy directly, without a fake function body."""
    spec = ExpressionStrategySpec(
        name=name,
        entry=entry,
        exit=exit,
        prepare_bars=prepare_bars,
        required_lookback=required_lookback,
    )
    registry.add(name, spec, **meta)
    return spec


def discover_plugins() -> None:
    """Import every plugin module so its ``@strategy`` decorators fire."""
    from screener.strategies import plugins

    autodiscover(plugins)
