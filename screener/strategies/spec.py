"""Strategy descriptor and decorator used by every plugin file.

A strategy comes in one of two flavors:

- **callable** (`fn(df) -> list[ResearchTrade]`) — the pine-port style used by
  `screener.research.pine_runner`. Register with ``@strategy("name") def fn(df)``.
- **expression** (entry/exit Pine strings) — used by the historical/rolling
  backtester. Register with ``register_expression_strategy(...)``.

Strategies that need bar prep before the backtester evaluates signals attach a
``prepare_bars`` hook and an optional ``required_lookback``. This replaces the
``if cfg.strategy_name == ...`` branches that used to live in the core.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import date
from typing import Any, Literal, TypeVar, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, SkipValidation, field_validator

from screener._registry import Registry
from screener.backtester.data import PriceFetcher
from screener.strategies.trades import ResearchTrade

StrategyFn = Callable[[pd.DataFrame], list[ResearchTrade]]
F = TypeVar("F", bound=Callable[..., Any])
V = TypeVar("V")


class PrepareCtx(BaseModel):
    """Inputs handed to a strategy's ``prepare_bars`` hook."""

    market: str
    benchmark: str
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
    exit: str | None = None
    prepare_bars: PrepareBarsFn | None = None
    required_lookback: LookbackFn | None = None

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

    def __init__(self, project: Callable[[StrategySpec], V | None]) -> None:
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
    exit: str | None = None,
    prepare_bars: PrepareBarsFn | None = None,
    required_lookback: LookbackFn | None = None,
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
    from screener.strategies.plugins import (  # noqa: F401
        adx_trend,
        accounting_anomalies,
        beta_volatility,
        calendar_seasonality,
        delivery_accumulation,
        earnings_momentum,
        keltner_squeeze_breakout,
        quality_defensive,
        reversal_52week,
        trend_technical,
        vcp_breakout,
        vol_expansion_breakout,
        vol_target_lowvol,
        volume_flow,
        awesome_oscillator,
        bb_breakout,
        bb_pattern,
        bollinger_reversion,
        breakout,
        cci,
        connors_rsi2,
        donchian_breakout,
        ema150_200_revenue,
        ema_trend,
        fifty_two_week_high,
        golden_cross,
        heikin_ashi,
        keltner,
        long_term_reversal,
        low_volatility,
        ma_cross,
        ma_cross_regime,
        ma_cross_st_entry,
        ma_cross_st_exit,
        macd_cross,
        macd_oscillator,
        macd_rsi,
        mark_minervini,
        minervini_filtered,
        mom_lowvol_combo,
        momentum_12_1,
        chandelier_breakout,
        momentum_quality,
        quality_mom_lowvol,
        supertrend_expr,
        turtle_breakout,
        value_garp,
        vwap_reversion,
        nifty_momentum,
        rs_trend,
        vwap_trend,
        parabolic_sar,
        rs_breakout,
        rs_momentum_regime,
        rsi_ema,
        rsi_pattern,
        rsi_reversion,
        shooting_star,
        short_term_reversal,
        stochastic,
        supertrend,
        supertrend_rsi,
        trading_range_break,
        turn_of_month,
        vivek_equity_tool,
        williams_r,
        dual_momentum,
        long_only_trend,
        momentum_riskmanaged,
        time_series_momentum,
    )


def resolve_strategy_spec(name: str | None) -> StrategySpec | None:
    """Resolve a registered or dynamic strategy through one canonical path."""
    if name is None:
        return None
    discover_plugins()
    from screener.strategies.combo import is_combo_strategy, resolve_combo_spec

    if is_combo_strategy(name):
        return resolve_combo_spec(name)
    return registry.get_optional(name)
