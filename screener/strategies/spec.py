"""Strategy descriptor and decorator used by every plugin file.

A strategy comes in one of two flavors:

- **callable** (`fn(df) -> list[ResearchTrade]`) — the pine-port style used by
  `screener.research.pine_runner`. Register with ``@strategy("name") def fn(df)``.
- **expression** (entry/exit Pine strings) — used by the historical/rolling
  backtester. Register with ``register_expression_strategy(...)``.

Strategies that need bar prep before the backtester evaluates signals attach a
``prepare_bars`` hook and an optional ``required_lookback``. This replaces the
``if cfg.strategy_name == ...`` branches that used to live in the core.

Expression strategies may also declare a ``StrategyProfile``: the candidate
gate defaults (liquidity filters, regime gate, blackout, sector
neutralisation) that screen and backtest will both load, mirroring
``SignalPanelInputs``. Stage 1 declares it; nothing consumes it yet.
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


class StrategyProfile(BaseModel):
    """Per-strategy candidate-gate defaults for screen and backtest.

    Mirrors the eligibility inputs of
    :class:`~screener.backtester.signal_panel.SignalPanelInputs` field for
    field, minus the run-scoped universe and venue fields collected in
    ``RUN_SCOPED_SIGNAL_PANEL_FIELDS``: a strategy declares how a candidate is
    judged, not which names or venue a run covers.

    The field list is derived, not restated: the partition against
    ``SIGNAL_PANEL_INPUT_FIELDS`` is asserted at import time in
    :mod:`screener.backtester.signal_panel` (which imports this module - the
    sanctioned backtester -> strategies direction), so a gate added to
    ``SignalPanelInputs`` cannot ship without either mirroring it here or
    classifying it as run-scoped. Scalar values are the effective
    ``BacktestConfig`` defaults, so an attached profile changes nothing until
    a caller resolves and applies it (stage 1 wires no caller).

    ``entry_expr``/``exit_expr`` are ``None`` when unset, meaning "the spec's
    own ``entry``/``exit`` stand" - they stay required on
    :class:`ExpressionStrategySpec`.
    """

    entry_expr: str | None = None
    exit_expr: str | None = None
    regime_filter: tuple[str, ...] = ()
    earnings_blackout_days: int | None = None
    sector_neutral: bool = False
    min_price: float | None = None
    min_avg_dollar_volume: float | None = None
    avg_dollar_volume_window: int = 20


# The shared baseline every strategy without its own profile resolves to.
# Equal to the effective BacktestConfig defaults by construction.
DEFAULT_STRATEGY_PROFILE = StrategyProfile()


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
    # Declared candidate-gate defaults. ``None`` keeps the plugin on the
    # effective defaults (``DEFAULT_STRATEGY_PROFILE``); nothing reads it yet.
    profile: StrategyProfile | None = None

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
    profile: StrategyProfile | None = None,
    **meta: Any,
) -> ExpressionStrategySpec:
    """Register an expression strategy directly, without a fake function body."""
    spec = ExpressionStrategySpec(
        name=name,
        entry=entry,
        exit=exit,
        prepare_bars=prepare_bars,
        required_lookback=required_lookback,
        profile=profile,
    )
    registry.add(name, spec, **meta)
    return spec


def discover_plugins() -> None:
    """Import every plugin module so its ``@strategy`` decorators fire."""
    from screener.strategies.plugins import (  # noqa: F401
        awesome_oscillator,
        bb_breakout,
        bb_pattern,
        breakout,
        donchian_breakout,
        ema150_200_revenue,
        ema_stack_lowvol,
        ema_trend,
        heikin_ashi,
        low_volatility,
        ma_cross,
        ma_cross_regime,
        ma_cross_st_entry,
        ma_cross_st_exit,
        macd_oscillator,
        macd_rsi,
        mark_minervini,
        minervini_filtered,
        mom_lowvol_combo,
        momentum_12_1,
        parabolic_sar,
        rs_breakout,
        rs_momentum_regime,
        rsi_ema,
        rsi_pattern,
        rsi_reversion,
        shooting_star,
        supertrend,
        supertrend_rsi,
        vivek_equity_tool,
    )


def resolve_strategy_spec(name: str | None) -> StrategySpec | None:
    """Resolve a registered or dynamic strategy through one canonical path.

    A name that is not a strategy but *is* a screen-only scorer raises rather
    than resolving to ``None``: silently reporting "unknown strategy" for
    ``value`` or ``quality`` hides the real reason those cannot be backtested.
    """
    if name is None:
        return None
    discover_plugins()
    from screener.strategies.combo import is_combo_strategy, resolve_combo_spec

    if is_combo_strategy(name):
        return resolve_combo_spec(name)
    spec = registry.get_optional(name)
    if spec is None:
        from screener.scoring import ensure_backtestable_scorer

        ensure_backtestable_scorer(name)
    return spec


def resolve_strategy_profile(
    spec: ExpressionStrategySpec | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> StrategyProfile:
    """Effective candidate gates: the declared profile, then explicit overrides.

    A spec without a ``profile`` resolves to :data:`DEFAULT_STRATEGY_PROFILE`,
    so the resolved value is total for every expression strategy. Overrides
    use ``SignalPanelInputs`` field names, win over both the defaults and any
    attached profile, and are validated by rebuilding the model; unknown keys
    raise instead of silently doing nothing. No CLI path calls this yet -
    stage 1 is additive only (D18).
    """
    base = (
        spec.profile
        if isinstance(spec, ExpressionStrategySpec) and spec.profile is not None
        else DEFAULT_STRATEGY_PROFILE
    )
    if not overrides:
        return base
    unknown = [key for key in overrides if key not in StrategyProfile.model_fields]
    if unknown:
        raise ValueError(
            f"unknown strategy-profile override(s): {sorted(unknown)}; "
            f"known gates: {sorted(StrategyProfile.model_fields)}"
        )
    return StrategyProfile(**{**base.model_dump(), **overrides})
