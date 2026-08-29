"""Strategy descriptor and decorator used by every plugin file.

A strategy comes in one of two flavors:

- **callable** (`fn(df) -> list[ResearchTrade]`) — the pine-port style used by
  `screener.research.pine_runner`. Register with ``@strategy("name") def fn(df)``.
- **expression** (entry/exit Pine strings) — used by the historical/rolling
  backtester. Register with ``register_expression_strategy(...)``.

Strategies that need bar prep before the backtester evaluates signals attach a
``prepare_bars`` hook. That hook requires ``required_lookback``: columns written
in ``prepare_bars`` are invisible to the entry expression, so fetch cannot size
history from the AST. ``low_volatility`` is the worked example - it writes
``vol_252`` from 253 bars behind an entry of ``vol_252 > 0``. This replaces the ``if cfg.strategy_name == ...``
branches that used to live in the core.

Expression strategies may also declare a ``StrategyProfile``: the candidate
gate defaults (liquidity filters, regime gate, blackout, sector
neutralisation) that screen and backtest both load, mirroring
``SignalPanelInputs``, plus an optional ``tv_prefilter`` naming the criterion
whose vendor filters cut the field before bars are fetched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from datetime import date
from typing import Any, Literal, TypeVar, cast

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    SkipValidation,
    field_validator,
    model_validator,
)

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

#: Builds one derived column from a single ticker's OHLCV frame. Pure and
#: bar-local on purpose: it sees no panel, no market and no fetcher, which is
#: what lets the same declaration serve the backtester and the pine_runner.
BarColumnFn = Callable[[pd.DataFrame], pd.Series]

#: Attribute a ``@bar_column`` recipe carries its warm-up on.
_BAR_COLUMN_LOOKBACK = "_bar_column_lookback"

#: Attribute marking a ``prepare_bars`` hook as derived from ``bar_columns``
#: rather than hand-written. Consumers that can apply the columns themselves
#: (the pine_runner) use it to tell "prep I can reproduce" from "prep I cannot".
_DERIVED_FROM_BAR_COLUMNS = "_derived_from_bar_columns"


def bar_column(required_lookback: int) -> Callable[[BarColumnFn], BarColumnFn]:
    """Declare how many bars of history a column recipe needs to be valid.

    The warm-up is invisible to the entry/exit AST - an expression naming
    ``bb_upper`` says nothing about the 350-bar window behind it - so each
    recipe carries its own, and ``register_expression_strategy`` folds the
    largest into the spec's ``required_lookback``. Leaving it undeclared is
    rejected at registration rather than defaulting to zero: a silent zero is
    what let a 350-bar Bollinger column ship behind a one-bar ``crossover``.

    The unit matches :func:`screener.backtester.pine.required_lookback`: the
    rolling window length, so a column built from ``rolling(20).shift(1)``
    declares 21.
    """
    if required_lookback < 0:
        raise ValueError("bar column lookback must not be negative")

    def _wrap(fn: BarColumnFn) -> BarColumnFn:
        setattr(fn, _BAR_COLUMN_LOOKBACK, required_lookback)
        return fn

    return _wrap


def bar_columns_lookback(bar_columns: Mapping[str, BarColumnFn]) -> int:
    """Largest warm-up any of the declared columns needs, in bars."""
    floor = 0
    for column, build in bar_columns.items():
        declared = getattr(build, _BAR_COLUMN_LOOKBACK, None)
        if declared is None:
            raise ValueError(
                f"bar column {column!r} does not declare its warm-up: "
                "decorate the recipe with @bar_column(n)"
            )
        floor = max(floor, int(declared))
    return floor


def apply_bar_columns(
    bar_columns: Mapping[str, BarColumnFn] | None, bars: pd.DataFrame
) -> pd.DataFrame:
    """Return ``bars`` with each declared column computed and attached."""
    if not bar_columns or bars is None or bars.empty:
        return bars
    frame = bars.copy()
    for column, build in bar_columns.items():
        frame[column] = build(frame)
    return frame


def _prepare_from_bar_columns(bar_columns: Mapping[str, BarColumnFn]) -> PrepareBarsFn:
    """Lift bar-local column builders into the panel-level prepare_bars hook.

    Declaring ``bar_columns`` therefore costs the backtester nothing new: it
    keeps calling ``prepare_bars`` exactly as before. The same mapping is read
    directly by ``screener.strategies.registry`` for the pine_runner, so the
    columns an expression reads are defined once rather than once per consumer.
    """

    def _prepare(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
        return {
            tv: apply_bar_columns(bar_columns, bars)
            for tv, bars in ctx.bars_by_tv.items()
        }

    setattr(_prepare, _DERIVED_FROM_BAR_COLUMNS, True)
    return _prepare


def is_derived_from_bar_columns(prepare_bars: PrepareBarsFn | None) -> bool:
    """True when the hook is just ``bar_columns`` lifted to the panel level."""
    return bool(getattr(prepare_bars, _DERIVED_FROM_BAR_COLUMNS, False))


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

    #: Name of the criterion in :mod:`screener.criteria` whose TradingView
    #: filters cut the field before bars are downloaded, or ``None`` for a
    #: strategy that has no vendor-side prefilter.
    #:
    #: This is the one field that is *not* mirrored from
    #: ``SignalPanelInputs``, and deliberately so: it names an optimisation,
    #: never a rule. The prefilter may only ever remove names the bar rules
    #: would have removed anyway, so a run with ``--universe`` (no prefilter)
    #: and a default run must reach the same candidates on the names both saw.
    #: A criterion *name* rather than the filter list itself, so this module
    #: stays free of the ``tradingview_screener`` import and the filters keep
    #: living exactly once, in ``screener/criteria/plugins/``.
    tv_prefilter: str | None = None


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
    # Bar-local derived columns the entry/exit expressions may reference by
    # name. Keeps the Pine grammar fixed (plan D10): a new indicator becomes a
    # column, never a new function in the parser.
    bar_columns: SkipValidation[Mapping[str, BarColumnFn]] | None = None
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

    @model_validator(mode="after")
    def _prepare_bars_requires_lookback(self) -> ExpressionStrategySpec:
        if self.prepare_bars is not None and self.required_lookback is None:
            raise ValueError(
                "prepare_bars requires required_lookback: columns written in "
                "prepare_bars are invisible to the entry expression, so fetch "
                "cannot size history from the AST"
            )
        return self


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
    bar_columns: Mapping[str, BarColumnFn] | None = None,
    **meta: Any,
) -> ExpressionStrategySpec:
    """Register an expression strategy directly, without a fake function body."""
    if bar_columns and prepare_bars is not None:
        # The two consumers would disagree: the backtester runs only the hook
        # (so the columns never exist and the expression raises PineNameError),
        # while the pine_runner applies only the columns. A hook that needs the
        # columns should call apply_bar_columns itself.
        raise ValueError(
            f"strategy {name!r} declares both bar_columns and prepare_bars; "
            "fold the columns into the hook with apply_bar_columns"
        )
    if bar_columns:
        # A bar column's warm-up is invisible to the entry/exit AST, so fold it
        # into the spec's own lookback. Without this the fetch window and the
        # candidate mask both size themselves off the expression alone, and a
        # long-window column is silently NaN over the whole backtest.
        column_floor = bar_columns_lookback(bar_columns)
        prepare_bars = _prepare_from_bar_columns(bar_columns)
        declared_lookback = required_lookback

        def _lookback_with_columns() -> int:
            explicit = declared_lookback() if declared_lookback else 0
            return max(column_floor, explicit)

        required_lookback = _lookback_with_columns
    spec = ExpressionStrategySpec(
        name=name,
        entry=entry,
        exit=exit,
        prepare_bars=prepare_bars,
        required_lookback=required_lookback,
        profile=profile,
        bar_columns=bar_columns,
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
