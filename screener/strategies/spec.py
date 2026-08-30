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

import numpy as np
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

#: Optional panel form of a recipe: the same column for many tickers at once,
#: over ``(bars, symbols)`` arrays keyed "open"/"high"/"low"/"close"/"volume".
#: It exists for the recipes whose indicator recurses bar by bar, where the
#: per-ticker form pays one Python step per bar *per ticker*. Only the shape
#: changes; the arithmetic is the same call into ``screener/indicators/``.
BarColumnPanelFn = Callable[[Mapping[str, np.ndarray]], np.ndarray]

#: Fields offered to a panel recipe, when every frame in the group has them.
_PANEL_FIELDS = ("open", "high", "low", "close", "volume")

#: Attribute a ``@bar_column`` recipe carries its warm-up on.
_BAR_COLUMN_LOOKBACK = "_bar_column_lookback"

#: Attribute a recipe carries its optional panel form on.
_BAR_COLUMN_PANEL = "_bar_column_panel"

#: Attribute marking a ``prepare_bars`` hook as derived from ``bar_columns``
#: rather than hand-written. Consumers that can apply the columns themselves
#: (the pine_runner) use it to tell "prep I can reproduce" from "prep I cannot".
_DERIVED_FROM_BAR_COLUMNS = "_derived_from_bar_columns"


def bar_column(
    required_lookback: int, panel: BarColumnPanelFn | None = None
) -> Callable[[BarColumnFn], BarColumnFn]:
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

    ``panel`` is an optional second form of the same recipe that builds the
    column for a whole group of tickers at once; see :data:`BarColumnPanelFn`.
    It is an optimisation only - a recipe without one is still correct, just
    slower - and :func:`apply_bar_columns_to_panel` is the only caller.
    """
    if required_lookback < 0:
        raise ValueError("bar column lookback must not be negative")

    def _wrap(fn: BarColumnFn) -> BarColumnFn:
        setattr(fn, _BAR_COLUMN_LOOKBACK, required_lookback)
        setattr(fn, _BAR_COLUMN_PANEL, panel)
        return fn

    return _wrap


def bar_column_panel_fn(build: BarColumnFn) -> BarColumnPanelFn | None:
    """The recipe's panel form, or None when it only has the per-ticker one."""
    return cast("BarColumnPanelFn | None", getattr(build, _BAR_COLUMN_PANEL, None))


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


#: How much taller than its tallest member a shared calendar may be before a
#: group is not worth stacking. A field is one calendar with symbols starting
#: and ending on different dates, so the union is a little taller than any one
#: of them; anything beyond this is a mixed field whose panel would be mostly
#: padding.
_MAX_PANEL_PADDING = 2.0


def _panel_groups(
    bars_by_tv: Mapping[str, pd.DataFrame],
) -> list[tuple[list[str], np.ndarray, int]]:
    """Frames that can be stacked, as ``(members, offsets, height)`` groups.

    Two passes, because most of a field shares one calendar exactly and the
    rest does not. The first groups frames with identical indexes, using
    :func:`screener.backtester.pine.panel_index_key` - the rule the engine's
    own panel paths use - and needs no padding at all. The second offers what
    is left to :func:`_aligned_panel_groups`, which pads. Running the union
    pass first would be worse than useless: one late listing's extra dates
    would put a hole in every frame that shares the main calendar.
    """
    from screener.backtester.pine import panel_index_key

    exact: dict[object, list[str]] = {}
    for tv, bars in bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        key = panel_index_key(bars.index)
        if key is not None:
            exact.setdefault(key, []).append(tv)

    groups: list[tuple[list[str], np.ndarray, int]] = []
    remaining: dict[str, pd.DataFrame] = {}
    for members in exact.values():
        if len(members) < 2:
            # A group of one saves nothing here, but it may still align with
            # the frames the exact rule could not group.
            remaining[members[0]] = bars_by_tv[members[0]]
            continue
        groups.append(
            (
                members,
                np.zeros(len(members), dtype=int),
                int(bars_by_tv[members[0]].shape[0]),
            )
        )
    return groups + _aligned_panel_groups(remaining)


def _aligned_panel_groups(
    bars_by_tv: Mapping[str, pd.DataFrame],
) -> list[tuple[list[str], np.ndarray, int]]:
    """Group frames whose calendars are contiguous slices of one calendar.

    Returns ``(members, offsets, height)`` per group: each member's bars occupy
    rows ``offset .. offset + len(bars)`` of a shared calendar of ``height``
    rows, so the group stacks into one array with the shorter histories padded.

    Requiring a *contiguous* slice is the safety rule. Symbols in one field
    share a trading calendar and differ only in when their history starts and
    ends; a frame with an interior hole is not a slice of anything and is left
    out, because padding cannot express a gap. Members keep their own indexes -
    nothing is relabelled - so unlike ``pine.panel_index_key`` this does not
    have to reject frames whose labels merely compare equal.
    """
    from screener.backtester.pine import is_naive_numpy_datetime_index

    by_calendar: dict[str, list[tuple[str, np.ndarray]]] = {}
    for tv, bars in bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        index = bars.index
        if not is_naive_numpy_datetime_index(index) or not index.is_unique:
            continue
        by_calendar.setdefault(str(index.dtype), []).append(
            (tv, index.to_numpy().view("i8"))
        )

    groups: list[tuple[list[str], np.ndarray, int]] = []
    for calendar in by_calendar.values():
        if len(calendar) < 2:
            continue
        union = np.unique(np.concatenate([ticks for _, ticks in calendar]))
        members: list[str] = []
        offsets: list[int] = []
        for tv, ticks in calendar:
            offset = int(np.searchsorted(union, ticks[0]))
            slot = union[offset : offset + ticks.size]
            if slot.size == ticks.size and np.array_equal(slot, ticks):
                members.append(tv)
                offsets.append(offset)
        if len(members) < 2:
            continue
        tallest = max(bars_by_tv[tv].shape[0] for tv in members)
        if union.size > _MAX_PANEL_PADDING * tallest:
            continue
        groups.append((members, np.asarray(offsets), int(union.size)))
    return groups


def apply_bar_columns_to_panel(
    bar_columns: Mapping[str, BarColumnFn] | None,
    bars_by_tv: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """``apply_bar_columns`` over many tickers, in one pass where a recipe allows it.

    A recipe that declares a panel form is built once per group of tickers on
    one calendar rather than once per ticker. That is what makes the two
    recursive recipes affordable: their indicators step through bars in Python,
    so the per-ticker form costs one step per bar per ticker, while the panel
    form costs one step per bar for the whole group.

    Grouping is :func:`_panel_groups`, and a symbol whose history is
    shorter than the group's calendar is padded with NaN at the ends. A panel
    recipe must therefore give a padded column exactly what the symbol's own
    frame gives - a NaN run outside the history means "no bars here", never "a
    bar with no price". Anything the grouping rejects, and any recipe with no
    panel form, falls back to the per-ticker path.
    """
    if not bar_columns:
        return dict(bars_by_tv)
    panel_builders = {}
    for column, build in bar_columns.items():
        panel_build = bar_column_panel_fn(build)
        if panel_build is not None:
            panel_builders[column] = panel_build
    if not panel_builders:
        return {
            tv: apply_bar_columns(bar_columns, bars) for tv, bars in bars_by_tv.items()
        }
    per_ticker = {
        column: build
        for column, build in bar_columns.items()
        if column not in panel_builders
    }

    prepared: dict[str, pd.DataFrame] = {}
    for members, offsets, height in _panel_groups(bars_by_tv):
        frames = [bars_by_tv[tv] for tv in members]
        fields: dict[str, np.ndarray] = {}
        for field in _PANEL_FIELDS:
            if not all(field in frame.columns for frame in frames):
                continue
            stacked = np.full((height, len(frames)), np.nan, dtype=float)
            for position, (frame, offset) in enumerate(zip(frames, offsets)):
                values = frame[field].to_numpy(dtype=float)
                stacked[offset : offset + values.size, position] = values
            fields[field] = stacked
        try:
            built = {column: build(fields) for column, build in panel_builders.items()}
        except KeyError:
            # A recipe wanted a field this group does not carry. Leave the
            # group to the per-ticker path, which raises where it always did.
            continue
        for position, (tv, frame, offset) in enumerate(zip(members, frames, offsets)):
            prepared_frame = apply_bar_columns(per_ticker, frame)
            if prepared_frame is frame:
                prepared_frame = frame.copy()
            rows = slice(offset, offset + frame.shape[0])
            for column, values in built.items():
                prepared_frame[column] = pd.Series(
                    values[rows, position], index=frame.index, dtype=float
                )
            prepared[tv] = prepared_frame

    for tv, bars in bars_by_tv.items():
        if tv not in prepared:
            prepared[tv] = apply_bar_columns(bar_columns, bars)
    return prepared


def _prepare_from_bar_columns(bar_columns: Mapping[str, BarColumnFn]) -> PrepareBarsFn:
    """Lift bar-local column builders into the panel-level prepare_bars hook.

    Declaring ``bar_columns`` therefore costs the backtester nothing new: it
    keeps calling ``prepare_bars`` exactly as before. The same mapping is read
    directly by ``screener.strategies.registry`` for the pine_runner, so the
    columns an expression reads are defined once rather than once per consumer.
    """

    def _prepare(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
        return apply_bar_columns_to_panel(bar_columns, ctx.bars_by_tv)

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
    ``BacktestConfig`` defaults, so an attached profile that restates them
    changes nothing. Both callers resolve and apply it: the screen in
    :mod:`screener.screen_candidates` and the rolling backtest in
    ``screener.backtester.workflow._effective_gates``.

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
    # effective defaults (``DEFAULT_STRATEGY_PROFILE``). Both the screen and
    # the rolling backtest read it, which is what keeps the two paths from
    # drifting by config instead of by code.
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
        ha_momentum,
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
    raise instead of silently doing nothing. The backtest CLI passes the flags
    the user actually typed as overrides, so a flag left at its option default
    lets the profile speak.
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
