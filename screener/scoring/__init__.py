"""Per-criterion ranking scores for the TradingView screen path.

Filters stay in ``screener.criteria``; this package owns the *ranking*
philosophy for each criterion name. Output is always written as
``setup_score`` so history, display, and CSV stay stable.

Every recipe declares its ``data_source``, and the two values are not
interchangeable:

* ``"snapshot"`` - reads TradingView's per-row snapshot columns (``RSI``,
  ``market_cap_basic``, ``relative_volume_10d_calc``, ``Perf.Y``, and every
  fundamental). A snapshot carries only *today's* value with no history and no
  point-in-time restatement, so replaying one through a backtest is lookahead.
  These recipes are screen-only, and :func:`ensure_backtestable_scorer` refuses
  them anywhere in the backtest path.
* ``"bars"`` - delegates to a shared price-only recipe in
  :mod:`screener.factors`, which the rolling backtester consumes through its
  own adapter. One formula, two consumers, identical numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal, TypeVar

import pandas as pd

from screener._registry import Registry
from screener.factors import PriceScoreSpec, get_price_score
from screener.scoring.bar_scores import DEFAULT_PRICE_ADJUSTMENT, PriceAdjustment

if TYPE_CHECKING:  # pragma: no cover - typing only
    from screener.backtester.data import PriceFetcher

ScoreFn = Callable[[pd.DataFrame], pd.Series]
_RegisteredScoreFn = TypeVar("_RegisteredScoreFn", bound=ScoreFn)

#: Where a recipe's inputs come from. ``"snapshot"`` recipes are screen-only;
#: only ``"bars"`` recipes are valid in the backtest path.
ScoreDataSource = Literal["snapshot", "bars"]
SNAPSHOT_SOURCE: ScoreDataSource = "snapshot"
BARS_SOURCE: ScoreDataSource = "bars"

OUTPUT_SCORE_COLUMN = "setup_score"
DEFAULT_SCORER_NAME = "ema"

registry: Registry[ScoreFn] = Registry("scorer")


@dataclass(frozen=True)
class ScoreSpec:
    """Resolved scorer used by the scanner plan + shape pipeline."""

    name: str
    columns: tuple[str, ...]
    score_fn: ScoreFn
    description: str = ""
    #: Declares what the recipe reads. ``"snapshot"`` means TradingView's
    #: as-of-today row, which is why such a recipe cannot be backtested.
    data_source: ScoreDataSource = SNAPSHOT_SOURCE
    #: Set when the recipe is bar-derived: the shared price-only spec from
    #: :mod:`screener.factors` that both the screen and the backtest evaluate.
    #: ``None`` means the recipe reads TradingView snapshot columns instead.
    bar_score: PriceScoreSpec | None = None


def _bar_derived_score_fn(name: str) -> ScoreFn:
    """Placeholder ``score_fn`` for a bar-derived recipe.

    A bar-derived recipe cannot be evaluated from a snapshot row, so calling it
    that way is a programming error, not a data gap.
    """

    def _fail(_df: pd.DataFrame) -> pd.Series:
        raise TypeError(
            f"scorer {name!r} is bar-derived: score it with apply_score(..., "
            "market=...) so its price history can be loaded, not from the "
            "TradingView snapshot row"
        )

    return _fail


def register_bar_scorer(
    name: str,
    price_score_name: str,
    *,
    description: str = "",
) -> None:
    """Register a criterion whose ranking recipe lives in :mod:`screener.factors`.

    ``columns`` stays empty on purpose: the recipe reads bars, so the scan does
    not need to request any extra TradingView column for it.
    """
    spec = get_price_score(price_score_name)
    registry.add(
        name,
        _bar_derived_score_fn(name),
        columns=(),
        description=description or spec.description,
        data_source=BARS_SOURCE,
        bar_score=spec,
    )


def scorer(
    name: str,
    *,
    columns: Sequence[str] = (),
    description: str = "",
    data_source: ScoreDataSource,
) -> Callable[[_RegisteredScoreFn], _RegisteredScoreFn]:
    """Register a ranking recipe for a criterion name.

    ``data_source`` has no default on purpose: whether a recipe can ever be
    replayed through history is a property its author must state, not one a
    reader should infer from which columns it happens to touch today.
    """

    def _wrap(fn: _RegisteredScoreFn) -> _RegisteredScoreFn:
        registry.add(
            name,
            fn,
            columns=tuple(columns),
            description=description,
            data_source=data_source,
        )
        return fn

    return _wrap


def get_scorer(name: str) -> ScoreSpec:
    """Look up one registered scorer by criterion name."""
    fn = registry.get(name)
    meta = registry.meta(name)
    return ScoreSpec(
        name=name,
        columns=tuple(meta.get("columns", ())),
        score_fn=fn,
        description=str(meta.get("description", "")),
        data_source=meta.get("data_source", SNAPSHOT_SOURCE),
        bar_score=meta.get("bar_score"),
    )


def default_scorer() -> ScoreSpec:
    """EMA setup score — historical default for ``order_by=setup_score``."""
    return get_scorer(DEFAULT_SCORER_NAME)


class IncompatibleScorerBlendError(ValueError):
    """Raised when a bar-derived scorer is averaged with another criterion.

    Both sides write ``setup_score`` on a 0-100 scale, but a bar-derived
    percentile is computed from price history over the scan's survivors and a
    snapshot composite is computed from TradingView columns. Averaging them
    would mix two incomparable rankings. The CLI turns this into a usage error
    so the refusal is not a traceback.
    """


def resolve_scorer(names: Sequence[str], *, strict: bool = True) -> ScoreSpec:
    """Resolve one or more criterion names into a single ranking recipe.

    * One name → that scorer.
    * Several → equal-weight average of each scorer; columns are the union.

    With ``strict=False`` an unregistered criterion name degrades to the
    default scorer instead of raising, so a missing ranking recipe cannot take
    down a whole screen (including ``--sort volume``, which never scores).
    """
    selected = tuple(names)
    if not selected:
        raise ValueError("resolve_scorer requires at least one criterion name")
    if not strict:
        try:
            return resolve_scorer(selected)
        except KeyError:
            return default_scorer()
    if len(selected) == 1:
        return get_scorer(selected[0])

    specs = [get_scorer(name) for name in selected]
    bar_derived = [spec.name for spec in specs if spec.bar_score is not None]
    if bar_derived:
        # A bar-derived recipe is scored from price history over the scan's
        # survivors; a snapshot recipe is a composite over TradingView columns.
        # Both write 0-100, but the percentiles are not over the same field, so
        # averaging them would blend two incomparable rankings.
        raise IncompatibleScorerBlendError(
            "cannot blend bar-derived scorer(s) "
            f"{sorted(bar_derived)} with other criteria: a bar-derived "
            "percentile is computed from price history, not from the snapshot "
            "row. Screen them one criterion at a time."
        )
    columns: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        for col in spec.columns:
            if col not in seen:
                seen.add(col)
                columns.append(col)

    descriptions = [s.description for s in specs if s.description]
    label = "+".join(selected)

    def blended(df: pd.DataFrame) -> pd.Series:
        parts = [spec.score_fn(df) for spec in specs]
        stacked = pd.concat(parts, axis=1)
        return stacked.mean(axis=1)

    return ScoreSpec(
        name=label,
        columns=tuple(columns),
        score_fn=blended,
        description="Equal blend: " + "; ".join(descriptions)
        if descriptions
        else f"Equal blend of {label}",
    )


class SnapshotOnlyScorerError(ValueError):
    """Raised when a screen-only scorer is used in the backtest path."""


def ensure_backtestable_scorer(name: str) -> None:
    """Refuse a snapshot-only scorer name anywhere in the backtest path.

    TradingView's snapshot columns are *today's* values: there is no history
    for them and no point-in-time restatement, so ranking a historical day by
    them would score that day with facts nobody had at the time. Only recipes
    whose ``data_source`` is ``"bars"`` are replayable.

    No-ops for a name that is not a registered scorer at all, so ordinary
    "unknown strategy" handling stays unchanged.
    """
    fn = registry.get_optional(name)
    if fn is None:
        return
    spec = get_scorer(name)
    if spec.data_source == BARS_SOURCE:
        return
    raise SnapshotOnlyScorerError(
        f"{name!r} is a screen-only scorer (data_source={spec.data_source!r}) "
        "and cannot be used in the backtest path: it reads TradingView "
        "snapshot columns, which carry only today's values, so replaying them "
        "through history is lookahead. Use a bar-derived factor strategy "
        f"instead (backtestable scorers: {sorted(backtestable_scorer_names())})."
    )


def backtestable_scorer_names() -> list[str]:
    """Names of the scorers whose recipe is bar-derived, hence replayable."""
    return [
        name
        for name in registry.names()
        if registry.meta(name).get("data_source") == BARS_SOURCE
    ]


def apply_score(
    df: pd.DataFrame,
    spec: ScoreSpec,
    *,
    market: str | None = None,
    as_of: date | None = None,
    fetcher: "PriceFetcher | None" = None,
    refresh: bool = False,
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT,
    strict: bool = False,
) -> pd.DataFrame:
    """Assign ``setup_score`` from ``spec`` without sorting or dropping columns.

    A bar-derived ``spec`` needs ``market`` so the scanned tickers' cached
    price history can be resolved; it also drops rows with too little history,
    because in the unified layer NaN means ineligible rather than rank-last.
    ``price_adjustment`` is used only by bar-derived recipes. It must match
    the backtest's ``--price-adjustment`` so the raw values are the same
    numbers. Snapshot recipes ignore it. TradingView serves one set of
    columns regardless of the adjustment.

    ``refresh`` is bar-derived only: it is forwarded to
    :func:`screener.backtester.data.build_price_fetcher` so the on-disk bar
    cache is asked to update. A failed download still merges leftover cache
    (the availability-first default). ``strict=True`` together with
    ``refresh=True`` refuses that merge and raises
    :class:`~screener.providers.StaleDataError` instead. ``strict`` without
    ``refresh`` is ignored here; that flag's scan-snapshot meaning lives
    on the TradingView fetch. Snapshot recipes ignore both flags.

    An empty frame short-circuits to an empty scored frame for every ``spec``,
    bar-derived included: with no rows there is no price history to resolve, so
    the missing ``market`` is not yet an error.
    """
    if df.empty:
        return df.assign(**{OUTPUT_SCORE_COLUMN: pd.Series(dtype=float)})
    if spec.bar_score is not None:
        if market is None:
            raise ValueError(
                f"scorer {spec.name!r} is bar-derived and needs a market to "
                "resolve price history; pass apply_score(..., market=...)"
            )
        from screener.scoring.bar_scores import apply_bar_score

        return apply_bar_score(
            df,
            spec.bar_score,
            market=market,
            output_column=OUTPUT_SCORE_COLUMN,
            as_of=as_of,
            fetcher=fetcher,
            refresh=refresh,
            price_adjustment=price_adjustment,
            strict=strict,
        )
    scores = spec.score_fn(df)
    return df.assign(**{OUTPUT_SCORE_COLUMN: pd.to_numeric(scores, errors="coerce")})


def _register_plugins() -> None:
    from screener.scoring.plugins import fundamental, technical  # noqa: F401


_register_plugins()

SCORERS: dict[str, ScoreFn] = registry.as_dict()

__all__ = [
    "BARS_SOURCE",
    "DEFAULT_PRICE_ADJUSTMENT",
    "DEFAULT_SCORER_NAME",
    "OUTPUT_SCORE_COLUMN",
    "PriceAdjustment",
    "SCORERS",
    "SNAPSHOT_SOURCE",
    "ScoreDataSource",
    "ScoreFn",
    "ScoreSpec",
    "IncompatibleScorerBlendError",
    "SnapshotOnlyScorerError",
    "apply_score",
    "backtestable_scorer_names",
    "default_scorer",
    "ensure_backtestable_scorer",
    "get_scorer",
    "register_bar_scorer",
    "registry",
    "resolve_scorer",
    "scorer",
]
