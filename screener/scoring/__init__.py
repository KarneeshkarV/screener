"""Per-criterion ranking scores for the TradingView screen path.

Filters stay in ``screener.criteria``; this package owns the *ranking*
philosophy for each criterion name. Output is always written as
``setup_score`` so history, display, and CSV stay stable.

A recipe comes in one of two flavours:

* **snapshot** - reads TradingView's per-row snapshot columns (``RSI``,
  ``market_cap_basic``, ``Perf.Y``, fundamentals). Screen-only.
* **bars** - delegates to a shared price-only recipe in
  :mod:`screener.factors`, which the rolling backtester consumes through its
  own adapter. One formula, two consumers, identical numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, TypeVar

import pandas as pd

from screener._registry import Registry
from screener.factors import PriceScoreSpec, get_price_score

if TYPE_CHECKING:  # pragma: no cover - typing only
    from screener.backtester.data import PriceFetcher

ScoreFn = Callable[[pd.DataFrame], pd.Series]
_RegisteredScoreFn = TypeVar("_RegisteredScoreFn", bound=ScoreFn)

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
    #: Set when the recipe is bar-derived: the shared price-only spec from
    #: :mod:`screener.factors` that both the screen and the backtest evaluate.
    #: ``None`` means the recipe reads TradingView snapshot columns instead.
    bar_score: PriceScoreSpec | None = None


def _snapshot_only_score_fn(name: str) -> ScoreFn:
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
        _snapshot_only_score_fn(name),
        columns=(),
        description=description or spec.description,
        bar_score=spec,
    )


def scorer(
    name: str,
    *,
    columns: Sequence[str] = (),
    description: str = "",
) -> Callable[[_RegisteredScoreFn], _RegisteredScoreFn]:
    """Register a ranking recipe for a criterion name."""

    def _wrap(fn: _RegisteredScoreFn) -> _RegisteredScoreFn:
        registry.add(
            name,
            fn,
            columns=tuple(columns),
            description=description,
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
        bar_score=meta.get("bar_score"),
    )


def default_scorer() -> ScoreSpec:
    """EMA setup score — historical default for ``order_by=setup_score``."""
    return get_scorer(DEFAULT_SCORER_NAME)


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
        # A bar-derived recipe produces a raw per-bar value, not a 0-100
        # snapshot composite, so averaging it with a snapshot recipe would
        # blend two incomparable units. Refuse loudly instead.
        raise ValueError(
            "cannot blend bar-derived scorer(s) "
            f"{sorted(bar_derived)} with other criteria: their scores are raw "
            "price-series values, not the 0-100 snapshot composites. Screen "
            "them one criterion at a time."
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


def apply_score(
    df: pd.DataFrame,
    spec: ScoreSpec,
    *,
    market: str | None = None,
    as_of: date | None = None,
    fetcher: "PriceFetcher | None" = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Assign ``setup_score`` from ``spec`` without sorting or dropping columns.

    A bar-derived ``spec`` needs ``market`` so the scanned tickers' cached
    price history can be resolved; it also drops rows with too little history,
    because in the unified layer NaN means ineligible rather than rank-last.
    """
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
        )
    if df.empty:
        return df.assign(**{OUTPUT_SCORE_COLUMN: pd.Series(dtype=float)})
    scores = spec.score_fn(df)
    return df.assign(**{OUTPUT_SCORE_COLUMN: pd.to_numeric(scores, errors="coerce")})


def _register_plugins() -> None:
    from screener.scoring.plugins import fundamental, technical  # noqa: F401


_register_plugins()

SCORERS: dict[str, ScoreFn] = registry.as_dict()

__all__ = [
    "DEFAULT_SCORER_NAME",
    "OUTPUT_SCORE_COLUMN",
    "SCORERS",
    "ScoreFn",
    "ScoreSpec",
    "apply_score",
    "default_scorer",
    "get_scorer",
    "register_bar_scorer",
    "registry",
    "resolve_scorer",
    "scorer",
]
