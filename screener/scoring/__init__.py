"""Per-criterion ranking scores for the TradingView screen path.

Filters stay in ``screener.criteria``; this package owns the *ranking*
philosophy for each criterion name. Output is always written as
``setup_score`` so history, display, and CSV stay stable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd

from screener._registry import Registry

ScoreFn = Callable[[pd.DataFrame], pd.Series]
_RegisteredScoreFn = TypeVar("_RegisteredScoreFn", bound=ScoreFn)

OUTPUT_SCORE_COLUMN = "setup_score"

registry: Registry[ScoreFn] = Registry("scorer")


@dataclass(frozen=True)
class ScoreSpec:
    """Resolved scorer used by the scanner plan + shape pipeline."""

    name: str
    columns: tuple[str, ...]
    score_fn: ScoreFn
    description: str = ""


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
    )


def resolve_scorer(names: Sequence[str]) -> ScoreSpec:
    """Resolve one or more criterion names into a single ranking recipe.

    * One name → that scorer.
    * Several → equal-weight average of each scorer; columns are the union.
    """
    selected = tuple(names)
    if not selected:
        raise ValueError("resolve_scorer requires at least one criterion name")
    if len(selected) == 1:
        return get_scorer(selected[0])

    specs = [get_scorer(name) for name in selected]
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


def apply_score(df: pd.DataFrame, spec: ScoreSpec) -> pd.DataFrame:
    """Assign ``setup_score`` from ``spec`` without sorting or dropping columns."""
    if df.empty:
        return df.assign(**{OUTPUT_SCORE_COLUMN: pd.Series(dtype=float)})
    scores = spec.score_fn(df)
    return df.assign(**{OUTPUT_SCORE_COLUMN: pd.to_numeric(scores, errors="coerce")})


def _register_plugins() -> None:
    from screener.scoring.plugins import fundamental, technical  # noqa: F401


_register_plugins()

SCORERS: dict[str, ScoreFn] = registry.as_dict()

__all__ = [
    "OUTPUT_SCORE_COLUMN",
    "SCORERS",
    "ScoreFn",
    "ScoreSpec",
    "apply_score",
    "get_scorer",
    "registry",
    "resolve_scorer",
    "scorer",
]
