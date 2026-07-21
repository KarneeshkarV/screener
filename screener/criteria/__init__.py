"""Composable TradingView filter criteria.

Every registered criterion has one shape: a zero-argument callable returning
TradingView filter expressions. Full command workflows are their own top-level
commands (e.g. ``garp``, ``mark-minervini``, ``options signals``) instead of
sharing this registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

from screener._registry import Registry


class FilterCriterionFn(Protocol):
    def __call__(self) -> list[Any]: ...


_RegisteredCriterion = TypeVar("_RegisteredCriterion", bound=FilterCriterionFn)


@dataclass(frozen=True)
class FilterCriteriaSelection:
    names: tuple[str, ...]
    label: str
    filters: list[Any]


registry: Registry[FilterCriterionFn] = Registry("criterion")


def criterion(
    name: str,
) -> Callable[[_RegisteredCriterion], _RegisteredCriterion]:
    """Register a composable TradingView filter criterion."""

    def _wrap(fn: _RegisteredCriterion) -> _RegisteredCriterion:
        registry.add(name, fn)
        return fn

    return _wrap


def combine(*filter_fns: FilterCriterionFn) -> FilterCriterionFn:
    """Return a function that merges filters from all supplied criteria."""

    def combined() -> list[Any]:
        filters: list[Any] = []
        for fn in filter_fns:
            filters.extend(fn())
        return filters

    return combined


def resolve_criteria(names: Sequence[str]) -> FilterCriteriaSelection:
    """Resolve criterion names into one composable filter selection."""
    selected = tuple(names)
    filter_fns = [registry.get(name) for name in selected]
    return FilterCriteriaSelection(
        names=selected,
        label="+".join(selected),
        filters=combine(*filter_fns)(),
    )


def _register_plugins() -> None:
    """Import plugin modules so their ``@criterion`` decorators fire."""
    from screener.criteria.plugins import fundamental, technical  # noqa: F401


_register_plugins()

CRITERIA: dict[str, FilterCriterionFn] = registry.as_dict()

__all__ = [
    "CRITERIA",
    "FilterCriteriaSelection",
    "FilterCriterionFn",
    "combine",
    "criterion",
    "registry",
    "resolve_criteria",
]
