"""Screening criteria registry and dispatch Interface.

A filter criterion is a zero-argument function returning TradingView filter
expressions. A pipeline criterion is a full workflow Adapter that owns its
scan, output, and provider behavior. This Module keeps that Interface explicit
so callers do not need to treat every criterion as the same callable shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, TypeVar, cast

from screener._registry import Registry, autodiscover


class FilterCriterionFn(Protocol):
    def __call__(self) -> list[Any]: ...


class PipelineCriterionFn(Protocol):
    def __call__(
        self,
        *,
        market: str,
        limit: int,
        output_csv: bool,
        refresh: bool,
        cache_ttl: str,
        **kwargs: Any,
    ) -> None: ...


CriterionCallable = Callable[..., Any]
CriterionFn = FilterCriterionFn | PipelineCriterionFn
_RegisteredCriterion = TypeVar("_RegisteredCriterion", bound=CriterionCallable)


class CriterionMode(str, Enum):
    FILTER = "filter"
    PIPELINE = "pipeline"


@dataclass(frozen=True)
class CriterionDefinition:
    name: str
    mode: CriterionMode
    fn: CriterionCallable

    @property
    def is_filter(self) -> bool:
        return self.mode is CriterionMode.FILTER

    @property
    def is_pipeline(self) -> bool:
        return self.mode is CriterionMode.PIPELINE

    def as_filter(self) -> FilterCriterionFn:
        if not self.is_filter:
            raise TypeError(f"criterion {self.name!r} is a pipeline criterion")
        return cast(FilterCriterionFn, self.fn)

    def as_pipeline(self) -> PipelineCriterionFn:
        if not self.is_pipeline:
            raise TypeError(f"criterion {self.name!r} is a filter criterion")
        return cast(PipelineCriterionFn, self.fn)


@dataclass(frozen=True)
class FilterCriteriaSelection:
    names: tuple[str, ...]
    label: str
    filters: list[Any]


@dataclass(frozen=True)
class PipelineCriteriaSelection:
    name: str
    runner: PipelineCriterionFn


CriteriaSelection = FilterCriteriaSelection | PipelineCriteriaSelection


class CriteriaSelectionError(ValueError):
    """Raised when selected criteria violate the dispatch Interface."""


registry: Registry[CriterionCallable] = Registry("criterion")


def criterion(
    name: str, **meta: Any
) -> Callable[[_RegisteredCriterion], _RegisteredCriterion]:
    """Decorator: ``@criterion("ema") def _(): return [...]``.

    Pass ``pipeline=True`` to register a full-pipeline criterion whose body
    runs the entire scan (rather than returning TV filter expressions).
    """

    def _wrap(fn: _RegisteredCriterion) -> _RegisteredCriterion:
        registry.add(name, fn, **meta)
        return fn

    return _wrap


def get_definition(name: str) -> CriterionDefinition:
    """Return the named criterion's explicit Interface definition."""
    mode = CriterionMode.PIPELINE if is_pipeline(name) else CriterionMode.FILTER
    return CriterionDefinition(name=name, mode=mode, fn=registry.get(name))


def definitions() -> dict[str, CriterionDefinition]:
    """Return all registered criteria keyed by name."""
    return {name: get_definition(name) for name in registry}


def filter_criteria() -> dict[str, FilterCriterionFn]:
    """Return only filter criteria through the public criteria Interface."""
    return {
        name: definition.as_filter()
        for name, definition in definitions().items()
        if definition.is_filter
    }


def pipeline_criteria() -> dict[str, PipelineCriterionFn]:
    """Return only pipeline criteria through the public criteria Interface."""
    return {
        name: definition.as_pipeline()
        for name, definition in definitions().items()
        if definition.is_pipeline
    }


def is_pipeline(name: str) -> bool:
    """True when the named criterion takes over execution from ``screen``."""
    return bool(registry.meta(name).get("pipeline"))


def combine(*filter_fns: FilterCriterionFn) -> FilterCriterionFn:
    """Return a function that merges filters from all given filter functions."""

    def combined() -> list[Any]:
        filters: list[Any] = []
        for fn in filter_fns:
            filters.extend(fn())
        return filters

    return combined


def resolve_criteria(names: Sequence[str]) -> CriteriaSelection:
    """Resolve selected criteria into one filter or pipeline dispatch plan."""
    selected = tuple(names)
    definitions_ = tuple(get_definition(name) for name in selected)
    pipelines = [definition for definition in definitions_ if definition.is_pipeline]
    if pipelines:
        if len(definitions_) > 1:
            raise CriteriaSelectionError(
                f"Pipeline criterion {pipelines[0].name!r} cannot be combined "
                f"with other -c values; got {list(selected)!r}."
            )
        pipeline = pipelines[0]
        return PipelineCriteriaSelection(
            name=pipeline.name,
            runner=pipeline.as_pipeline(),
        )

    filter_fns = [definition.as_filter() for definition in definitions_]
    return FilterCriteriaSelection(
        names=selected,
        label="+".join(selected),
        filters=combine(*filter_fns)(),
    )


def _discover() -> None:
    from screener.criteria import plugins

    autodiscover(plugins)


_discover()


CRITERIA: dict[str, CriterionCallable] = registry.as_dict()

__all__ = [
    "CRITERIA",
    "CriteriaSelection",
    "CriteriaSelectionError",
    "CriterionCallable",
    "CriterionDefinition",
    "CriterionFn",
    "CriterionMode",
    "FilterCriteriaSelection",
    "FilterCriterionFn",
    "PipelineCriteriaSelection",
    "PipelineCriterionFn",
    "combine",
    "criterion",
    "definitions",
    "filter_criteria",
    "get_definition",
    "is_pipeline",
    "pipeline_criteria",
    "registry",
    "resolve_criteria",
]
