"""Screening criteria registry + ``combine`` helper.

A criterion is normally a zero-arg function returning a list of TradingView
filter expressions, merged into a single ``Query().where(*filters)`` call.
Drop a new file in ``screener/criteria/plugins/`` with ``@criterion("name")``
and it will be registered automatically — no edits to this file are needed.

Some scans cannot be expressed as a TV filter list (they need per-ticker
enrichment, history, or external providers). These register with
``@criterion("name", pipeline=True)`` and the decorated function takes
screen's shared options instead of returning filters. The ``screen`` command
dispatches to the pipeline when one is selected via ``-c``.
"""

from __future__ import annotations

from typing import Any, Callable

from screener._registry import Registry, autodiscover

CriterionFn = Callable[..., Any]

registry: Registry[CriterionFn] = Registry("criterion")


def criterion(name: str, **meta: Any) -> Callable[[CriterionFn], CriterionFn]:
    """Decorator: ``@criterion("ema") def _(): return [...]``.

    Pass ``pipeline=True`` to register a full-pipeline criterion whose body
    runs the entire scan (rather than returning TV filter expressions).
    """
    return registry.register(name, **meta)


def is_pipeline(name: str) -> bool:
    """True when the named criterion takes over execution from ``screen``."""
    return bool(registry.meta(name).get("pipeline"))


def combine(*filter_fns: CriterionFn) -> CriterionFn:
    """Return a function that merges filters from all given filter functions."""

    def combined() -> list:
        filters: list = []
        for fn in filter_fns:
            filters.extend(fn())
        return filters

    return combined


def _discover() -> None:
    from screener.criteria import plugins

    autodiscover(plugins)


_discover()


CRITERIA: dict[str, CriterionFn] = registry.as_dict()

__all__ = [
    "CRITERIA",
    "CriterionFn",
    "combine",
    "criterion",
    "is_pipeline",
    "registry",
]
