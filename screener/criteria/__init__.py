"""TradingView screening criteria registry + ``combine`` helper.

A criterion is a zero-arg function returning a list of TradingView filter
expressions. Drop a new file in ``screener/criteria/plugins/`` with
``@criterion("name")`` and it will be registered automatically — no edits to
this file are needed.
"""

from __future__ import annotations

from typing import Callable

from screener._registry import Registry, autodiscover

CriterionFn = Callable[[], list]

registry: Registry[CriterionFn] = Registry("criterion")


def criterion(name: str, **meta) -> Callable[[CriterionFn], CriterionFn]:
    """Decorator: ``@criterion("ema") def _(): return [...]``."""
    return registry.register(name, **meta)


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

__all__ = ["CRITERIA", "CriterionFn", "combine", "criterion", "registry"]
