"""Indicator registry. Public ``indicator(name)`` decorator + autodiscovery.

Drop a new file in ``screener/indicators/plugins/`` with ``@indicator("name")``
and it's available via ``get_indicator("name")`` and re-exported by
``screener.indicators.numpy`` under its legacy ``_name`` alias if applicable.
"""

from __future__ import annotations

from typing import Any, Callable

from screener._registry import Registry, autodiscover

IndicatorFn = Callable[..., Any]

registry: Registry[IndicatorFn] = Registry("indicator")


def indicator(name: str, **meta) -> Callable[[IndicatorFn], IndicatorFn]:
    """Decorator: ``@indicator("ema") def ema(x, n): ...``."""
    return registry.register(name, **meta)


def get_indicator(name: str) -> IndicatorFn:
    return registry.get(name)


def _discover() -> None:
    from screener.indicators import plugins

    autodiscover(plugins)


_discover()


__all__ = ["IndicatorFn", "get_indicator", "indicator", "registry"]
