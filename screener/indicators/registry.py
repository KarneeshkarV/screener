"""Indicator registry. Public ``indicator(name)`` decorator + explicit registration.

Drop a new file in ``screener/indicators/plugins/`` with ``@indicator("name")``,
import it from ``_register_plugins`` below, and it's available via
``registry.get("name")``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

from screener._registry import Registry

IndicatorFn = Callable[..., Any]

#: The decorator hands the function straight back, so it must return the exact
#: type it was given: erasing it to ``IndicatorFn`` would turn every call to a
#: decorated indicator into an ``Any``.
F = TypeVar("F", bound=IndicatorFn)

registry: Registry[IndicatorFn] = Registry("indicator")


def indicator(name: str, **meta) -> Callable[[F], F]:
    """Decorator: ``@indicator("ema") def ema(x, n): ...``."""
    return cast(Callable[[F], F], registry.register(name, **meta))


def _register_plugins() -> None:
    """Import plugin modules so their ``@indicator`` decorators fire."""
    from screener.indicators.plugins import (  # noqa: F401
        atr,
        bollinger_bands,
        ema,
        heikin_ashi,
        rma,
        rsi,
        sar,
        sma,
        stdev,
        supertrend,
    )


_register_plugins()


__all__ = ["IndicatorFn", "indicator", "registry"]
