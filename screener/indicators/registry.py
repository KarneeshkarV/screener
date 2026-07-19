"""Indicator registry. Public ``indicator(name)`` decorator + explicit registration.

Drop a new file in ``screener/indicators/plugins/`` with ``@indicator("name")``,
import it from ``_register_plugins`` below, and it's available via
``registry.get("name")``.
"""

from __future__ import annotations

from typing import Any, Callable

from screener._registry import Registry

IndicatorFn = Callable[..., Any]

registry: Registry[IndicatorFn] = Registry("indicator")


def indicator(name: str, **meta) -> Callable[[IndicatorFn], IndicatorFn]:
    """Decorator: ``@indicator("ema") def ema(x, n): ...``."""
    return registry.register(name, **meta)


def _register_plugins() -> None:
    """Import plugin modules so their ``@indicator`` decorators fire."""
    from screener.indicators.plugins import (  # noqa: F401
        atr,
        bollinger_bands,
        ema,
        rma,
        rsi,
        sar,
        sma,
        stdev,
        supertrend,
    )


_register_plugins()


__all__ = ["IndicatorFn", "indicator", "registry"]
