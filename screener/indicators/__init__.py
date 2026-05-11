"""Shared technical indicators used by research strategies.

Indicators self-register via ``@indicator(...)`` in plugin modules under
``screener.indicators.plugins``. Importing this package triggers discovery.
"""

from screener.indicators.registry import (
    IndicatorFn,
    get_indicator,
    indicator,
    registry,
)

__all__ = ["IndicatorFn", "get_indicator", "indicator", "registry"]
