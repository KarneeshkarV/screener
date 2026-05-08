"""Compatibility exports for named strategy expression shortcuts."""
from screener.strategies.expressions import (
    NAMED_STRATEGIES,
    NamedStrategy,
    resolve_strategy,
)

STRATEGIES = NAMED_STRATEGIES

__all__ = ["NAMED_STRATEGIES", "NamedStrategy", "STRATEGIES", "resolve_strategy"]
