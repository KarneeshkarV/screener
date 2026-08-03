"""Callable-strategy view over the unified strategy registry.

``STRATEGIES`` is a live, read-only projection of
``screener.strategies.spec.registry`` restricted to callable strategies
(``fn(df) -> list[Trade]``), used by the pine_runner. It holds no state of its
own — every access re-reads the one registry, so it can never drift from it.
Expression-flavored strategies are surfaced through
``screener.strategies.expressions``.

Add a new strategy by dropping a plugin file in ``screener/strategies/plugins/``
with an ``@strategy(...)`` decorator. No edits to this file are needed.
"""

from __future__ import annotations

from screener.strategies.base import StrategyFn
from screener.strategies.spec import (
    CallableStrategySpec,
    DerivedView,
    StrategySpec,
    discover_plugins,
)

discover_plugins()


def _callable_of(spec: StrategySpec) -> StrategyFn | None:
    if not isinstance(spec, CallableStrategySpec):
        return None
    return spec.callable_fn


STRATEGIES: DerivedView[StrategyFn] = DerivedView(_callable_of)
