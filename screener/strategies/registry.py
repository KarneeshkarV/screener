"""Registry for callable research strategies.

Backwards-compatible view over the unified ``screener.strategies.spec.registry``.
``STRATEGIES`` and friends remain a plain dict for the pine_runner; expression-
flavored strategies are surfaced separately through
``screener.strategies.expressions``.

Add a new strategy by dropping a plugin file in ``screener/strategies/plugins/``
with an ``@strategy(...)`` decorator. No edits to this file are needed.
"""

from __future__ import annotations

from collections.abc import Iterator

from screener.strategies.base import StrategyFn
from screener.strategies.spec import discover_plugins, registry

discover_plugins()


STRATEGIES: dict[str, StrategyFn] = {
    name: spec.callable_fn
    for name, spec in registry.items()
    if spec.callable_fn is not None
}


def get_strategy(name: str) -> StrategyFn:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(STRATEGIES)}"
        ) from None


def iter_strategies() -> Iterator[tuple[str, StrategyFn]]:
    return iter(STRATEGIES.items())
