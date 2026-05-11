"""Named Pine-like strategy expression shortcuts.

Backwards-compatible view over the unified ``screener.strategies.spec.registry``.
Add a new entry/exit Pine strategy by dropping a plugin file in
``screener/strategies/plugins/`` with ``@strategy("name", entry="...", exit="...")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from screener.strategies.spec import discover_plugins, registry

discover_plugins()


@dataclass(frozen=True)
class NamedStrategy:
    entry: str
    exit: Optional[str]


NAMED_STRATEGIES: dict[str, NamedStrategy] = {
    name: NamedStrategy(entry=spec.entry, exit=spec.exit)
    for name, spec in registry.items()
    if spec.entry is not None
}


def resolve_strategy(name: str) -> NamedStrategy:
    try:
        return NAMED_STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(NAMED_STRATEGIES)}"
        ) from None
