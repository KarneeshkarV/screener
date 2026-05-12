"""Named Pine-like strategy expression shortcuts.

Backwards-compatible view over the unified ``screener.strategies.spec.registry``.
Add a new entry/exit Pine strategy by dropping a plugin file in
``screener/strategies/plugins/`` with ``@strategy("name", entry="...", exit="...")``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from screener.strategies.spec import discover_plugins, registry

discover_plugins()


class NamedStrategy(BaseModel):
    entry: str
    exit: Optional[str]

    model_config = ConfigDict(frozen=True)

    @field_validator("entry")
    @classmethod
    def _normalize_entry(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("entry must not be empty")
        return normalized


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
