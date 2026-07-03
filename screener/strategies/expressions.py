"""Named Pine-like strategy expression view over the unified registry.

``NAMED_STRATEGIES`` is a live, read-only projection of
``screener.strategies.spec.registry`` restricted to expression strategies
(entry/exit Pine strings), used by the historical/rolling backtester. It holds
no state of its own — every access re-reads the one registry, so it can never
drift from it.

Add a new entry/exit Pine strategy by dropping a plugin file in
``screener/strategies/plugins/`` with ``@strategy("name", entry="...", exit="...")``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from screener.strategies.spec import (
    DerivedView,
    StrategySpec,
    discover_plugins,
)

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


def _named_of(spec: StrategySpec) -> Optional[NamedStrategy]:
    if spec.entry is None:
        return None
    return NamedStrategy(entry=spec.entry, exit=spec.exit)


NAMED_STRATEGIES: DerivedView[NamedStrategy] = DerivedView(_named_of)


def resolve_strategy(name: str) -> NamedStrategy:
    try:
        return NAMED_STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(NAMED_STRATEGIES)}"
        ) from None
