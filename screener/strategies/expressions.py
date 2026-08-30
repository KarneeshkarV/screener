"""Named Pine-like strategy expression view over the unified registry.

``NAMED_STRATEGIES`` is a live, read-only projection of
``screener.strategies.spec.registry`` restricted to expression strategies
(entry/exit Pine strings), used by the historical/rolling backtester. It holds
no state of its own — every access re-reads the one registry, so it can never
drift from it.

Add a new entry/exit Pine strategy by dropping a plugin file in
``screener/strategies/plugins/`` with a ``register_expression_strategy(...)`` call.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from screener.strategies.spec import (
    DerivedView,
    ExpressionStrategySpec,
    StrategySpec,
    discover_plugins,
)

discover_plugins()


class NamedStrategy(BaseModel):
    entry: str
    exit: str | None

    model_config = ConfigDict(frozen=True)

    @field_validator("entry")
    @classmethod
    def _normalize_entry(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("entry must not be empty")
        return normalized


def _named_from_expression(spec: ExpressionStrategySpec) -> NamedStrategy:
    """The rules a run executes: the profile's overrides, else the spec's own.

    ``screen_candidates`` reads the pair the same way, so a profile that
    restates a rule cannot move one path without moving the other.
    """
    profile = spec.profile
    if profile is None:
        return NamedStrategy(entry=spec.entry, exit=spec.exit)
    return NamedStrategy(
        entry=profile.entry_expr or spec.entry,
        exit=profile.exit_expr or spec.exit,
    )


def _named_of(spec: StrategySpec) -> NamedStrategy | None:
    if not isinstance(spec, ExpressionStrategySpec):
        return None
    return _named_from_expression(spec)


NAMED_STRATEGIES: DerivedView[NamedStrategy] = DerivedView(_named_of)


def resolve_strategy(name: str) -> NamedStrategy:
    from screener.strategies.spec import resolve_strategy_spec

    try:
        spec = resolve_strategy_spec(name)
    except ValueError as exc:
        raise KeyError(str(exc)) from exc
    if not isinstance(spec, ExpressionStrategySpec):
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {sorted(NAMED_STRATEGIES)}"
        ) from None
    return _named_from_expression(spec)
