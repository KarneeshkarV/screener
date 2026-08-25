"""Callable-strategy view over the unified strategy registry.

``STRATEGIES`` is a live, read-only projection of
``screener.strategies.spec.registry`` that hands the pine_runner a
``fn(df) -> list[ResearchTrade]`` for *every* registered strategy, whichever
flavour it was declared in. It holds no state of its own - every access
re-reads the one registry, so it can never drift from it.

A callable strategy projects to its own function. An expression strategy
projects to a function synthesised from its entry/exit expressions: parse,
evaluate against the frame, then walk the two boolean masks with the same
:func:`~screener.strategies.trades._walk` every callable already uses.

That synthesis is what lets a strategy be *converted* from callable to
expression without disappearing from this view. The names here are referenced
by saved configs and CLI invocations (see ``tests/test_strategy_registry.py``),
so dropping one is a breaking change; deriving the callable instead means one
definition serves the pine runner and the backtester at once, which is the
point of ``docs/plans/unify-screen-backtest.md``.

Add a new strategy by dropping a plugin file in ``screener/strategies/plugins/``
with an ``@strategy(...)`` or ``register_expression_strategy(...)`` call. No
edits to this file are needed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel

from screener.backtester.pine import SERIES_NAMES, Name, Node, evaluate, parse
from screener.strategies.base import StrategyFn
from screener.strategies.spec import (
    CallableStrategySpec,
    DerivedView,
    ExpressionStrategySpec,
    StrategySpec,
    discover_plugins,
)
from screener.strategies.trades import ResearchTrade, _walk

discover_plugins()


@lru_cache(maxsize=None)
def _parsed(expr: str) -> Node:
    """Parse once per distinct expression; the AST is immutable and reusable."""
    return parse(expr)


def _mask(expr: str | None, bars: pd.DataFrame) -> np.ndarray:
    """Evaluate ``expr`` to a bool array, or all-False when there is no expr.

    A missing exit expression means "never exit on a signal", which leaves
    ``_walk`` to close the open position on the final bar, matching how the
    backtester treats an expression strategy with no exit rule.
    """
    if expr is None:
        return np.zeros(len(bars), dtype=bool)
    series = evaluate(_parsed(expr), bars)
    return series.fillna(False).to_numpy(dtype=bool)


def _identifiers(node: Node) -> set[str]:
    """Every identifier the expression reads, found by walking the AST."""
    if isinstance(node, Name):
        return {node.name}
    found: set[str] = set()
    for value in node.__dict__.values():
        if isinstance(value, BaseModel):
            found |= _identifiers(cast(Node, value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, BaseModel):
                    found |= _identifiers(cast(Node, item))
    return found


def _is_self_contained(spec: ExpressionStrategySpec) -> bool:
    """True when a bare OHLCV frame supplies every name the expression reads.

    The pine_runner hands over one frame with no panel, no market and no
    fetcher, so an expression that reads a prepared column (``rank_score``), a
    cross-sectional rank (``rs_rank``) or a merged fundamental
    (``revenue_up_3q``) cannot be evaluated there. Deciding that statically
    from the AST keeps the view honest: such a strategy is simply absent rather
    than present and raising ``PineNameError`` on first use.
    """
    names = _identifiers(_parsed(spec.entry))
    if spec.exit is not None:
        names |= _identifiers(_parsed(spec.exit))
    return names <= SERIES_NAMES


@lru_cache(maxsize=None)
def _expression_callable(name: str, entry: str, exit_expr: str | None) -> StrategyFn:
    """Synthesise the pine_runner's callable from an expression strategy.

    Keyed on the three values that determine behaviour rather than on the spec
    object, which carries an unhashable profile. Memoising matters: without it
    every read of :data:`STRATEGIES` builds a fresh function, so the view stops
    being a projection in any useful sense
    (``dict(STRATEGIES.items()) != dict(STRATEGIES)``) and callers that cache or
    compare entries by identity break.
    """

    def _run(df: pd.DataFrame) -> list[ResearchTrade]:
        return _walk(
            _mask(entry, df),
            _mask(exit_expr, df),
            df["close"].to_numpy(dtype=float),
            df["date"].values,
        )

    _run.__name__ = f"strat_{name}"
    _run.__doc__ = f"Expression strategy {name!r}: entry={entry!r}."
    return _run


def _callable_of(spec: StrategySpec) -> StrategyFn | None:
    if isinstance(spec, CallableStrategySpec):
        return spec.callable_fn
    if isinstance(spec, ExpressionStrategySpec) and _is_self_contained(spec):
        return _expression_callable(spec.name, spec.entry, spec.exit)
    return None


STRATEGIES: DerivedView[StrategyFn] = DerivedView(_callable_of)
