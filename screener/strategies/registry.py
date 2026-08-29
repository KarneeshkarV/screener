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

from screener.backtester.pine import (
    FUNC_NAMES,
    SERIES_NAMES,
    Name,
    Node,
    PineError,
    _call_names,
    evaluate,
    parse,
)
from screener.strategies.base import StrategyFn
from screener.strategies.spec import (
    CallableStrategySpec,
    apply_bar_columns,
    DerivedView,
    ExpressionStrategySpec,
    StrategySpec,
    discover_plugins,
    is_derived_from_bar_columns,
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

    A hand-written ``prepare_bars`` is disqualifying on its own, whatever the
    AST says. The synthesised callable can replay ``bar_columns`` but not a
    panel-level hook, so a strategy whose entry reads only OHLCV while its prep
    supplies the rest would otherwise project to a callable that silently skips
    that prep - indistinguishable from the strategy it was derived from.
    """
    if spec.prepare_bars is not None and not is_derived_from_bar_columns(
        spec.prepare_bars
    ):
        return False
    try:
        asts = [_parsed(spec.entry)]
        if spec.exit is not None:
            asts.append(_parsed(spec.exit))
    except PineError:
        # An unparseable expression is excluded, not raised. This runs during
        # iteration of STRATEGIES, and a market run that walks every strategy
        # must not abort on one bad registration - the backtester still reports
        # the syntax error when that strategy is the one being run.
        return False
    names: set[str] = set()
    called: set[str] = set()
    for ast in asts:
        names |= _identifiers(ast)
        called |= _call_names(ast)
    # An unknown *function* passes the identifier check but raises PineNameError
    # inside the pine_runner's per-ticker loop, which catches only ValueError
    # and friends. Reject it here, where the answer is "absent from the view".
    if not called <= FUNC_NAMES:
        return False
    # Declared bar-local columns are self-contained by construction: each is a
    # pure function of the one frame the pine_runner already has.
    available = SERIES_NAMES | set(spec.bar_columns or ())
    return names <= available


#: Built callables, keyed by the identity of the spec they were built from.
#: Memoising matters: without it every read of :data:`STRATEGIES` builds a fresh
#: function, so the view stops being a projection in any useful sense
#: (``dict(STRATEGIES.items()) != dict(STRATEGIES)``) and callers that cache or
#: compare entries by identity break.
#:
#: Keyed by identity rather than by name because a name can be re-registered:
#: ``Registry.remove`` exists for temporary registrations, so ``foo`` removed
#: and re-added as a different spec would otherwise get the first spec's stale
#: closure back. The spec is held alongside the callable so its ``id`` stays
#: reserved for as long as the entry does.
_SYNTHESISED: dict[int, tuple[ExpressionStrategySpec, StrategyFn]] = {}


def _expression_callable(spec: ExpressionStrategySpec) -> StrategyFn:
    """Synthesise the pine_runner's callable from an expression strategy."""
    cached = _SYNTHESISED.get(id(spec))
    if cached is not None:
        return cached[1]

    def _run(df: pd.DataFrame) -> list[ResearchTrade]:
        bars = apply_bar_columns(spec.bar_columns, df)
        return _walk(
            _mask(spec.entry, bars),
            _mask(spec.exit, bars),
            bars["close"].to_numpy(dtype=float),
            bars["date"].values,
        )

    _run.__name__ = f"strat_{spec.name}"
    _run.__doc__ = f"Expression strategy {spec.name!r}: entry={spec.entry!r}."
    _SYNTHESISED[id(spec)] = (spec, _run)
    return _run


def _callable_of(spec: StrategySpec) -> StrategyFn | None:
    if isinstance(spec, CallableStrategySpec):
        return spec.callable_fn
    if isinstance(spec, ExpressionStrategySpec) and _is_self_contained(spec):
        return _expression_callable(spec)
    return None


STRATEGIES: DerivedView[StrategyFn] = DerivedView(_callable_of)
