"""A declared bar column carries its warm-up into the strategy's lookback.

The entry/exit AST cannot reveal how much history a derived column needs: an
expression naming ``bb_upper`` looks like a one-bar ``crossover`` while the
column behind it is a 350-bar Bollinger band. These tests pin that each recipe
declares its warm-up and that ``register_expression_strategy`` folds the
largest declaration into ``required_lookback``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener._registry import Registry
from screener.backtester.pine import parse, required_lookback
from screener.strategies import bar_column_recipes as _cols
from screener.strategies import spec as spec_module
from screener.strategies.spec import (
    ExpressionStrategySpec,
    bar_column,
    bar_columns_lookback,
    discover_plugins,
    register_expression_strategy,
    registry,
)


def _expression_specs_with_columns() -> dict[str, ExpressionStrategySpec]:
    discover_plugins()
    return {
        name: spec
        for name, spec in registry.items()
        if isinstance(spec, ExpressionStrategySpec) and spec.bar_columns
    }


def _ast_lookback(spec: ExpressionStrategySpec) -> int:
    lookback = required_lookback(parse(spec.entry))
    if spec.exit is not None:
        lookback = max(lookback, required_lookback(parse(spec.exit)))
    return lookback


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> Registry:
    """An empty registry, so a test registration cannot leak into the real one."""
    fresh: Registry = Registry("strategy")
    monkeypatch.setattr(spec_module, "registry", fresh)
    return fresh


@pytest.mark.parametrize("name", sorted(_expression_specs_with_columns()))
def test_spec_lookback_covers_its_bar_columns(name: str) -> None:
    """The declared floor covers the columns; the AST floor is added by callers.

    ``prepare_strategy_bars`` hands its floor back for the caller to combine
    with the expression's own (``max(lookback, strategy_lookback)``), so the
    spec only owns what the AST cannot see.
    """
    spec = _expression_specs_with_columns()[name]
    assert spec.required_lookback is not None, f"{name}: no required_lookback"
    assert spec.required_lookback() >= bar_columns_lookback(spec.bar_columns)
    effective = max(spec.required_lookback(), _ast_lookback(spec))
    assert effective >= bar_columns_lookback(spec.bar_columns)


def test_bb_breakout_lookback_reaches_the_bollinger_window() -> None:
    """The regression this guards: a 350-bar column behind a 1-bar crossover."""
    spec = _expression_specs_with_columns()["bb_breakout"]
    assert _ast_lookback(spec) == 1
    assert spec.required_lookback() == 350


def test_undeclared_column_is_rejected_at_registration(
    scratch_registry: Registry,
) -> None:
    def unlabelled(bars: pd.DataFrame) -> pd.Series:
        return bars["close"].rolling(200).mean()

    with pytest.raises(ValueError, match="does not declare its warm-up"):
        register_expression_strategy(
            "_test_undeclared_column",
            entry="close > slow",
            bar_columns={"slow": unlabelled},
        )
    assert scratch_registry.get_optional("_test_undeclared_column") is None


def test_declared_lookback_and_columns_take_the_larger(
    scratch_registry: Registry,
) -> None:
    @bar_column(30)
    def slow(bars: pd.DataFrame) -> pd.Series:
        return bars["close"].rolling(30).mean()

    spec = register_expression_strategy(
        "_test_column_vs_declared",
        entry="close > slow",
        bar_columns={"slow": slow},
        required_lookback=lambda: 400,
    )
    assert spec.required_lookback() == 400

    @bar_column(500)
    def very_slow(bars: pd.DataFrame) -> pd.Series:
        return bars["close"].rolling(500).mean()

    wide = register_expression_strategy(
        "_test_columns_win",
        entry="close > slow",
        bar_columns={"slow": very_slow},
        required_lookback=lambda: 5,
    )
    assert wide.required_lookback() == 500


def test_bar_column_lookback_matches_when_the_column_turns_valid() -> None:
    """The declared warm-up is not smaller than the recipe's real NaN run."""
    n = 800
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    bars = pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", periods=n, freq="D"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adj_close": close,
            "volume": np.full(n, 10_000.0),
        }
    )
    for name, spec in _expression_specs_with_columns().items():
        for column, build in spec.bar_columns.items():
            values = build(bars).to_numpy(dtype=float)
            valid = np.flatnonzero(~np.isnan(values))
            assert valid.size, f"{name}.{column}: never valid on the fixture"
            first_valid_bars = int(valid[0]) + 1
            declared = bar_columns_lookback({column: build})
            assert declared >= first_valid_bars, (
                f"{name}.{column}: declared {declared} bars but the column is "
                f"NaN until bar {first_valid_bars}"
            )


def test_negative_lookback_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        bar_column(-1)


def test_every_recipe_declares_a_warm_up() -> None:
    """No public recipe ships undecorated - the failure mode is silent."""
    recipes = {
        name: value
        for name, value in vars(_cols).items()
        if callable(value)
        and not name.startswith("_")
        and getattr(value, "__module__", None) == _cols.__name__
    }
    assert recipes
    bar_columns_lookback(recipes)


def test_bar_columns_and_prepare_bars_together_are_rejected(
    scratch_registry: Registry,
) -> None:
    """The two consumers would disagree about which prep actually ran."""

    @bar_column(3)
    def mid(bars: pd.DataFrame) -> pd.Series:
        return (bars["high"] + bars["low"]) / 2.0

    with pytest.raises(ValueError, match="both bar_columns and prepare_bars"):
        register_expression_strategy(
            "_test_columns_and_hook",
            entry="close > mid",
            bar_columns={"mid": mid},
            prepare_bars=lambda ctx: ctx.bars_by_tv,
        )
    assert scratch_registry.get_optional("_test_columns_and_hook") is None
