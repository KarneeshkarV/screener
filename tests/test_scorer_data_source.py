"""Every scorer declares where its inputs come from, and only bars replay.

TradingView snapshot columns carry one as-of-today value, so ranking a past
day by them is lookahead. These tests pin the classification and the refusal.
"""

from __future__ import annotations

import pytest

from screener.scoring import (
    BARS_SOURCE,
    SCORERS,
    SNAPSHOT_SOURCE,
    SnapshotOnlyScorerError,
    backtestable_scorer_names,
    ensure_backtestable_scorer,
    get_scorer,
)
from screener.strategies.combo import validate_combo_components
from screener.strategies.spec import resolve_strategy_spec

_SNAPSHOT_ONLY = (
    "value",
    "undervalued",
    "quality",
    "cheap_quality",
    "dividend",
    "momentum_value",
    "ema",
    "ema_breakout",
    "breakout",
    "above_avg_volume",
    "near_52_high",
    "intraday_breakout",
    "intraday_momentum",
    "mark_minervini",
)
_BAR_DERIVED = ("momentum_12_1",)


def test_every_scorer_is_classified() -> None:
    classified = set(_SNAPSHOT_ONLY) | set(_BAR_DERIVED)
    assert sorted(SCORERS) == sorted(classified)
    for name in SCORERS:
        assert get_scorer(name).data_source in {SNAPSHOT_SOURCE, BARS_SOURCE}, name


@pytest.mark.parametrize("name", _SNAPSHOT_ONLY)
def test_snapshot_recipes_are_declared_snapshot(name: str) -> None:
    assert get_scorer(name).data_source == SNAPSHOT_SOURCE


@pytest.mark.parametrize("name", _BAR_DERIVED)
def test_bar_recipes_are_declared_bars(name: str) -> None:
    spec = get_scorer(name)
    assert spec.data_source == BARS_SOURCE
    assert spec.bar_score is not None


def test_backtestable_scorer_names_are_exactly_the_bar_derived_ones() -> None:
    assert sorted(backtestable_scorer_names()) == sorted(_BAR_DERIVED)


@pytest.mark.parametrize("name", ("value", "quality", "dividend", "intraday_momentum"))
def test_snapshot_scorer_is_refused_in_the_backtest_path(name: str) -> None:
    with pytest.raises(SnapshotOnlyScorerError) as excinfo:
        ensure_backtestable_scorer(name)
    message = str(excinfo.value)
    assert name in message
    assert "snapshot" in message
    assert "lookahead" in message


def test_resolving_a_snapshot_scorer_as_a_strategy_explains_why() -> None:
    # ``value`` is a criterion/scorer name but never a strategy, so the
    # backtest path must say *why* rather than "unknown strategy".
    with pytest.raises(SnapshotOnlyScorerError, match="value"):
        resolve_strategy_spec("value")


def test_snapshot_scorer_is_refused_as_a_combo_component() -> None:
    with pytest.raises(SnapshotOnlyScorerError, match="quality"):
        validate_combo_components([("quality", 1.0)])


def test_bar_derived_scorer_passes_the_backtest_guard() -> None:
    ensure_backtestable_scorer("momentum_12_1")


def test_unknown_names_are_left_to_normal_handling() -> None:
    ensure_backtestable_scorer("definitely_not_a_scorer")
    assert resolve_strategy_spec("definitely_not_a_scorer") is None
