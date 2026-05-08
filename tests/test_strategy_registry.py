from __future__ import annotations

from screener.backtester import pine_runner
from screener.strategies.registry import STRATEGIES, get_strategy, iter_strategies


def test_strategy_registry_preserves_pine_runner_names():
    assert set(STRATEGIES) == set(pine_runner.STRATEGIES)
    assert dict(iter_strategies()) == STRATEGIES


def test_strategy_registry_lookup_returns_callable():
    strategy = get_strategy("ma_cross")

    assert strategy is STRATEGIES["ma_cross"]
    assert callable(strategy)


def test_backtester_pine_runner_reexports_legacy_helpers():
    assert pine_runner._ema is not None
    assert pine_runner._rsi is not None
    assert pine_runner.load_universe is not None
    assert pine_runner.strat_ma_cross is STRATEGIES["ma_cross"]
