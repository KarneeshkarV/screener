from datetime import date

import pytest
from pydantic import ValidationError

from screener.backtester.models import (
    BacktestConfig,
    DataPolicy,
    ExecutionPolicy,
    PortfolioPolicy,
    SignalPolicy,
    UniversePolicy,
)
from screener.backtester.slippage import FixedBpsSlippage


def _nested_config() -> BacktestConfig:
    return BacktestConfig(
        market="america",
        as_of=date(2025, 1, 31),
        benchmark="SPY",
        universe=UniversePolicy(tickers=("AAPL", "MSFT"), max_universe=50),
        signals=SignalPolicy(
            strategy_name="momentum",
            entry_expr="close > sma(close, 20)",
            exit_expr=None,
        ),
        data=DataPolicy(interval="1h", price_adjustment="splits_only"),
        execution=ExecutionPolicy(
            hold=10,
            stop_loss=0.08,
            take_profit=0.2,
            trailing_stop=None,
            slippage_bps=5.0,
            commission_bps=1.0,
        ),
        portfolio=PortfolioPolicy(top=5, initial_capital=100_000.0),
    )


def test_backtest_config_composes_focused_policies() -> None:
    config = _nested_config()

    assert config.universe.tickers == ("AAPL", "MSFT")
    assert config.signals.strategy_name == "momentum"
    assert config.data.interval == "1h"
    assert config.execution.hold == 10
    assert config.portfolio.top == 5
    assert isinstance(config.execution.slippage_model, FixedBpsSlippage)
    assert config.execution.slippage_model.bps == 5.0


def test_flat_backtest_config_input_remains_a_public_boundary() -> None:
    nested = _nested_config()
    flat = nested.to_flat_dict()

    rebuilt = BacktestConfig.model_validate(flat)

    assert rebuilt == nested
    assert rebuilt.tickers == nested.universe.tickers
    assert rebuilt.entry_expr == nested.signals.entry_expr
    assert rebuilt.interval == nested.data.interval
    assert rebuilt.hold == nested.execution.hold
    assert rebuilt.top == nested.portfolio.top


def test_flat_model_copy_updates_the_owning_policy() -> None:
    original = _nested_config()

    updated = original.model_copy(update={"hold": 30, "top": 10})

    assert updated.execution.hold == 30
    assert updated.portfolio.top == 10
    assert original.execution.hold == 10
    assert original.portfolio.top == 5


def test_interval_validation_lives_with_data_policy() -> None:
    with pytest.raises(ValidationError, match="unsupported interval"):
        DataPolicy(interval="2h")


def test_unknown_flat_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="bogus_flat_field"):
        BacktestConfig.model_validate(
            {**_nested_config().to_flat_dict(), "bogus_flat_field": True}
        )
