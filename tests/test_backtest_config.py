from datetime import date

import pytest
from pydantic import ValidationError

from screener.backtester.models import BacktestConfig
from screener.backtester.slippage import FixedBpsSlippage


def _config() -> BacktestConfig:
    return BacktestConfig(
        market="america",
        as_of=date(2025, 1, 31),
        benchmark="SPY",
        tickers=("AAPL", "MSFT"),
        max_universe=50,
        strategy_name="momentum",
        entry_expr="close > sma(close, 20)",
        exit_expr=None,
        interval="1h",
        price_adjustment="splits_only",
        hold=10,
        stop_loss=0.08,
        take_profit=0.2,
        trailing_stop=None,
        slippage_bps=5.0,
        commission_bps=1.0,
        top=5,
        initial_capital=100_000.0,
    )


def test_backtest_config_is_flat() -> None:
    config = _config()

    assert config.tickers == ("AAPL", "MSFT")
    assert config.strategy_name == "momentum"
    assert config.interval == "1h"
    assert config.hold == 10
    assert config.top == 5
    assert isinstance(config.slippage_model, FixedBpsSlippage)
    assert config.slippage_model.bps == 5.0


def test_flat_model_copy_updates_fields() -> None:
    original = _config()

    updated = original.model_copy(update={"hold": 30, "top": 10})

    assert updated.hold == 30
    assert updated.top == 10
    assert original.hold == 10
    assert original.top == 5


def _flat_dict() -> dict:
    # ``slippage_model`` is a Protocol/arbitrary type that does not round-trip
    # through ``model_dump``; drop it so ``_default_slippage`` rebuilds it.
    return _config().model_dump(exclude={"slippage_model"})


def test_interval_validation() -> None:
    with pytest.raises(ValidationError, match="unsupported interval"):
        BacktestConfig.model_validate({**_flat_dict(), "interval": "2h"})


def test_unknown_flat_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="bogus_flat_field"):
        BacktestConfig.model_validate({**_flat_dict(), "bogus_flat_field": True})
