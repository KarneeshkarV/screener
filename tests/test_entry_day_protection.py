"""Entry-day protection and ambiguous-bar fills through both portfolio engines."""

from datetime import date

import pandas as pd
import pytest

from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from tests.conftest import StubPriceFetcher


def protection_run(
    engine,
    *,
    spike_day=1,
    order="moo",
    partials=(),
    low=80.0,
    capital=100_000,
    take_profit=None,
    entry_limit_bps=None,
    slippage_bps=0,
):
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000.0,
            "entry_signal": [1.0, 0, 0, 0, 0, 0],
        },
        index=pd.bdate_range("2024-01-01", periods=6),
    )
    bars.loc[bars.index[spike_day], ["high", "low"]] = [120.0, low]
    cfg = BacktestConfig(
        market="india",
        as_of=date(2024, 1, 1),
        benchmark="^NSEI",
        tickers=("TEST.NS",),
        entry_expr="entry_signal > 0",
        exit_expr=None,
        hold=2,
        stop_loss=0.1,
        take_profit=take_profit,
        trailing_stop=None,
        top=1,
        initial_capital=capital,
        slippage_bps=slippage_bps,
        commission_bps=0,
        entry_order_type=order,
        partial_exits=partials,
        entry_limit_bps=entry_limit_bps,
    )
    fetcher = StubPriceFetcher({"TEST.NS": bars, "^NSEI": bars})
    if engine == "historical":
        return run_backtest(cfg, fetcher)
    return run_rolling_backtest(
        cfg, fetcher, start_date=bars.index[0].date(), end_date=bars.index[-1].date()
    )


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_open_entry_honors_stop_on_entry_day(engine):
    result = protection_run(engine)
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].exit_date == date(2024, 1, 2)
    assert result.trades[0].exit_price == pytest.approx(90)
    assert result.metrics["final_equity"] == pytest.approx(90_000)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_close_entry_does_not_use_pre_entry_intraday_low(engine):
    result = protection_run(engine, order="moc")
    assert result.trades[0].exit_reason == "time"
    assert result.metrics["final_equity"] == pytest.approx(100_000)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_stop_precedes_ambiguous_partial_profit_target(engine):
    result = protection_run(engine, spike_day=2, partials=((0.1, 0.5),))
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop"
    assert result.metrics["final_equity"] == pytest.approx(90_000)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_new_breakeven_stop_does_not_reuse_pre_target_low(engine):
    result = protection_run(engine, spike_day=2, partials=((0.1, 0.5),), low=95)
    assert result.trades[0].exit_reason == "target"
    # The original 90 stop did not trigger. The new 100 stop starts next bar.
    assert result.trades[-1].exit_date > date(2024, 1, 3)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_entry_skips_unaffordable_whole_share(engine):
    result = protection_run(engine, capital=99)
    assert not result.trades
    assert result.metrics["final_equity"] == pytest.approx(99)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_partial_exit_keeps_integer_shares_and_cash(engine):
    result = protection_run(engine, capital=350, low=95, partials=((0.1, 0.5),))
    assert [trade.shares for trade in result.trades] == [1, 2]
    assert result.metrics["final_equity"] == pytest.approx(360)


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_one_share_partial_does_not_raise_stop_without_a_sale(engine):
    result = protection_run(engine, capital=150, low=95, partials=((0.1, 0.5),))
    assert len(result.trades) == 1
    assert result.trades[0].shares == 1
    assert result.trades[0].exit_reason == "time"


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_entry_day_target_only_for_entry_at_open(engine):
    result = protection_run(engine, take_profit=0.1, low=95)
    assert result.trades[0].exit_reason == "target"
    assert result.trades[0].exit_date == date(2024, 1, 2)
    limit = protection_run(
        engine, order="limit", entry_limit_bps=400, take_profit=0.1, low=95
    )
    assert limit.trades[0].exit_reason == "time"


@pytest.mark.parametrize("engine", ["historical", "rolling"])
def test_india_slippage_cannot_create_zero_share_position(engine):
    result = protection_run(engine, capital=100, slippage_bps=10)
    assert not result.trades
    assert result.metrics["final_equity"] == pytest.approx(100)
