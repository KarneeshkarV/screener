"""Fetching must cover causal gate history even when the entry rule is short."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester import BacktestConfig, run_rolling_backtest
from screener.backtester.rolling_simulation import prepare_rolling_backtest
from tests.conftest import StubPriceFetcher


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 + np.arange(len(index)) * 0.1
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 10_000.0,
        },
        index=index,
    )


def _config(index: pd.DatetimeIndex, **changes) -> BacktestConfig:
    values = dict(
        market="us",
        as_of=index[-1].date(),
        benchmark="SPY",
        tickers=("AAA",),
        entry_expr="close > 0",
        exit_expr=None,
        hold=20,
        top=1,
        initial_capital=1_000,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0,
        commission_bps=0,
    )
    values.update(changes)
    return BacktestConfig(**values)


def test_dynamic_universe_has_full_adv_history_on_requested_first_day():
    index = pd.bdate_range("2020-01-01", periods=1_000)
    frame = _frame(index)
    cfg = _config(
        index,
        dynamic_universe_size=1,
        dynamic_universe_lookback=300,
        dynamic_universe_rebalance="daily",
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher({"AAA": frame, "SPY": frame}),
        start_date=index[700].date(),
        end_date=index[760].date(),
    )
    assert result.trades[0].signal_date == index[700].date()


def test_intraday_regime_has_full_benchmark_history_on_first_day():
    index = pd.DatetimeIndex(
        [
            day + pd.Timedelta(hours=hour)
            for day in pd.bdate_range("2024-01-02", periods=80)
            for hour in range(14, 21)
        ]
    )
    frame = _frame(index)
    cfg = _config(index, interval="1h", regime_filter=("bull",))
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher({"AAA": frame, "SPY": frame}),
        start_date=index[300].date(),
        end_date=index[400].date(),
    )
    assert result.trades
    assert result.trades[0].signal_date.date() == index[300].date()


@pytest.mark.parametrize(
    "field", ["avg_dollar_volume_window", "sizing_atr_window", "sizing_vol_window"]
)
def test_large_filter_and_sizing_windows_are_fetched_and_fingerprinted(field):
    index = pd.bdate_range("2020-01-01", periods=1_000)
    frame = _frame(index)
    cfg = _config(index, **{field: 300})
    prepared = prepare_rolling_backtest(
        cfg,
        StubPriceFetcher({"AAA": frame, "SPY": frame}),
        start_date=index[700].date(),
        end_date=index[760].date(),
    )
    assert (prepared.bars_by_tv["AAA"].index < index[700]).sum() >= 300
    assert not prepared.supports(cfg.model_copy(update={field: 400}))
