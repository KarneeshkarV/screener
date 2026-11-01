"""Limit orders must wait for bars instead of peeking at future fills."""

from datetime import date

import pandas as pd

from screener.backtester import BacktestConfig, run_backtest, run_rolling_backtest
from tests.conftest import StubPriceFetcher


def test_unfilled_top_limit_order_does_not_select_lower_ranked_future_fill() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)

    def bars(low: float, volume: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [100.0] * len(dates),
                "high": [101.0] * len(dates),
                "low": [low] * len(dates),
                "close": [100.0] * len(dates),
                "volume": [volume] * len(dates),
            },
            index=dates,
        )

    config = BacktestConfig(
        market="us",
        as_of=date(2024, 1, 9),
        benchmark="SPY",
        tickers=("AAA", "BBB"),
        entry_expr="close > 50",
        exit_expr=None,
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0,
        commission_bps=0,
        top=1,
        initial_capital=1000,
        entry_order_type="limit",
        entry_limit_bps=1200,
        min_price=0,
        min_avg_dollar_volume=0,
    )
    fetcher = StubPriceFetcher(
        {"AAA": bars(95, 10_000), "BBB": bars(87, 5_000), "SPY": bars(95, 1000)}
    )

    result = run_rolling_backtest(
        config, fetcher, start_date=dates[0].date(), end_date=dates[-1].date()
    )

    assert result.trades == []
    assert result.metrics["final_equity"] == 1000


def test_historical_limit_order_fills_only_when_later_bar_touches() -> None:
    dates = pd.bdate_range("2024-01-02", periods=7)
    lows = [99.0, 95.0, 95.0, 95.0, 87.0, 95.0, 95.0]
    bars = pd.DataFrame(
        {
            "open": [100.0] * len(dates),
            "high": [101.0] * len(dates),
            "low": lows,
            "close": [100.0] * len(dates),
            "volume": [10_000.0] * len(dates),
        },
        index=dates,
    )
    config = BacktestConfig(
        market="us",
        as_of=dates[0].date(),
        benchmark="SPY",
        tickers=("AAA",),
        entry_expr="close > 50",
        exit_expr=None,
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0,
        commission_bps=0,
        top=1,
        initial_capital=1000,
        entry_order_type="limit",
        entry_limit_bps=1200,
        min_price=0,
        min_avg_dollar_volume=0,
    )

    result = run_backtest(config, StubPriceFetcher({"AAA": bars, "SPY": bars}))

    assert len(result.trades) == 1
    assert result.trades[0].signal_date == dates[0].date()
    assert result.trades[0].entry_date == dates[4].date()
