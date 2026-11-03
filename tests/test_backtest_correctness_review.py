"""Regression tests for the backtest correctness review fixes."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from screener.backtester import run_backtest
from screener.backtester.metrics import (
    _sortino,
    equity_curve_sharpe,
    periods_per_year_for_interval,
)
from screener.backtester.models import BacktestConfig
from tests.conftest import StubPriceFetcher


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=1,
        entry_expr="close > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _flat_bars(n: int, start: str = "2024-01-01", price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def _stop_on(bars: pd.DataFrame, pos: int) -> None:
    bars.iat[pos, bars.columns.get_loc("low")] = 50.0
    bars.iat[pos, bars.columns.get_loc("close")] = 60.0


def test_historical_metrics_do_not_depend_on_future_data_length():
    # Same trades; one provider holds 60 more bars after the last exit.
    short = _flat_bars(120)
    long = _flat_bars(180)
    as_of = short.index[100].date()
    results = []
    for bars in (short, long):
        fetcher = StubPriceFetcher({"AAA": bars, "SPY": bars.copy()})
        results.append(
            run_backtest(_cfg(as_of=as_of, hold=3, tickers=("AAA",)), fetcher)
        )
    short_res, long_res = results
    assert [t.exit_date for t in short_res.trades] == [
        t.exit_date for t in long_res.trades
    ]
    assert short_res.equity_curve.index.equals(long_res.equity_curve.index)
    assert short_res.equity_curve.index[-1].date() == short_res.trades[-1].exit_date
    for key in ("cagr", "exposure", "sharpe", "vol_annual"):
        assert short_res.metrics[key] == long_res.metrics[key]


class _RecordingFetcher(StubPriceFetcher):
    def __init__(self, data):
        super().__init__(data)
        self.starts: list[date] = []

    def fetch(self, tickers, start, end):
        self.starts.append(start)
        return super().fetch(tickers, start, end)


def test_historical_fetch_covers_liquidity_and_sizing_windows():
    bars = _flat_bars(1200, start="2020-01-01")
    as_of = bars.index[1100].date()
    fetcher = _RecordingFetcher({"AAA": bars, "SPY": bars.copy()})
    run_backtest(
        _cfg(as_of=as_of, tickers=("AAA",), avg_dollar_volume_window=300), fetcher
    )
    # 3 * 300 + 30 calendar days, as the rolling panel requests.
    assert (pd.Timestamp(as_of) - pd.Timestamp(fetcher.starts[0])).days >= 930


def test_reserve_with_false_signal_on_free_day_stays_queued():
    n = 60
    active = _flat_bars(n)
    reserve = _flat_bars(n)
    as_of_pos = 10
    _stop_on(active, as_of_pos + 2)
    # The reserve signal is false on the day the slot frees, then true later.
    reserve["close"] = reserve["close"].where(
        (reserve.index < active.index[as_of_pos + 1])
        | (reserve.index > active.index[as_of_pos + 5]),
        -1.0,
    )
    reserve.iloc[as_of_pos, reserve.columns.get_loc("volume")] = 1.0
    fetcher = StubPriceFetcher({"AAA": active, "BBB": reserve, "SPY": active.copy()})
    result = run_backtest(
        _cfg(
            as_of=active.index[as_of_pos].date(),
            hold=20,
            stop_loss=0.05,
            tickers=("AAA", "BBB"),
        ),
        fetcher,
    )
    tickers = [t.ticker for t in sorted(result.trades, key=lambda t: t.entry_date)]
    assert tickers[0] == "AAA"
    assert "BBB" in tickers
    bbb = next(t for t in result.trades if t.ticker == "BBB")
    assert bbb.entry_date > active.index[as_of_pos + 5].date()


def test_reentry_that_never_fires_does_not_block_reserve_rotation():
    n = 60
    active = _flat_bars(n)
    reserve = _flat_bars(n)
    as_of_pos = 10
    _stop_on(active, as_of_pos + 2)
    # AAA never signals again after the stop.
    active["close"] = active["close"].where(
        active.index <= active.index[as_of_pos + 2], -1.0
    )
    reserve.iloc[as_of_pos, reserve.columns.get_loc("volume")] = 1.0
    fetcher = StubPriceFetcher({"AAA": active, "BBB": reserve, "SPY": reserve.copy()})
    result = run_backtest(
        _cfg(
            as_of=active.index[as_of_pos].date(),
            hold=20,
            stop_loss=0.05,
            tickers=("AAA", "BBB"),
            allow_reentry=True,
            max_reentries=1,
        ),
        fetcher,
    )
    assert "BBB" in {t.ticker for t in result.trades}


def test_allow_reentry_without_count_warns():
    bars = _flat_bars(60)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": bars.copy()})
    result = run_backtest(
        _cfg(
            as_of=bars.index[10].date(),
            tickers=("AAA",),
            allow_reentry=True,
            max_reentries=0,
        ),
        fetcher,
    )
    assert any("max_reentries is 0" in w for w in result.warnings)


def test_historical_metrics_include_unique_tickers():
    bars = _flat_bars(60)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": bars.copy()})
    result = run_backtest(_cfg(as_of=bars.index[10].date(), tickers=("AAA",)), fetcher)
    assert result.metrics["unique_tickers"] == 1


@pytest.mark.parametrize(
    ("interval", "bars"),
    [("1d", 1), ("1h", 7), ("30m", 13), ("15m", 25), ("5m", 75), ("1m", 375)],
)
def test_india_annualization_uses_nse_session(interval, bars):
    assert periods_per_year_for_interval(interval, "india") == 252 * bars
    assert periods_per_year_for_interval(interval, "INDIA") == 252 * bars


def test_flat_curve_below_positive_hurdle_has_undefined_sharpe():
    zeros = pd.Series([0.0] * 252)
    assert math.isnan(equity_curve_sharpe(zeros, rf=0.06))
    assert equity_curve_sharpe(zeros, rf=0.0) == 0.0
    # Sortino on the same series still reports the shortfall.
    assert _sortino(zeros, rf=0.06) == pytest.approx(-math.sqrt(252))
