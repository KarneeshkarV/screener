"""Fetch windows must cover strategy-prepared columns, not just the AST.

Backtests used to size history from the entry and exit expressions only.
``low_volatility`` writes ``vol_252`` in ``prepare_bars`` (253 prior bars)
while its entry expression only names the column. Too little history leaves
that column as NaN and valid trades never fire. The probe below is the same
shape with a longer window, so the gap between the two warmups is unmistakable.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, timedelta

import pandas as pd
import pytest

from screener.backtester.core import strategy_required_lookback
from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import parse, required_lookback
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.warmup import _warmup_days_for_interval
from screener.strategies.spec import register_expression_strategy, registry
from tests.conftest import StubPriceFetcher, make_bars

_PROBE = "unit_prepared_column_lookback_probe"
_PROBE_PERIOD = 350
_PROBE_COLUMN = "probe_sma"
_ENTRY = f"{_PROBE_COLUMN} > 0"


class _RecordingFetcher:
    def __init__(self, inner: StubPriceFetcher) -> None:
        self._inner = inner
        self.starts: list[date] = []

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        self.starts.append(start)
        return self._inner.fetch(tickers, start, end)


def _prepare_probe_column(ctx):
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame[_PROBE_COLUMN] = close.rolling(
            _PROBE_PERIOD, min_periods=_PROBE_PERIOD
        ).mean()
        out[tv] = frame
    return out


@pytest.fixture(autouse=True, scope="module")
def _probe_strategy() -> Iterator[str]:
    """Register the probe for this module only.

    The strategy registry is process-global, so a probe left behind would show
    up in every later test that enumerates registered strategies.
    """
    register_expression_strategy(
        _PROBE,
        entry=_ENTRY,
        exit=None,
        prepare_bars=_prepare_probe_column,
        required_lookback=lambda: _PROBE_PERIOD,
    )
    yield _PROBE
    registry.remove(_PROBE)


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 6, 3),
        hold=5,
        top=1,
        entry_expr=_ENTRY,
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        strategy_name=_PROBE,
        tickers=("AAA",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _panel(n: int = 800) -> dict[str, pd.DataFrame]:
    bars = make_bars(start="2021-01-04", n=n, open_base=100.0, seed=1)
    return {"AAA": bars, "SPY": bars.copy()}


def test_probe_entry_expression_does_not_encode_the_prepare_window() -> None:
    assert required_lookback(parse(_ENTRY)) == 0
    assert strategy_required_lookback(_PROBE) == _PROBE_PERIOD
    # low_volatility is the real case: 252 rolling returns plus the bar
    # pct_change consumes, none of it visible in ``vol_252 > 0``.
    assert strategy_required_lookback("low_volatility") == 253
    # bb_breakout is the other shape a hidden window takes: a declared bar
    # column. Its 350-period Bollinger bands are invisible to an entry of
    # ``crossover(close, bb_upper)``, so the floor comes from the column.
    assert strategy_required_lookback("bb_breakout") == 350
    assert strategy_required_lookback(None) == 0


def test_unresolvable_strategy_reports_instead_of_dropping_the_floor() -> None:
    warnings: list[str] = []
    assert strategy_required_lookback("combo:", warnings) == 0
    assert warnings and "strategy lookback unavailable" in warnings[0]


def test_historical_fetch_covers_strategy_required_lookback() -> None:
    as_of = date(2024, 6, 3)
    fetcher = _RecordingFetcher(StubPriceFetcher(_panel()))
    run_backtest(_cfg(as_of=as_of), fetcher)

    assert fetcher.starts
    expected = as_of - timedelta(days=_warmup_days_for_interval(_PROBE_PERIOD, "1d"))
    expression_only = as_of - timedelta(days=_warmup_days_for_interval(0, "1d"))
    assert min(fetcher.starts) == expected
    assert expected < expression_only


def test_rolling_fetch_covers_strategy_required_lookback() -> None:
    start = date(2024, 1, 2)
    end = date(2024, 3, 1)
    fetcher = _RecordingFetcher(StubPriceFetcher(_panel()))
    run_rolling_backtest(_cfg(as_of=end), fetcher, start_date=start, end_date=end)

    assert fetcher.starts
    expected = start - timedelta(days=_warmup_days_for_interval(_PROBE_PERIOD, "1d"))
    expression_only = start - timedelta(days=_warmup_days_for_interval(0, "1d"))
    assert min(fetcher.starts) == expected
    assert expected < expression_only


def test_rolling_takes_trades_once_prepared_columns_have_history() -> None:
    # Short window: expression-only warmup (~365 calendar days) cannot fill a
    # 350-bar rolling column, so every signal stays NaN and no trade fires.
    bars = _panel(n=900)
    index = bars["AAA"].index
    start = index[-40].date()
    end = index[-1].date()
    result = run_rolling_backtest(
        _cfg(as_of=end),
        StubPriceFetcher(bars),
        start_date=start,
        end_date=end,
    )
    assert result.trades, "probe_sma stayed NaN; fetch window was too short"
