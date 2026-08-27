"""Fetch windows must cover strategy-prepared columns, not just the AST.

Backtests used to size history from the entry and exit expressions only.
A strategy such as a Bollinger Band breakout writes ``bb_upper`` in
``prepare_bars`` (about 350 prior bars) while the expression only names the
column. Too little history leaves that column as NaN and valid trades never
fire.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd

from screener.backtester.core import strategy_required_lookback
from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import parse, required_lookback
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.warmup import _warmup_days_for_interval
from screener.strategies.spec import register_expression_strategy, registry
from tests.conftest import StubPriceFetcher, make_bars

_PROBE = "unit_bb_column_lookback_probe"
_BB_PERIOD = 350
_ENTRY = "bb_upper > 0"


class _RecordingFetcher:
    def __init__(self, inner: StubPriceFetcher) -> None:
        self._inner = inner
        self.starts: list[date] = []

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        self.starts.append(start)
        return self._inner.fetch(tickers, start, end)


def _ensure_probe() -> str:
    if registry.get_optional(_PROBE) is None:
        register_expression_strategy(
            _PROBE,
            entry=_ENTRY,
            exit=None,
            prepare_bars=_prepare_bb_upper,
            required_lookback=lambda: _BB_PERIOD,
        )
    return _PROBE


def _prepare_bb_upper(ctx):
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["bb_upper"] = close.rolling(_BB_PERIOD, min_periods=_BB_PERIOD).mean()
        out[tv] = frame
    return out


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
        strategy_name=_ensure_probe(),
        tickers=("AAA",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _panel(n: int = 800) -> dict[str, pd.DataFrame]:
    bars = make_bars(start="2021-01-04", n=n, open_base=100.0, seed=1)
    return {"AAA": bars, "SPY": bars.copy()}


def test_probe_entry_expression_does_not_encode_the_bb_window() -> None:
    assert required_lookback(parse(_ENTRY)) == 0
    assert strategy_required_lookback(_ensure_probe()) == _BB_PERIOD
    assert strategy_required_lookback("bb_breakout") == 0
    assert strategy_required_lookback("low_volatility") == 253
    assert strategy_required_lookback(None) == 0


def test_historical_fetch_covers_strategy_required_lookback() -> None:
    as_of = date(2024, 6, 3)
    fetcher = _RecordingFetcher(StubPriceFetcher(_panel()))
    run_backtest(_cfg(as_of=as_of), fetcher)

    assert fetcher.starts
    expected = as_of - timedelta(days=_warmup_days_for_interval(_BB_PERIOD, "1d"))
    expression_only = as_of - timedelta(days=_warmup_days_for_interval(0, "1d"))
    assert min(fetcher.starts) == expected
    assert expected < expression_only


def test_rolling_fetch_covers_strategy_required_lookback() -> None:
    start = date(2024, 1, 2)
    end = date(2024, 3, 1)
    fetcher = _RecordingFetcher(StubPriceFetcher(_panel()))
    run_rolling_backtest(_cfg(as_of=end), fetcher, start_date=start, end_date=end)

    assert fetcher.starts
    expected = start - timedelta(days=_warmup_days_for_interval(_BB_PERIOD, "1d"))
    expression_only = start - timedelta(days=_warmup_days_for_interval(0, "1d"))
    assert min(fetcher.starts) == expected
    assert expected < expression_only


def test_rolling_takes_trades_once_bb_columns_have_history() -> None:
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
    assert result.trades, "bb_upper stayed NaN; fetch window was too short"
