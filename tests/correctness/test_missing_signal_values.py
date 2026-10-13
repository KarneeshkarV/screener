"""Unknown expression operands must never invent an entry or exit signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester import BacktestConfig, run_rolling_backtest
from screener.backtester.pine import (
    evaluate,
    evaluate_panel,
    evaluate_panel_many,
    parse,
)
from tests.conftest import StubPriceFetcher


def _bars() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=8)
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000.0},
        index=index,
    )
    frame["known_value"] = [np.nan, 0.0, 1.0, np.inf, -np.inf, 0.0, 1.0, 0.0]
    return frame


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("known_value != 0", [False, False, True, False, False, False, True, False]),
        (
            "known_value and close > 0",
            [False, False, True, False, False, False, True, False],
        ),
        ("not known_value", [False, True, False, False, False, True, False, True]),
        (
            "not (known_value == 0)",
            [False, False, True, False, False, False, True, False],
        ),
        (
            "not (known_value != 0)",
            [False, True, False, False, False, True, False, True],
        ),
        (
            "known_value or close < 0",
            [False, False, True, False, False, False, True, False],
        ),
        ("known_value or close > 0", [True] * 8),
        (
            "not (known_value and close > 0)",
            [False, True, False, False, False, True, False, True],
        ),
        ("known_value > 0", [False, False, True, False, False, False, True, False]),
        (
            "(1 / known_value) != 0",
            [False, False, True, False, False, False, True, False],
        ),
        (
            "not crossover(known_value, 0.5)",
            [False, False, False, False, False, False, False, True],
        ),
        (
            "not crossunder(known_value, 0.5)",
            [False, False, True, False, False, False, True, False],
        ),
    ],
)
def test_missing_operands_preserve_unknown_until_final_signal(expression, expected):
    frame = _bars()
    ast = parse(expression)
    single = evaluate(ast, frame)
    assert single.dtype == bool
    assert single.tolist() == expected
    panel = evaluate_panel(ast, {"AAA": frame, "BBB": frame.copy()})
    arrays = evaluate_panel_many(
        (ast,), {"AAA": frame, "BBB": frame.copy()}, as_bool_array=True
    )[0]
    for ticker in ("AAA", "BBB"):
        assert panel[ticker].tolist() == expected
        assert arrays[ticker].tolist() == expected


def test_missing_fundamental_never_creates_prepublication_trade():
    frame = _bars()
    frame["revenue_up_3q"] = [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg = BacktestConfig(
        market="us",
        as_of=frame.index[-1].date(),
        benchmark="SPY",
        tickers=("AAA",),
        entry_expr="revenue_up_3q != 0",
        exit_expr=None,
        hold=10,
        top=1,
        initial_capital=1_000,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0,
        commission_bps=0,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher({"AAA": frame, "SPY": frame}),
        start_date=frame.index[0].date(),
        end_date=frame.index[-1].date(),
    )
    assert result.trades
    assert result.trades[0].signal_date == frame.index[3].date()
    assert result.trades[0].entry_date == frame.index[4].date()
