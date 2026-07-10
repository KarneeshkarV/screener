from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from click.testing import CliRunner

from main import cli
from screener import minervini
from screener.screen_alias_plugins import mark_minervini as mark_minervini_plugin
from screener.backtester.pine import evaluate, parse
from screener.minervini import (
    MINERVINI_ENTRY_EXPR,
    add_rs_rank_column,
    evaluate_symbol,
    prepare_backtest_frames,
    required_history_bars,
)
from screener.strategies.expressions import resolve_strategy


def _bars(start: float, end: float, n: int = 320) -> pd.DataFrame:
    idx = pd.bdate_range(end="2026-04-30", periods=n)
    close = pd.Series(np.linspace(start, end, n), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 100_000.0,
        },
        index=idx,
    )


def test_evaluate_symbol_matches_minervini_template() -> None:
    prepared = add_rs_rank_column(
        {
            "LEADER": _bars(50.0, 180.0),
            "MID": _bars(80.0, 120.0),
            "LAGGARD": _bars(120.0, 100.0),
        }
    )

    row = evaluate_symbol("LEADER", prepared["LEADER"], date(2026, 4, 30))
    laggard = evaluate_symbol("LAGGARD", prepared["LAGGARD"], date(2026, 4, 30))

    assert row is not None
    assert row.symbol == "LEADER"
    assert row.rs_rank >= 70.0
    assert row.close > row.sma50 > row.sma150 > row.sma200
    assert laggard is None


def test_mark_minervini_strategy_expression_is_backtestable() -> None:
    prepared = prepare_backtest_frames(
        {
            "LEADER": _bars(50.0, 180.0),
            "MID": _bars(80.0, 120.0),
            "LAGGARD": _bars(120.0, 100.0),
        }
    )

    signal = evaluate(parse(MINERVINI_ENTRY_EXPR), prepared["LEADER"])

    assert len(prepared["LEADER"]) >= required_history_bars()
    assert bool(signal.iloc[-1]) is True


def test_strategy_registry_exposes_mark_minervini() -> None:
    strategy = resolve_strategy("mark_minervini")

    assert strategy.entry == MINERVINI_ENTRY_EXPR
    assert strategy.exit is not None


def test_screen_pipeline_runs_with_stubbed_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        mark_minervini_plugin,
        "scan_minervini",
        lambda *args, **kwargs: [
            minervini.MinerviniRow(
                symbol="LEADER",
                close=180.0,
                sma50=170.0,
                sma150=140.0,
                sma200=130.0,
                pct_above_low_52w=260.0,
                pct_below_high_52w=-1.0,
                rs_rank=100.0,
                as_of=date(2026, 4, 30),
            )
        ],
    )

    res = CliRunner().invoke(
        cli,
        ["screen", "-c", "mark-minervini", "-m", "us", "-n", "1"],
    )

    assert res.exit_code == 0, res.output
    assert "Mark Minervini Screen" in res.output
    assert "LEADER" in res.output
    assert "proprietary RS Rating" in res.output
