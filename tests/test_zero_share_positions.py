"""Regression: equal_slot cash exhaustion must not park zero-share ghost lots.

Under fixed ``equal_slot``, a refill can see ``entry_budget == 0`` after earlier
fills consume cash. Opening that lot still occupied a slot and excluded the
ticker until exit, which blocked later eligible candidates. The zero-share
guard must apply for every sizing rule, not only ``reinvested_equal_slot``.
"""

from __future__ import annotations

import csv
import io
from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.slippage import VolumeImpactSlippage
from screener.cli import cli
from tests.conftest import StubPriceFetcher

_N = 35
_INDEX = pd.bdate_range("2024-01-01", periods=_N)


def _ohlcv(close: np.ndarray, volume: float | np.ndarray) -> pd.DataFrame:
    close_arr = np.asarray(close, dtype=float)
    open_arr = np.concatenate([[close_arr[0]], close_arr[:-1]])
    vol = (
        np.asarray(volume, dtype=float)
        if not np.isscalar(volume)
        else np.full(_N, float(volume))
    )
    return pd.DataFrame(
        {
            "open": open_arr,
            "high": np.maximum(open_arr, close_arr) + 1.0,
            "low": np.minimum(open_arr, close_arr) - 1.0,
            "close": close_arr,
            "volume": vol,
        },
        index=_INDEX,
    )


def _ghost_panel() -> dict[str, pd.DataFrame]:
    """Lossy book that parks a CCC ghost under the ungated equal_slot bug.

    First cycle deploys three full slots, then prices crash so exit cash cannot
    refill all three. CCC would open at ``shares=0`` and stay active, so the
    early AAA exit cannot give CCC a real fill and DDD takes the cash instead.
    """
    base = np.full(_N, 100.0)
    base[8:] = 40.0
    aaa = base.copy()
    # Exit + stay below the entry floor so AAA is not immediately reselected.
    aaa[19:23] = 30.0
    return {
        "AAA": _ohlcv(aaa, 4_000_000.0),
        "BBB": _ohlcv(base, 3_000_000.0),
        "CCC": _ohlcv(base, 2_000_000.0),
        "DDD": _ohlcv(base, 1_000_000.0),
        "SPY": _ohlcv(np.linspace(400.0, 420.0, _N), 10_000_000.0),
    }


def _rolling_ghost_cfg(**overrides) -> BacktestConfig:
    base = dict(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=10,
        top=3,
        strategy_name=None,
        entry_expr="close >= 40",
        exit_expr="close < 35",
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=30_000.0,
        benchmark="SPY",
        tickers=("AAA", "BBB", "CCC", "DDD"),
        sizing_rule="equal_slot",
        cost_model="flat",
    )
    base.update(overrides)
    return BacktestConfig(**base)


def _assert_no_ghost_trades(trades) -> None:
    ghosts = [t for t in trades if t.shares <= 0.0]
    assert ghosts == [], f"zero-share ghost trades: {ghosts!r}"


def test_cli_rolling_equal_slot_cash_exhaustion_has_no_ghost_trades():
    """Offline rolling CLI with stub prices must not emit shares=0 ledger rows."""
    fetcher = StubPriceFetcher(_ghost_panel())
    result = CliRunner().invoke(
        cli,
        [
            "--no-agent",
            "backtest-rolling",
            "--tickers",
            "AAA,BBB,CCC,DDD",
            "--start",
            _INDEX[0].date().isoformat(),
            "--end",
            _INDEX[-1].date().isoformat(),
            "--hold",
            "10",
            "--top",
            "3",
            "--entry",
            "close >= 40",
            "--exit",
            "close < 35",
            "--initial-capital",
            "30000",
            "--sizing",
            "equal_slot",
            "--slippage-bps",
            "0",
            "--commission-bps",
            "0",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--csv",
        ],
        obj=fetcher,
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(io.StringIO(result.output)))
    assert rows, "expected trade CSV rows from --csv"
    ghosts = [row for row in rows if float(row["shares"]) == 0.0]
    assert ghosts == [], f"CLI ghost rows: {ghosts!r}"

    # After AAA's early exit frees cash, higher-ranked CCC must be able to take
    # a real fill. Under the bug CCC is still parked as a ghost so DDD wins.
    post = [
        row
        for row in rows
        if row["entry_date"] >= "2024-01-27" and float(row["shares"]) > 0.0
    ]
    post_tickers = {row["ticker"] for row in post}
    assert "CCC" in post_tickers
    assert "DDD" not in post_tickers


def test_rolling_equal_slot_skips_zero_budget_and_lets_later_candidate_fill():
    result = run_rolling_backtest(
        _rolling_ghost_cfg(),
        StubPriceFetcher(_ghost_panel()),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    _assert_no_ghost_trades(result.trades)

    post_ccc = [
        t
        for t in result.trades
        if t.ticker == "CCC" and t.shares > 0.0 and t.entry_date >= date(2024, 1, 27)
    ]
    assert post_ccc, "expected CCC to receive a real fill once cash returns"
    assert all(t.entry_cost > 0.0 for t in post_ccc)

    # Intentional fractional-share sizing must remain intact.
    assert any(t.shares > 0.0 and t.shares != int(t.shares) for t in result.trades)


def test_rolling_volume_impact_equal_slot_skips_zero_share_quotes():
    result = run_rolling_backtest(
        _rolling_ghost_cfg(slippage_model=VolumeImpactSlippage(k=0.1)),
        StubPriceFetcher(_ghost_panel()),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )
    _assert_no_ghost_trades(result.trades)
    assert any(t.shares > 0.0 for t in result.trades)


def test_historical_reserve_refill_skips_zero_share_equal_slot():
    """Reserve promotion after a lossy exit must not park a shares=0 lot."""
    n = 60
    idx = pd.bdate_range("2024-01-01", periods=n)
    as_of_i = 20
    closes = np.full(n, 100.0)
    closes[as_of_i + 2 :] = 40.0

    def frame(volume: float) -> pd.DataFrame:
        open_arr = np.concatenate([[closes[0]], closes[:-1]])
        return pd.DataFrame(
            {
                "open": open_arr,
                "high": np.maximum(open_arr, closes) + 1.0,
                "low": np.minimum(open_arr, closes) - 1.0,
                "close": closes,
                "volume": np.full(n, volume),
            },
            index=idx,
        )

    names = ("A1", "A2", "A3", "R1", "R2", "R3")
    volumes = (5e6, 4e6, 3e6, 2e6, 1.5e6, 1e6)
    panel = {name: frame(vol) for name, vol in zip(names, volumes, strict=True)}
    panel["SPY"] = frame(1e7)
    cfg = BacktestConfig(
        market="us",
        as_of=idx[as_of_i].date(),
        hold=5,
        top=3,
        strategy_name=None,
        entry_expr="close > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=30_000.0,
        benchmark="SPY",
        tickers=names,
        sizing_rule="equal_slot",
        reserve_multiple=3,
        reinvest=True,
    )
    result = run_backtest(cfg, StubPriceFetcher(panel))
    _assert_no_ghost_trades(result.trades)
    # First lossy refill previously opened R3 at shares=0; it must stay out.
    assert all(t.ticker != "R3" or t.shares > 0.0 for t in result.trades)
    assert not any(t.ticker == "R3" for t in result.trades)


def test_tiny_positive_cash_still_opens_fractional_shares():
    """Exact-zero is the bug; a tiny positive budget must still buy fractions."""
    from screener.backtester.portfolio import Portfolio

    portfolio = Portfolio(0.01, 1)
    position = portfolio.open("AAA", _INDEX[0].date(), 100.0, budget=0.01)
    assert position.shares > 0.0
    assert position.shares == pytest.approx(0.01 / 100.0)
