from __future__ import annotations

import logging
import os
import time
from datetime import date
from types import SimpleNamespace

import pandas as pd
from click.testing import CliRunner

from screener import universes
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import backtest_rolling
from screener.backtester.rolling_simulation import run_rolling_backtest

_SP500_HTML = """
<html><body>
<table>
<tr><th>Symbol</th><th>Security</th><th>Date added</th></tr>
<tr><td>AAA</td><td>Alpha Corp</td><td>2010-01-15</td></tr>
<tr><td>BB.B</td><td>Beta Inc</td><td>2024-06-03</td></tr>
<tr><td>CCC</td><td>Gamma Ltd</td><td></td></tr>
</table>
</body></html>
"""


def _patch_sp500_page(monkeypatch, tmp_path, counter: dict[str, int]) -> None:
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)

    def fake_get(url, **kwargs):
        counter["fetches"] += 1
        return SimpleNamespace(text=_SP500_HTML, raise_for_status=lambda: None)

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=fake_get))


def test_sp500_membership_parses_date_added(tmp_path, monkeypatch):
    counter = {"fetches": 0}
    _patch_sp500_page(monkeypatch, tmp_path, counter)

    membership = universes.load_sp500_membership(as_of=date(2026, 6, 10))

    assert membership == {
        "AAA": date(2010, 1, 15),
        "BB-B": date(2024, 6, 3),
        "CCC": None,
    }


def test_sp500_membership_uses_cache(tmp_path, monkeypatch):
    counter = {"fetches": 0}
    _patch_sp500_page(monkeypatch, tmp_path, counter)
    as_of = date(2026, 6, 10)

    first = universes.load_sp500_membership(as_of=as_of)
    second = universes.load_sp500_membership(as_of=as_of)
    assert counter["fetches"] == 1
    assert first == second

    universes.load_sp500_membership(as_of=as_of, use_cache=False)
    assert counter["fetches"] == 2


def test_sp500_membership_cache_for_today_expires(tmp_path, monkeypatch):
    counter = {"fetches": 0}
    _patch_sp500_page(monkeypatch, tmp_path, counter)
    today = date.today()

    universes.load_sp500_membership(as_of=today)
    path = universes._membership_cache_path("sp500", today)
    stale_mtime = time.time() - universes._UNIVERSE_CACHE_TTL_SECONDS - 60
    os.utime(path, (stale_mtime, stale_mtime))

    universes.load_sp500_membership(as_of=today)
    assert counter["fetches"] == 2


def test_sp500_membership_serves_stale_cache_when_wikipedia_is_down(
    tmp_path, monkeypatch, caplog
):
    counter = {"fetches": 0}
    _patch_sp500_page(monkeypatch, tmp_path, counter)
    today = date.today()

    expected = universes.load_sp500_membership(as_of=today)
    path = universes._membership_cache_path("sp500", today)
    stale_mtime = time.time() - universes._UNIVERSE_CACHE_TTL_SECONDS - 60
    os.utime(path, (stale_mtime, stale_mtime))

    def boom(url, **kwargs):
        raise RuntimeError("wikipedia is down")

    monkeypatch.setattr(universes, "requests", SimpleNamespace(get=boom))

    with caplog.at_level(logging.WARNING, logger=universes.LOG.name):
        served = universes.load_sp500_membership(as_of=today)

    assert served == expected
    assert "Serving stale sp500 membership cache" in caplog.text


def _trend_bars(start: str = "2024-01-01", n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    close = pd.Series(
        [100.0 + i for i in range(n)],
        index=idx,
        dtype=float,
    )
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(100_000.0, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


def _pit_cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=3,
        top=2,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("AAA", "BBB"),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def test_rolling_backtest_suppresses_entries_before_date_added(stub_fetcher_factory):
    added = date(2024, 2, 15)
    fetcher = stub_fetcher_factory(
        {"AAA": _trend_bars(), "BBB": _trend_bars(), "SPY": _trend_bars()}
    )

    baseline = run_rolling_backtest(
        _pit_cfg(),
        fetcher,
        start_date=date(2024, 2, 1),
        end_date=date(2024, 3, 1),
    )
    baseline_bbb = [t for t in baseline.trades if t.ticker == "BBB"]
    assert baseline_bbb and baseline_bbb[0].entry_date < added

    result = run_rolling_backtest(
        _pit_cfg(membership_added=(("BBB", added),)),
        fetcher,
        start_date=date(2024, 2, 1),
        end_date=date(2024, 3, 1),
    )
    bbb_trades = [t for t in result.trades if t.ticker == "BBB"]
    assert bbb_trades, "BBB should still enter after its date added"
    assert all(t.entry_date >= added for t in bbb_trades)
    bbb_selection = result.selection[result.selection["ticker"] == "BBB"]
    assert (bbb_selection["signal_date"] >= added).all()
    assert any(t.ticker == "AAA" and t.entry_date < added for t in result.trades), (
        "unrestricted symbols should be unaffected"
    )


def test_rolling_backtest_applies_snapshot_membership_windows(stub_fetcher_factory):
    switched = date(2024, 2, 15)
    fetcher = stub_fetcher_factory(
        {"AAA": _trend_bars(), "BBB": _trend_bars(), "SPY": _trend_bars()}
    )
    cfg = _pit_cfg(
        membership_windows=(
            ("AAA", date(2024, 1, 1), switched),
            ("BBB", switched, None),
        )
    )

    result = run_rolling_backtest(
        cfg,
        fetcher,
        start_date=date(2024, 2, 1),
        end_date=date(2024, 3, 1),
    )

    aaa = result.selection[result.selection["ticker"] == "AAA"]
    bbb = result.selection[result.selection["ticker"] == "BBB"]
    assert not aaa.empty and (aaa["signal_date"] < switched).all()
    assert not bbb.empty and (bbb["signal_date"] >= switched).all()


def test_point_in_time_rejects_explicit_ticker_universe():
    runner = CliRunner()
    result = runner.invoke(
        backtest_rolling,
        ["--tickers", "AAA", "--entry", "close > sma(close, 3)", "--point-in-time"],
    )
    assert result.exit_code != 0
    assert "--point-in-time is unavailable" in result.output
    assert "--tickers supplies one fixed list" in result.output


def test_default_point_in_time_does_not_reject_ticker_universe():
    """The flag is on by default, so a ticker list must downgrade, not fail."""
    runner = CliRunner()
    result = runner.invoke(
        backtest_rolling,
        ["--tickers", "AAA", "--entry", "close > sma(close, 3)"],
    )
    assert "--point-in-time is unavailable" not in result.output


def _request(**overrides):
    """A minimal rolling BacktestRequest; overrides name what the test is about."""
    from datetime import datetime

    from screener.backtester.workflow import BacktestRequest

    params = dict(
        mode="rolling",
        context_obj=None,
        market="us",
        hold=20,
        top=10,
        entry_expr="close > 0",
        exit_expr=None,
        strategy_name=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        cost_model="flat",
        initial_capital=100_000.0,
        benchmark=None,
        tickers=None,
        universe=None,
        universe_config=None,
        universe_file=None,
        max_universe=0,
        min_price=None,
        min_avg_dollar_volume=None,
        adv_window=20,
        slippage_model="fixed",
        half_spread_bps=0.0,
        vol_impact_k=0.1,
        no_gap_fills=False,
        entry_order="moo",
        entry_limit_bps=None,
        partial_exit_args=(),
        price_adjustment="full",
        interval="1d",
        output_csv=False,
        report_path=None,
        open_report=False,
        sizing_rule="equal_slot",
        sizing_risk_pct=0.01,
        sizing_position_pct=0.1,
        sizing_atr_window=14,
        sizing_atr_multiple=2.0,
        sizing_vol_window=20,
        intraday_only=False,
        start_arg=datetime(2024, 1, 1),
        end_arg=datetime(2024, 6, 1),
        point_in_time=True,
    )
    params.update(overrides)
    return BacktestRequest(**params)


def test_the_ticker_downgrade_reports_survivorship_bias():
    """A silent downgrade is the dangerous one: the run still looks unbiased.

    The universe branch never runs for --tickers, so this note has nothing to
    append to and has to be created. It was missing entirely before.
    """
    from screener.backtester.workflow import resolve_backtest_run

    note = resolve_backtest_run(_request(tickers="AAA,BBB")).universe_note or ""

    assert "survivorship bias" in note
    assert "--tickers" in note


def test_the_universe_file_downgrade_names_the_file_not_the_ticker_flag():
    from screener.backtester.workflow import resolve_backtest_run

    note = resolve_backtest_run(_request(universe_file="names.txt")).universe_note or ""

    assert "--universe-file supplies one fixed list" in note


def test_a_defaulted_point_in_time_degrades_when_membership_history_fails(monkeypatch):
    """sp500 rebuilds history from dozens of web fetches; offline that raises.

    Point-in-time is on by default, so this run never asked for that history
    and must fall back to the current list instead of aborting.
    """
    from screener.backtester import workflow as workflow_mod
    from screener.universes import UniverseSelection

    def flaky(name, **kwargs):
        if kwargs.get("point_in_time"):
            raise OSError("wikipedia is unreachable")
        return UniverseSelection(name, "us", "SPY", ("AAA", "BBB"), "cached list")

    monkeypatch.setattr(workflow_mod, "load_universe_selection", flaky)

    run = workflow_mod.resolve_backtest_run(_request(universe="sp500"))
    note = run.universe_note or ""

    assert "membership history is unavailable" in note
    assert "wikipedia is unreachable" in note
    assert "point-in-time is inactive" in note


def test_a_typed_point_in_time_still_fails_when_membership_history_fails(monkeypatch):
    """Typing the flag means asking for it, so silently dropping it would lie."""
    import click
    import pytest

    from screener.backtester import workflow as workflow_mod

    def always_fails(name, **kwargs):
        raise OSError("wikipedia is unreachable")

    monkeypatch.setattr(workflow_mod, "load_universe_selection", always_fails)

    with pytest.raises(click.UsageError, match="wikipedia is unreachable"):
        workflow_mod.resolve_backtest_run(
            _request(universe="sp500", point_in_time_was_explicit=True)
        )
