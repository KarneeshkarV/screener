from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
from click.testing import CliRunner

from screener import universes
from screener.backtester.models import BacktestConfig
from screener.backtester.rolling import backtest_rolling, run_rolling_backtest


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


def test_point_in_time_rejects_explicit_ticker_universe():
    runner = CliRunner()
    result = runner.invoke(
        backtest_rolling,
        ["--tickers", "AAA", "--entry", "close > sma(close, 3)", "--point-in-time"],
    )
    assert result.exit_code != 0
    assert "--point-in-time requires an index universe" in result.output
