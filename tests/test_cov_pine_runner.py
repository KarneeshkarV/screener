"""Offline coverage tests for ``screener.research.pine_runner``.

All tests are deterministic and offline: every provider/fetcher/scan seam is
monkeypatched so nothing touches the network.
"""

from __future__ import annotations

import json
import runpy
from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.research.pine_runner import cli as cli_mod
from screener.research.pine_runner import data as data_mod
from screener.research.pine_runner import output as output_mod
from screener.research.pine_runner import run as run_mod
from screener.research.pine_runner.run import (
    MarketRun,
    _compound,
    _run_ticker,
    run_market,
)
from screener.strategies.trades import Trade

from tests.conftest import StubPriceFetcher, make_bars


# ── data.py ─────────────────────────────────────────────────────────


def _ohlcv_frame(n: int = 60, seed: int = 1) -> pd.DataFrame:
    """A make_bars frame with a DatetimeIndex (as a price fetcher returns)."""
    return make_bars(n=n, seed=seed)


def test_fetch_ohlcv_uses_module_fetcher_and_adds_adj_close(monkeypatch):
    frame = _ohlcv_frame()
    stub = StubPriceFetcher({"AAA.NS": frame})
    monkeypatch.setattr(data_mod, "_FETCHER", stub)
    monkeypatch.setattr(data_mod, "tv_to_yf", lambda t, m: "AAA.NS")

    out = data_mod.fetch_ohlcv("AAA", date(2024, 1, 1), date(2025, 1, 1), "india")

    assert out is not None
    assert "date" in out.columns
    # adj_close synthesized from close when absent
    assert "adj_close" in out.columns
    assert (out["adj_close"] == out["close"]).all()


def test_fetch_ohlcv_caret_symbol_skips_tv_to_yf(monkeypatch):
    frame = _ohlcv_frame()
    stub = StubPriceFetcher({"^NSEI": frame})
    monkeypatch.setattr(data_mod, "_FETCHER", stub)

    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("tv_to_yf should not be called for ^ symbols")

    monkeypatch.setattr(data_mod, "tv_to_yf", _boom)

    out = data_mod.fetch_ohlcv("^NSEI", date(2024, 1, 1), date(2025, 1, 1), "us")
    assert out is not None


def test_fetch_ohlcv_keeps_existing_adj_close(monkeypatch):
    frame = _ohlcv_frame()
    frame = frame.copy()
    frame["adj_close"] = frame["close"] * 0.5
    stub = StubPriceFetcher({"AAA": frame})
    monkeypatch.setattr(data_mod, "_FETCHER", stub)
    monkeypatch.setattr(data_mod, "tv_to_yf", lambda t, m: "AAA")

    out = data_mod.fetch_ohlcv("AAA", date(2024, 1, 1), date(2025, 1, 1), "us")
    assert out is not None
    # original adj_close preserved (not overwritten by close)
    assert not (out["adj_close"] == out["close"]).all()


def test_fetch_ohlcv_refresh_builds_fresh_fetcher(monkeypatch):
    frame = _ohlcv_frame()
    fresh = StubPriceFetcher({"AAA": frame})
    calls = {}

    def fake_build(refresh=False):
        calls["refresh"] = refresh
        return fresh

    monkeypatch.setattr(data_mod, "build_price_fetcher", fake_build)
    monkeypatch.setattr(data_mod, "tv_to_yf", lambda t, m: "AAA")

    out = data_mod.fetch_ohlcv(
        "AAA", date(2024, 1, 1), date(2025, 1, 1), "us", refresh=True
    )
    assert out is not None
    assert calls["refresh"] is True


def test_fetch_ohlcv_returns_none_when_missing(monkeypatch):
    stub = StubPriceFetcher({})  # no data → empty frame
    monkeypatch.setattr(data_mod, "_FETCHER", stub)
    monkeypatch.setattr(data_mod, "tv_to_yf", lambda t, m: "ZZZ")

    out = data_mod.fetch_ohlcv("ZZZ", date(2024, 1, 1), date(2025, 1, 1), "us")
    assert out is None


def test_load_universe_us_price_floor(monkeypatch):
    captured = {}

    def fake_scan(*, market, filters, limit, order_by):
        captured["market"] = market
        captured["limit"] = limit
        df = pd.DataFrame({"name": ["AAA", "BBB", None]})
        return 3, df

    monkeypatch.setattr(data_mod, "_tv_scan", fake_scan)
    out = data_mod.load_universe("us")
    assert out == ["AAA", "BBB"]
    assert captured["market"] == "us"
    assert captured["limit"] == 500


def test_load_universe_india_uses_default_branch(monkeypatch):
    def fake_scan(*, market, filters, limit, order_by):
        return 0, pd.DataFrame({"name": ["XYZ"]})

    monkeypatch.setattr(data_mod, "_tv_scan", fake_scan)
    # "fr" is unknown → exercises the .get default price floor path
    assert data_mod.load_universe("india") == ["XYZ"]


# ── run.py ──────────────────────────────────────────────────────────


def _trade(entry_idx, exit_idx, entry_px, exit_px, entry_date, exit_date):
    return Trade(
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_px=entry_px,
        exit_px=exit_px,
        entry_date=pd.Timestamp(entry_date),
        exit_date=pd.Timestamp(exit_date),
    )


def test_compound():
    t1 = _trade(0, 1, 100.0, 110.0, "2024-01-01", "2024-01-02")  # +10%
    t2 = _trade(2, 3, 100.0, 90.0, "2024-01-03", "2024-01-04")  # -10%
    # 1.1 * 0.9 - 1 = -0.01
    assert _compound([t1, t2]) == pytest.approx(-0.01)
    assert _compound([]) == 0.0


def _df_with_date(n=60, start="2024-01-01"):
    bars = make_bars(n=n, start=start)
    df = bars.reset_index().rename(columns={"index": "date"})
    return df


def test_run_ticker_too_short_returns_none():
    df = _df_with_date(n=40)
    assert _run_ticker(df, pd.Timestamp("2024-01-01"), lambda d: []) is None


def test_run_ticker_aggregates_in_window_trades():
    df = _df_with_date(n=60, start="2024-01-01")
    window_start = pd.Timestamp(df["date"].iloc[30])
    before = pd.Timestamp(df["date"].iloc[5])
    after_entry = pd.Timestamp(df["date"].iloc[40])
    after_exit = pd.Timestamp(df["date"].iloc[45])

    def strat(d):
        # one trade before window (excluded), one inside (counted, winning)
        return [
            _trade(5, 10, 100.0, 90.0, before, df["date"].iloc[10]),
            _trade(40, 45, 100.0, 120.0, after_entry, after_exit),
        ]

    res = _run_ticker(df, window_start, strat)
    assert res is not None
    assert res["n_trades"] == 1  # only in-window
    assert res["wins"] == 1
    assert res["exposure"] == 5  # 45 - 40
    assert res["total_return"] == pytest.approx(0.2)
    assert res["n_bars"] == int((pd.to_datetime(df["date"]) >= window_start).sum())


def _patch_run_market_seams(monkeypatch, *, universe, ohlcv, bench_df, strategies):
    monkeypatch.setattr(run_mod, "load_universe", lambda market: list(universe))

    def fake_fetch(t, start, end, market, refresh=False):
        if t == "SPY" or t == "^NSEI":
            return bench_df
        return ohlcv.get(t)

    monkeypatch.setattr(run_mod, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(run_mod, "STRATEGIES", strategies)
    return fake_fetch


def test_run_market_full_flow(monkeypatch):
    df_a = _df_with_date(n=60)
    df_b = _df_with_date(n=60, start="2024-02-01")
    window_start = pd.Timestamp(date.today()) - pd.DateOffset(years=3)
    window_start = window_start.normalize()

    entry_d = pd.Timestamp(date.today()) - pd.DateOffset(years=1)
    exit_d = pd.Timestamp(date.today()) - pd.DateOffset(months=6)

    def good_strat(d):
        return [_trade(40, 45, 100.0, 130.0, entry_d, exit_d)]

    def boom_strat(d):
        raise ValueError("kaboom")

    # benchmark frame with adj_close spanning the window (both bars in-window)
    bench = pd.DataFrame(
        {
            "date": pd.to_datetime([window_start + pd.Timedelta(days=1), date.today()]),
            "adj_close": [100.0, 150.0],
            "close": [100.0, 150.0],
        }
    )

    df_short = _df_with_date(n=40)  # < 50 bars → _run_ticker returns None

    _patch_run_market_seams(
        monkeypatch,
        universe=["AAA", "BBB", "CCC", "DDD"],
        # CCC missing → returns None; DDD too short → _run_ticker returns None
        ohlcv={"AAA": df_a, "BBB": df_b, "DDD": df_short},
        bench_df=bench,
        strategies={"good": good_strat, "bad": boom_strat},
    )

    res = run_market(market="us", years=3, limit=0, refresh=False)
    assert isinstance(res, MarketRun)
    assert res.benchmark_symbol == "SPY"
    assert res.benchmark_return == pytest.approx(0.5)
    # AAA, BBB raise in boom_strat; DDD is too short so _run_ticker returns None
    # before the strategy runs → bad only errors twice.
    assert res.error_counts["bad"] == 2
    # good: AAA, BBB produce results; DDD returns None (res is None → skipped)
    assert len(res.per_strategy["good"]) == 2
    assert all("ticker" in r for r in res.per_strategy["good"])


def test_run_market_limit_caps_universe(monkeypatch):
    df_a = _df_with_date(n=60)
    bench = pd.DataFrame(
        {"date": pd.to_datetime([date.today()]), "adj_close": [100.0], "close": [100.0]}
    )
    seen = {}

    monkeypatch.setattr(run_mod, "load_universe", lambda market: ["AAA", "BBB", "CCC"])

    def fake_fetch(t, start, end, market, refresh=False):
        seen[t] = True
        if t == "SPY":
            return bench
        return df_a if t == "AAA" else None

    monkeypatch.setattr(run_mod, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(run_mod, "STRATEGIES", {"good": lambda d: []})

    res = run_market(market="us", years=3, limit=1, refresh=False)
    # only AAA fetched from the universe (plus the SPY benchmark)
    assert "BBB" not in seen and "CCC" not in seen
    assert res.market == "us"


def test_run_market_benchmark_missing_warns(monkeypatch):
    monkeypatch.setattr(run_mod, "load_universe", lambda market: ["AAA"])

    def fake_fetch(t, start, end, market, refresh=False):
        if t == "^NSEI":
            return None  # benchmark missing
        return None  # AAA also has no data

    monkeypatch.setattr(run_mod, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(run_mod, "STRATEGIES", {"s": lambda d: []})

    res = run_market(market="india", years=2, limit=0, refresh=False)
    assert res.benchmark_symbol == "^NSEI"
    assert res.benchmark_return is None
    assert res.per_strategy["s"] == []


def test_run_market_benchmark_single_bar_stays_none(monkeypatch):
    # benchmark frame with only one in-window bar → len(b) <= 1 branch
    bench = pd.DataFrame(
        {
            "date": pd.to_datetime([date.today()]),
            "adj_close": [100.0],
            "close": [100.0],
        }
    )
    monkeypatch.setattr(run_mod, "load_universe", lambda market: ["AAA"])

    def fake_fetch(t, start, end, market, refresh=False):
        if t == "SPY":
            return bench
        return None

    monkeypatch.setattr(run_mod, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(run_mod, "STRATEGIES", {"s": lambda d: []})

    res = run_market(market="us", years=3, limit=0, refresh=False)
    assert res.benchmark_return is None


def test_marketrun_rejects_empty_market():
    with pytest.raises(ValueError):
        MarketRun(
            market="   ",
            today=date.today(),
            window_start=pd.Timestamp("2020-01-01"),
            benchmark_symbol="SPY",
            benchmark_return=None,
            per_strategy={},
            error_counts={},
        )


# ── output.py ───────────────────────────────────────────────────────


def _make_result(*, per_strategy, error_counts, benchmark_return=0.1):
    return MarketRun(
        market="us",
        today=date.today(),
        window_start=pd.Timestamp("2023-01-01"),
        benchmark_symbol="SPY",
        benchmark_return=benchmark_return,
        per_strategy=per_strategy,
        error_counts=error_counts,
    )


def _result_row(
    ticker="AAA", n_trades=2, wins=1, total_return=0.2, exposure=10, n_bars=50
):
    return {
        "ticker": ticker,
        "n_trades": n_trades,
        "n_bars": n_bars,
        "exposure": exposure,
        "total_return": total_return,
        "wins": wins,
        "trades": [],
    }


def test_print_market_table_with_results(monkeypatch, capsys):
    strategies = {"good": lambda d: [], "empty": lambda d: []}
    monkeypatch.setattr(output_mod, "STRATEGIES", strategies)
    result = _make_result(
        per_strategy={
            "good": [_result_row("AAA"), _result_row("BBB", total_return=0.5)],
            "empty": [],  # exercises the "no results" branch
        },
        error_counts={"good": 0, "empty": 3},
        benchmark_return=0.1,
    )
    output_mod.print_market_table(result)
    out = capsys.readouterr().out
    assert "good" in out
    assert "no results" in out
    assert "Best in this market:" in out
    assert "highest alpha" in out


def test_print_market_table_no_benchmark_and_zero_trades(monkeypatch, capsys):
    strategies = {"s": lambda d: []}
    monkeypatch.setattr(output_mod, "STRATEGIES", strategies)
    # total_trades == 0 → win is nan; benchmark_return None → alpha nan ('-' rendered)
    row = _result_row("AAA", n_trades=0, wins=0, total_return=0.0)
    result = _make_result(
        per_strategy={"s": [row]},
        error_counts={"s": 0},
        benchmark_return=None,
    )
    output_mod.print_market_table(result)
    out = capsys.readouterr().out
    assert "bench=SPY=-" in out
    assert "Best in this market:" in out


def test_write_trades_json(tmp_path, monkeypatch):
    strategies = {"s": lambda d: []}
    monkeypatch.setattr(output_mod, "STRATEGIES", strategies)
    result = _make_result(
        per_strategy={
            "s": [
                _result_row("AAA", n_trades=2, total_return=0.3),
                _result_row("BBB", n_trades=0, total_return=0.9),  # filtered (0 trades)
                _result_row("CCC", n_trades=1, total_return=0.5),
            ]
        },
        error_counts={"s": 0},
    )
    path = tmp_path / "trades.json"
    output_mod.write_trades_json(result, str(path))

    payload = json.loads(path.read_text())
    assert payload["market"] == "us"
    s = payload["strategies"]["s"]
    assert s["n_tickers_traded"] == 2  # BBB excluded
    # sorted by return desc → CCC (0.5) before AAA (0.3)
    assert [t["ticker"] for t in s["tickers"]] == ["CCC", "AAA"]
    assert s["tickers"][0]["return"] == 0.5


# ── cli.py ──────────────────────────────────────────────────────────


def test_cli_help():
    res = CliRunner().invoke(cli_mod.main, ["--help"])
    assert res.exit_code == 0
    assert "--market" in res.output
    assert "--trades-json" in res.output


def test_cli_runs_without_trades_json(monkeypatch):
    sentinel = object()
    calls = {}

    def fake_run_market(*, market, years, limit, refresh):
        calls["run"] = (market, years, limit, refresh)
        return sentinel

    def fake_print(result):
        calls["print"] = result

    def fake_write(result, path):  # pragma: no cover - must not be called here
        raise AssertionError("write_trades_json should not run without --trades-json")

    monkeypatch.setattr(cli_mod, "run_market", fake_run_market)
    monkeypatch.setattr(cli_mod, "print_market_table", fake_print)
    monkeypatch.setattr(cli_mod, "write_trades_json", fake_write)

    res = CliRunner().invoke(
        cli_mod.main, ["--market", "india", "--years", "2", "--limit", "5", "--refresh"]
    )
    assert res.exit_code == 0, res.output
    assert calls["run"] == ("india", 2, 5, True)
    assert calls["print"] is sentinel


def test_cli_runs_with_trades_json(monkeypatch, tmp_path):
    sentinel = object()
    written = {}

    monkeypatch.setattr(cli_mod, "run_market", lambda **k: sentinel)
    monkeypatch.setattr(cli_mod, "print_market_table", lambda r: None)
    monkeypatch.setattr(
        cli_mod, "write_trades_json", lambda r, p: written.update(result=r, path=p)
    )

    path = tmp_path / "out.json"
    res = CliRunner().invoke(cli_mod.main, ["--trades-json", str(path)])
    assert res.exit_code == 0, res.output
    assert written["result"] is sentinel
    assert written["path"] == str(path)


# ── __main__.py ─────────────────────────────────────────────────────


def test_main_module_importable(monkeypatch):
    import importlib
    import sys

    # Patch the cli.main so importing the module is a no-op and offline.
    monkeypatch.setattr(cli_mod, "main", lambda *a, **k: None)
    # Force a fresh execution so the import-time line is measured. Importing it
    # only runs the import statement; the ``if __name__ == "__main__"`` guarded
    # call to main() is excluded from coverage by the config.
    sys.modules.pop("screener.research.pine_runner.__main__", None)
    mod = importlib.import_module("screener.research.pine_runner.__main__")
    assert mod.main is not None
    # Also drive the script path explicitly via runpy for good measure.
    sys.modules.pop("screener.research.pine_runner.__main__", None)
    runpy.run_module("screener.research.pine_runner.__main__", run_name="not_main")
