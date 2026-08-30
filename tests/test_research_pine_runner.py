from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.research.pine_runner import cli, data, output, run
from screener.strategies.trades import ResearchTrade

_AS_OF = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def test_fetch_ohlcv_normalizes_index_and_adj_close(monkeypatch):
    frame = pd.DataFrame(
        {"close": [10.0, 11.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    class Fetcher:
        def fetch(self, symbols, start, end):
            assert symbols == ["AAA"]
            assert start == date(2024, 1, 1)
            assert end == date(2024, 1, 2)
            return {"AAA": frame}

    monkeypatch.setattr(data, "_FETCHER", Fetcher())
    monkeypatch.setattr(data, "tv_to_yf", lambda ticker, market: ticker)

    out = data.fetch_ohlcv("AAA", date(2024, 1, 1), date(2024, 1, 2), "us")

    assert out is not None
    assert out.columns.tolist() == ["date", "close", "adj_close"]
    assert out["adj_close"].tolist() == [10.0, 11.0]


def test_fetch_ohlcv_refresh_uses_new_fetcher_and_handles_empty(monkeypatch):
    class Fetcher:
        def fetch(self, symbols, start, end):
            return {"AAA.NS": pd.DataFrame()}

    monkeypatch.setattr(data, "build_price_fetcher", lambda refresh=False: Fetcher())
    monkeypatch.setattr(data, "tv_to_yf", lambda ticker, market: f"{ticker}.NS")

    assert (
        data.fetch_ohlcv("AAA", date(2024, 1, 1), date(2024, 1, 2), "india", True)
        is None
    )


def test_load_universe_reads_tradingview_names(monkeypatch):
    captured = {}

    def fake_scan(*, market, filters, limit, order_by):
        captured.update(
            {"market": market, "filters": filters, "limit": limit, "order_by": order_by}
        )
        return 2, pd.DataFrame({"name": ["AAA", None, "BBB"]}), _AS_OF

    monkeypatch.setattr(data, "_tv_scan", fake_scan)

    assert data.load_universe("india") == ["AAA", "BBB"]
    assert captured["market"] == "india"
    assert captured["limit"] == 500
    assert captured["order_by"] == "volume"


def test_load_universe_named_universe_skips_tv_scan(monkeypatch):
    class FakeUniverse:
        symbols = ("RELIANCE", "TCS")

    monkeypatch.setattr(
        data,
        "load_current_universe",
        lambda name: (
            FakeUniverse()
            if name == "nifty500"
            else (_ for _ in ()).throw(AssertionError)
        ),
    )

    assert data.load_universe("india", universe="nifty500") == ["RELIANCE", "TCS"]


def test_compound_and_run_ticker_filter_window():
    bars = _bars(60)
    trades = [
        _trade(1, 2, 10, 12, "2020-01-02"),
        _trade(55, 56, 20, 18, "2024-02-25"),
    ]

    assert run._compound(trades) == pytest.approx((1.2 * 0.9) - 1)
    out = run._run_ticker(
        bars,
        pd.Timestamp("2024-02-01"),
        lambda df: trades,
    )

    assert out is not None
    assert out["n_trades"] == 1
    assert out["wins"] == 0
    assert out["total_return"] == pytest.approx(-0.1)
    assert out["exposure"] == 1
    assert out["n_bars"] == 29
    assert run._run_ticker(_bars(20), pd.Timestamp("2024-01-01"), lambda df: []) is None


def test_market_run_validation():
    with pytest.raises(ValueError, match="value must not be empty"):
        run.MarketRun(
            market=" ",
            today=date(2024, 1, 1),
            window_start=pd.Timestamp("2023-01-01"),
            benchmark_symbol="SPY",
            benchmark_return=None,
            per_strategy={},
            error_counts={},
        )


def test_run_market_aggregates_successes_errors_and_benchmark(monkeypatch):
    strategies = {
        "ok": lambda df: [_trade(55, 56, 10, 12, "2024-02-25")],
        "bad": lambda df: (_ for _ in ()).throw(ValueError("boom")),
    }
    monkeypatch.setattr(run, "STRATEGIES", strategies)
    monkeypatch.setattr(
        run, "load_universe", lambda market, universe=None: ["AAA", "EMPTY", "ERR"]
    )

    def fake_fetch(ticker, start, end, market, refresh=False):
        if ticker == "EMPTY":
            return pd.DataFrame()
        if ticker == "ERR":
            return None
        if ticker == "SPY":
            return pd.DataFrame(
                {
                    "date": pd.date_range(
                        date.today() - pd.Timedelta(days=10), periods=3
                    ),
                    "adj_close": [100.0, 105.0, 110.0],
                    "close": [100.0, 105.0, 110.0],
                }
            )
        return _bars(60)

    monkeypatch.setattr(run, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(run, "BENCHMARKS", {"us": "SPY"})

    result = run.run_market(market="us", years=1, limit=2, refresh=True)

    assert result.market == "us"
    assert result.benchmark_symbol == "SPY"
    assert result.benchmark_return == pytest.approx(0.1)
    assert [row["ticker"] for row in result.per_strategy["ok"]] == ["AAA"]
    assert result.error_counts["bad"] == 1


def test_run_market_handles_missing_benchmark(monkeypatch):
    monkeypatch.setattr(run, "STRATEGIES", {"ok": lambda df: []})
    monkeypatch.setattr(run, "load_universe", lambda market, universe=None: ["SHORT"])

    def fake_fetch(ticker, *args, **kwargs):
        if ticker == "SHORT":
            return _bars(20)
        return None

    monkeypatch.setattr(run, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(run, "BENCHMARKS", {"india": "^NSEI"})

    result = run.run_market(market="india", years=1, limit=0, refresh=False)

    assert result.benchmark_return is None
    assert result.per_strategy == {"ok": []}


def test_print_market_table_and_write_trades_json(tmp_path, monkeypatch, capsys):
    result = run.MarketRun(
        market="us",
        today=date(2024, 3, 1),
        window_start=pd.Timestamp("2023-03-01"),
        benchmark_symbol="SPY",
        benchmark_return=0.10,
        per_strategy={
            "ok": [
                {
                    "ticker": "AAA",
                    "total_return": 0.25,
                    "n_trades": 2,
                    "wins": 1,
                    "exposure": 10,
                    "n_bars": 100,
                },
                {
                    "ticker": "BBB",
                    "total_return": -0.05,
                    "n_trades": 0,
                    "wins": 0,
                    "exposure": 5,
                    "n_bars": 100,
                },
            ],
            "empty": [],
        },
        error_counts={"ok": 0, "empty": 2},
    )
    path = tmp_path / "trades.json"

    output.print_market_table(result)
    output.write_trades_json(result, str(path))

    out = capsys.readouterr().out
    assert "US  |  window 2023-03-01 -> 2024-03-01" in out
    assert "Best in this market" in out
    assert "empty               no results" in out
    payload = json.loads(path.read_text())
    assert payload["strategies"]["ok"]["tickers"] == [
        {"ticker": "AAA", "n_trades": 2, "wins": 1, "return": 0.25}
    ]


def test_print_market_table_handles_missing_benchmark(monkeypatch, capsys):
    result = run.MarketRun(
        market="india",
        today=date(2024, 3, 1),
        window_start=pd.Timestamp("2023-03-01"),
        benchmark_symbol="^NSEI",
        benchmark_return=None,
        per_strategy={
            "ok": [
                {
                    "ticker": "AAA",
                    "total_return": 0.25,
                    "n_trades": 0,
                    "wins": 0,
                    "exposure": 0,
                    "n_bars": 0,
                }
            ]
        },
        error_counts={"ok": 0},
    )

    output.print_market_table(result)

    assert "bench=^NSEI=-" in capsys.readouterr().out


def test_cli_main_runs_market_and_writes_json(monkeypatch, tmp_path):
    result = run.MarketRun(
        market="us",
        today=date(2024, 3, 1),
        window_start=pd.Timestamp("2023-03-01"),
        benchmark_symbol="SPY",
        benchmark_return=None,
        per_strategy={},
        error_counts={},
    )
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "run_market", lambda **kwargs: result)
    monkeypatch.setattr(
        cli, "print_market_table", lambda value: calls.append(("print", value))
    )
    monkeypatch.setattr(
        cli,
        "write_trades_json",
        lambda value, path: calls.append(("json", (value, path))),
    )

    path = tmp_path / "out.json"
    res = CliRunner().invoke(
        cli.main,
        [
            "--market",
            "us",
            "--years",
            "2",
            "--limit",
            "5",
            "--refresh",
            "--trades-json",
            str(path),
        ],
    )

    assert res.exit_code == 0, res.output
    assert calls == [("print", result), ("json", (result, str(path)))]


def _bars(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "close": [float(i + 1) for i in range(n)],
            "adj_close": [float(i + 1) for i in range(n)],
        }
    )


def _trade(
    entry_idx: int, exit_idx: int, entry_px: float, exit_px: float, entry_date: str
) -> ResearchTrade:
    return ResearchTrade(
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_px=entry_px,
        exit_px=exit_px,
        entry_date=pd.Timestamp(entry_date),
        exit_date=pd.Timestamp(entry_date) + pd.Timedelta(days=1),
        return_pct=exit_px / entry_px - 1.0 if entry_px > 0 else 0.0,
    )
