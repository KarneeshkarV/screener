"""Offline coverage tests for garp / rs_breakout modules and their commands.

Extends the existing ``tests/test_garp.py`` / ``tests/test_rs_breakout.py`` /
``tests/test_seasonality.py`` suites to drive the target modules to full line
coverage. Everything here is deterministic and offline: providers, scanners,
fetchers and HTTP calls are stubbed via monkeypatch.
"""

from __future__ import annotations


from datetime import date


from pathlib import Path


import numpy as np


import pandas as pd


import pytest


from click.testing import CliRunner


from rich.console import Console


from screener import rs_breakout as rs_module


from screener.cli import cli


from screener.commands import rs_breakout as rs_cli


from screener.commands import screen as screen_cli  # noqa: F401  (import for cov)


def _us_passing_row(name="AAA"):
    return {
        "name": name,
        "description": "Alpha",
        "market_cap": 2.0e9,
        "sales": 5.0e9,
        "peg": 1.2,
        "sales_growth_5y": 18.0,
        "operating_profit_growth": 12.0,
        "eps_growth_5y": 16.0,
        "roe_5y": 17.0,
        "roce_or_roic": 18.0,
        "quarterly_profit_growth": 20.0,
    }


def _bars(n=90, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(np.linspace(100.0, 150.0, n), index=idx)
    openp = close.shift(1).fillna(100.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(100_000.0, index=idx),
        }
    )


def _delivery_panel_frame(symbol="AAA", n=30):
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame(
        {
            "SYMBOL": symbol,
            "date": idx,
            "DELIV_PER": np.linspace(40.0, 60.0, n),
        }
    )


def _trend_bars(start=100.0, end=150.0, volume=100_000.0, n=90):
    idx = pd.bdate_range(end="2026-04-30", periods=n)
    close = pd.Series(
        [start + (end - start) * i / (n - 1) for i in range(n)],
        index=idx,
        dtype=float,
    )
    openp = close.shift(1).fillna(start)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(volume, index=idx, dtype=float),
        }
    )


def _result_with_rows():
    bars = _trend_bars(100.0, 150.0)
    bars.iloc[-1, bars.columns.get_loc("volume")] = 200_000.0
    benchmark = _trend_bars(100.0, 110.0)
    panel = pd.DataFrame(
        [
            {"SYMBOL": "AAA", "date": date(2026, 4, 29), "DELIV_PER": 45.0},
            {"SYMBOL": "AAA", "date": date(2026, 4, 30), "DELIV_PER": 55.0},
        ]
    )
    result = rs_module.scan_rs_breakouts(
        {"AAA": bars}, benchmark, date(2026, 4, 30), delivery_panel=panel
    )
    assert result.full, "expected a full-bucket row for rendering coverage"
    return result


def test_join_microstructure_panels_tz_index_and_zero_overlap(monkeypatch) -> None:
    bars = _bars(n=40)
    benchmark = _bars(n=40)
    benchmark["close"] = benchmark["close"] * 0.5
    frame = rs_module.build_signal_frame(bars, benchmark["close"], symbol="NSE:AAA")
    # tz-aware frame index exercises the tz_localize(None) branch (line 561).
    frame.index = pd.DatetimeIndex(frame.index).tz_localize("UTC")
    prepared = {"NSE:AAA": frame}

    # panels have non-NaN data but on dates that do NOT overlap the frame,
    # so after reindex+shift every joined value is NaN -> logger.debug paths.
    far_idx = bars.index - pd.Timedelta(days=5000)
    oc = pd.DataFrame(
        {
            "SYMBOL": "AAA",
            "as_of": far_idx,
            "call_put_oi_ratio": np.linspace(1.0, 2.0, len(bars)),
            "pcr": np.linspace(0.5, 1.5, len(bars)),
        }
    )
    fd_metric = pd.DataFrame(
        {
            "fii_5d_net": np.linspace(1.0, 2.0, len(bars)),
            "dii_5d_net": np.linspace(1.0, 2.0, len(bars)),
            "fii_trend": np.linspace(1.0, 2.0, len(bars)),
        },
        index=far_idx,
    )

    import screener.cache as cache_mod
    import screener.unusual_volume.fii_dii as fii_dii_mod

    def fake_read_frame(path):
        name = str(path)
        if "option_chain" in name:
            return oc
        if "fii_dii" in name:
            return pd.DataFrame({"date": far_idx})
        return pd.DataFrame()

    monkeypatch.setattr(cache_mod, "read_frame", fake_read_frame)
    monkeypatch.setattr(cache_mod, "panel_path", lambda name: Path(f"/tmp/{name}"))
    monkeypatch.setattr(fii_dii_mod, "fii_dii_metric_series", lambda df: fd_metric)

    rs_module._join_microstructure_panels(prepared)
    out = prepared["NSE:AAA"]
    assert out["call_put_oi_ratio"].isna().all()
    assert out["fii_5d_net"].isna().all()


def test_join_microstructure_panels_missing_columns(monkeypatch) -> None:
    bars = _bars(n=40)
    benchmark = _bars(n=40)
    benchmark["close"] = benchmark["close"] * 0.5
    prepared = {
        "NSE:AAA": rs_module.build_signal_frame(
            bars, benchmark["close"], symbol="NSE:AAA"
        ),
    }
    import screener.cache as cache_mod

    # both panels empty -> the "else" NaN-fill branches run
    monkeypatch.setattr(cache_mod, "read_frame", lambda path: pd.DataFrame())
    monkeypatch.setattr(cache_mod, "panel_path", lambda name: Path(f"/tmp/{name}"))

    rs_module._join_microstructure_panels(prepared)
    frame = prepared["NSE:AAA"]
    assert frame["call_put_oi_ratio"].isna().all()
    assert frame["fii_5d_net"].isna().all()


def test_render_result_and_buckets() -> None:
    result = _result_with_rows()
    console = Console(record=True, width=200)
    rs_module.render_result(result, console, limit=5, market="india")
    text = console.export_text()
    assert "RS Breakout Screen" in text


def test_write_markdown_outputs_table(tmp_path) -> None:
    result = _result_with_rows()
    path = tmp_path / "out.md"
    rs_module.write_markdown(result, path, market="india")
    text = path.read_text()
    assert "RS Breakout Screen" in text
    assert "Ticker" in text


def test_fmt_float_handles_none_and_nan() -> None:
    assert rs_module._fmt_float(None) == "-"
    assert rs_module._fmt_float(float("nan")) == "-"
    assert rs_module._fmt_float(1.234) == "1.23"


def test_rs_request_validators_reject_empty() -> None:
    from screener.commands.rs_breakout import RsBreakoutRequest

    with pytest.raises(ValueError):
        RsBreakoutRequest(
            market="  ",
            as_of=date(2026, 4, 30),
            universe=["AAA"],
            benchmark="^NSEI",
            history_days=10,
            require_delivery=False,
        )
    with pytest.raises(ValueError):
        RsBreakoutRequest(
            market="india",
            as_of=date(2026, 4, 30),
            universe=["  ", ""],
            benchmark="^NSEI",
            history_days=10,
            require_delivery=False,
        )


def test_resolve_universe_from_tickers() -> None:
    out = rs_cli.resolve_universe("india", "AAA, BBB ,", None, 10)
    assert out == ["AAA", "BBB"]


def test_resolve_universe_from_file(tmp_path) -> None:
    path = tmp_path / "u.txt"
    path.write_text("AAA\n  \nBBB\n")
    out = rs_cli.resolve_universe("india", None, str(path), 10)
    assert out == ["AAA", "BBB"]


def test_resolve_universe_missing_file_errors() -> None:
    import click

    with pytest.raises(click.UsageError, match="not found"):
        rs_cli.resolve_universe("india", None, "/no/such/file.txt", 10)


def test_resolve_universe_falls_back_to_scan(monkeypatch) -> None:
    monkeypatch.setattr(rs_cli, "load_universe", lambda *a, **k: ["X", "Y"])
    out = rs_cli.resolve_universe("us", None, None, 10)
    assert out == ["X", "Y"]


def test_load_universe_calls_scan(monkeypatch) -> None:
    captured: dict = {}

    def fake_scan(*, market, filters, limit, order_by, cache_ttl, refresh):
        captured["limit"] = limit
        return 2, pd.DataFrame({"name": ["AAA", None, "BBB"]})

    monkeypatch.setattr(rs_cli, "scan", fake_scan)
    # limit 0 -> broad 5000
    out = rs_cli.load_universe("india", 0)
    assert out == ["AAA", "BBB"]
    assert captured["limit"] == 5000


def test_run_rs_breakout_screen_empty_universe_errors(monkeypatch) -> None:
    import click

    monkeypatch.setattr(rs_cli, "resolve_universe", lambda *a, **k: [])
    with pytest.raises(click.UsageError, match="Empty universe"):
        rs_cli.run_rs_breakout_screen(
            "india",
            as_of=date(2026, 4, 30),
            benchmark=None,
            history_days=220,
            cache_ttl=None,
            refresh=False,
            console=Console(),
        )


def test_run_rs_breakout_screen_builds_fetcher(monkeypatch) -> None:
    from tests.conftest import StubPriceFetcher

    bars = _bars()
    bars.iloc[-1, bars.columns.get_loc("volume")] = 200_000.0
    benchmark = _bars()
    benchmark["close"] = benchmark["close"] * 0.5
    fetcher = StubPriceFetcher({"AAA.NS": bars, "^NSEI": benchmark})

    monkeypatch.setattr(rs_cli, "build_price_fetcher", lambda refresh: fetcher)
    monkeypatch.setattr(
        rs_cli,
        "load_india_delivery_for_scan",
        lambda symbols, as_of: pd.DataFrame(),
    )

    result = rs_cli.run_rs_breakout_screen(
        "india",
        as_of=bars.index[-1].date(),
        benchmark=None,
        history_days=220,
        cache_ttl=None,
        refresh=False,
        console=Console(),
        tickers="AAA",
    )
    assert result.as_of == bars.index[-1].date()


def test_run_rs_breakout_scan_delivery_failure(monkeypatch) -> None:
    from screener.commands.rs_breakout import RsBreakoutRequest
    from tests.conftest import StubPriceFetcher

    bars = _bars()
    benchmark = _bars()
    benchmark["close"] = benchmark["close"] * 0.5
    fetcher = StubPriceFetcher({"AAA.NS": bars, "^NSEI": benchmark})

    def boom(universe, as_of):
        raise RuntimeError("delivery down")

    monkeypatch.setattr(rs_cli, "load_india_delivery_for_scan", boom)
    request = RsBreakoutRequest(
        market="india",
        as_of=bars.index[-1].date(),
        universe=["AAA"],
        benchmark="^NSEI",
        history_days=220,
        require_delivery=True,
    )
    console = Console(record=True, width=200)
    result = rs_cli.run_rs_breakout_scan(request, fetcher, console)
    assert "Delivery data load failed" in console.export_text()
    assert result is not None


def test_write_default_outputs(tmp_path) -> None:
    result = _result_with_rows()
    json_path = tmp_path / "x.json"
    md_path = tmp_path / "x.md"
    j, m = rs_cli.write_default_outputs(result, "india", str(json_path), str(md_path))
    assert j == str(json_path)
    assert m == str(md_path)
    assert json_path.exists() and md_path.exists()


def test_rs_breakout_cli_writes_output_files(monkeypatch, tmp_path) -> None:
    from tests.conftest import StubPriceFetcher

    bars = _bars()
    bars.iloc[-1, bars.columns.get_loc("volume")] = 200_000.0
    benchmark = _bars()
    benchmark["close"] = benchmark["close"] * 0.5
    fetcher = StubPriceFetcher({"AAA.NS": bars, "^NSEI": benchmark})

    monkeypatch.setattr(
        rs_cli,
        "load_india_delivery_for_scan",
        lambda symbols, as_of: pd.DataFrame(),
    )

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(
            cli,
            [
                "rs-breakout",
                "--tickers",
                "AAA",
                "--as-of",
                bars.index[-1].date().isoformat(),
            ],
            obj=fetcher,
        )
        assert res.exit_code == 0, res.output
        assert "Wrote" in res.output


def test_screen_command_default_path(monkeypatch) -> None:
    df = pd.DataFrame({"name": ["AAA", "BBB"]})
    monkeypatch.setattr("screener.commands.screen.scan", lambda **k: (2, df))
    monkeypatch.setattr("screener.commands.screen.history.save_run", lambda *a: 1)
    monkeypatch.setattr(
        "screener.commands.screen.history.previous_run", lambda *a, **k: None
    )
    captured: dict = {}

    def fake_print_results(df, total, market, label, *, added, removed, first_run):
        captured["first_run"] = first_run

    monkeypatch.setattr("screener.commands.screen.print_results", fake_print_results)

    res = CliRunner().invoke(cli, ["screen", "-c", "ema", "-m", "us"])
    assert res.exit_code == 0, res.output
    assert captured["first_run"] is True


def test_screen_command_with_previous_run_diff(monkeypatch) -> None:
    df = pd.DataFrame({"name": ["AAA"]})
    prev = pd.DataFrame({"name": ["BBB"]})
    monkeypatch.setattr("screener.commands.screen.scan", lambda **k: (1, df))
    monkeypatch.setattr("screener.commands.screen.history.save_run", lambda *a: 2)
    monkeypatch.setattr(
        "screener.commands.screen.history.previous_run", lambda *a, **k: prev
    )
    monkeypatch.setattr(
        "screener.commands.screen.history.diff",
        lambda cur, prv: (["AAA"], ["BBB"]),
    )
    captured: dict = {}

    def fake_print_results(df, total, market, label, *, added, removed, first_run):
        captured["added"] = added
        captured["first_run"] = first_run

    monkeypatch.setattr("screener.commands.screen.print_results", fake_print_results)

    res = CliRunner().invoke(cli, ["screen", "-c", "ema"])
    assert res.exit_code == 0, res.output
    assert captured["added"] == ["AAA"]
    assert captured["first_run"] is False


def test_screen_command_csv_output(monkeypatch) -> None:
    df = pd.DataFrame({"name": ["AAA"]})
    monkeypatch.setattr("screener.commands.screen.scan", lambda **k: (1, df))
    captured: dict = {}
    monkeypatch.setattr(
        "screener.commands.screen.print_csv",
        lambda d: captured.setdefault("csv", True),
    )

    res = CliRunner().invoke(cli, ["screen", "-c", "ema", "--csv"])
    assert res.exit_code == 0, res.output
    assert captured["csv"] is True


def test_screen_command_pipeline_dispatch(monkeypatch) -> None:
    captured: dict = {}

    def fake_runner(*, market, limit, output_csv, refresh, cache_ttl):
        captured["market"] = market

    monkeypatch.setattr(
        "screener.screen_aliases.SCREEN_ALIASES",
        {"rs-breakout": fake_runner},
    )

    res = CliRunner().invoke(cli, ["screen", "-c", "rs-breakout", "-m", "india"])
    assert res.exit_code == 0, res.output
    assert captured["market"] == "india"


def test_screen_command_pipeline_combined_rejected() -> None:
    res = CliRunner().invoke(cli, ["screen", "-c", "rs-breakout", "-c", "ema"])
    assert res.exit_code != 0
    assert "cannot be combined" in res.output


def test_garp_cli_no_universe(monkeypatch) -> None:
    monkeypatch.setattr("screener.commands.garp.run_garp_screen", lambda *a, **k: None)
    res = CliRunner().invoke(cli, ["garp", "-m", "india"])
    assert res.exit_code == 0, res.output
    assert "No tickers returned" in res.output


def test_garp_cli_table_output(monkeypatch) -> None:
    from screener.garp import add_garp_score

    results = add_garp_score(pd.DataFrame([_us_passing_row()]))
    monkeypatch.setattr(
        "screener.commands.garp.run_garp_screen", lambda *a, **k: results
    )
    captured: dict = {}
    monkeypatch.setattr(
        "screener.commands.garp.print_garp_results",
        lambda results, market: captured.setdefault("market", market),
    )
    res = CliRunner().invoke(cli, ["garp", "-m", "us"])
    assert res.exit_code == 0, res.output
    assert captured["market"] == "us"


def test_seasonality_cli_rejects_bad_years() -> None:
    res = CliRunner().invoke(cli, ["seasonality", "AAA", "--years", "0"])
    assert res.exit_code != 0
    assert "--years must be >= 1" in res.output


def test_seasonality_cli_value_error(monkeypatch) -> None:
    from tests.conftest import StubPriceFetcher

    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=600)
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        },
        index=idx,
    )
    fetcher = StubPriceFetcher({"AAA": bars})

    import screener.commands.seasonality as seas_mod

    def boom(bars, ticker):
        raise ValueError("bad seasonality")

    monkeypatch.setattr(seas_mod, "compute_seasonality", boom)
    res = CliRunner().invoke(cli, ["seasonality", "AAA", "--years", "2"], obj=fetcher)
    assert res.exit_code != 0
    assert "bad seasonality" in res.output
