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


from screener import garp as garp_module


from screener import rs_breakout as rs_module


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


def test_num_handles_none_garbage_and_nan() -> None:
    assert garp_module.to_number(None) is None
    assert garp_module.to_number("1,234.5%") == pytest.approx(1234.5)
    assert garp_module.to_number("not-a-number") is None
    assert garp_module.to_number(float("nan")) is None


def test_first_num_returns_none_when_no_key_matches() -> None:
    assert garp_module.first_number({"a": "x"}, "missing", "alsomissing") is None
    assert garp_module.first_number({"PEG": "1.5"}, "peg") == pytest.approx(1.5)


def test_pct_change_guards_and_cagr_guards() -> None:
    assert garp_module.pct_change(None, 1.0) is None
    assert garp_module.pct_change(1.0, 0.0) is None
    assert garp_module.pct_change(120.0, 100.0) == pytest.approx(20.0)
    # cagr guards: non-positive / zero years
    assert garp_module._cagr(None, 1.0, 4) is None
    assert garp_module._cagr(1.0, -1.0, 4) is None
    assert garp_module._cagr(2.0, 1.0, 0) is None
    assert garp_module._cagr(2.0, 1.0, 1) == pytest.approx(100.0)


def test_series_from_statement_empty_and_missing_rows() -> None:
    assert garp_module._series_from_statement(None, ["x"]).empty
    assert garp_module._series_from_statement(pd.DataFrame(), ["x"]).empty
    df = pd.DataFrame({"c": [1.0]}, index=["Total Revenue"]).T
    # row not present -> empty series
    assert garp_module._series_from_statement(df, ["Nope"]).empty


def test_average_ratio_empty_and_no_valid_pairs() -> None:
    assert (
        garp_module._average_ratio(pd.Series(dtype=float), pd.Series([1.0]), 3) is None
    )
    num = pd.Series({"a": 10.0})
    den = pd.Series({"a": 0.0})  # zero denominator -> skipped
    assert garp_module._average_ratio(num, den, 3) is None


def test_add_garp_score_empty_frame() -> None:
    out = garp_module.add_garp_score(pd.DataFrame())
    assert "garp_score" in out.columns
    assert out.empty


@pytest.mark.parametrize("market", ["india", "us"])
def test_load_garp_universe_both_markets(monkeypatch, market) -> None:
    captured: dict = {}

    def fake_scan(*, market, filters, limit, order_by, cache_ttl, refresh):
        captured["market"] = market
        captured["filters"] = filters
        return 1, pd.DataFrame({"name": ["AAA"]})

    monkeypatch.setattr(garp_module, "scan", fake_scan)
    df = garp_module.load_garp_universe(market, 10, cache_ttl=None, refresh=False)
    assert list(df["name"]) == ["AAA"]
    assert captured["market"] == market
    assert len(captured["filters"]) == 3


def test_fetch_india_sections_uses_openscreener(monkeypatch) -> None:
    class FakeStock:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def fetch(self, section: str):
            return {"section": section} if section == "ratios" else None

    import sys
    import types

    fake_mod = types.ModuleType("openscreener")
    fake_mod.Stock = FakeStock
    monkeypatch.setitem(sys.modules, "openscreener", fake_mod)

    out = garp_module._fetch_india_sections("AAA")
    assert out["ratios"] == {"section": "ratios"}
    # None payloads coerced to {}
    assert out["profit_loss"] == {}
    assert out["quarterly_results"] == {}


def test_india_row_maps_metrics() -> None:
    payload = {
        "ratios": {
            "market_capitalization": "1500",
            "sales": "1600",
            "peg_ratio": "1.2",
            "sales_growth_5years": "18",
            "operating_profit_growth": "12",
            "eps_growth_5years": "16",
            "average_return_on_equity_5years": "17",
            "average_return_on_capital_employed_3years": "18",
            "expected_quarterly_net_profit": "120",
        },
        "profit_loss": {},
        "quarterly_results": {"net_profit_3quarters_back": "100"},
    }
    row = garp_module._india_row("AAA", "Alpha", payload)
    assert row["name"] == "AAA"
    assert row["market_cap"] == pytest.approx(1500.0)
    assert row["quarterly_profit_growth"] == pytest.approx(20.0)


def test_india_row_non_dict_sections_default_empty() -> None:
    # payload sections that aren't dicts get coerced to {}
    row = garp_module._india_row(
        "AAA", None, {"ratios": "x", "profit_loss": 5, "quarterly_results": None}
    )
    assert row["description"] == ""
    assert row["market_cap"] is None


def test_screen_india_garp_offline(monkeypatch) -> None:
    universe = pd.DataFrame(
        {"name": ["AAA", "", "BBB"], "description": ["Alpha", "", "Beta"]}
    )

    def fake_cached_json_call(*args, **kwargs):
        # Run the fetch lambda so its body counts; ignore result.
        kwargs["fetch"]()
        return {
            "ratios": {
                "market_capitalization": 1500.0,
                "sales": 1600.0,
                "peg_ratio": 1.2,
                "sales_growth_5years": 18.0,
                "operating_profit_growth": 12.0,
                "eps_growth_5years": 16.0,
                "average_return_on_equity_5years": 17.0,
                "average_return_on_capital_employed_3years": 18.0,
                "expected_quarterly_net_profit": 120.0,
            },
            "profit_loss": {},
            "quarterly_results": {"net_profit_3quarters_back": 100.0},
        }

    monkeypatch.setattr(garp_module, "cached_json_call", fake_cached_json_call)
    monkeypatch.setattr(garp_module, "_fetch_india_sections", lambda symbol: {})

    out = garp_module.screen_india_garp(
        universe, limit=10, workers=2, cache_ttl=None, refresh=False
    )
    assert set(out["name"]) == {"AAA", "BBB"}


def test_screen_india_garp_swallows_fetch_errors(monkeypatch) -> None:
    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(garp_module, "cached_json_call", boom)
    out = garp_module.screen_india_garp(
        universe, limit=10, workers=1, cache_ttl=None, refresh=False
    )
    assert out.empty


def test_us_row_yfinance_with_balance_failure(monkeypatch) -> None:
    dates = pd.to_datetime(
        ["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"]
    )
    income = pd.DataFrame(
        [
            [5.0e9, 4.5e9, 4.0e9, 3.5e9, 2.5e9],
            [1.2e9, 1.0e9, 0.9e9, 0.8e9, 0.7e9],
            [8.0e8, 7.0e8, 6.0e8, 5.0e8, 4.0e8],
            [4.0e9, 3.5e9, 3.0e9, 2.5e9, 2.0e9],
        ],
        index=[
            "Total Revenue",
            "Operating Income",
            "Net Income",
            "Stockholders Equity",
        ],
        columns=dates,
    )

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.info = {}
            self.income_stmt = income
            self.earnings_estimate = pd.DataFrame()  # empty -> no quarterly eps

        @property
        def balance_sheet(self):
            raise RuntimeError("no balance sheet")

    import sys
    import types

    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    row = garp_module._us_row("AAA", None)
    # description falls back to "" because info has no shortName
    assert row["description"] == ""
    assert row["expected_quarterly_profit"] is None
    assert row["sales"] == pytest.approx(5.0e9)


def test_fmp_api_key_resolves_via_fmp() -> None:
    import screener.fmp as fmp_mod

    assert garp_module.resolve_api_key is fmp_mod.resolve_api_key


def test_fmp_get_parses_json(monkeypatch) -> None:
    import json

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps([{"x": 1}]).encode("utf-8")

    import screener.fmp

    monkeypatch.setattr(
        screener.fmp.urllib.request, "urlopen", lambda req, timeout=20: FakeResp()
    )
    out = garp_module._fmp_get("profile/AAA", {"limit": 1}, "key")
    assert out == [{"x": 1}]


def test_fetch_fmp_us_sections_invokes_get(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(path, params, api_key):
        calls.append(path)
        return [{"path": path}]

    monkeypatch.setattr(garp_module, "_fmp_get", fake_get)
    out = garp_module._fetch_fmp_us_sections("AAA", "key")
    assert set(out.keys()) == {
        "profile",
        "ratios_ttm",
        "income_annual",
        "balance_annual",
        "income_quarterly",
        "estimates_quarterly",
    }
    assert len(calls) == 6


def test_fmp_list_filters_non_dict() -> None:
    assert garp_module._fmp_list({"k": "x"}, "k") == []
    assert garp_module._fmp_list({"k": [1, {"a": 1}, "z"]}, "k") == [{"a": 1}]


def test_fmp_series_skips_dupes_and_bad_values() -> None:
    series = garp_module._fmp_series(
        [
            {"date": "2025-12-31", "v": "10"},
            {"date": "2025-12-31", "v": "20"},  # dupe date -> skipped
            {"date": None, "v": "30"},  # missing date -> skipped
            {"date": "2024-12-31", "v": None},  # bad value -> skipped
            {"date": "2023-12-31", "v": "5"},
        ],
        "v",
    )
    assert list(series.index) == ["2025-12-31", "2023-12-31"]


def test_fmp_quarterly_eps_no_quarterly_income() -> None:
    expected, year_ago = garp_module._fmp_quarterly_eps([], [])
    assert expected is None and year_ago is None


def test_fmp_quarterly_eps_no_year_ago_match() -> None:
    estimates = [{"date": "2026-06-30", "estimatedEpsAvg": 1.4}]
    # quarterly income exists but no entry within +/-60 days of (estimate - 1yr)
    quarterly = [{"date": "2025-01-01", "eps": 0.9}]
    expected, year_ago = garp_module._fmp_quarterly_eps(estimates, quarterly)
    assert expected == pytest.approx(1.4)
    assert year_ago is None


def test_fmp_quarterly_eps_skips_entry_with_none_eps() -> None:
    estimates = [{"date": "2026-06-30", "estimatedEpsAvg": 1.4}]
    # one in-window entry has eps=None (skip via continue), the other matches
    quarterly = [
        {"date": "2025-12-31", "eps": None},
        {"date": "2025-06-30", "eps": 0.9},
    ]
    expected, year_ago = garp_module._fmp_quarterly_eps(estimates, quarterly)
    assert expected == pytest.approx(1.4)
    assert year_ago == pytest.approx(0.9)


def test_fmp_us_row_non_dict_payload() -> None:
    assert garp_module._fmp_us_row("AAA", "Alpha", ["not", "a", "dict"]) is None


def test_fetch_fmp_us_cached_uses_provider(monkeypatch) -> None:
    captured: dict = {}

    class FakeProvider:
        def fetch(self, key, fetch, *, refresh, fallback, ttl_seconds, operation):
            captured["key"] = key
            captured["operation"] = operation
            return fetch()

    monkeypatch.setattr(garp_module, "_FMP_US_PROVIDER", FakeProvider())
    monkeypatch.setattr(
        garp_module, "_fetch_fmp_us_sections", lambda symbol, api_key: {"ok": symbol}
    )
    out = garp_module._fetch_fmp_us_cached("AAA", "key", cache_ttl=None, refresh=False)
    # the fetch lambda calls _fetch_fmp_us_sections
    assert captured["key"] == ("us", "AAA")
    assert captured["operation"] == "garp fundamentals AAA"
    assert out is not None


def test_screen_us_garp_swallows_resolve_errors(monkeypatch) -> None:
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: None)

    def boom(symbol, description):
        raise RuntimeError("boom")

    monkeypatch.setattr(garp_module, "_us_row", boom)
    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})
    out = garp_module.screen_us_garp(
        universe, limit=10, workers=1, cache_ttl=None, refresh=False
    )
    assert out.empty


def test_run_garp_screen_us_branch(monkeypatch) -> None:
    universe = pd.DataFrame({"name": ["AAA"], "description": ["Alpha"]})
    monkeypatch.setattr(garp_module, "load_garp_universe", lambda *a, **k: universe)
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: None)
    monkeypatch.setattr(
        garp_module, "_us_row", lambda symbol, description: _us_passing_row(symbol)
    )

    out = garp_module.run_garp_screen(
        "us", 50, limit=10, workers=1, cache_ttl=None, refresh=False
    )
    assert out is not None
    assert list(out["name"]) == ["AAA"]


def test_row_validators_and_to_dict() -> None:
    row = rs_module.RsBreakoutRow(
        symbol=" AAA ",
        date=date(2026, 4, 30),
        close=100.0,
        rs_55=1.0,
        supertrend=90.0,
        previous_week_high=99.0,
        volume=100.0,
        avg_volume_20d=80.0,
        volume_ratio=1.25,
        delivery_pct=50.0,
        previous_delivery_pct=40.0,
    )
    assert row.symbol == "AAA"
    d = row.to_dict()
    assert d["symbol"] == "AAA"
    assert d["date"] == "2026-04-30"

    with pytest.raises(ValueError, match="symbol must not be empty"):
        rs_module.RsBreakoutRow(
            symbol="   ",
            date=date(2026, 4, 30),
            close=1.0,
            rs_55=1.0,
            supertrend=1.0,
            previous_week_high=None,
            volume=1.0,
            avg_volume_20d=1.0,
            volume_ratio=1.0,
            delivery_pct=None,
            previous_delivery_pct=None,
        )


def test_result_benchmark_validator_rejects_blank() -> None:
    with pytest.raises(ValueError, match="benchmark must not be empty"):
        rs_module.RsBreakoutResult(
            as_of=date(2026, 4, 30), benchmark="  ", full=[], relaxed=[]
        )


def test_evaluate_symbol_nan_rs_at_last_bar() -> None:
    # benchmark misaligned with stock dates so the last bar has NaN rs.
    bars = _bars()
    benchmark = bars["close"].copy()
    benchmark.index = benchmark.index - pd.Timedelta(days=400)
    out = rs_module.evaluate_symbol("AAA", bars, benchmark, bars.index[-1].date())
    assert out is None


def test_normalize_bars_empty_and_no_date_column() -> None:
    assert rs_module.normalize_bars(pd.DataFrame(), date(2026, 1, 1)).empty
    # non-datetime index, no "date" column -> empty
    df = pd.DataFrame({"close": [1.0]})
    assert rs_module.normalize_bars(df, date(2026, 1, 1)).empty


def test_normalize_bars_from_date_column_and_missing_cols() -> None:
    df = pd.DataFrame({"date": ["2026-01-01"], "close": [1.0]})
    # has date column but missing OHLCV columns -> empty
    assert rs_module.normalize_bars(df, date(2026, 1, 2)).empty
    full = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1.0, 2.0],
        }
    )
    out = rs_module.normalize_bars(full, date(2026, 1, 2))
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 2


def test_supertrend_empty_returns_empty() -> None:
    assert rs_module.supertrend(pd.DataFrame()).empty


def test_previous_completed_week_high_empty_and_no_week() -> None:
    assert (
        rs_module.previous_completed_week_high(pd.DataFrame(), date(2026, 1, 1)) is None
    )
    # bars only in current week -> no previous week data -> None
    idx = pd.bdate_range("2026-04-27", periods=3)  # Mon-Wed
    bars = pd.DataFrame({"high": [1.0, 2.0, 3.0]}, index=idx)
    assert rs_module.previous_completed_week_high(bars, date(2026, 4, 29)) is None


def test_delivery_lookup_empty_and_single_value() -> None:
    assert rs_module.delivery_lookup(pd.DataFrame()) == {}
    panel = pd.DataFrame(
        [
            {"SYMBOL": "aaa", "date": date(2026, 1, 1), "DELIV_PER": 50.0},
            {"SYMBOL": "bbb", "date": date(2026, 1, 1), "DELIV_PER": float("nan")},
        ]
    )
    out = rs_module.delivery_lookup(panel)
    # bbb all-NaN -> dropped; aaa has single value -> prev is None
    assert out == {"AAA": (50.0, None)}


def test_evaluate_symbol_too_short_history() -> None:
    short = _bars(n=10)
    assert (
        rs_module.evaluate_symbol("AAA", short, short["close"], date(2026, 4, 30))
        is None
    )


def test_evaluate_symbol_base_fail_returns_none() -> None:
    # strong benchmark, flat stock -> rs negative -> base fail
    bars = _bars()
    benchmark = _bars()
    benchmark["close"] = benchmark["close"] * 100.0
    out = rs_module.evaluate_symbol(
        "AAA", bars, benchmark["close"], bars.index[-1].date()
    )
    assert out is None


def test_evaluate_symbol_zero_avg_volume_returns_none() -> None:
    bars = _bars()
    bars["volume"] = 0.0
    benchmark = _bars()
    benchmark["close"] = benchmark["close"] * 0.5
    out = rs_module.evaluate_symbol(
        "AAA", bars, benchmark["close"], bars.index[-1].date()
    )
    assert out is None


def test_scan_rs_breakouts_empty_benchmark_raises() -> None:
    with pytest.raises(ValueError, match="Benchmark OHLCV data is empty"):
        rs_module.scan_rs_breakouts({}, pd.DataFrame(), date(2026, 4, 30))


def test_india_symbol_variants() -> None:
    assert rs_module.india_symbol("nse:reliance") == "RELIANCE"
    assert rs_module.india_symbol("RELIANCE.NS") == "RELIANCE"
    assert rs_module.india_symbol("RELIANCE.BO") == "RELIANCE"


def test_required_history_bars() -> None:
    assert rs_module.required_history_bars() == 56


def test_fetch_price_data_handles_fetch_exception(monkeypatch) -> None:
    bars = _bars()

    class FlakyFetcher:
        def fetch(self, tickers, start, end):
            if tickers == ["^NSEI"]:
                return {"^NSEI": bars}
            raise ValueError("boom")

    bars_by_symbol, benchmark = rs_module.fetch_price_data(
        ["AAA"], "india", date(2026, 4, 30), FlakyFetcher(), max_workers=1
    )
    assert bars_by_symbol["AAA"].empty
    assert not benchmark.empty


def test_load_india_delivery_for_scan(monkeypatch) -> None:
    captured: dict = {}

    def fake_panel(symbols, as_of, history_days):
        captured["symbols"] = symbols
        captured["history_days"] = history_days
        return pd.DataFrame({"SYMBOL": symbols})

    monkeypatch.setattr(rs_module, "load_delivery_panel", fake_panel)
    out = rs_module.load_india_delivery_for_scan(
        ["NSE:AAA", "BBB.NS"], date(2026, 4, 30)
    )
    assert captured["symbols"] == ["AAA", "BBB"]
    assert captured["history_days"] == 14
    assert not out.empty


def test_previous_completed_week_high_series_empty_and_values() -> None:
    assert rs_module.previous_completed_week_high_series(pd.DataFrame()).empty
    bars = _bars(n=30)
    series = rs_module.previous_completed_week_high_series(bars)
    assert len(series) == len(bars)
    assert series.notna().any()


def test_delivery_series_for_symbol_empty_panel_and_match() -> None:
    bars = _bars(n=30)
    idx = pd.DatetimeIndex(bars.index)
    empty = rs_module._delivery_series_for_symbol(None, "AAA", idx)
    assert empty["delivery_pct"].isna().all()

    # no panel rows match the symbol -> empty path
    panel = _delivery_panel_frame("ZZZ")
    none_match = rs_module._delivery_series_for_symbol(panel, "AAA", idx)
    assert none_match["delivery_pct"].isna().all()

    panel = _delivery_panel_frame("AAA")
    matched = rs_module._delivery_series_for_symbol(panel, "AAA", idx)
    assert matched["delivery_pct"].notna().any()


def test_build_signal_frame_empty_and_full() -> None:
    assert rs_module.build_signal_frame(None, pd.Series(dtype=float)).empty
    bars = _bars()
    benchmark = _bars()
    benchmark["close"] = benchmark["close"] * 0.5
    out = rs_module.build_signal_frame(
        bars,
        benchmark["close"],
        delivery_panel=_delivery_panel_frame("AAA", n=len(bars)),
        symbol="AAA",
        require_delivery=True,
    )
    assert "rs_breakout_entry" in out.columns
    assert "delivery_spike" in out.columns


def test_prepare_backtest_frames_no_benchmark_passthrough() -> None:
    bars = _bars(n=30)
    out = rs_module.prepare_backtest_frames({"AAA": bars}, pd.DataFrame(), market="us")
    # benchmark empty -> raw copy returned
    assert out["AAA"].equals(bars)


def test_prepare_backtest_frames_us_branch() -> None:
    bars = _bars()
    benchmark = _bars()
    benchmark["close"] = benchmark["close"] * 0.5
    out = rs_module.prepare_backtest_frames({"AAA": bars}, benchmark, market="us")
    assert "rs_breakout_entry" in out["AAA"].columns


def test_prepare_backtest_frames_india_joins_micro(monkeypatch) -> None:
    bars = _bars()
    benchmark = _bars()
    benchmark["close"] = benchmark["close"] * 0.5

    calls: list = []

    def fake_join(prepared):
        calls.append(set(prepared))

    monkeypatch.setattr(rs_module, "_join_microstructure_panels", fake_join)
    out = rs_module.prepare_backtest_frames(
        {"NSE:AAA": bars}, benchmark, market="india"
    )
    assert calls == [{"NSE:AAA"}]
    assert "rs_breakout_entry" in out["NSE:AAA"].columns


def test_join_microstructure_panels_with_panels(monkeypatch) -> None:
    bars = _bars(n=40)
    benchmark = _bars(n=40)
    benchmark["close"] = benchmark["close"] * 0.5
    prepared = {
        "NSE:AAA": rs_module.build_signal_frame(
            bars, benchmark["close"], symbol="NSE:AAA"
        ),
        "NSE:EMPTY": pd.DataFrame(),  # exercises empty-frame skip
    }

    oc = pd.DataFrame(
        {
            "SYMBOL": "AAA",
            "as_of": bars.index,
            "call_put_oi_ratio": np.linspace(1.0, 2.0, len(bars)),
            "pcr": np.linspace(0.5, 1.5, len(bars)),
        }
    )
    fd_raw = pd.DataFrame({"date": bars.index})

    fd_metric = pd.DataFrame(
        {
            "fii_5d_net": np.linspace(1.0, 2.0, len(bars)),
            "dii_5d_net": np.linspace(1.0, 2.0, len(bars)),
            "fii_trend": np.linspace(1.0, 2.0, len(bars)),
        },
        index=bars.index,
    )

    import screener.cache as cache_mod

    def fake_read_frame(path):
        name = str(path)
        if "option_chain" in name:
            return oc
        if "fii_dii" in name:
            return fd_raw
        return pd.DataFrame()

    monkeypatch.setattr(cache_mod, "read_frame", fake_read_frame)
    monkeypatch.setattr(cache_mod, "panel_path", lambda name: Path(f"/tmp/{name}"))

    import screener.unusual_volume.fii_dii as fii_dii_mod

    monkeypatch.setattr(fii_dii_mod, "fii_dii_metric_series", lambda df: fd_metric)

    rs_module._join_microstructure_panels(prepared)
    frame = prepared["NSE:AAA"]
    for col in (
        "call_put_oi_ratio",
        "pcr",
        "fii_5d_net",
        "dii_5d_net",
        "fii_trend",
    ):
        assert col in frame.columns
