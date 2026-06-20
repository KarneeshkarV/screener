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

from screener import garp as garp_module
from screener import rs_breakout as rs_module
from screener.cli import cli
from screener.commands import rs_breakout as rs_cli
from screener.commands import screen as screen_cli  # noqa: F401  (import for cov)


# ── garp.py numeric helpers ─────────────────────────────────────────────────


def test_num_handles_none_garbage_and_nan() -> None:
    assert garp_module._num(None) is None
    assert garp_module._num("1,234.5%") == pytest.approx(1234.5)
    assert garp_module._num("not-a-number") is None
    assert garp_module._num(float("nan")) is None


def test_first_num_returns_none_when_no_key_matches() -> None:
    assert garp_module._first_num({"a": "x"}, "missing", "alsomissing") is None
    assert garp_module._first_num({"PEG": "1.5"}, "peg") == pytest.approx(1.5)


def test_pct_change_guards_and_cagr_guards() -> None:
    assert garp_module._pct_change(None, 1.0) is None
    assert garp_module._pct_change(1.0, 0.0) is None
    assert garp_module._pct_change(120.0, 100.0) == pytest.approx(20.0)
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


# ── garp.py universe loading (stub the scanner) ─────────────────────────────


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


# ── garp.py india fetch + row mapping ───────────────────────────────────────


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


# ── garp.py US yfinance path ────────────────────────────────────────────────


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


# ── garp.py FMP helpers ─────────────────────────────────────────────────────


def test_fmp_api_key_delegates(monkeypatch) -> None:
    import screener.insiders as insiders

    monkeypatch.setattr(insiders, "_fmp_api_key", lambda: "abc")
    assert garp_module._fmp_api_key() == "abc"


def test_fmp_get_parses_json(monkeypatch) -> None:
    import json

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps([{"x": 1}]).encode("utf-8")

    monkeypatch.setattr(
        garp_module.urllib.request, "urlopen", lambda req, timeout=20: FakeResp()
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


# ── garp.py US screen orchestration / run_garp_screen ───────────────────────


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


def test_screen_us_garp_swallows_resolve_errors(monkeypatch) -> None:
    monkeypatch.setattr(garp_module, "_fmp_api_key", lambda: None)

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
    monkeypatch.setattr(garp_module, "_fmp_api_key", lambda: None)
    monkeypatch.setattr(
        garp_module, "_us_row", lambda symbol, description: _us_passing_row(symbol)
    )

    out = garp_module.run_garp_screen(
        "us", 50, limit=10, workers=1, cache_ttl=None, refresh=False
    )
    assert out is not None
    assert list(out["name"]) == ["AAA"]


# ── rs_breakout.py edge cases ───────────────────────────────────────────────


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


# ── rs_breakout.py backtest-frame builders ──────────────────────────────────


def test_previous_completed_week_high_series_empty_and_values() -> None:
    assert rs_module.previous_completed_week_high_series(pd.DataFrame()).empty
    bars = _bars(n=30)
    series = rs_module.previous_completed_week_high_series(bars)
    assert len(series) == len(bars)
    assert series.notna().any()


def _delivery_panel_frame(symbol="AAA", n=30):
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame(
        {
            "SYMBOL": symbol,
            "date": idx,
            "DELIV_PER": np.linspace(40.0, 60.0, n),
        }
    )


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


# ── rs_breakout.py rendering / writers ──────────────────────────────────────


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


# ── commands/rs_breakout.py ─────────────────────────────────────────────────


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


# ── commands/screen.py ──────────────────────────────────────────────────────


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
        "screener.commands.screen.criteria_registry.get",
        lambda name: fake_runner,
    )

    res = CliRunner().invoke(cli, ["screen", "-c", "rs-breakout", "-m", "india"])
    assert res.exit_code == 0, res.output
    assert captured["market"] == "india"


def test_screen_command_pipeline_combined_rejected() -> None:
    res = CliRunner().invoke(cli, ["screen", "-c", "rs-breakout", "-c", "ema"])
    assert res.exit_code != 0
    assert "cannot be combined" in res.output


# ── commands/garp.py & seasonality.py uncovered branches ────────────────────


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
