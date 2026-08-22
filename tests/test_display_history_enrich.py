from __future__ import annotations

import builtins
import sys
import types
from datetime import UTC, datetime

import pandas as pd

from screener import display, enrich, history


def test_display_value_formatters_cover_numeric_tiers():
    assert display._format_value("change", 1.25) == "+1.25%"
    assert display._format_value("volume", 2_500_000) == "2.50M"
    assert display._format_value("volume", 2_500) == "2.5K"
    assert display._format_value("volume", 25) == "25"
    assert display._format_value("market_cap_basic", 2_500_000_000) == "2.50B"
    assert display._format_value("market_cap_basic", 2_500_000_000, "india") == (
        "250.00 Cr"
    )
    assert display._format_value("sales", 7_800_000, "india") == "78.0 Lakh"
    assert display._format_value("close", 12.345) == "12.35"
    assert display._format_value("RSI", 55.678) == "55.68"
    assert display._format_value("sales", 2_500_000_000) == "2.50B"
    assert display._format_value("unknown", "abc") == "abc"
    assert display._format_value("unknown", None) == "-"
    assert display._format_value("unknown", float("nan")) == "-"


def test_display_insider_and_institutional_formatters_cover_tiers():
    assert display._format_insider("promoter_pct_latest", 44.444) == "44.44%"
    assert display._format_insider("promoter_change", -1.25) == "-1.25"
    assert display._format_insider("yf_net_pct_6m", 0.01234) == "+1.234%"
    assert display._format_insider("fmp_net_shares_6m", 1_500_000) == "+1.50M"
    assert display._format_insider("yf_net_shares_6m", -1_500) == "-1.5K"
    assert display._format_insider("yf_net_shares_6m", 15) == "+15"
    assert display._format_insider("yf_total_held", 1_500_000) == "1.5M"
    assert display._format_insider("fmp_buy_shares_6m", 1_500) == "1.5K"
    assert display._format_insider("fmp_sell_shares_6m", 15) == "15"
    assert display._format_insider("yf_buy_trans_6m", 3) == "3"
    assert display._format_insider("close", 12.3) == "12.30"
    assert display._format_insider("close", None) == "-"

    assert display._format_institutional("holders", 1234) == "1,234"
    assert display._format_institutional("qoq_change_pct", -2.5) == "-2.50%"
    assert display._format_institutional("total_shares", 2_500_000_000) == "2.50B"
    assert display._format_institutional("total_shares", 2_500_000) == "2.50M"
    assert display._format_institutional("qoq_change_shares", 2_500) == "+2.5K"
    assert display._format_institutional("qoq_change_shares", 25) == "+25"
    assert display._format_institutional("symbol", "AAA") == "AAA"
    assert display._format_institutional("holders", None) == "-"


def test_print_result_tables_and_diffs(capsys):
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "name": "AAA",
                "description": "Alpha Corp",
                "close": 12.345,
                "change": 1.5,
                "volume": 2_500_000,
                "market_cap_basic": 3_000_000_000,
                "setup_score": 8.5,
                "RSI": 62.1,
            }
        ]
    )

    display.print_results(
        df,
        total=3,
        market="us",
        criteria_name="ema",
        added=["AAA"],
        removed=["OLD"],
    )
    display.print_results(df, total=1, market="us", criteria_name="ema")
    display.print_results(df, total=1, market="us", criteria_name="ema", first_run=True)
    display.print_results(
        df[["name", "description", "close"]],
        total=1,
        market="india",
        criteria_name="value",
    )
    display.print_csv(df)

    out = capsys.readouterr().out
    assert "EMA" in out
    assert "Diff vs previous run" in out
    assert "No changes since last run" in out
    assert "saved as baseline" in out
    assert "Alpha Corp" in out


def test_print_specialized_result_tables(capsys):
    garp_df = pd.DataFrame(
        [
            {
                "name": "AAA",
                "description": "Alpha",
                "garp_score": 9.1,
                "peg": 1.2,
                "sales": 2_000_000_000,
                "sales_growth_5y": 12.3,
                "operating_profit_growth": 13.4,
                "eps_growth_5y": 14.5,
                "roe_5y": 15.6,
                "roce_or_roic": 16.7,
                "quarterly_profit_growth": 17.8,
            }
        ]
    )
    india_insider = pd.DataFrame(
        [
            {
                "name": "AAA",
                "close": 12.3,
                "promoter_pct_prev": 40.0,
                "promoter_pct_latest": 41.5,
                "promoter_change": 1.5,
                "latest_quarter": "2025Q4",
                "fii_pct_latest": 10.0,
                "dii_pct_latest": 11.0,
            }
        ]
    )
    us_insider = pd.DataFrame(
        [
            {
                "name": "AAA",
                "description": "Alpha",
                "close": 12.3,
                "fmp_net_shares_6m": 1500,
                "fmp_buy_shares_6m": 2000,
                "fmp_sell_shares_6m": 500,
                "yf_net_shares_6m": -50,
                "yf_net_pct_6m": -0.01,
                "yf_total_held": 10_000,
                "yf_buy_trans_6m": 2,
                "yf_sell_trans_6m": 1,
            }
        ]
    )
    institutional = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "holders": 12,
                "total_shares": 1_500_000,
                "qoq_change_shares": -2500,
                "qoq_change_pct": -1.2,
            }
        ]
    )

    display.print_garp_results(garp_df, "india")
    display.print_insider_results(
        india_insider, "india", universe_size=200, match_count=1
    )
    display.print_insider_results(us_insider, "us", universe_size=200, match_count=1)
    display.print_institutional_results(institutional)

    out = capsys.readouterr().out
    assert "GARP" in out
    assert "Promoter buys" in out
    assert "Insider buys" in out
    assert "Institutional ownership" in out


def test_history_save_previous_and_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    timestamps = iter(
        [
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC),
        ]
    )

    class FakeDateTime:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return next(timestamps)

    monkeypatch.setattr(history, "datetime", FakeDateTime)
    first = pd.DataFrame(
        [
            {
                "name": "AAA",
                "description": "Alpha",
                "close": "12.3",
                "change": "bad",
                "volume": 1000,
                "market_cap_basic": float("nan"),
                "setup_score": 8,
            },
            {"name": "", "description": "blank"},
        ]
    )
    second = pd.DataFrame([{"name": "BBB", "description": None, "close": 22.0}])

    first_id = history.save_run("us", "ema", 2, first)
    second_id = history.save_run("us", "ema", 1, second)

    assert first_id < second_id
    previous = history.previous_run("us", "ema", second_id)
    assert previous is not None
    assert previous["ticker"].tolist() == ["AAA"]
    assert history.previous_run("us", "ema", first_id) is None
    assert history.diff(second, previous) == (["BBB"], ["AAA"])
    assert history.diff(pd.DataFrame(), previous) == ([], ["AAA"])
    assert history.diff(second, pd.DataFrame()) == (["BBB"], [])


def test_history_to_float_handles_bad_inputs():
    assert history._to_float(None) is None
    assert history._to_float("bad") is None
    assert history._to_float(float("nan")) is None
    assert history._to_float("12.5") == 12.5


def test_enrich_fundamentals_non_india_empty_and_import_error(monkeypatch):
    df = pd.DataFrame({"name": ["AAA"]})
    assert enrich.enrich_fundamentals(df, "us") is df
    assert enrich.enrich_fundamentals(pd.DataFrame({"name": []}), "india").empty

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openscreener":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert enrich.enrich_fundamentals(df, "india") is df


def test_enrich_fundamentals_success_and_fetch_failure(monkeypatch):
    class FakeBatch:
        def fetch(self, key: str) -> dict:
            assert key == "ratios"
            return {
                "AAA": {
                    "stock_p_e": 12.5,
                    "roce_percent": 22.0,
                    "return_on_equity": 18.0,
                }
            }

    class FakeStock:
        @staticmethod
        def batch(symbols: list[str]) -> FakeBatch:
            assert symbols == ["AAA"]
            return FakeBatch()

    module = types.SimpleNamespace(Stock=FakeStock)
    monkeypatch.setitem(sys.modules, "openscreener", module)

    enriched = enrich.enrich_fundamentals(pd.DataFrame({"name": ["AAA"]}), "india")

    assert enriched.loc[0, "P/E"] == 12.5
    assert enriched.loc[0, "ROCE%"] == 22.0
    assert enriched.loc[0, "ROE%"] == 18.0

    class FailingStock:
        @staticmethod
        def batch(symbols: list[str]) -> object:
            raise RuntimeError("offline")

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FailingStock)
    )
    original = pd.DataFrame({"name": ["AAA"]})
    assert enrich.enrich_fundamentals(original, "india") is original
