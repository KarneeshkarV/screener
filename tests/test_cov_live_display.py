"""Offline coverage tests for live_strategies, display, and enrich modules.

These tests are fully offline and deterministic: every provider/fetcher/universe
seam is stubbed or monkeypatched. No network access occurs.
"""

from __future__ import annotations

import sys
import types
from datetime import date

import numpy as np
import pandas as pd

import screener.commands.live_strategies as live
import screener.display as display
import screener.enrich as enrich
from screener.universes import Universe

from tests.conftest import StubPriceFetcher, make_bars


# --------------------------------------------------------------------------- #
# live_strategies.py
# --------------------------------------------------------------------------- #


def _patch_universe(monkeypatch, symbols):
    """Make load_current_universe return a deterministic offline Universe."""

    def fake_load(name, *, as_of=None, use_cache=True):
        return Universe(
            name=name,
            symbols=tuple(symbols),
            source="test",
            cached_path="/tmp/test-universe.json",
        )

    monkeypatch.setattr(live, "load_current_universe", fake_load)


def _make_panel_data():
    """Two tickers with a clear breakout/cross on the final bars + volume spike."""
    n = 80
    # AAA: steady uptrend so a Donchian breakout + OBV cross fires near the end.
    aaa = make_bars(n=n, seed=1, open_base=100.0, drift=0.3)
    # Force a fresh breakout on the last bar with a big volume spike.
    aaa.iat[n - 1, aaa.columns.get_loc("close")] = float(aaa["high"].max() + 50)
    aaa.iat[n - 1, aaa.columns.get_loc("volume")] = 5_000_000.0
    # BBB: also trending up but more modest.
    bbb = make_bars(n=n, seed=2, open_base=50.0, drift=0.2)
    bbb.iat[n - 1, bbb.columns.get_loc("close")] = float(bbb["high"].max() + 20)
    bbb.iat[n - 1, bbb.columns.get_loc("volume")] = 5_000_000.0
    return {"AAA": aaa, "BBB": bbb}


def test_vol_breakout_live_runs(monkeypatch):
    data = _make_panel_data()
    _patch_universe(monkeypatch, ["AAA", "BBB"])
    fetcher = StubPriceFetcher(data)
    as_of = data["AAA"].index[-1].date()
    # window small so a breakout is detectable within the fetched window.
    live.run_vol_breakout_live(
        market="us",
        as_of=as_of,
        window=10,
        hold=15,
        vol_ma=5,
        vol_mult=1.0,
        limit=30,
        fetcher=fetcher,
    )


def test_vol_breakout_live_empty(monkeypatch):
    """No entries / no active positions -> the '(none)' branches render."""
    n = 80
    # Flat series so nothing breaks out and volume never exceeds the MA.
    flat = make_bars(n=n, seed=5, open_base=100.0, drift=0.0)
    flat["volume"] = 10_000.0
    _patch_universe(monkeypatch, ["AAA"])
    fetcher = StubPriceFetcher({"AAA": flat})
    as_of = flat.index[-1].date()
    live.run_vol_breakout_live(
        market="us",
        as_of=as_of,
        window=50,
        hold=15,
        vol_ma=20,
        vol_mult=100.0,  # impossible volume requirement -> no entries
        limit=30,
        fetcher=fetcher,
    )


def test_vol_breakout_live_skips_nonfinite_last_px(monkeypatch):
    """Cover the active-position guard that rejects non-finite last price.

    A breakout fires a couple of bars before the end, but the final bar's close
    is NaN -> last_px is non-finite -> the entry is skipped.
    """
    n = 80
    aaa = make_bars(n=n, seed=1, open_base=100.0, drift=0.3)
    # Breakout on the second-to-last bar (finite entry price within hold window).
    aaa.iat[n - 2, aaa.columns.get_loc("close")] = float(aaa["high"].max() + 50)
    aaa.iat[n - 2, aaa.columns.get_loc("volume")] = 5_000_000.0
    # Final bar close is NaN -> last_px non-finite -> guard skips the position.
    aaa.iat[n - 1, aaa.columns.get_loc("close")] = float("nan")
    _patch_universe(monkeypatch, ["AAA"])
    fetcher = StubPriceFetcher({"AAA": aaa})
    as_of = aaa.index[-1].date()
    live.run_vol_breakout_live(
        market="us",
        as_of=as_of,
        window=10,
        hold=15,
        vol_ma=5,
        vol_mult=1.0,
        limit=30,
        fetcher=fetcher,
    )


def test_obv_trend_live_runs(monkeypatch):
    # India market: tv_to_yf maps "AAA" -> "AAA.NS", so key data on the yf symbol.
    raw = _make_panel_data()
    data = {f"{k}.NS": v for k, v in raw.items()}
    _patch_universe(monkeypatch, ["AAA", "BBB"])
    fetcher = StubPriceFetcher(data)
    as_of = data["AAA.NS"].index[-1].date()
    live.run_obv_trend_live(
        market="india",
        as_of=as_of,
        ema_window=5,
        limit=30,
        fetcher=fetcher,
    )


def test_obv_trend_live_empty(monkeypatch):
    """Falling series so OBV never crosses above its EMA -> '(none)' branches."""
    n = 80
    falling = make_bars(n=n, seed=7, open_base=200.0, drift=-0.5)
    _patch_universe(monkeypatch, ["AAA"])
    fetcher = StubPriceFetcher({"AAA": falling})
    as_of = falling.index[-1].date()
    live.run_obv_trend_live(
        market="us",
        as_of=as_of,
        ema_window=10,
        limit=30,
        fetcher=fetcher,
    )


def test_obv_trend_live_exit_after_entry(monkeypatch):
    """Series that crosses up then down so last_e <= last_x -> position skipped."""
    n = 80
    # Up then down: OBV rises then falls, producing a cross-up followed by a
    # cross-down, so the most recent exit post-dates the most recent entry.
    up = make_bars(n=n // 2, seed=3, open_base=100.0, drift=0.4)
    down = make_bars(
        n=n // 2, seed=4, open_base=float(up["close"].iloc[-1]), drift=-0.6
    )
    combined = pd.concat([up, down])
    combined.index = pd.bdate_range("2024-01-01", periods=len(combined))
    _patch_universe(monkeypatch, ["AAA"])
    fetcher = StubPriceFetcher({"AAA": combined})
    as_of = combined.index[-1].date()
    live.run_obv_trend_live(
        market="us",
        as_of=as_of,
        ema_window=5,
        limit=30,
        fetcher=fetcher,
    )


def test_obv_trend_live_skips_nonfinite_last_px(monkeypatch):
    """Cover the obv-trend guard rejecting a non-finite last price."""
    n = 80
    aaa = make_bars(n=n, seed=1, open_base=100.0, drift=0.4)
    # Strong uptrend produces an OBV cross-up; NaN final close -> last_px NaN.
    aaa.iat[n - 1, aaa.columns.get_loc("close")] = float("nan")
    _patch_universe(monkeypatch, ["AAA"])
    fetcher = StubPriceFetcher({"AAA": aaa})
    as_of = aaa.index[-1].date()
    live.run_obv_trend_live(
        market="us",
        as_of=as_of,
        ema_window=5,
        limit=30,
        fetcher=fetcher,
    )


def test_market_to_universe():
    assert live._market_to_universe("us") == "sp500"
    assert live._market_to_universe("india") == "nifty50"


def test_crossed_above_np_basic():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 2.0])
    out = live._crossed_above_np(a, b)
    # a goes 1->2->3, b flat 2: crosses above at index 2 (prev 2<=2, now 3>2)
    assert out.tolist() == [False, False, True]


def test_vol_breakout_command_via_click(monkeypatch):
    """Exercise the Click command wrapper, including --as-of parsing."""
    from click.testing import CliRunner

    data = _make_panel_data()
    _patch_universe(monkeypatch, ["AAA", "BBB"])
    fetcher = StubPriceFetcher(data)
    as_of = data["AAA"].index[-1].date()
    runner = CliRunner()
    res = runner.invoke(
        live.vol_breakout_live,
        [
            "--market",
            "us",
            "--as-of",
            as_of.isoformat(),
            "--window",
            "10",
            "--hold",
            "15",
            "--vol-ma",
            "5",
            "--vol-mult",
            "1.0",
            "-n",
            "30",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output


def test_obv_trend_command_via_click(monkeypatch):
    from click.testing import CliRunner

    raw = _make_panel_data()
    data = {f"{k}.NS": v for k, v in raw.items()}
    _patch_universe(monkeypatch, ["AAA", "BBB"])
    fetcher = StubPriceFetcher(data)
    as_of = data["AAA.NS"].index[-1].date()
    runner = CliRunner()
    res = runner.invoke(
        live.obv_trend_live,
        [
            "--market",
            "india",
            "--as-of",
            as_of.isoformat(),
            "--ema-window",
            "5",
            "-n",
            "30",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output


def test_vol_breakout_command_default_as_of(monkeypatch):
    """as_of_arg is None -> the date.today() branch is taken."""
    from click.testing import CliRunner

    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(live, "run_vol_breakout_live", fake_run)
    runner = CliRunner()
    res = runner.invoke(live.vol_breakout_live, [], obj=object())
    assert res.exit_code == 0, res.output
    assert calls["as_of"] == date.today()


def test_obv_trend_command_default_as_of(monkeypatch):
    from click.testing import CliRunner

    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(live, "run_obv_trend_live", fake_run)
    runner = CliRunner()
    res = runner.invoke(live.obv_trend_live, [], obj=object())
    assert res.exit_code == 0, res.output
    assert calls["as_of"] == date.today()


def test_run_functions_build_fetcher_when_none(monkeypatch):
    """fetcher=None branch -> build_price_fetcher() is invoked (then stubbed out)."""
    raw = _make_panel_data()
    # Provide both bare (us) and .NS (india) keys so a single fetcher serves both.
    data = dict(raw)
    data.update({f"{k}.NS": v for k, v in raw.items()})
    _patch_universe(monkeypatch, ["AAA", "BBB"])
    fetcher = StubPriceFetcher(data)
    monkeypatch.setattr(live, "build_price_fetcher", lambda: fetcher)
    as_of = raw["AAA"].index[-1].date()
    live.run_vol_breakout_live(
        market="us", as_of=as_of, window=10, hold=15, vol_ma=5, vol_mult=1.0
    )
    live.run_obv_trend_live(market="india", as_of=as_of, ema_window=5)


# --------------------------------------------------------------------------- #
# display.py
# --------------------------------------------------------------------------- #


def test_format_value_all_branches():
    assert display._format_value("change", None) == "-"
    assert display._format_value("change", float("nan")) == "-"
    # change -> fmt_pct
    assert "%" in display._format_value("change", 1.23)
    # volume tiers
    assert display._format_value("volume", 2_500_000) == "2.5M"
    assert display._format_value("volume", 2_500) == "2.5K"
    assert display._format_value("volume", 500) == "500"
    # market cap
    assert display._format_value("market_cap_basic", 1_000_000_000) != "-"
    # price columns
    assert display._format_value("close", 12.345) == "12.35"
    assert display._format_value("EMA5", 12.345) == "12.35"
    # 2dp numeric block
    assert display._format_value("setup_score", 3.14159) == "3.14"
    assert display._format_value("P/E", 10.0) == "10.00"
    # sales -> fmt_mcap
    assert display._format_value("sales", 5_000_000_000) != "-"
    # fallthrough -> str
    assert display._format_value("name", "AAPL") == "AAPL"


def _screen_df(extra_cols=False):
    base = {
        "ticker": ["NASDAQ:AAPL", "NYSE:IBM"],
        "name": ["AAPL", "IBM"],
        "close": [180.5, 140.25],
        "change": [1.2, -0.5],
        "volume": [2_500_000, 800],
        "market_cap_basic": [3_000_000_000_000, 120_000_000_000],
        "setup_score": [8.5, 6.0],
    }
    if extra_cols:
        base["description"] = ["Apple Inc", "IBM Corp"]
        base["EMA5"] = [179.0, 139.0]
        base["EMA20"] = [175.0, 138.0]
        base["EMA100"] = [170.0, 137.0]
        base["price_earnings_ttm"] = [28.0, 22.0]
    return pd.DataFrame(base)


def test_print_results_basic():
    df = _screen_df()
    display.print_results(df, total=100, market="us", criteria_name="momentum")


def test_print_results_wide_drops_description(capsys):
    df = _screen_df(extra_cols=True)
    display.print_results(
        df,
        total=100,
        market="us",
        criteria_name="momentum",
        added=["XXX"],
        removed=["YYY"],
        first_run=False,
    )


def test_print_results_keeps_description_when_narrow():
    """<=8 columns -> 'description' is kept and its column branch (min/max width)."""
    df = pd.DataFrame(
        {
            "ticker": ["NASDAQ:AAPL"],
            "name": ["AAPL"],
            "description": ["Apple Inc"],
            "close": [180.5],
        }
    )
    display.print_results(df, total=1, market="us", criteria_name="momentum")


def test_print_results_first_run():
    df = _screen_df()
    display.print_results(
        df, total=5, market="india", criteria_name="garp", first_run=True
    )


def test_print_results_no_changes():
    df = _screen_df()
    display.print_results(
        df, total=5, market="us", criteria_name="garp", added=[], removed=[]
    )


def test_print_diff_added_only():
    display._print_diff("us", "x", ["AAA", "BBB"], [], first_run=False)


def test_print_diff_removed_only():
    display._print_diff("us", "x", [], ["CCC"], first_run=False)


def test_print_csv(capsys):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    display.print_csv(df)
    out = capsys.readouterr().out
    assert "a,b" in out


def test_print_garp_results():
    df = pd.DataFrame(
        {
            "name": ["AAA"],
            "description": ["Alpha Corp"],
            "garp_score": [7.5],
            "peg": [1.2],
            "sales": [5_000_000_000],
            "sales_growth_5y": [15.0],
            "operating_profit_growth": [12.0],
            "eps_growth_5y": [18.0],
            "roe_5y": [22.0],
            "roce_or_roic": [25.0],
            "quarterly_profit_growth": [30.0],
        }
    )
    display.print_garp_results(df, market="india")


def test_print_garp_results_missing_columns():
    df = pd.DataFrame({"name": ["AAA"], "garp_score": [7.5]})
    display.print_garp_results(df, market="us")


def test_format_insider_all_branches():
    assert display._format_insider("promoter_pct_latest", None) == "-"
    assert display._format_insider("promoter_pct_latest", 12.0) == "12.00%"
    assert display._format_insider("promoter_change", 1.5) == "+1.50"
    assert display._format_insider("yf_net_pct_6m", 0.01234) == "+1.234%"
    # net shares signed buckets
    assert display._format_insider("yf_net_shares_6m", 2_500_000) == "+2.50M"
    assert display._format_insider("fmp_net_shares_6m", 5_000) == "+5.0K"
    assert display._format_insider("yf_net_shares_6m", -300) == "-300"
    # held / buy / sell unsigned buckets
    assert display._format_insider("yf_total_held", 3_000_000) == "3.0M"
    assert display._format_insider("fmp_buy_shares_6m", 4_000) == "4.0K"
    assert display._format_insider("fmp_sell_shares_6m", 250) == "250"
    # transaction counts
    assert display._format_insider("yf_buy_trans_6m", 7) == "7"
    assert display._format_insider("yf_sell_trans_6m", 3) == "3"
    # fallthrough to _format_value
    assert display._format_insider("close", 12.34) == "12.34"


def test_print_insider_results_india():
    df = pd.DataFrame(
        {
            "name": ["AAA"],
            "close": [123.4],
            "promoter_pct_prev": [50.0],
            "promoter_pct_latest": [55.0],
            "promoter_change": [5.0],
            "latest_quarter": ["2025Q1"],
            "fii_pct_latest": [10.0],
            "dii_pct_latest": [8.0],
        }
    )
    display.print_insider_results(df, market="india", universe_size=50, match_count=1)


def test_print_insider_results_us():
    df = pd.DataFrame(
        {
            "name": ["AAA"],
            "description": ["Alpha Corp"],
            "close": [123.4],
            "fmp_net_shares_6m": [1_000_000],
            "fmp_buy_shares_6m": [2_000_000],
            "fmp_sell_shares_6m": [1_000_000],
            "yf_net_shares_6m": [500_000],
            "yf_net_pct_6m": [0.012],
            "yf_total_held": [10_000_000],
            "yf_buy_trans_6m": [5],
            "yf_sell_trans_6m": [2],
        }
    )
    display.print_insider_results(df, market="us", universe_size=500, match_count=1)


def test_format_institutional_all_branches():
    assert display._format_institutional("holders", None) == "-"
    assert display._format_institutional("holders", 1234) == "1,234"
    assert display._format_institutional("qoq_change_pct", 2.5) == "+2.50%"
    # billions / millions / thousands / small with sign rules
    assert display._format_institutional("total_shares", 2_500_000_000) == "2.50B"
    assert display._format_institutional("total_shares", 3_000_000) == "3.00M"
    assert display._format_institutional("total_shares", 4_000) == "4.0K"
    assert display._format_institutional("total_shares", 250) == "250"
    # qoq_change_shares carries an explicit + for positives
    assert display._format_institutional("qoq_change_shares", 2_500_000) == "+2.50M"
    assert display._format_institutional("qoq_change_shares", -2_500_000) == "-2.50M"
    # fallthrough
    assert display._format_institutional("symbol", "AAPL") == "AAPL"


def test_print_institutional_results():
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "holders": [100, 200],
            "total_shares": [5_000_000, 8_000_000],
            "qoq_change_shares": [1_000_000, -500_000],
            "qoq_change_pct": [5.0, -2.0],
        }
    )
    display.print_institutional_results(df)


# --------------------------------------------------------------------------- #
# enrich.py
# --------------------------------------------------------------------------- #


def test_enrich_non_india_passthrough():
    df = pd.DataFrame({"name": ["AAA"]})
    out = enrich.enrich_fundamentals(df, market="us")
    assert out is df


def test_enrich_import_error(monkeypatch):
    """openscreener not importable -> df returned unchanged."""
    # Ensure import fails by blocking the module name.
    monkeypatch.setitem(sys.modules, "openscreener", None)
    df = pd.DataFrame({"name": ["AAA"]})
    out = enrich.enrich_fundamentals(df, market="india")
    assert out is df


def _install_fake_openscreener(monkeypatch, *, batch_factory):
    mod = types.ModuleType("openscreener")

    class Stock:
        @staticmethod
        def batch(symbols):
            return batch_factory(symbols)

    mod.Stock = Stock
    monkeypatch.setitem(sys.modules, "openscreener", mod)


def test_enrich_empty_symbols(monkeypatch):
    """Empty symbol list short-circuits before any network call."""
    _install_fake_openscreener(monkeypatch, batch_factory=lambda s: None)
    df = pd.DataFrame({"name": []})
    out = enrich.enrich_fundamentals(df, market="india")
    assert out is df


def test_enrich_fetch_raises(monkeypatch):
    """batch.fetch raising a handled error -> df returned unchanged."""

    class FailingBatch:
        def fetch(self, kind):
            raise ConnectionError("boom")

    _install_fake_openscreener(monkeypatch, batch_factory=lambda s: FailingBatch())
    df = pd.DataFrame({"name": ["AAA"]})
    out = enrich.enrich_fundamentals(df, market="india")
    assert out is df


def test_enrich_success(monkeypatch):
    """Happy path -> ratios merged onto the frame."""

    class OkBatch:
        def fetch(self, kind):
            assert kind == "ratios"
            return {
                "AAA": {
                    "stock_p_e": 15.0,
                    "roce_percent": 22.0,
                    "return_on_equity": 18.0,
                },
                # BBB intentionally missing -> data.get returns {} -> None fields
            }

    _install_fake_openscreener(monkeypatch, batch_factory=lambda s: OkBatch())
    df = pd.DataFrame({"name": ["AAA", "BBB"]})
    out = enrich.enrich_fundamentals(df, market="india")
    assert "P/E" in out.columns
    assert out.loc[out["name"] == "AAA", "P/E"].iloc[0] == 15.0
    assert pd.isna(out.loc[out["name"] == "BBB", "P/E"].iloc[0])
