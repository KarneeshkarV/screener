"""Offline line-coverage tests for insiders / conviction / commands.insiders / pledge.

All tests are deterministic and never touch the network: every provider seam,
HTTP/urlopen, FMP/NSE/openscreener call and scanner fetch is stubbed or
monkeypatched. CLI flows use ``click.testing.CliRunner``.
"""

from __future__ import annotations


import json


from datetime import date


import pandas as pd


import pytest


from screener import conviction as conviction_mod


from screener import insiders as insiders_mod


from screener import pledge as pledge_mod


from screener.commands import insiders as cmd_insiders


from screener.providers import FakeProvider


from tests.conftest import make_bars


class _FakeYfTicker:
    def __init__(self, purchases):
        self.insider_purchases = purchases


def _purchases_frame():
    return pd.DataFrame(
        {
            "Insider Purchases Last 6m": [
                "Net Shares Purchased (Sold)",
                "% Net Shares Purchased (Sold)",
                "Total Insider Shares Held",
                "Purchases",
                "Sales",
            ],
            "Shares": [1000.0, 5.0, 50000.0, None, None],
            "Trans": [None, None, None, 3.0, 1.0],
        }
    )


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class _FakeStock:
    def __init__(self, rows):
        self._rows = rows

    def __call__(self, name, scraper=None):
        return self

    def shareholding_quarterly(self):
        return self._rows


def _install_openscreener(monkeypatch, rows):
    import sys
    import types

    mod = types.ModuleType("openscreener")
    stock = _FakeStock(rows)
    mod.Stock = lambda name, scraper=None: stock
    monkeypatch.setitem(sys.modules, "openscreener", mod)


class _Ev:
    def __init__(self, symbol):
        self.symbol = symbol
        self.pledge_pct = None


class _VolEvent:
    def __init__(self, rvol, z, direction, strength="strong"):
        self.rvol = rvol
        self.z_score = z
        self.direction = direction
        self.strength = strength


def _stub_scanner(monkeypatch, universe, total=500):
    monkeypatch.setattr(
        cmd_insiders,
        "get_scanner_data_cached",
        lambda *a, **k: (total, universe.copy()),
    )
    monkeypatch.setattr(cmd_insiders, "_dedupe_listings", lambda df: df)


def _base_universe():
    return pd.DataFrame(
        [
            {
                "name": "ACME",
                "description": "Acme Inc",
                "close": 100.0,
                "change": 1.0,
                "volume": 50000,
                "market_cap_basic": 1e9,
            }
        ]
    )


def test_row_value_none_for_empty_and_missing_label():
    assert insiders_mod._row_value(None, "x", "Shares") is None
    assert insiders_mod._row_value(pd.DataFrame(), "x", "Shares") is None
    # No match for the requested label row.
    df = pd.DataFrame({"Insider Purchases Last 6m": ["Purchases"], "Shares": [10]})
    assert insiders_mod._row_value(df, "Sales", "Shares") is None


def test_row_value_handles_na_and_non_numeric():
    df = pd.DataFrame(
        {
            "Insider Purchases Last 6m": ["Net Shares Purchased (Sold)", "Purchases"],
            "Shares": [None, "not-a-number"],
        }
    )
    # NA value -> None
    assert insiders_mod._row_value(df, "Net Shares Purchased (Sold)", "Shares") is None
    # Non-numeric -> None (TypeError/ValueError swallowed)
    assert insiders_mod._row_value(df, "Purchases", "Shares") is None


def test_row_value_returns_float():
    df = pd.DataFrame({"Insider Purchases Last 6m": ["Purchases"], "Shares": ["1234"]})
    assert insiders_mod._row_value(df, "Purchases", "Shares") == 1234.0


def test_fetch_yf_one_builds_row(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_YF_INSIDER_PROVIDER", FakeProvider())
    monkeypatch.setattr(
        insiders_mod.yf, "Ticker", lambda sym: _FakeYfTicker(_purchases_frame())
    )
    out = insiders_mod._fetch_yf_one("Acme", "ACME", cache_ttl=None, refresh=True)
    assert out["name"] == "Acme"
    assert out["yf_symbol"] == "ACME"
    assert out["yf_net_shares_6m"] == 1000.0
    assert out["yf_buy_trans_6m"] == 3.0
    assert out["yf_sell_trans_6m"] == 1.0


def test_fetch_yf_one_none_when_empty(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_YF_INSIDER_PROVIDER", FakeProvider())
    monkeypatch.setattr(
        insiders_mod.yf, "Ticker", lambda sym: _FakeYfTicker(pd.DataFrame())
    )
    assert (
        insiders_mod._fetch_yf_one("Acme", "ACME", cache_ttl=None, refresh=True) is None
    )


def test_fetch_yfinance_insiders_empty_universe():
    assert insiders_mod.fetch_yfinance_insiders(pd.DataFrame(), "us").empty


def test_fetch_yfinance_insiders_collects_rows(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_YF_INSIDER_PROVIDER", FakeProvider())

    def fake_ticker(sym):
        # Only ACME returns data; the other yields an empty frame -> dropped.
        if sym == "ACME":
            return _FakeYfTicker(_purchases_frame())
        return _FakeYfTicker(pd.DataFrame())

    monkeypatch.setattr(insiders_mod.yf, "Ticker", fake_ticker)
    universe = pd.DataFrame(
        [
            {"name": "Acme", "ticker": "NASDAQ:ACME"},
            {"name": "Beta", "ticker": "NASDAQ:BETA"},
        ]
    )
    df = insiders_mod.fetch_yfinance_insiders(universe, "us", max_workers=2)
    assert list(df["name"]) == ["Acme"]


def test_fmp_api_key_from_env(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "envkey")
    assert insiders_mod.resolve_api_key() == "envkey"


def test_fmp_api_key_loads_env_file(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    called = {"n": 0}

    def fake_load_env():
        called["n"] += 1
        import os

        os.environ["FMP_API_KEY"] = "fromfile"

    import screener.backtester.data as bt_data

    monkeypatch.setattr(bt_data, "load_env_file", fake_load_env)
    assert insiders_mod.resolve_api_key() == "fromfile"
    assert called["n"] == 1


def test_fmp_api_key_import_failure(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "screener.backtester.data":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert insiders_mod.resolve_api_key() is None


def test_fetch_fmp_insider_one_non_list_payload(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_FMP_INSIDER_PROVIDER", FakeProvider())
    monkeypatch.setattr(
        insiders_mod.urllib.request,
        "urlopen",
        lambda req, timeout=20: _Resp({"Error Message": "bad key"}),
    )
    out = insiders_mod._fetch_fmp_insider_one(
        "AAA", "AAA", api_key="k", cache_ttl=None, refresh=True
    )
    assert out is None


def test_fetch_fmp_insider_one_no_aggregatable_rows(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_FMP_INSIDER_PROVIDER", FakeProvider())
    # Rows in window but all award/gift -> aggregate returns None.
    d = (pd.Timestamp.now().normalize() - pd.Timedelta(days=5)).date().isoformat()
    row = {
        "transactionDate": d,
        "acquistionOrDisposition": "A",
        "transactionType": "A-Award",
        "securitiesTransacted": 100,
    }

    def fake_urlopen(req, timeout=20):
        import urllib.parse

        page = int(
            urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)["page"][0]
        )
        if page == 0:
            return _Resp([row])
        return _Resp([])

    monkeypatch.setattr(insiders_mod.urllib.request, "urlopen", fake_urlopen)
    out = insiders_mod._fetch_fmp_insider_one(
        "AAA", "AAA", api_key="k", cache_ttl=None, refresh=True
    )
    assert out is None


def test_fetch_fmp_insiders_empty_universe():
    assert insiders_mod.fetch_fmp_insiders(pd.DataFrame(), "us").empty


def test_fetch_fmp_insiders_no_api_key(monkeypatch):
    monkeypatch.setattr(insiders_mod, "resolve_api_key", lambda: None)
    universe = pd.DataFrame([{"name": "Acme", "ticker": "NASDAQ:ACME"}])
    assert insiders_mod.fetch_fmp_insiders(universe, "us").empty


def test_fetch_fmp_insiders_collects(monkeypatch):
    monkeypatch.setattr(insiders_mod, "resolve_api_key", lambda: "k")
    monkeypatch.setattr(
        insiders_mod,
        "_fetch_fmp_insider_one",
        lambda name, symbol, *, api_key, cache_ttl, refresh: (
            {"name": name, "fmp_symbol": symbol, "fmp_net_shares_6m": 100.0}
            if name == "Acme"
            else None
        ),
    )
    universe = pd.DataFrame(
        [
            {"name": "Acme", "ticker": "NASDAQ:ACME"},
            {"name": "Beta", "ticker": "NASDAQ:BETA"},
        ]
    )
    df = insiders_mod.fetch_fmp_insiders(universe, "us", max_workers=2)
    assert list(df["name"]) == ["Acme"]


def test_http_scraper_fetch_page(monkeypatch):
    captured = {}

    def fake_resilience(breaker, op, fn, *, fallback):
        captured["breaker"] = breaker
        return fn()

    monkeypatch.setattr(insiders_mod, "call_with_resilience", fake_resilience)
    monkeypatch.setattr(
        insiders_mod.urllib.request,
        "urlopen",
        lambda req, timeout=20: _Resp("<html>page</html>"),
    )
    scraper = insiders_mod._HttpScraper()
    html = scraper.fetch_page("reliance")
    assert "page" in html
    assert captured["breaker"] == "screener-in"


def test_http_scraper_fetch_pages(monkeypatch):
    monkeypatch.setattr(
        insiders_mod._HttpScraper, "fetch_page", lambda self, sym: f"html-{sym}"
    )
    out = insiders_mod._HttpScraper().fetch_pages(["aa", "bb"])
    assert out == {"AA": "html-aa", "BB": "html-bb"}


def test_fetch_openscreener_one_happy(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_OPENSCREENER_PROVIDER", FakeProvider())
    rows = [
        {"date": "Dec 2025", "promoters": 50.0, "fiis": 10.0, "diis": 5.0},
        {"date": "Mar 2026", "promoters": 51.5, "fiis": 11.0, "diis": 6.0},
    ]
    _install_openscreener(monkeypatch, rows)
    out = insiders_mod._fetch_openscreener_one("ACME", cache_ttl=None, refresh=True)
    assert out["promoter_pct_latest"] == 51.5
    assert out["promoter_pct_prev"] == 50.0
    assert out["promoter_change"] == 1.5
    assert out["latest_quarter"] == "Mar 2026"
    assert out["fii_pct_latest"] == 11.0


def test_fetch_openscreener_one_import_error(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_OPENSCREENER_PROVIDER", FakeProvider())
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "openscreener":
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert (
        insiders_mod._fetch_openscreener_one("ACME", cache_ttl=None, refresh=True)
        is None
    )


def test_fetch_openscreener_one_empty_rows(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_OPENSCREENER_PROVIDER", FakeProvider())
    _install_openscreener(monkeypatch, [])
    assert (
        insiders_mod._fetch_openscreener_one("ACME", cache_ttl=None, refresh=True)
        is None
    )


def test_fetch_openscreener_one_single_row(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_OPENSCREENER_PROVIDER", FakeProvider())
    _install_openscreener(monkeypatch, [{"date": "Mar 2026", "promoters": 50.0}])
    assert (
        insiders_mod._fetch_openscreener_one("ACME", cache_ttl=None, refresh=True)
        is None
    )


def test_fetch_openscreener_one_missing_promoter(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_OPENSCREENER_PROVIDER", FakeProvider())
    _install_openscreener(
        monkeypatch,
        [
            {"date": "Dec 2025", "promoters": None},
            {"date": "Mar 2026", "promoters": 50.0},
        ],
    )
    assert (
        insiders_mod._fetch_openscreener_one("ACME", cache_ttl=None, refresh=True)
        is None
    )


def test_fetch_openscreener_one_non_numeric_promoter(monkeypatch):
    monkeypatch.setattr(insiders_mod, "_OPENSCREENER_PROVIDER", FakeProvider())
    _install_openscreener(
        monkeypatch,
        [
            {"date": "Dec 2025", "promoters": "x"},
            {"date": "Mar 2026", "promoters": "y"},
        ],
    )
    assert (
        insiders_mod._fetch_openscreener_one("ACME", cache_ttl=None, refresh=True)
        is None
    )


def test_fetch_openscreener_promoters_empty_universe():
    assert insiders_mod.fetch_openscreener_promoters(pd.DataFrame()).empty


def test_fetch_openscreener_promoters_collects(monkeypatch):
    monkeypatch.setattr(
        insiders_mod,
        "_fetch_openscreener_one",
        lambda name, *, cache_ttl, refresh: (
            {"name": name, "promoter_change": 1.0} if name == "Acme" else None
        ),
    )
    universe = pd.DataFrame([{"name": "Acme"}, {"name": "Beta"}])
    df = insiders_mod.fetch_openscreener_promoters(universe, max_workers=2)
    assert list(df["name"]) == ["Acme"]


def test_filter_empty_returns_input():
    df = pd.DataFrame()
    assert insiders_mod.filter_promoter_increased(df, "us") is df


def test_filter_india_promoter_change():
    df = pd.DataFrame(
        [
            {"name": "A", "promoter_change": 0.5},
            {"name": "B", "promoter_change": -0.2},
        ]
    )
    out = insiders_mod.filter_promoter_increased(
        df, "india", min_promoter_change_pct=0.1
    )
    assert list(out["name"]) == ["A"]


def test_filter_india_require_both():
    df = pd.DataFrame(
        [
            {"name": "A", "promoter_change": 1.0, "yf_net_shares_6m": 10.0},
            {"name": "B", "promoter_change": 1.0, "yf_net_shares_6m": -5.0},
        ]
    )
    out = insiders_mod.filter_promoter_increased(df, "india", require_both=True)
    assert list(out["name"]) == ["A"]


def test_filter_us_no_fmp_column_min_pct():
    df = pd.DataFrame(
        [
            {"name": "A", "yf_net_shares_6m": 10.0, "yf_net_pct_6m": 5.0},
            {"name": "B", "yf_net_shares_6m": 10.0, "yf_net_pct_6m": 1.0},
        ]
    )
    out = insiders_mod.filter_promoter_increased(df, "us", min_yf_net_pct=3.0)
    assert list(out["name"]) == ["A"]


def test_as_float_variants():
    assert pledge_mod._as_float(None) is None
    assert pledge_mod._as_float("12.5%") == 12.5
    assert pledge_mod._as_float("1,234") == 1234.0
    assert pledge_mod._as_float("abc") is None


def test_as_pct_bounds():
    assert pledge_mod._as_pct("50") == 50.0
    assert pledge_mod._as_pct("-1") is None
    assert pledge_mod._as_pct("101") is None
    assert pledge_mod._as_pct("bad") is None


def test_fetch_nse_pledge_primary_key(monkeypatch):
    monkeypatch.setattr(
        pledge_mod,
        "nse_cached_json",
        lambda *a, **k: {"data": [{"per. of Promoter Holding Shares pledge": "12.34"}]},
    )
    assert pledge_mod.fetch_nse_pledge("ACME") == 12.34


def test_fetch_nse_pledge_list_payload_and_fallback_key(monkeypatch):
    # raw is a bare list (not dict); latest dict uses a fallback "pledge" key.
    monkeypatch.setattr(
        pledge_mod,
        "nse_cached_json",
        lambda *a, **k: [{"somePledgePercent": "7.5"}],
    )
    assert pledge_mod.fetch_nse_pledge("ACME") == 7.5


def test_fetch_nse_pledge_none_when_no_rows(monkeypatch):
    monkeypatch.setattr(pledge_mod, "nse_cached_json", lambda *a, **k: {"data": []})
    assert pledge_mod.fetch_nse_pledge("ACME") is None
    monkeypatch.setattr(pledge_mod, "nse_cached_json", lambda *a, **k: "garbage")
    assert pledge_mod.fetch_nse_pledge("ACME") is None


def test_fetch_nse_pledge_latest_not_dict(monkeypatch):
    monkeypatch.setattr(pledge_mod, "nse_cached_json", lambda *a, **k: {"data": ["x"]})
    assert pledge_mod.fetch_nse_pledge("ACME") is None


def test_fetch_nse_pledge_no_matching_key(monkeypatch):
    monkeypatch.setattr(
        pledge_mod, "nse_cached_json", lambda *a, **k: {"data": [{"unrelated": 1}]}
    )
    assert pledge_mod.fetch_nse_pledge("ACME") is None


def test_fetch_openscreener_pledge_match(monkeypatch):
    monkeypatch.setattr(pledge_mod, "_OSC_PLEDGE_PROVIDER", FakeProvider())
    monkeypatch.setattr(
        pledge_mod._HttpScraper,
        "fetch_page",
        lambda self, name: "Pledged percentage</span> ... 12.34%",
    )
    assert pledge_mod.fetch_openscreener_pledge("ACME") == 12.34


def test_fetch_openscreener_pledge_no_html(monkeypatch):
    monkeypatch.setattr(pledge_mod, "_OSC_PLEDGE_PROVIDER", FakeProvider())
    monkeypatch.setattr(pledge_mod._HttpScraper, "fetch_page", lambda self, name: "")
    assert pledge_mod.fetch_openscreener_pledge("ACME") is None


def test_fetch_openscreener_pledge_no_match(monkeypatch):
    monkeypatch.setattr(pledge_mod, "_OSC_PLEDGE_PROVIDER", FakeProvider())
    monkeypatch.setattr(
        pledge_mod._HttpScraper, "fetch_page", lambda self, name: "no pledge here"
    )
    assert pledge_mod.fetch_openscreener_pledge("ACME") is None


def test_resolve_pledge_pct_prefers_nse(monkeypatch):
    monkeypatch.setattr(pledge_mod, "fetch_nse_pledge", lambda sym, *, refresh: 5.0)
    monkeypatch.setattr(
        pledge_mod,
        "fetch_openscreener_pledge",
        lambda name, *, refresh: pytest.fail("should not call osc"),
    )
    assert pledge_mod.resolve_pledge_pct("ACME", "ACME") == 5.0


def test_resolve_pledge_pct_falls_back(monkeypatch):
    monkeypatch.setattr(pledge_mod, "fetch_nse_pledge", lambda sym, *, refresh: None)
    monkeypatch.setattr(
        pledge_mod, "fetch_openscreener_pledge", lambda name, *, refresh: 9.0
    )
    assert pledge_mod.resolve_pledge_pct("ACME", "ACME") == 9.0


def test_overlay_pledge_empty_noop():
    assert pledge_mod.overlay_pledge([]) is None


def test_overlay_pledge_mutates(monkeypatch):
    monkeypatch.setattr(
        pledge_mod,
        "resolve_pledge_pct",
        lambda sym, name, *, refresh: 12.0 if sym == "AAA" else None,
    )
    events = [_Ev("aaa"), _Ev("bbb")]
    pledge_mod.overlay_pledge(events, max_workers=2)
    assert events[0].pledge_pct == 12.0
    assert events[1].pledge_pct is None


def test_quarter_public_date_none_label():
    assert conviction_mod._quarter_public_date(None) is None


def test_quarter_public_date_unparseable():
    assert conviction_mod._quarter_public_date("not a date") is None


def test_quarter_public_date_ok():
    pub = conviction_mod._quarter_public_date("Mar 2024")
    assert pub == date(2024, 5, 15)


def test_rsi_points_buckets():
    assert conviction_mod._rsi_points(90) == 5.0
    assert conviction_mod._rsi_points(78) == 15.0
    assert conviction_mod._rsi_points(60) == 30.0
    assert conviction_mod._rsi_points(48) == 20.0
    assert conviction_mod._rsi_points(38) == 10.0
    assert conviction_mod._rsi_points(20) == 0.0


def test_score_trend_insufficient_bars():
    res = conviction_mod.score_trend(make_bars(n=10), pd.Series(dtype=float))
    assert res.status == "skipped"
    assert "insufficient" in res.reason


def test_score_trend_no_benchmark_renormalizes():
    bars = make_bars(n=80, drift=0.4, seed=3)
    res = conviction_mod.score_trend(bars, pd.Series(dtype=float))
    assert res.status == "ok"
    assert "RS63 n/a" in res.evidence


def test_score_trend_with_benchmark():
    bars = make_bars(n=120, drift=0.4, seed=3)
    bench = make_bars(n=120, drift=0.05, seed=9, open_base=400.0)
    res = conviction_mod.score_trend(bars, bench["close"].astype(float))
    assert res.status == "ok"
    assert "vs benchmark" in res.evidence


def test_score_breakout_insufficient():
    res = conviction_mod.score_breakout(
        make_bars(n=5), pd.Series(dtype=float), date(2026, 1, 2)
    )
    assert res.status == "skipped"


def test_score_breakout_no_benchmark():
    bars = make_bars(n=300, drift=0.3, seed=4)
    as_of = bars.index[-1].date()
    res = conviction_mod.score_breakout(bars, pd.Series(dtype=float), as_of)
    assert res.status == "ok"
    assert "RS55 n/a" in res.evidence


def test_score_breakout_with_benchmark():
    bars = make_bars(n=300, drift=0.3, seed=4)
    bench = make_bars(n=300, drift=0.02, seed=11, open_base=400.0)
    as_of = bars.index[-1].date()
    res = conviction_mod.score_breakout(bars, bench["close"].astype(float), as_of)
    assert res.status == "ok"
    assert "RS55" in res.evidence


def test_score_breakout_below_pivot_downtrend():
    # A persistent downtrend puts price below the previous-week pivot and the
    # SuperTrend, exercising the "below pivot" / "overhead" branches.
    bars = make_bars(n=300, drift=-0.4, seed=5)
    as_of = bars.index[-1].date()
    res = conviction_mod.score_breakout(bars, pd.Series(dtype=float), as_of)
    assert res.status == "ok"


def test_score_breakout_no_pivot(monkeypatch):
    bars = make_bars(n=300, drift=0.2, seed=4)
    as_of = bars.index[-1].date()
    monkeypatch.setattr(
        conviction_mod, "previous_completed_week_high", lambda df, d: None
    )
    res = conviction_mod.score_breakout(bars, pd.Series(dtype=float), as_of)
    assert "no pivot" in res.evidence


def test_score_breakout_near_pivot(monkeypatch):
    bars = make_bars(n=300, drift=0.2, seed=4)
    as_of = bars.index[-1].date()
    close = float(conviction_mod.normalize_bars(bars, as_of)["close"].iloc[-1])
    # Pivot just above close but within 3% -> "near pivot".
    monkeypatch.setattr(
        conviction_mod,
        "previous_completed_week_high",
        lambda df, d: close * 1.01,
    )
    res = conviction_mod.score_breakout(bars, pd.Series(dtype=float), as_of)
    assert "near pivot" in res.evidence


def test_load_smart_money_us(monkeypatch):
    monkeypatch.setattr(
        conviction_mod,
        "load_insider_aggregate",
        lambda symbol, *, api_key, cache_ttl, refresh: {"fmp_net_shares_6m": 1.0},
    )
    out = conviction_mod._load_smart_money_us(
        "AAPL", "k", cache_ttl=None, refresh=False
    )
    assert out == {"fmp_net_shares_6m": 1.0}


def test_score_volume_skipped(monkeypatch):
    monkeypatch.setattr(conviction_mod, "detect_ticker", lambda *a, **k: None)
    res = conviction_mod.score_volume("X", make_bars(n=100), date(2026, 1, 2))
    assert res.status == "skipped"


def test_score_volume_with_delivery_increase(monkeypatch):
    monkeypatch.setattr(
        conviction_mod,
        "detect_ticker",
        lambda *a, **k: _VolEvent(3.0, 2.0, "BUYING"),
    )
    res = conviction_mod.score_volume(
        "X", make_bars(n=100), date(2026, 1, 2), delivery=(60.0, 50.0)
    )
    assert res.status == "ok"
    assert "delivery 50.0%→60.0%" in res.evidence


def test_score_volume_delivery_latest_only(monkeypatch):
    monkeypatch.setattr(
        conviction_mod,
        "detect_ticker",
        lambda *a, **k: _VolEvent(float("inf"), float("nan"), "SELLING"),
    )
    res = conviction_mod.score_volume(
        "X", make_bars(n=100), date(2026, 1, 2), delivery=(40.0, None)
    )
    assert res.status == "ok"
    assert "delivery 40.0%" in res.evidence


def test_score_smart_money_us():
    res = conviction_mod._score_smart_money_us(
        {
            "fmp_buy_shares_6m": 100.0,
            "fmp_sell_shares_6m": 100.0,
            "fmp_net_shares_6m": 0.0,
            "fmp_buy_trans_6m": 2,
            "fmp_sell_trans_6m": 2,
        }
    )
    assert res.status == "ok"
    assert res.score == 50.0


def test_score_smart_money_us_zero_total():
    res = conviction_mod._score_smart_money_us({})
    assert res.score == 50.0


def test_score_smart_money_india_no_change():
    res = conviction_mod._score_smart_money_india({"promoter_change": None})
    assert res.status == "skipped"


def test_score_smart_money_india_change_only():
    res = conviction_mod._score_smart_money_india({"promoter_change": 1.0})
    assert res.status == "ok"
    assert res.evidence == "promoter +1.00pp"
