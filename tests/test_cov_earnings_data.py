"""Offline line-coverage tests for ``screener.earnings_backtest.data``.

All network/provider access (yfinance, jugaad_data, openscreener, requests,
NSELive) is stubbed via monkeypatch / injected fake modules. No disk cache is
touched unless the test explicitly drives the cache helpers in a tmp dir.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from screener.earnings_backtest import data as ebd

# Captured before the autouse fixture stubs the module attribute.
_REAL_INSTALL_PATCH = ebd._install_yfinance_timeout_patch


# ── helpers ──────────────────────────────────────────────────────────────


def _earnings_df(dates, eps_est=None, reported=None, surprise=None):
    idx = pd.to_datetime(dates)
    cols = {
        "EPS Estimate": eps_est if eps_est is not None else [float("nan")] * len(idx),
        "Reported EPS": reported if reported is not None else [float("nan")] * len(idx),
        "Surprise(%)": surprise if surprise is not None else [float("nan")] * len(idx),
    }
    return pd.DataFrame(cols, index=idx)


class _FakeTicker:
    """Configurable stand-in for ``yfinance.Ticker``."""

    def __init__(
        self,
        *,
        earnings_dates=None,
        upgrades=None,
        options=None,
        chain=None,
        raise_on=None,
    ):
        self._earnings_dates = earnings_dates
        self._upgrades = upgrades
        self._options = options
        self._chain = chain
        self._raise_on = raise_on or set()

    @property
    def earnings_dates(self):
        if "earnings_dates" in self._raise_on:
            raise RuntimeError("boom earnings")
        return self._earnings_dates

    @property
    def upgrades_downgrades(self):
        if "upgrades" in self._raise_on:
            raise RuntimeError("boom upgrades")
        return self._upgrades

    @property
    def options(self):
        if "options" in self._raise_on:
            raise RuntimeError("boom options")
        return self._options

    def option_chain(self, expiry):
        return self._chain


class _Chain:
    def __init__(self, calls, puts):
        self.calls = calls
        self.puts = puts


@pytest.fixture(autouse=True)
def _no_yf_patch(monkeypatch):
    """Neutralise side-effecting yfinance global helpers for every test."""
    monkeypatch.setattr(ebd, "_install_yfinance_timeout_patch", lambda: None)
    monkeypatch.setattr(ebd, "_configure_yfinance", lambda: None)


@pytest.fixture
def no_disk_cache(monkeypatch):
    monkeypatch.setattr(ebd, "_read_json_cache", lambda path, max_age: (False, None))
    monkeypatch.setattr(ebd, "_write_json_cache", lambda path, value: None)


# ── _install_yfinance_timeout_patch ──────────────────────────────────────


def test_install_yfinance_timeout_patch_already_patched(monkeypatch):
    monkeypatch.setattr(ebd, "_YFINANCE_TIMEOUT_PATCHED", True)
    # Returns early without touching yfinance internals.
    _REAL_INSTALL_PATCH()


def test_install_yfinance_timeout_patch_wraps_get(monkeypatch):
    # Restore the real function (the autouse fixture stubs it to a no-op).
    import yfinance.data as yf_data

    calls = {}

    def orig_get(self, url, params=None, timeout=30):
        calls["get"] = timeout
        return "got"

    def orig_cache_get(self, url, params=None, timeout=30):
        calls["cache_get"] = timeout
        return "cached"

    monkeypatch.setattr(yf_data.YfData, "get", orig_get, raising=False)
    monkeypatch.setattr(yf_data.YfData, "cache_get", orig_cache_get, raising=False)
    monkeypatch.setattr(ebd, "_YFINANCE_TIMEOUT_PATCHED", False)

    _REAL_INSTALL_PATCH()

    # The wrappers cap the timeout to YFINANCE_TIMEOUT_SECONDS.
    dummy = types.SimpleNamespace()
    assert yf_data.YfData.get(dummy, "http://x", timeout=30) == "got"
    assert calls["get"] == ebd.YFINANCE_TIMEOUT_SECONDS
    assert yf_data.YfData.cache_get(dummy, "http://x", timeout=None) == "cached"
    assert calls["cache_get"] == ebd.YFINANCE_TIMEOUT_SECONDS


def test_install_yfinance_timeout_patch_handles_exception(monkeypatch):
    monkeypatch.setattr(ebd, "_YFINANCE_TIMEOUT_PATCHED", False)
    import yfinance.data as yf_data

    monkeypatch.setattr(ebd, "_YFINANCE_TIMEOUT_PATCHED", False)

    # Force the body to raise by removing ``YfData`` -> AttributeError on
    # ``yf_data.YfData.get``, exercising the broad ``except`` handler.
    monkeypatch.delattr(yf_data, "YfData", raising=False)
    _REAL_INSTALL_PATCH()  # swallowed, logged at debug
    # The except branch leaves the flag untouched (still False).
    assert ebd._YFINANCE_TIMEOUT_PATCHED is False


# ── _safe_key / cache path ───────────────────────────────────────────────


def test_safe_key_sanitises():
    assert ebd._safe_key("AAPL.NS_3") == "AAPL.NS_3"
    assert ebd._safe_key("a b/c") == "a_b_c"


def test_json_cache_path():
    p = ebd._json_cache_path("earnings_yf", "AAPL_3")
    assert p.name == "AAPL_3.json"
    assert p.parent.name == "earnings_yf"


# ── _read_json_cache / _write_json_cache ─────────────────────────────────


def test_read_json_cache_missing(tmp_path):
    hit, val = ebd._read_json_cache(tmp_path / "nope.json", 30)
    assert hit is False and val is None


def test_read_json_cache_stale(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"value": 1}))
    old = date.today() - timedelta(days=100)
    import os

    ts = pd.Timestamp(old).timestamp()
    os.utime(p, (ts, ts))
    hit, val = ebd._read_json_cache(p, 30)
    assert hit is False and val is None


def test_read_json_cache_fresh(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"value": [1, 2, 3]}))
    hit, val = ebd._read_json_cache(p, 30)
    assert hit is True and val == [1, 2, 3]


def test_read_json_cache_bad_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("not json{{{")
    hit, val = ebd._read_json_cache(p, 30)
    assert hit is False and val is None


def test_write_json_cache_roundtrip(tmp_path):
    p = tmp_path / "sub" / "c.json"
    ebd._write_json_cache(p, {"a": 1})
    assert json.loads(p.read_text())["value"] == {"a": 1}


def test_write_json_cache_error(monkeypatch):
    # Unwritable path (parent is a file) -> exception swallowed.
    bad = Path("/proc/cannot/write/c.json")
    ebd._write_json_cache(bad, {"a": 1})  # no raise


# ── _jsonable ────────────────────────────────────────────────────────────


def test_jsonable_variants():
    assert ebd._jsonable(None) is None
    assert ebd._jsonable("s") == "s"
    assert ebd._jsonable(True) is True
    assert ebd._jsonable(3) == 3
    assert ebd._jsonable(2.5) == 2.5
    assert ebd._jsonable(float("nan")) is None
    assert ebd._jsonable({"k": 1, 2: "v"}) == {"k": 1, "2": "v"}
    assert ebd._jsonable([1, (2, 3)]) == [1, [2, 3]]
    # numpy scalar has .item()
    assert ebd._jsonable(np.int64(5)) == 5

    # fallback to str()
    class Weird:
        def __str__(self):
            return "weird"

    assert ebd._jsonable(Weird()) == "weird"


# ── _earnings_to_records / _earnings_from_records ────────────────────────


def test_earnings_records_roundtrip():
    df = _earnings_df(["2024-01-15"], eps_est=[1.0], reported=[1.2], surprise=[20.0])
    recs = ebd._earnings_to_records(df)
    assert recs[0]["earnings_date"] == "2024-01-15"
    assert recs[0]["reported_eps"] == 1.2
    back = ebd._earnings_from_records(recs)
    assert back is not None
    assert "EPS Estimate" in back.columns
    assert pd.Timestamp("2024-01-15") in back.index


def test_earnings_from_records_empty():
    assert ebd._earnings_from_records([]) is None


# ── Universe loaders ─────────────────────────────────────────────────────


def test_load_sp500(monkeypatch):
    fake_univ = types.SimpleNamespace(symbols=["AAA", "BBB"])
    universes = types.ModuleType("screener.universes")
    universes.load_current_universe = lambda name: fake_univ
    monkeypatch.setitem(sys.modules, "screener.universes", universes)
    assert ebd.load_sp500() == ["AAA", "BBB"]


def test_load_universe_dispatch(monkeypatch):
    monkeypatch.setattr(ebd, "load_sp500", lambda: ["US"])
    monkeypatch.setattr(ebd, "load_nifty500", lambda: ["IN.NS"])
    assert ebd.load_universe("us") == ["US"]
    assert ebd.load_universe("india") == ["IN.NS"]
    with pytest.raises(ValueError):
        ebd.load_universe("mars")


def test_load_nifty500_from_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(ebd, "CACHE_DIR", cache_dir)
    (cache_dir / "nifty500_symbols.txt").write_text("AAA.NS\nBBB.NS")
    assert ebd.load_nifty500() == ["AAA.NS", "BBB.NS"]


def test_load_nifty500_fetch(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(ebd, "CACHE_DIR", cache_dir)

    csv = "Symbol\nreliance\ntcs\n"
    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda url, headers=None, timeout=None: types.SimpleNamespace(
        text=csv, raise_for_status=lambda: None
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    resilience = types.ModuleType("screener.resilience")
    resilience.call_with_resilience = lambda *a, **kw: a[2]()  # call the fetch fn
    monkeypatch.setitem(sys.modules, "screener.resilience", resilience)

    syms = ebd.load_nifty500()
    assert syms == ["RELIANCE.NS", "TCS.NS"]
    # Cache file was written.
    assert (cache_dir / "nifty500_symbols.txt").exists()


def test_load_nifty500_unavailable(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache2"
    monkeypatch.setattr(ebd, "CACHE_DIR", cache_dir)
    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError())
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    resilience = types.ModuleType("screener.resilience")
    resilience.call_with_resilience = lambda *a, **kw: None  # fallback
    monkeypatch.setitem(sys.modules, "screener.resilience", resilience)
    with pytest.raises(RuntimeError):
        ebd.load_nifty500()


def test_load_nifty500_stale_cache_refetch(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache3"
    cache_dir.mkdir()
    monkeypatch.setattr(ebd, "CACHE_DIR", cache_dir)
    cache_file = cache_dir / "nifty500_symbols.txt"
    cache_file.write_text("OLD.NS")
    import os

    old_ts = pd.Timestamp(date.today() - timedelta(days=30)).timestamp()
    os.utime(cache_file, (old_ts, old_ts))

    csv = "SYMBOL\nnew\n"
    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda url, headers=None, timeout=None: types.SimpleNamespace(
        text=csv, raise_for_status=lambda: None
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    resilience = types.ModuleType("screener.resilience")
    resilience.call_with_resilience = lambda *a, **kw: a[2]()
    monkeypatch.setitem(sys.modules, "screener.resilience", resilience)

    assert ebd.load_nifty500() == ["NEW.NS"]


# ── fetch_earnings_dates_yf ──────────────────────────────────────────────


def test_fetch_earnings_dates_yf_cache_hit(monkeypatch):
    recs = ebd._earnings_to_records(_earnings_df(["2024-01-15"]))
    monkeypatch.setattr(ebd, "_read_json_cache", lambda p, m: (True, recs))
    out = ebd.fetch_earnings_dates_yf("AAPL")
    assert out is not None
    assert pd.Timestamp("2024-01-15") in out.index


def test_fetch_earnings_dates_yf_success(monkeypatch, no_disk_cache):
    recent = (date.today() - timedelta(days=30)).isoformat()
    old = (date.today() - timedelta(days=5000)).isoformat()
    df = _earnings_df(
        [recent, old], eps_est=[1.0, 2.0], reported=[1.1, 2.1], surprise=[10.0, 5.0]
    )
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(earnings_dates=df))
    out = ebd.fetch_earnings_dates_yf("AAPL", years=3)
    assert out is not None
    # Old row beyond cutoff dropped.
    assert pd.Timestamp(recent) in out.index
    assert pd.Timestamp(old) not in out.index


def test_fetch_earnings_dates_yf_empty(monkeypatch, no_disk_cache):
    monkeypatch.setattr(
        ebd.yf, "Ticker", lambda tk: _FakeTicker(earnings_dates=pd.DataFrame())
    )
    assert ebd.fetch_earnings_dates_yf("AAPL") is None


def test_fetch_earnings_dates_yf_none(monkeypatch, no_disk_cache):
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(earnings_dates=None))
    assert ebd.fetch_earnings_dates_yf("AAPL") is None


def test_fetch_earnings_dates_yf_all_filtered_returns_none(monkeypatch, no_disk_cache):
    old = (date.today() - timedelta(days=5000)).isoformat()
    df = _earnings_df([old])
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(earnings_dates=df))
    assert ebd.fetch_earnings_dates_yf("AAPL", years=1) is None


def test_fetch_earnings_dates_yf_exception(monkeypatch, no_disk_cache):
    monkeypatch.setattr(
        ebd.yf, "Ticker", lambda tk: _FakeTicker(raise_on={"earnings_dates"})
    )
    assert ebd.fetch_earnings_dates_yf("AAPL") is None


# ── fetch_earnings_dates_nse ─────────────────────────────────────────────


def _inject_nselive(monkeypatch, instance):
    module = types.ModuleType("jugaad_data.nse")
    module.NSELive = lambda: instance
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", module)
    parent = types.ModuleType("jugaad_data")
    monkeypatch.setitem(sys.modules, "jugaad_data", parent)


def test_fetch_earnings_dates_nse_cache_hit_data(monkeypatch):
    cached = [{"ticker": "AAA.NS", "earnings_date": "2024-01-15", "desc": "x"}]
    monkeypatch.setattr(ebd, "_read_json_cache", lambda p, m: (True, cached))
    out = ebd.fetch_earnings_dates_nse()
    assert out is not None
    assert out["ticker"].iloc[0] == "AAA.NS"


def test_fetch_earnings_dates_nse_cache_hit_empty(monkeypatch):
    monkeypatch.setattr(ebd, "_read_json_cache", lambda p, m: (True, []))
    assert ebd.fetch_earnings_dates_nse() is None


def test_fetch_earnings_dates_nse_success(monkeypatch, no_disk_cache):
    anns = [
        {
            "desc": "Financial Results Q4",
            "attchmntText": "",
            "symbol": "RELIANCE",
            "sort_date": "2024-05-25 10:00:00",
        },
        {
            "desc": "irrelevant board meeting",
            "symbol": "X",
            "sort_date": "2024-05-01",
        },  # filtered out
        {
            "desc": "earnings",
            "attchmntText": "",
            "symbol": "",
            "sort_date": "2024-05-01",
        },  # missing symbol
        {
            "desc": "quarterly result",
            "attchmntText": "",
            "symbol": "TCS",
            "sort_date": "notadate",
        },  # unparseable date -> skipped
    ]
    nse = types.SimpleNamespace(corporate_announcements=lambda: anns)
    _inject_nselive(monkeypatch, nse)
    out = ebd.fetch_earnings_dates_nse()
    assert out is not None
    assert list(out["ticker"]) == ["RELIANCE.NS"]


def test_fetch_earnings_dates_nse_no_announcements(monkeypatch, no_disk_cache):
    nse = types.SimpleNamespace(corporate_announcements=lambda: [])
    _inject_nselive(monkeypatch, nse)
    assert ebd.fetch_earnings_dates_nse() is None


def test_fetch_earnings_dates_nse_no_matching_rows(monkeypatch, no_disk_cache):
    anns = [{"desc": "board meeting", "symbol": "X", "sort_date": "2024-01-01"}]
    nse = types.SimpleNamespace(corporate_announcements=lambda: anns)
    _inject_nselive(monkeypatch, nse)
    assert ebd.fetch_earnings_dates_nse() is None


def test_fetch_earnings_dates_nse_exception(monkeypatch, no_disk_cache):
    # NSELive() raises -> warning + None.
    module = types.ModuleType("jugaad_data.nse")

    def boom():
        raise RuntimeError("nse down")

    module.NSELive = boom
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", module)
    monkeypatch.setitem(sys.modules, "jugaad_data", types.ModuleType("jugaad_data"))
    assert ebd.fetch_earnings_dates_nse() is None


# ── _earnings_rows_for_ticker / _fetch_yf_earnings_rows ──────────────────


def test_earnings_rows_for_ticker_none(monkeypatch):
    monkeypatch.setattr(ebd, "fetch_earnings_dates_yf", lambda t, years: None)
    assert ebd._earnings_rows_for_ticker("AAA", 3) == []


def test_earnings_rows_for_ticker_rows(monkeypatch):
    df = _earnings_df(["2024-01-15"], eps_est=[1.0], reported=[1.1], surprise=[10.0])
    monkeypatch.setattr(ebd, "fetch_earnings_dates_yf", lambda t, years: df)
    rows = ebd._earnings_rows_for_ticker("AAA", 3)
    assert rows[0]["ticker"] == "AAA"
    assert rows[0]["earnings_date"] == date(2024, 1, 15)


def test_fetch_yf_earnings_rows(monkeypatch):
    def fake(t, years):
        return _earnings_df(["2024-01-15"]) if t == "GOOD" else None

    monkeypatch.setattr(ebd, "fetch_earnings_dates_yf", fake)
    rows = ebd._fetch_yf_earnings_rows(["GOOD", "BAD"], 3, 50)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "GOOD"


def test_fetch_yf_earnings_rows_handles_future_exception(monkeypatch):
    def boom(t, years):
        raise RuntimeError("worker boom")

    monkeypatch.setattr(ebd, "_earnings_rows_for_ticker", boom)
    rows = ebd._fetch_yf_earnings_rows(["AAA"], 3, 50)
    assert rows == []


# ── fetch_earnings_dates_openscreener ────────────────────────────────────


def _inject_openscreener(monkeypatch, stock_cls):
    module = types.ModuleType("openscreener")
    module.Stock = stock_cls
    monkeypatch.setitem(sys.modules, "openscreener", module)
    insiders = types.ModuleType("screener.insiders")
    insiders._HttpScraper = lambda: object()
    monkeypatch.setitem(sys.modules, "screener.insiders", insiders)


def _stock_factory(payload):
    class _Stock:
        def __init__(self, symbol, scraper=None):
            self.symbol = symbol

        def fetch(self, section):
            return payload

    return _Stock


def test_openscreener_cache_hit(monkeypatch):
    recs = [
        {
            "earnings_date": "2024-05-30",
            "period_end": "2024-03-31",
            "eps_estimate": None,
            "reported_eps": 5.0,
            "surprise_pct": None,
        }
    ]
    monkeypatch.setattr(ebd, "_read_json_cache", lambda p, m: (True, recs))
    out = ebd.fetch_earnings_dates_openscreener("RELIANCE.NS")
    assert out is not None
    assert pd.Timestamp("2024-05-30") in out.index


def test_openscreener_success(monkeypatch, no_disk_cache):
    payload = {
        "quarterly_results": [
            {"date": "Mar 2024", "eps": 12.0},
            {"date": "bad-label", "eps": 1.0},  # unparseable -> skipped
            {"eps": 9.0},  # no date -> skipped
            "not a dict",  # skipped
            {"date": "Jan 2000", "eps": 0.0},  # before cutoff -> skipped
        ]
    }
    _inject_openscreener(monkeypatch, _stock_factory(payload))
    out = ebd.fetch_earnings_dates_openscreener("RELIANCE.NS", years=5)
    assert out is not None
    expected = pd.Timestamp("2024-03-31") + pd.Timedelta(
        days=ebd.INDIA_EARNINGS_FILING_LAG_DAYS
    )
    assert expected in out.index
    assert len(out.index) == 1


def test_openscreener_payload_not_dict(monkeypatch, no_disk_cache):
    _inject_openscreener(monkeypatch, _stock_factory(["nope"]))
    assert ebd.fetch_earnings_dates_openscreener("R.NS") is None


def test_openscreener_no_quarterly(monkeypatch, no_disk_cache):
    _inject_openscreener(monkeypatch, _stock_factory({"quarterly_results": []}))
    assert ebd.fetch_earnings_dates_openscreener("R.NS") is None


def test_openscreener_quarterly_not_list(monkeypatch, no_disk_cache):
    _inject_openscreener(monkeypatch, _stock_factory({"quarterly_results": {"x": 1}}))
    assert ebd.fetch_earnings_dates_openscreener("R.NS") is None


def test_openscreener_exception(monkeypatch, no_disk_cache):
    class _Boom:
        def __init__(self, symbol, scraper=None):
            raise RuntimeError("osc down")

    _inject_openscreener(monkeypatch, _Boom)
    assert ebd.fetch_earnings_dates_openscreener("R.NS") is None


# ── _openscreener_earnings_rows_for_ticker / _fetch_openscreener_rows ─────


def test_openscreener_rows_for_ticker_none(monkeypatch):
    monkeypatch.setattr(ebd, "fetch_earnings_dates_openscreener", lambda t, years: None)
    assert ebd._openscreener_earnings_rows_for_ticker("R.NS", 3) == []


def test_openscreener_rows_for_ticker_rows(monkeypatch):
    df = pd.DataFrame(
        {
            "period_end": ["2024-03-31"],
            "EPS Estimate": [float("nan")],
            "Reported EPS": [12.0],
            "Surprise(%)": [float("nan")],
        },
        index=pd.to_datetime(["2024-05-30"]),
    )
    monkeypatch.setattr(ebd, "fetch_earnings_dates_openscreener", lambda t, years: df)
    rows = ebd._openscreener_earnings_rows_for_ticker("R.NS", 3)
    assert rows[0]["ticker"] == "R.NS"
    assert rows[0]["period_end"] == "2024-03-31"
    assert rows[0]["earnings_date"] == date(2024, 5, 30)


def test_fetch_openscreener_rows(monkeypatch):
    df = pd.DataFrame(
        {
            "period_end": ["2024-03-31"],
            "EPS Estimate": [float("nan")],
            "Reported EPS": [12.0],
            "Surprise(%)": [float("nan")],
        },
        index=pd.to_datetime(["2024-05-30"]),
    )
    monkeypatch.setattr(
        ebd,
        "fetch_earnings_dates_openscreener",
        lambda t, years: df if t == "GOOD.NS" else None,
    )
    rows = ebd._fetch_openscreener_earnings_rows(["GOOD.NS", "BAD.NS"], 3, 50)
    assert len(rows) == 1


def test_fetch_openscreener_rows_exception(monkeypatch):
    monkeypatch.setattr(
        ebd,
        "_openscreener_earnings_rows_for_ticker",
        lambda t, years: (_ for _ in ()).throw(RuntimeError()),
    )
    rows = ebd._fetch_openscreener_earnings_rows(["A.NS"], 3, 50)
    assert rows == []


# ── collect_earnings_events ──────────────────────────────────────────────


def test_collect_us(monkeypatch):
    monkeypatch.setattr(
        ebd,
        "_fetch_yf_earnings_rows",
        lambda batch, years, bs: [
            {
                "ticker": t,
                "earnings_date": date(2024, 1, 1),
                "eps_estimate": 1.0,
                "reported_eps": 1.1,
                "surprise_pct": 10.0,
            }
            for t in batch
        ],
    )
    out = ebd.collect_earnings_events(["AAA", "BBB"], batch_size=1, market="us")
    assert set(out["ticker"]) == {"AAA", "BBB"}


def test_collect_us_empty(monkeypatch):
    monkeypatch.setattr(ebd, "_fetch_yf_earnings_rows", lambda *a, **kw: [])
    out = ebd.collect_earnings_events([], market="us")
    assert out.empty
    assert list(out.columns) == [
        "ticker",
        "earnings_date",
        "eps_estimate",
        "reported_eps",
        "surprise_pct",
    ]


def test_collect_india_with_nse_and_dedup(monkeypatch):
    nse_date = pd.Timestamp("2024-05-25")
    nse_df = pd.DataFrame(
        {
            "ticker": ["RELIANCE.NS", "OTHER.NS"],
            "earnings_date": [nse_date, nse_date],
            "desc": ["x", "y"],
        }
    )
    monkeypatch.setattr(ebd, "fetch_earnings_dates_nse", lambda: nse_df)
    # openscreener returns Mar-2024 (deduped) and Dec-2023 (kept) for RELIANCE.
    osc_rows = [
        {
            "ticker": "RELIANCE.NS",
            "earnings_date": date(2024, 5, 30),
            "period_end": "2024-03-31",
            "eps_estimate": float("nan"),
            "reported_eps": 12.0,
            "surprise_pct": float("nan"),
        },
        {
            "ticker": "RELIANCE.NS",
            "earnings_date": date(2024, 2, 29),
            "period_end": "2023-12-31",
            "eps_estimate": float("nan"),
            "reported_eps": 10.0,
            "surprise_pct": float("nan"),
        },
        {
            "ticker": "RELIANCE.NS",
            "earnings_date": date(2024, 1, 1),
            "period_end": None,
            "eps_estimate": float("nan"),
            "reported_eps": 1.0,
            "surprise_pct": float("nan"),
        },  # pe None branch
    ]
    monkeypatch.setattr(
        ebd, "_fetch_openscreener_earnings_rows", lambda batch, years, bs: osc_rows
    )
    out = ebd.collect_earnings_events(
        ["RELIANCE.NS"], years=5, batch_size=50, market="india"
    )
    rel = out[out["ticker"] == "RELIANCE.NS"]
    dates = set(pd.to_datetime(rel["earnings_date"]))
    assert nse_date in dates
    # Mar-2024 quarter osc estimate deduped away.
    assert pd.Timestamp("2024-05-30") not in dates
    # Dec-2023 estimate retained.
    assert pd.Timestamp("2024-02-29") in dates
    # pe-None row retained.
    assert pd.Timestamp("2024-01-01") in dates


def test_collect_india_no_nse(monkeypatch):
    monkeypatch.setattr(ebd, "fetch_earnings_dates_nse", lambda: None)
    monkeypatch.setattr(
        ebd,
        "_fetch_openscreener_earnings_rows",
        lambda batch, years, bs: [
            {
                "ticker": "R.NS",
                "earnings_date": date(2024, 5, 30),
                "period_end": "2024-03-31",
                "eps_estimate": float("nan"),
                "reported_eps": 12.0,
                "surprise_pct": float("nan"),
            }
        ],
    )
    out = ebd.collect_earnings_events(["R.NS"], market="india")
    assert "R.NS" in set(out["ticker"])


# ── fetch_analyst_sentiment ──────────────────────────────────────────────


def test_analyst_sentiment_india_none():
    assert ebd.fetch_analyst_sentiment("X.NS", market="india") is None


def test_analyst_sentiment_cache_hit(monkeypatch):
    monkeypatch.setattr(ebd, "_read_json_cache", lambda p, m: (True, {"net": 3}))
    assert ebd.fetch_analyst_sentiment("AAPL") == {"net": 3}


def test_analyst_sentiment_action_col(monkeypatch, no_disk_cache):
    ud = pd.DataFrame({"Action": ["up", "up", "reit", "down"]})
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(upgrades=ud))
    out = ebd.fetch_analyst_sentiment("AAPL")
    assert out["upgrades"] == 2.5  # 2 up + 0.5*1 reit
    assert out["downgrades"] == 1
    assert out["net"] == 1.5
    assert out["grade_counts"]


def test_analyst_sentiment_tograde_col(monkeypatch, no_disk_cache):
    ud = pd.DataFrame({"ToGrade": ["Buy", "Outperform", "Sell"]})
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(upgrades=ud))
    out = ebd.fetch_analyst_sentiment("AAPL")
    assert out["upgrades"] == 2
    assert out["downgrades"] == 1
    assert out["grade_counts"] == {}


def test_analyst_sentiment_unknown_cols(monkeypatch, no_disk_cache):
    ud = pd.DataFrame({"Other": [1, 2]})
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(upgrades=ud))
    assert ebd.fetch_analyst_sentiment("AAPL") is None


def test_analyst_sentiment_empty(monkeypatch, no_disk_cache):
    monkeypatch.setattr(
        ebd.yf, "Ticker", lambda tk: _FakeTicker(upgrades=pd.DataFrame())
    )
    assert ebd.fetch_analyst_sentiment("AAPL") is None


def test_analyst_sentiment_none_ud(monkeypatch, no_disk_cache):
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(upgrades=None))
    assert ebd.fetch_analyst_sentiment("AAPL") is None


def test_analyst_sentiment_exception(monkeypatch, no_disk_cache):
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(raise_on={"upgrades"}))
    assert ebd.fetch_analyst_sentiment("AAPL") is None


# ── fetch_iv_sentiment_yf ────────────────────────────────────────────────


def _opt_df(volume=None, oi=None, iv=None, n=2):
    cols = {}
    if volume is not None:
        cols["volume"] = volume
    if oi is not None:
        cols["openInterest"] = oi
    if iv is not None:
        cols["impliedVolatility"] = iv
    if not cols:
        return pd.DataFrame(index=range(n))
    return pd.DataFrame(cols)


def test_iv_yf_cache_hit(monkeypatch):
    monkeypatch.setattr(ebd, "_read_json_cache", lambda p, m: (True, {"pc_ratio": 1.0}))
    assert ebd.fetch_iv_sentiment_yf("AAPL") == {"pc_ratio": 1.0}


def test_iv_yf_success_with_volume(monkeypatch, no_disk_cache):
    today = date.today()
    far = (today + timedelta(days=10)).isoformat()
    calls = _opt_df(volume=[100, 200], oi=[10, 20], iv=[0.40, 0.42])
    puts = _opt_df(volume=[50, 60], oi=[5, 6], iv=[0.50, 0.52])
    chain = _Chain(calls, puts)
    monkeypatch.setattr(
        ebd.yf, "Ticker", lambda tk: _FakeTicker(options=[far], chain=chain)
    )
    out = ebd.fetch_iv_sentiment_yf("AAPL")
    assert out["total_calls"] == 300
    assert out["total_puts"] == 110
    assert out["pc_ratio"] == round(110 / 300, 4)
    assert out["median_iv"] > 0


def test_iv_yf_no_volume_uses_oi(monkeypatch, no_disk_cache):
    today = date.today()
    near = today.isoformat()  # < 5 days -> target_expiry stays None, uses dates[0]
    calls = _opt_df(oi=[0, 0])  # no volume col, total_calls = len(calls) = 2
    puts = _opt_df(oi=[5, 6])
    # Force total_calls path: volume col absent -> total_calls = len(calls)=2 (>0)
    chain = _Chain(calls, puts)
    monkeypatch.setattr(
        ebd.yf, "Ticker", lambda tk: _FakeTicker(options=[near], chain=chain)
    )
    out = ebd.fetch_iv_sentiment_yf("AAPL")
    assert out is not None


def test_iv_yf_zero_calls_oi_branch(monkeypatch, no_disk_cache):
    today = date.today()
    far = (today + timedelta(days=10)).isoformat()
    # volume present but all zero -> total_calls == 0 -> OI-based pc_ratio.
    calls = _opt_df(volume=[0, 0], oi=[10, 20])
    puts = _opt_df(volume=[0, 0], oi=[5, 5])
    chain = _Chain(calls, puts)
    monkeypatch.setattr(
        ebd.yf, "Ticker", lambda tk: _FakeTicker(options=[far], chain=chain)
    )
    out = ebd.fetch_iv_sentiment_yf("AAPL")
    assert out["pc_ratio"] == round(10 / 30, 4)


def test_iv_yf_no_options(monkeypatch, no_disk_cache):
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(options=[]))
    assert ebd.fetch_iv_sentiment_yf("AAPL") is None


def test_iv_yf_empty_chain(monkeypatch, no_disk_cache):
    today = date.today()
    far = (today + timedelta(days=10)).isoformat()
    chain = _Chain(pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(
        ebd.yf, "Ticker", lambda tk: _FakeTicker(options=[far], chain=chain)
    )
    assert ebd.fetch_iv_sentiment_yf("AAPL") is None


def test_iv_yf_no_iv_cols(monkeypatch, no_disk_cache):
    today = date.today()
    far = (today + timedelta(days=10)).isoformat()
    calls = _opt_df(volume=[1, 1])
    puts = _opt_df(volume=[1, 1])
    chain = _Chain(calls, puts)
    monkeypatch.setattr(
        ebd.yf, "Ticker", lambda tk: _FakeTicker(options=[far], chain=chain)
    )
    out = ebd.fetch_iv_sentiment_yf("AAPL")
    assert out["median_iv"] != out["median_iv"] or np.isnan(out["median_iv"])  # nan


def test_iv_yf_exception(monkeypatch, no_disk_cache):
    monkeypatch.setattr(ebd.yf, "Ticker", lambda tk: _FakeTicker(raise_on={"options"}))
    assert ebd.fetch_iv_sentiment_yf("AAPL") is None


# ── fetch_iv_sentiment_nse ───────────────────────────────────────────────


def test_iv_nse_cache_hit(monkeypatch):
    monkeypatch.setattr(ebd, "_read_json_cache", lambda p, m: (True, {"pc_ratio": 2.0}))
    assert ebd.fetch_iv_sentiment_nse("RELIANCE") == {"pc_ratio": 2.0}


def test_iv_nse_success(monkeypatch, no_disk_cache):
    oc = {
        "records": {
            "data": [
                {
                    "CE": {
                        "openInterest": 100,
                        "totalTradedVolume": 10,
                        "impliedVolatility": 20.0,
                    },
                    "PE": {
                        "openInterest": 50,
                        "totalTradedVolume": 5,
                        "impliedVolatility": 25.0,
                    },
                },
                {
                    "CE": {
                        "openInterest": 0,
                        "totalTradedVolume": 0,
                        "impliedVolatility": 0,
                    },
                    "PE": {},
                },  # empty PE
            ]
        }
    }
    nse = types.SimpleNamespace(equities_option_chain=lambda s: oc)
    _inject_nselive(monkeypatch, nse)
    out = ebd.fetch_iv_sentiment_nse("RELIANCE")
    assert out["pc_ratio"] == round(50 / 100, 4)
    assert out["median_iv"] is not None
    assert out["total_calls"] == 10
    assert out["total_puts"] == 5


def test_iv_nse_zero_ce_oi(monkeypatch, no_disk_cache):
    oc = {
        "records": {
            "data": [
                {
                    "CE": {"openInterest": 0, "totalTradedVolume": 0},
                    "PE": {"openInterest": 10, "totalTradedVolume": 1},
                },
            ]
        }
    }
    nse = types.SimpleNamespace(equities_option_chain=lambda s: oc)
    _inject_nselive(monkeypatch, nse)
    out = ebd.fetch_iv_sentiment_nse("RELIANCE")
    assert out["pc_ratio"] == 1.0  # ce oi == 0 -> default 1.0
    assert out["median_iv"] is None  # no iv vals -> nan -> None


def test_iv_nse_iv_outlier_filtered(monkeypatch, no_disk_cache):
    oc = {
        "records": {
            "data": [
                {
                    "CE": {
                        "openInterest": 1,
                        "totalTradedVolume": 1,
                        "impliedVolatility": 600.0,
                    },
                    "PE": {
                        "openInterest": 1,
                        "totalTradedVolume": 1,
                        "impliedVolatility": 600.0,
                    },
                },
            ]
        }
    }
    nse = types.SimpleNamespace(equities_option_chain=lambda s: oc)
    _inject_nselive(monkeypatch, nse)
    out = ebd.fetch_iv_sentiment_nse("RELIANCE")
    assert out["median_iv"] is None  # 600 filtered out (>=500)


def test_iv_nse_no_oc(monkeypatch, no_disk_cache):
    nse = types.SimpleNamespace(equities_option_chain=lambda s: None)
    _inject_nselive(monkeypatch, nse)
    assert ebd.fetch_iv_sentiment_nse("RELIANCE") is None


def test_iv_nse_no_records_key(monkeypatch, no_disk_cache):
    nse = types.SimpleNamespace(equities_option_chain=lambda s: {"foo": 1})
    _inject_nselive(monkeypatch, nse)
    assert ebd.fetch_iv_sentiment_nse("RELIANCE") is None


def test_iv_nse_empty_data(monkeypatch, no_disk_cache):
    nse = types.SimpleNamespace(
        equities_option_chain=lambda s: {"records": {"data": []}}
    )
    _inject_nselive(monkeypatch, nse)
    assert ebd.fetch_iv_sentiment_nse("RELIANCE") is None


def test_iv_nse_exception(monkeypatch, no_disk_cache):
    module = types.ModuleType("jugaad_data.nse")
    module.NSELive = lambda: (_ for _ in ()).throw(RuntimeError("nse boom"))
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", module)
    monkeypatch.setitem(sys.modules, "jugaad_data", types.ModuleType("jugaad_data"))
    assert ebd.fetch_iv_sentiment_nse("RELIANCE") is None


# ── fetch_iv_sentiment dispatch ──────────────────────────────────────────


def test_iv_dispatch_india(monkeypatch):
    monkeypatch.setattr(ebd, "fetch_iv_sentiment_nse", lambda symbol: {"sym": symbol})
    assert ebd.fetch_iv_sentiment("RELIANCE.NS", market="india") == {"sym": "RELIANCE"}


def test_iv_dispatch_us(monkeypatch):
    monkeypatch.setattr(ebd, "fetch_iv_sentiment_yf", lambda ticker: {"t": ticker})
    assert ebd.fetch_iv_sentiment("AAPL") == {"t": "AAPL"}


# ── fetch_price_data ─────────────────────────────────────────────────────


def test_fetch_price_data_with_fetcher():
    from tests.conftest import make_bars, StubPriceFetcher

    bars = make_bars(n=10)
    fetcher = StubPriceFetcher({"AAA": bars, "EMPTY": pd.DataFrame()})
    out = ebd.fetch_price_data(
        ["AAA", "EMPTY"],
        date(2024, 1, 1),
        date(2024, 3, 1),
        fetcher=fetcher,
        batch_size=1,
    )
    assert "AAA" in out
    # Empty frame kept in all_data (update happens before the cleanup of local).
    assert "EMPTY" in out


def test_fetch_price_data_default_fetcher(monkeypatch):
    captured = {}

    class _Fetcher:
        def __init__(self, auto_adjust=True):
            captured["auto_adjust"] = auto_adjust

        def fetch(self, batch, start, end):
            return {t: pd.DataFrame({"close": [1.0]}) for t in batch}

    monkeypatch.setattr(ebd, "YFinancePriceFetcher", _Fetcher)
    out = ebd.fetch_price_data(["AAA"], date(2024, 1, 1), date(2024, 2, 1))
    assert "AAA" in out
    assert captured["auto_adjust"] is True
