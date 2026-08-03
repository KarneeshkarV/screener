"""Behavioral tests for earnings-date & IV-sentiment acquisition.

Covers ``screener.earnings_backtest.earnings_dates`` and ``.sentiment``: the
point-in-time date resolution across yfinance/NSE/openscreener/FMP sources, the
Indian filing-lag floor, multi-source de-duplication and enrichment, and the
implied-volatility / analyst sentiment scoring. All provider access (yfinance,
jugaad_data, openscreener, requests, NSELive) is stubbed via monkeypatch or
injected fakes so the tests are deterministic and never touch the network.
"""

from __future__ import annotations

import sys
import types
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from screener.earnings_backtest import data as ebd
from screener.earnings_backtest import earnings_dates, sentiment
from screener.earnings_backtest.common import jsonable as _jsonable

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
    monkeypatch.setattr(earnings_dates, "_configure_yfinance", lambda: None)
    monkeypatch.setattr(sentiment, "_configure_yfinance", lambda: None)


@pytest.fixture
def no_disk_cache(monkeypatch):
    def direct(*args, **kwargs):
        return kwargs["fetch"]()

    monkeypatch.setattr(earnings_dates, "cached_json_call", direct)
    monkeypatch.setattr(sentiment, "cached_json_call", direct)


# ── _jsonable ────────────────────────────────────────────────────────────


def test_jsonable_variants():
    assert _jsonable(None) is None
    assert _jsonable("s") == "s"
    assert _jsonable(True) is True
    assert _jsonable(3) == 3
    assert _jsonable(2.5) == 2.5
    assert _jsonable(float("nan")) is None
    assert _jsonable({"k": 1, 2: "v"}) == {"k": 1, "2": "v"}
    assert _jsonable([1, (2, 3)]) == [1, [2, 3]]
    # numpy scalar has .item()
    assert _jsonable(np.int64(5)) == 5

    # fallback to str()
    class Weird:
        def __str__(self):
            return "weird"

    assert _jsonable(Weird()) == "weird"


# ── _earnings_to_records / _earnings_from_records ────────────────────────


def test_earnings_records_roundtrip():
    df = _earnings_df(["2024-01-15"], eps_est=[1.0], reported=[1.2], surprise=[20.0])
    recs = earnings_dates._earnings_to_records(df)
    assert recs[0]["earnings_date"] == "2024-01-15"
    assert recs[0]["reported_eps"] == 1.2
    back = earnings_dates._earnings_from_records(recs)
    assert back is not None
    assert "EPS Estimate" in back.columns
    assert pd.Timestamp("2024-01-15") in back.index


def test_earnings_from_records_empty():
    assert earnings_dates._earnings_from_records([]) is None


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


def test_load_nifty500_delegates_to_universes(monkeypatch):
    fake_univ = types.SimpleNamespace(symbols=("AAA.NS", "BBB.NS"))
    universes = types.ModuleType("screener.universes")
    universes.load_current_universe = lambda name: fake_univ
    monkeypatch.setitem(sys.modules, "screener.universes", universes)
    assert ebd.load_nifty500() == ["AAA.NS", "BBB.NS"]


# ── fetch_earnings_dates_yf ──────────────────────────────────────────────


def test_fetch_earnings_dates_yf_cache_hit(monkeypatch):
    recs = earnings_dates._earnings_to_records(_earnings_df(["2024-01-15"]))
    monkeypatch.setattr(earnings_dates, "cached_json_call", lambda *a, **kw: recs)
    out = earnings_dates.fetch_earnings_dates_yf("AAPL")
    assert out is not None
    assert pd.Timestamp("2024-01-15") in out.index


def test_fetch_earnings_dates_yf_success(monkeypatch, no_disk_cache):
    recent = (date.today() - timedelta(days=30)).isoformat()
    old = (date.today() - timedelta(days=5000)).isoformat()
    df = _earnings_df(
        [recent, old], eps_est=[1.0, 2.0], reported=[1.1, 2.1], surprise=[10.0, 5.0]
    )
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(earnings_dates=df))
    out = earnings_dates.fetch_earnings_dates_yf("AAPL", years=3)
    assert out is not None
    # Old row beyond cutoff dropped.
    assert pd.Timestamp(recent) in out.index
    assert pd.Timestamp(old) not in out.index


def test_fetch_earnings_dates_yf_empty(monkeypatch, no_disk_cache):
    monkeypatch.setattr(
        yf, "Ticker", lambda tk: _FakeTicker(earnings_dates=pd.DataFrame())
    )
    assert earnings_dates.fetch_earnings_dates_yf("AAPL") is None


def test_fetch_earnings_dates_yf_none(monkeypatch, no_disk_cache):
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(earnings_dates=None))
    assert earnings_dates.fetch_earnings_dates_yf("AAPL") is None


def test_fetch_earnings_dates_yf_all_filtered_returns_none(monkeypatch, no_disk_cache):
    old = (date.today() - timedelta(days=5000)).isoformat()
    df = _earnings_df([old])
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(earnings_dates=df))
    assert earnings_dates.fetch_earnings_dates_yf("AAPL", years=1) is None


def test_fetch_earnings_dates_yf_exception(monkeypatch, no_disk_cache):
    monkeypatch.setattr(
        yf, "Ticker", lambda tk: _FakeTicker(raise_on={"earnings_dates"})
    )
    assert earnings_dates.fetch_earnings_dates_yf("AAPL") is None


# ── fetch_earnings_dates_nse ─────────────────────────────────────────────


def _inject_nselive(monkeypatch, instance):
    module = types.ModuleType("jugaad_data.nse")
    module.NSELive = lambda: instance
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", module)
    parent = types.ModuleType("jugaad_data")
    monkeypatch.setitem(sys.modules, "jugaad_data", parent)


def test_fetch_earnings_dates_nse_cache_hit_data(monkeypatch):
    cached = [{"ticker": "AAA.NS", "earnings_date": "2024-01-15", "desc": "x"}]
    monkeypatch.setattr(earnings_dates, "cached_json_call", lambda *a, **kw: cached)
    out = earnings_dates.fetch_earnings_dates_nse()
    assert out is not None
    assert out["ticker"].iloc[0] == "AAA.NS"


def test_fetch_earnings_dates_nse_cache_hit_empty(monkeypatch):
    monkeypatch.setattr(earnings_dates, "cached_json_call", lambda *a, **kw: [])
    assert earnings_dates.fetch_earnings_dates_nse() is None


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
    out = earnings_dates.fetch_earnings_dates_nse()
    assert out is not None
    assert list(out["ticker"]) == ["RELIANCE.NS"]


def test_fetch_earnings_dates_nse_no_announcements(monkeypatch, no_disk_cache):
    nse = types.SimpleNamespace(corporate_announcements=list)
    _inject_nselive(monkeypatch, nse)
    assert earnings_dates.fetch_earnings_dates_nse() is None


def test_fetch_earnings_dates_nse_no_matching_rows(monkeypatch, no_disk_cache):
    anns = [{"desc": "board meeting", "symbol": "X", "sort_date": "2024-01-01"}]
    nse = types.SimpleNamespace(corporate_announcements=lambda: anns)
    _inject_nselive(monkeypatch, nse)
    assert earnings_dates.fetch_earnings_dates_nse() is None


def test_fetch_earnings_dates_nse_exception(monkeypatch, no_disk_cache):
    # NSELive() raises -> warning + None.
    module = types.ModuleType("jugaad_data.nse")

    def boom():
        raise RuntimeError("nse down")

    module.NSELive = boom
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", module)
    monkeypatch.setitem(sys.modules, "jugaad_data", types.ModuleType("jugaad_data"))
    assert earnings_dates.fetch_earnings_dates_nse() is None


# ── _earnings_rows_for_ticker / _fetch_yf_earnings_rows ──────────────────


def test_earnings_rows_for_ticker_none(monkeypatch):
    monkeypatch.setattr(
        earnings_dates, "fetch_earnings_dates_yf", lambda t, years: None
    )
    assert earnings_dates._earnings_rows_for_ticker("AAA", 3) == []


def test_earnings_rows_for_ticker_rows(monkeypatch):
    df = _earnings_df(["2024-01-15"], eps_est=[1.0], reported=[1.1], surprise=[10.0])
    monkeypatch.setattr(earnings_dates, "fetch_earnings_dates_yf", lambda t, years: df)
    rows = earnings_dates._earnings_rows_for_ticker("AAA", 3)
    assert rows[0]["ticker"] == "AAA"
    assert rows[0]["earnings_date"] == date(2024, 1, 15)


def test_fetch_yf_earnings_rows(monkeypatch):
    def fake(t, years):
        return _earnings_df(["2024-01-15"]) if t == "GOOD" else None

    monkeypatch.setattr(earnings_dates, "fetch_earnings_dates_yf", fake)
    rows = earnings_dates._fetch_yf_earnings_rows(["GOOD", "BAD"], 3, 50)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "GOOD"


def test_fetch_yf_earnings_rows_handles_future_exception(monkeypatch):
    def boom(t, years):
        raise RuntimeError("worker boom")

    monkeypatch.setattr(earnings_dates, "_earnings_rows_for_ticker", boom)
    rows = earnings_dates._fetch_yf_earnings_rows(["AAA"], 3, 50)
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
    monkeypatch.setattr(earnings_dates, "cached_json_call", lambda *a, **kw: recs)
    out = earnings_dates.fetch_earnings_dates_openscreener("RELIANCE.NS")
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
    out = earnings_dates.fetch_earnings_dates_openscreener("RELIANCE.NS", years=5)
    assert out is not None
    expected = pd.Timestamp("2024-03-31") + pd.Timedelta(
        days=earnings_dates.INDIA_EARNINGS_FILING_LAG_DAYS
    )
    assert expected in out.index
    assert len(out.index) == 1


def test_openscreener_payload_not_dict(monkeypatch, no_disk_cache):
    _inject_openscreener(monkeypatch, _stock_factory(["nope"]))
    assert earnings_dates.fetch_earnings_dates_openscreener("R.NS") is None


def test_openscreener_no_quarterly(monkeypatch, no_disk_cache):
    _inject_openscreener(monkeypatch, _stock_factory({"quarterly_results": []}))
    assert earnings_dates.fetch_earnings_dates_openscreener("R.NS") is None


def test_openscreener_quarterly_not_list(monkeypatch, no_disk_cache):
    _inject_openscreener(monkeypatch, _stock_factory({"quarterly_results": {"x": 1}}))
    assert earnings_dates.fetch_earnings_dates_openscreener("R.NS") is None


def test_openscreener_exception(monkeypatch, no_disk_cache):
    class _Boom:
        def __init__(self, symbol, scraper=None):
            raise RuntimeError("osc down")

    _inject_openscreener(monkeypatch, _Boom)
    assert earnings_dates.fetch_earnings_dates_openscreener("R.NS") is None


# ── _openscreener_earnings_rows_for_ticker / _fetch_openscreener_rows ─────


def test_openscreener_rows_for_ticker_none(monkeypatch):
    monkeypatch.setattr(
        earnings_dates, "fetch_earnings_dates_openscreener", lambda t, years: None
    )
    assert earnings_dates._openscreener_earnings_rows_for_ticker("R.NS", 3) == []


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
    monkeypatch.setattr(
        earnings_dates, "fetch_earnings_dates_openscreener", lambda t, years: df
    )
    rows = earnings_dates._openscreener_earnings_rows_for_ticker("R.NS", 3)
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
        earnings_dates,
        "fetch_earnings_dates_openscreener",
        lambda t, years: df if t == "GOOD.NS" else None,
    )
    rows = earnings_dates._fetch_openscreener_earnings_rows(
        ["GOOD.NS", "BAD.NS"], 3, 50
    )
    assert len(rows) == 1


def test_fetch_openscreener_rows_exception(monkeypatch):
    monkeypatch.setattr(
        earnings_dates,
        "_openscreener_earnings_rows_for_ticker",
        lambda t, years: (_ for _ in ()).throw(RuntimeError()),
    )
    rows = earnings_dates._fetch_openscreener_earnings_rows(["A.NS"], 3, 50)
    assert rows == []


# ── fetch_earnings_dates_fmp ─────────────────────────────────────────────


def _inject_fmp(monkeypatch, *, api_key="k", payload=None, boom=False):
    """Patch screener.fmp key resolution + client for the FMP earnings source."""
    from screener import fmp

    monkeypatch.setattr(fmp, "resolve_api_key", lambda: api_key)

    class _Client:
        def __init__(self, key, *, base_url=None, **kwargs):
            assert key == api_key

        def get(self, path, params=None):
            if boom:
                raise RuntimeError("fmp down")
            assert path.startswith("historical/earning_calendar/")
            return payload

    monkeypatch.setattr(fmp, "FmpClient", _Client)


def test_fetch_earnings_dates_fmp_cache_hit(monkeypatch):
    recs = earnings_dates._earnings_to_records(
        _earnings_df(["2024-01-15"], eps_est=[1.0], reported=[1.2], surprise=[20.0])
    )
    monkeypatch.setattr(earnings_dates, "cached_json_call", lambda *a, **kw: recs)
    out = earnings_dates.fetch_earnings_dates_fmp("RELIANCE.NS")
    assert out is not None
    assert pd.Timestamp("2024-01-15") in out.index


def test_fetch_earnings_dates_fmp_success(monkeypatch, no_disk_cache):
    recent = (date.today() - timedelta(days=30)).isoformat()
    old = (date.today() - timedelta(days=5000)).isoformat()
    payload = [
        {"date": recent, "eps": 1.2, "epsEstimated": 1.0},  # surprise 20%
        {"date": recent, "eps": 1.2, "epsEstimated": 0},  # zero estimate -> None
        {"date": recent, "eps": None, "epsEstimated": 1.0},  # no eps -> None
        {"date": old, "eps": 2.0, "epsEstimated": 1.0},  # before cutoff
        {"date": "", "eps": 1.0, "epsEstimated": 1.0},  # missing date
        {"date": "not-a-date", "eps": 1.0, "epsEstimated": 1.0},  # unparseable
        "not a dict",  # skipped
    ]
    _inject_fmp(monkeypatch, payload=payload)
    out = earnings_dates.fetch_earnings_dates_fmp("RELIANCE.NS", years=3)
    assert out is not None
    assert len(out) == 3
    surprises = out["Surprise(%)"].tolist()
    assert surprises[0] == pytest.approx(20.0)
    assert pd.isna(surprises[1]) and pd.isna(surprises[2])
    assert pd.Timestamp(old) not in out.index


def test_fetch_earnings_dates_fmp_negative_estimate_surprise(
    monkeypatch, no_disk_cache
):
    recent = (date.today() - timedelta(days=30)).isoformat()
    # eps 0.5 vs estimate -1.0: (0.5 - -1.0)/|-1.0| = +150% (abs denominator).
    payload = [{"date": recent, "eps": 0.5, "epsEstimated": -1.0}]
    _inject_fmp(monkeypatch, payload=payload)
    out = earnings_dates.fetch_earnings_dates_fmp("RELIANCE.NS")
    assert out["Surprise(%)"].iloc[0] == pytest.approx(150.0)


def test_fetch_earnings_dates_fmp_no_api_key(monkeypatch, no_disk_cache):
    _inject_fmp(monkeypatch, api_key=None)
    assert earnings_dates.fetch_earnings_dates_fmp("RELIANCE.NS") is None


def test_fetch_earnings_dates_fmp_empty_payload(monkeypatch, no_disk_cache):
    _inject_fmp(monkeypatch, payload=[])
    assert earnings_dates.fetch_earnings_dates_fmp("RELIANCE.NS") is None


def test_fetch_earnings_dates_fmp_non_list_payload(monkeypatch, no_disk_cache):
    _inject_fmp(monkeypatch, payload={"error": "nope"})
    assert earnings_dates.fetch_earnings_dates_fmp("RELIANCE.NS") is None


def test_fetch_earnings_dates_fmp_request_exception(monkeypatch, no_disk_cache):
    _inject_fmp(monkeypatch, boom=True)
    assert earnings_dates.fetch_earnings_dates_fmp("RELIANCE.NS") is None


def test_fetch_earnings_dates_fmp_cache_exception(monkeypatch):
    def raise_cache(*a, **kw):
        raise RuntimeError("cache io")

    monkeypatch.setattr(earnings_dates, "cached_json_call", raise_cache)
    assert earnings_dates.fetch_earnings_dates_fmp("RELIANCE.NS") is None


# ── _fmp_earnings_rows_for_ticker / _fetch_fmp_earnings_rows ─────────────


def test_fmp_rows_for_ticker_none(monkeypatch):
    monkeypatch.setattr(
        earnings_dates, "fetch_earnings_dates_fmp", lambda t, years: None
    )
    assert earnings_dates._fmp_earnings_rows_for_ticker("R.NS", 3) == []


def test_fmp_rows_for_ticker_rows(monkeypatch):
    df = _earnings_df(["2024-01-15"], eps_est=[1.0], reported=[1.2], surprise=[20.0])
    monkeypatch.setattr(earnings_dates, "fetch_earnings_dates_fmp", lambda t, years: df)
    rows = earnings_dates._fmp_earnings_rows_for_ticker("R.NS", 3)
    assert rows[0]["ticker"] == "R.NS"
    assert rows[0]["earnings_date"] == date(2024, 1, 15)
    assert rows[0]["surprise_pct"] == 20.0


def test_fetch_fmp_earnings_rows(monkeypatch):
    df = _earnings_df(["2024-01-15"], surprise=[20.0])
    monkeypatch.setattr(
        earnings_dates,
        "fetch_earnings_dates_fmp",
        lambda t, years: df if t == "GOOD.NS" else None,
    )
    rows = earnings_dates._fetch_fmp_earnings_rows(["GOOD.NS", "BAD.NS"], 3, 50)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "GOOD.NS"


def test_fetch_fmp_earnings_rows_handles_future_exception(monkeypatch):
    def boom(t, years):
        raise RuntimeError("worker boom")

    monkeypatch.setattr(earnings_dates, "_fmp_earnings_rows_for_ticker", boom)
    assert earnings_dates._fetch_fmp_earnings_rows(["A.NS"], 3, 50) == []


# ── collect_earnings_events ──────────────────────────────────────────────


def test_collect_us(monkeypatch):
    monkeypatch.setattr(
        earnings_dates,
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
    out = earnings_dates.collect_earnings_events(
        ["AAA", "BBB"], batch_size=1, market="us"
    )
    assert set(out["ticker"]) == {"AAA", "BBB"}


def test_collect_us_empty(monkeypatch):
    monkeypatch.setattr(earnings_dates, "_fetch_yf_earnings_rows", lambda *a, **kw: [])
    out = earnings_dates.collect_earnings_events([], market="us")
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
    monkeypatch.setattr(earnings_dates, "fetch_earnings_dates_nse", lambda: nse_df)
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
        earnings_dates,
        "_fetch_openscreener_earnings_rows",
        lambda batch, years, bs: osc_rows,
    )
    out = earnings_dates.collect_earnings_events(
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
    monkeypatch.setattr(earnings_dates, "fetch_earnings_dates_nse", lambda: None)
    monkeypatch.setattr(
        earnings_dates,
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
    out = earnings_dates.collect_earnings_events(["R.NS"], market="india")
    assert "R.NS" in set(out["ticker"])


# ── collect_earnings_events (India, surprise_source="fmp") ───────────────


def _recent_quarter_dates():
    """Two announcement dates in the recent past reporting different quarters."""
    today = pd.Timestamp(date.today()).normalize()
    ann = today - pd.Timedelta(days=90)
    prior_ann = ann - pd.Timedelta(days=95)  # previous fiscal quarter
    return ann, prior_ann


def test_collect_india_fmp_enriches_nse_dates(monkeypatch):
    ann, prior_ann = _recent_quarter_dates()
    nse_df = pd.DataFrame(
        {
            "ticker": ["RELIANCE.NS", "OTHER.NS", "NOFMP.NS"],
            "earnings_date": [ann, ann, ann],
            "desc": ["x", "y", "z"],
        }
    )
    monkeypatch.setattr(earnings_dates, "fetch_earnings_dates_nse", lambda: nse_df)
    # FMP: same fiscal quarter as the RELIANCE NSE announcement (a few days
    # earlier) plus an FMP-only prior quarter. Nothing for NOFMP.NS.
    fmp_rows = [
        {
            "ticker": "RELIANCE.NS",
            "earnings_date": (ann - pd.Timedelta(days=4)).date(),
            "eps_estimate": 1.0,
            "reported_eps": 1.2,
            "surprise_pct": 20.0,
        },
        {
            "ticker": "RELIANCE.NS",
            "earnings_date": prior_ann.date(),
            "eps_estimate": 2.0,
            "reported_eps": 1.8,
            "surprise_pct": -10.0,
        },
    ]
    monkeypatch.setattr(
        earnings_dates,
        "_fetch_fmp_earnings_rows",
        lambda batch, years, bs: fmp_rows,
    )
    out = earnings_dates.collect_earnings_events(
        ["RELIANCE.NS", "NOFMP.NS"], market="india", surprise_source="fmp"
    )

    rel = out[out["ticker"] == "RELIANCE.NS"]
    # Same-quarter row: NSE announcement date preferred, FMP surprise attached.
    enriched = rel[pd.to_datetime(rel["earnings_date"]) == ann]
    assert len(enriched) == 1
    assert enriched["surprise_pct"].iloc[0] == 20.0
    # FMP's own (earlier) date for the matched quarter is NOT double-counted.
    assert pd.Timestamp(ann - pd.Timedelta(days=4)) not in set(
        pd.to_datetime(rel["earnings_date"])
    )
    # FMP-only prior quarter kept with FMP's date and surprise.
    fmp_only = rel[pd.to_datetime(rel["earnings_date"]) == prior_ann]
    assert len(fmp_only) == 1
    assert fmp_only["surprise_pct"].iloc[0] == -10.0
    # NSE announcement without any FMP match keeps a NaN surprise.
    nofmp = out[out["ticker"] == "NOFMP.NS"]
    assert len(nofmp) == 1
    assert pd.isna(nofmp["surprise_pct"].iloc[0])
    # OTHER.NS is outside the requested ticker universe -> excluded.
    assert "OTHER.NS" not in set(out["ticker"])


def test_collect_india_fmp_without_nse(monkeypatch):
    ann, _ = _recent_quarter_dates()
    monkeypatch.setattr(earnings_dates, "fetch_earnings_dates_nse", lambda: None)
    monkeypatch.setattr(
        earnings_dates,
        "_fetch_fmp_earnings_rows",
        lambda batch, years, bs: [
            {
                "ticker": "R.NS",
                "earnings_date": ann.date(),
                "eps_estimate": 1.0,
                "reported_eps": 1.1,
                "surprise_pct": 10.0,
            }
        ],
    )
    out = earnings_dates.collect_earnings_events(
        ["R.NS"], market="india", surprise_source="fmp"
    )
    assert list(out["ticker"]) == ["R.NS"]
    assert out["surprise_pct"].iloc[0] == 10.0


def test_collect_india_fmp_empty(monkeypatch):
    monkeypatch.setattr(earnings_dates, "fetch_earnings_dates_nse", lambda: None)
    monkeypatch.setattr(
        earnings_dates, "_fetch_fmp_earnings_rows", lambda batch, years, bs: []
    )
    out = earnings_dates.collect_earnings_events(
        ["R.NS"], market="india", surprise_source="fmp"
    )
    assert out.empty


def test_collect_us_ignores_surprise_source(monkeypatch):
    monkeypatch.setattr(
        earnings_dates,
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
    out = earnings_dates.collect_earnings_events(
        ["AAA"], market="us", surprise_source="fmp"
    )
    assert set(out["ticker"]) == {"AAA"}


# ── fetch_analyst_sentiment ──────────────────────────────────────────────


def test_analyst_sentiment_india_none():
    assert sentiment.fetch_analyst_sentiment("X.NS", market="india") is None


def test_analyst_sentiment_cache_hit(monkeypatch):
    monkeypatch.setattr(sentiment, "cached_json_call", lambda *a, **kw: {"net": 3})
    assert sentiment.fetch_analyst_sentiment("AAPL") == {"net": 3}


def test_analyst_sentiment_action_col(monkeypatch, no_disk_cache):
    ud = pd.DataFrame({"Action": ["up", "up", "reit", "down"]})
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(upgrades=ud))
    out = sentiment.fetch_analyst_sentiment("AAPL")
    assert out["upgrades"] == 2.5  # 2 up + 0.5*1 reit
    assert out["downgrades"] == 1
    assert out["net"] == 1.5
    assert out["grade_counts"]


def test_analyst_sentiment_tograde_col(monkeypatch, no_disk_cache):
    ud = pd.DataFrame({"ToGrade": ["Buy", "Outperform", "Sell"]})
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(upgrades=ud))
    out = sentiment.fetch_analyst_sentiment("AAPL")
    assert out["upgrades"] == 2
    assert out["downgrades"] == 1
    assert out["grade_counts"] == {}


def test_analyst_sentiment_unknown_cols(monkeypatch, no_disk_cache):
    ud = pd.DataFrame({"Other": [1, 2]})
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(upgrades=ud))
    assert sentiment.fetch_analyst_sentiment("AAPL") is None


def test_analyst_sentiment_empty(monkeypatch, no_disk_cache):
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(upgrades=pd.DataFrame()))
    assert sentiment.fetch_analyst_sentiment("AAPL") is None


def test_analyst_sentiment_none_ud(monkeypatch, no_disk_cache):
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(upgrades=None))
    assert sentiment.fetch_analyst_sentiment("AAPL") is None


def test_analyst_sentiment_exception(monkeypatch, no_disk_cache):
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(raise_on={"upgrades"}))
    assert sentiment.fetch_analyst_sentiment("AAPL") is None


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
    monkeypatch.setattr(
        sentiment, "cached_json_call", lambda *a, **kw: {"pc_ratio": 1.0}
    )
    assert sentiment.fetch_iv_sentiment_yf("AAPL") == {"pc_ratio": 1.0}


def test_iv_yf_success_with_volume(monkeypatch, no_disk_cache):
    today = date.today()
    far = (today + timedelta(days=10)).isoformat()
    calls = _opt_df(volume=[100, 200], oi=[10, 20], iv=[0.40, 0.42])
    puts = _opt_df(volume=[50, 60], oi=[5, 6], iv=[0.50, 0.52])
    chain = _Chain(calls, puts)
    monkeypatch.setattr(
        yf, "Ticker", lambda tk: _FakeTicker(options=[far], chain=chain)
    )
    out = sentiment.fetch_iv_sentiment_yf("AAPL")
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
        yf, "Ticker", lambda tk: _FakeTicker(options=[near], chain=chain)
    )
    out = sentiment.fetch_iv_sentiment_yf("AAPL")
    assert out is not None


def test_iv_yf_zero_calls_oi_branch(monkeypatch, no_disk_cache):
    today = date.today()
    far = (today + timedelta(days=10)).isoformat()
    # volume present but all zero -> total_calls == 0 -> OI-based pc_ratio.
    calls = _opt_df(volume=[0, 0], oi=[10, 20])
    puts = _opt_df(volume=[0, 0], oi=[5, 5])
    chain = _Chain(calls, puts)
    monkeypatch.setattr(
        yf, "Ticker", lambda tk: _FakeTicker(options=[far], chain=chain)
    )
    out = sentiment.fetch_iv_sentiment_yf("AAPL")
    assert out["pc_ratio"] == round(10 / 30, 4)


def test_iv_yf_no_options(monkeypatch, no_disk_cache):
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(options=[]))
    assert sentiment.fetch_iv_sentiment_yf("AAPL") is None


def test_iv_yf_empty_chain(monkeypatch, no_disk_cache):
    today = date.today()
    far = (today + timedelta(days=10)).isoformat()
    chain = _Chain(pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(
        yf, "Ticker", lambda tk: _FakeTicker(options=[far], chain=chain)
    )
    assert sentiment.fetch_iv_sentiment_yf("AAPL") is None


def test_iv_yf_no_iv_cols(monkeypatch, no_disk_cache):
    today = date.today()
    far = (today + timedelta(days=10)).isoformat()
    calls = _opt_df(volume=[1, 1])
    puts = _opt_df(volume=[1, 1])
    chain = _Chain(calls, puts)
    monkeypatch.setattr(
        yf, "Ticker", lambda tk: _FakeTicker(options=[far], chain=chain)
    )
    out = sentiment.fetch_iv_sentiment_yf("AAPL")
    assert out["median_iv"] != out["median_iv"] or np.isnan(out["median_iv"])  # nan


def test_iv_yf_exception(monkeypatch, no_disk_cache):
    monkeypatch.setattr(yf, "Ticker", lambda tk: _FakeTicker(raise_on={"options"}))
    assert sentiment.fetch_iv_sentiment_yf("AAPL") is None


# ── fetch_iv_sentiment_nse ───────────────────────────────────────────────


def test_iv_nse_cache_hit(monkeypatch):
    monkeypatch.setattr(
        sentiment, "cached_json_call", lambda *a, **kw: {"pc_ratio": 2.0}
    )
    assert sentiment.fetch_iv_sentiment_nse("RELIANCE") == {"pc_ratio": 2.0}


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
    monkeypatch.setattr(sentiment, "fetch_option_chain", lambda s: oc)
    out = sentiment.fetch_iv_sentiment_nse("RELIANCE")
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
    monkeypatch.setattr(sentiment, "fetch_option_chain", lambda s: oc)
    out = sentiment.fetch_iv_sentiment_nse("RELIANCE")
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
    monkeypatch.setattr(sentiment, "fetch_option_chain", lambda s: oc)
    out = sentiment.fetch_iv_sentiment_nse("RELIANCE")
    assert out["median_iv"] is None  # 600 filtered out (>=500)


def test_iv_nse_no_oc(monkeypatch, no_disk_cache):
    monkeypatch.setattr(sentiment, "fetch_option_chain", lambda s: None)
    assert sentiment.fetch_iv_sentiment_nse("RELIANCE") is None


def test_iv_nse_no_records_key(monkeypatch, no_disk_cache):
    monkeypatch.setattr(sentiment, "fetch_option_chain", lambda s: {"foo": 1})
    assert sentiment.fetch_iv_sentiment_nse("RELIANCE") is None


def test_iv_nse_empty_data(monkeypatch, no_disk_cache):
    monkeypatch.setattr(
        sentiment, "fetch_option_chain", lambda s: {"records": {"data": []}}
    )
    assert sentiment.fetch_iv_sentiment_nse("RELIANCE") is None


def test_iv_nse_exception(monkeypatch, no_disk_cache):
    def boom(_symbol):
        raise RuntimeError("nse boom")

    monkeypatch.setattr(sentiment, "fetch_option_chain", boom)
    assert sentiment.fetch_iv_sentiment_nse("RELIANCE") is None


# ── fetch_iv_sentiment dispatch ──────────────────────────────────────────


def test_iv_dispatch_india(monkeypatch):
    monkeypatch.setattr(
        sentiment, "fetch_iv_sentiment_nse", lambda symbol: {"sym": symbol}
    )
    assert sentiment.fetch_iv_sentiment("RELIANCE.NS", market="india") == {
        "sym": "RELIANCE"
    }


def test_iv_dispatch_us(monkeypatch):
    monkeypatch.setattr(
        sentiment, "fetch_iv_sentiment_yf", lambda ticker: {"t": ticker}
    )
    assert sentiment.fetch_iv_sentiment("AAPL") == {"t": "AAPL"}


# ── fetch_price_data ─────────────────────────────────────────────────────


def test_fetch_price_data_with_fetcher():
    from tests.conftest import StubPriceFetcher, make_bars

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
