"""Offline line-coverage tests for the unusual-volume service stack.

Covers ``service``, ``cli``, ``nse_client``, ``delivery``, ``fii_dii`` and
``option_chain``. Everything is stubbed/monkeypatched — no network, no disk
caches beyond ``tmp_path``, fully deterministic.
"""

from __future__ import annotations


import io


import sys


import types


from datetime import date


import pandas as pd


import pytest


from rich.console import Console


from screener import cache


from screener.unusual_volume import (
    cli as uv_cli,
    delivery,
    fii_dii,
    nse_client,
    option_chain,
    service,
)


from screener.unusual_volume.buildup import BuildupScore


from screener.unusual_volume.detector import Event


def _console() -> Console:
    return Console(file=io.StringIO())


def _event(symbol: str = "RELIANCE", d: date = date(2026, 5, 15), **over) -> Event:
    base = dict(
        symbol=symbol,
        date=d,
        close=2500.0,
        pct_change=1.0,
        volume=150_000.0,
        avg_volume_20d=50_000.0,
        rvol=3.0,
        rvol_5d=3.0,
        rvol_50d=3.0,
        rvol_90d=3.0,
        z_score=2.5,
        pct_rank_252d=0.9,
        direction="BUYING",
        strength="HIGH",
    )
    base.update(over)
    return Event(**base)


def _bars(n: int = 30, as_of: date = date(2026, 5, 15)) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(as_of), periods=n)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [100_000.0] * n,
        },
        index=idx,
    )


class _Fetcher:
    def __init__(self, frames=None, exc=None):
        self._frames = frames or {}
        self._exc = exc

    def fetch(self, syms, start, end):
        if self._exc is not None:
            raise self._exc
        return self._frames


def _score(sym="AAA"):
    return BuildupScore(
        symbol=sym,
        as_of=date(2026, 5, 15),
        window=20,
        range_compression=0.7,
        updown_volume=0.6,
        higher_lows=0.6,
        sustained_delivery=None,
        close_near_high=0.7,
        composite=0.65,
        flags=["compression"],
    )


def _req(**over):
    base = dict(
        market="india",
        as_of=date(2026, 5, 15),
        universe=["NSE:RELIANCE"],
        min_rvol=0.0,
        min_z=0.0,
        strength_floor="MODERATE",
        min_avg_volume=0.0,
        min_market_cap=0.0,
        include_fno_ban=True,
        deep_india=False,
        buildup_enabled=False,
        buildup_window=20,
        buildup_min_score=0.0,
        option_chain=False,
        fii_dii=False,
        pledge=False,
    )
    base.update(over)
    return service.UnusualVolumeRequest(**base)


def _patch_run(monkeypatch, result):
    monkeypatch.setattr(uv_cli, "_resolve_universe", lambda m, t, f: ["AAA"])
    monkeypatch.setattr(uv_cli, "run_unusual_volume_scan", lambda req, console: result)


def _result(events, fetched=1, liquid=1):
    return service.UnusualVolumeResult(
        events=events, fetched_count=fetched, liquid_count=liquid
    )


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


def _reset_tls():
    for name in ("session", "primed", "primed_pages"):
        if hasattr(nse_client._tls, name):
            delattr(nse_client._tls, name)


def test_parse_holiday_payload_variants():
    raw = {
        "CM": [
            {"tradingDate": "26-Jan-2026"},
            {"date": "15-Aug-2026"},
            {"tradingDate": "not-a-date"},  # unparseable → skipped
            {"foo": "bar"},  # no date keys
            "notadict",  # skipped
        ],
        "BAD": "notalist",
    }
    out = nse_client._parse_holiday_payload(raw)
    assert date(2026, 1, 26) in out
    assert date(2026, 8, 15) in out
    assert nse_client._parse_holiday_payload("notadict") == set()


def test_calendar_load_holidays(monkeypatch):
    cal = nse_client.TradingCalendar()
    monkeypatch.setattr(
        nse_client,
        "nse_cached_json",
        lambda *a, **k: {"CM": [{"tradingDate": "26-Jan-2026"}]},
    )
    assert cal._holiday_set() == {date(2026, 1, 26)}
    # cached on second call (no refetch path)
    assert cal._holiday_set() == {date(2026, 1, 26)}


def test_calendar_last_trading_day_fallback(monkeypatch):
    cal = nse_client.TradingCalendar()
    cal._holidays = set()
    # all candidates are holidays-or-weekend so lookback exhausts → returns d
    monkeypatch.setattr(cal, "is_trading_day", lambda d: False)
    d = date(2026, 1, 7)
    assert cal.last_trading_day_on_or_before(d, lookback=3) == d


def test_module_level_calendar_shortcuts(monkeypatch):
    monkeypatch.setattr(nse_client._CALENDAR, "is_trading_day", lambda d: True)
    assert nse_client.is_trading_day(date(2026, 1, 5)) is True
    monkeypatch.setattr(
        nse_client._CALENDAR,
        "last_trading_day_on_or_before",
        lambda d, lookback=7: date(2026, 1, 2),
    )
    assert nse_client.last_trading_day_on_or_before(date(2026, 1, 3)) == date(
        2026, 1, 2
    )


def test_nse_cached_json_delegates(monkeypatch):
    captured = {}

    def fake_cached(ns, kp, *, ttl_seconds, refresh, fetch):
        captured["ns"] = ns
        return fetch()

    monkeypatch.setattr(nse_client, "cached_json_call", fake_cached)
    monkeypatch.setattr(
        nse_client, "fetch_nse_json", lambda url, op, extra_prime_page=None: {"ok": 1}
    )
    out = nse_client.nse_cached_json("ns", ("k",), "url", "op")
    assert out == {"ok": 1} and captured["ns"] == "ns"


def test_load_one_day_path_none(monkeypatch):
    fake = types.SimpleNamespace(full_bhavcopy_save=lambda dt, d: None)
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", fake)
    monkeypatch.setattr(
        delivery, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    assert delivery._load_one_day(date(2026, 5, 15)) is None


def test_load_one_day_missing_file(monkeypatch):
    fake = types.SimpleNamespace(full_bhavcopy_save=lambda dt, d: "/no/such.csv")
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", fake)
    monkeypatch.setattr(
        delivery, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    assert delivery._load_one_day(date(2026, 5, 15)) is None


def test_load_one_day_parses_csv(monkeypatch, tmp_path):
    csv = tmp_path / "bhav.csv"
    csv.write_text(
        " SYMBOL, SERIES, DATE1, TTL_TRD_QNTY, DELIV_QTY, DELIV_PER\n"
        "RELIANCE,EQ,15-May-2026,1000,500,50\n"
        "GOVTSEC,GS,15-May-2026,10,5,50\n"
    )
    fake = types.SimpleNamespace(full_bhavcopy_save=lambda dt, d: str(csv))
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", fake)
    monkeypatch.setattr(
        delivery, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    monkeypatch.setattr(delivery, "CACHE_DIR", tmp_path / "cache")
    out = delivery._load_one_day(date(2026, 5, 15))
    assert out is not None
    assert list(out["SYMBOL"]) == ["RELIANCE"]  # GS series filtered out


def test_load_one_day_missing_columns(monkeypatch, tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("SYMBOL,SERIES\nRELIANCE,EQ\n")
    fake = types.SimpleNamespace(full_bhavcopy_save=lambda dt, d: str(csv))
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", fake)
    monkeypatch.setattr(
        delivery, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    monkeypatch.setattr(delivery, "CACHE_DIR", tmp_path / "cache")
    assert delivery._load_one_day(date(2026, 5, 15)) is None


def test_load_one_day_parse_error(monkeypatch, tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_bytes(b"\x00\x01\x02")
    fake = types.SimpleNamespace(full_bhavcopy_save=lambda dt, d: str(bad))
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", fake)
    monkeypatch.setattr(
        delivery, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    monkeypatch.setattr(delivery, "CACHE_DIR", tmp_path / "cache")

    def boom(path):
        raise pd.errors.ParserError("bad")

    monkeypatch.setattr(delivery.pd, "read_csv", boom)
    assert delivery._load_one_day(date(2026, 5, 15)) is None


def test_load_delivery_panel_aggregates(monkeypatch):
    monkeypatch.setattr(delivery, "is_trading_day", lambda d: d.weekday() < 5)

    def fake_one(dt):
        return pd.DataFrame(
            [
                {
                    "SYMBOL": "RELIANCE",
                    "date": dt,
                    "TTL_TRD_QNTY": 1.0,
                    "DELIV_QTY": 1.0,
                    "DELIV_PER": 50.0,
                },
                {
                    "SYMBOL": "OTHER",
                    "date": dt,
                    "TTL_TRD_QNTY": 1.0,
                    "DELIV_QTY": 1.0,
                    "DELIV_PER": 50.0,
                },
            ]
        )

    monkeypatch.setattr(delivery, "_load_one_day", fake_one)
    panel = delivery.load_delivery_panel(
        ["reliance"], date(2026, 5, 15), history_days=3
    )
    assert set(panel["SYMBOL"]) == {"RELIANCE"}


def test_load_delivery_panel_empty(monkeypatch):
    monkeypatch.setattr(delivery, "is_trading_day", lambda d: True)
    monkeypatch.setattr(delivery, "_load_one_day", lambda dt: None)
    panel = delivery.load_delivery_panel(["AAA"], date(2026, 5, 15), history_days=2)
    assert panel.empty
    assert "DELIV_PER" in panel.columns


def test_delivery_notes_branches():
    assert delivery._delivery_notes(3.0, None, "BUYING") == ""
    assert delivery._delivery_notes(3.0, float("nan"), "BUYING") == ""
    assert "speculative" in delivery._delivery_notes(3.5, 10.0, "BUYING")
    note = delivery._delivery_notes(4.0, 70.0, "SELLING")
    assert "strong institutional footprint" in note
    assert "long-holder distribution" in note


def test_overlay_events_empty_inputs():
    assert delivery.overlay_events([], pd.DataFrame()) == []
    evs = [_event("AAA")]
    # non-empty events but empty panel → compute returns empty → returns events
    assert delivery.overlay_events(evs, pd.DataFrame()) is evs


def test_overlay_events_missing_key_skips():
    panel = pd.DataFrame(
        [
            {
                "SYMBOL": "RELIANCE",
                "date": date(2026, 5, 15),
                "TTL_TRD_QNTY": 1.0,
                "DELIV_QTY": 1.0,
                "DELIV_PER": 50.0,
            }
        ]
    )
    ev = _event("NOTINPANEL", date(2026, 5, 15))
    delivery.overlay_events([ev], panel)
    assert ev.delivery_pct is None


def test_overlay_events_duplicate_rows_dataframe_branch():
    # two rows same key remain after dedupe? dedupe keeps last; but the
    # isinstance(row, DataFrame) branch needs duplicate index entries.
    rows = []
    for d in [date(2026, 5, 14), date(2026, 5, 15)]:
        for q in (40_000.0, 41_000.0):
            rows.append(
                {
                    "SYMBOL": "RELIANCE",
                    "date": d,
                    "TTL_TRD_QNTY": 100_000.0,
                    "DELIV_QTY": q,
                    "DELIV_PER": 50.0,
                }
            )
    panel = pd.DataFrame(rows)
    ev = _event("RELIANCE", date(2026, 5, 15))
    delivery.overlay_events([ev], panel)
    assert ev.delivery_pct == 50.0


def test_quiet_accumulation_empty_panel():
    assert (
        delivery.quiet_accumulation_events({}, pd.DataFrame(), date(2026, 5, 15), 1.5)
        == []
    )


def test_quiet_accumulation_various_skips(monkeypatch):
    as_of = date(2026, 5, 15)
    panel = pd.DataFrame(
        [
            # high delivery rvol on as_of for several symbols (sorted ascending)
            *[
                {
                    "SYMBOL": s,
                    "date": (pd.Timestamp(as_of) - pd.Timedelta(days=k)).date(),
                    "TTL_TRD_QNTY": 100_000.0,
                    "DELIV_QTY": (60_000.0 if k == 0 else 20_000.0),
                    "DELIV_PER": 50.0,
                }
                for s in (
                    "WITHBARS",
                    "NOBARS",
                    "EMPTYBARS",
                    "NODATE",
                    "EXISTING",
                    "HIVOL",
                )
                for k in range(8, -1, -1)
            ]
        ]
    )
    idx = pd.bdate_range(end=pd.Timestamp(as_of), periods=30)
    good = pd.DataFrame({"close": [100.0] * 30, "volume": [1000.0] * 30}, index=idx)
    # HIVOL: last-bar volume RVOL above threshold → skipped
    hivol = good.copy()
    hivol.iloc[-1, hivol.columns.get_loc("volume")] = 10_000_000.0
    nodate = pd.DataFrame({"close": [100.0], "volume": [1.0]})  # no index, no date col
    bars_by_symbol = {
        "WITHBARS": good,
        "EMPTYBARS": pd.DataFrame(),
        "NODATE": nodate,
        "EXISTING": good,
        "HIVOL": hivol,
        # NOBARS intentionally absent
    }
    existing = [_event("EXISTING", as_of)]
    out = delivery.quiet_accumulation_events(
        bars_by_symbol, panel, as_of, min_rvol_skip=2.0, existing_events=existing
    )
    syms = {e.symbol for e in out}
    assert "WITHBARS" in syms
    assert "EXISTING" not in syms  # already detected
    assert "NOBARS" not in syms
    assert "EMPTYBARS" not in syms
    assert "HIVOL" not in syms


def test_quiet_accumulation_date_column_index(monkeypatch):
    as_of = date(2026, 5, 15)
    panel = pd.DataFrame(
        [
            {
                "SYMBOL": "DATECOL",
                "date": (pd.Timestamp(as_of) - pd.Timedelta(days=k)).date(),
                "TTL_TRD_QNTY": 100_000.0,
                "DELIV_QTY": (60_000.0 if k == 0 else 20_000.0),
                "DELIV_PER": 50.0,
            }
            for k in range(8, -1, -1)
        ]
    )
    dates = pd.bdate_range(end=pd.Timestamp(as_of), periods=5)
    bars = pd.DataFrame(
        {"date": [d.date() for d in dates], "close": [100.0] * 5, "volume": [10.0] * 5}
    )
    out = delivery.quiet_accumulation_events(
        {"DATECOL": bars}, panel, as_of, min_rvol_skip=2.0
    )
    assert any(e.symbol == "DATECOL" for e in out)


def test_quiet_accumulation_empty_after_asof_filter():
    # bars exist but all dates after filter → df empty
    as_of = date(2026, 5, 15)
    panel = pd.DataFrame(
        [
            {
                "SYMBOL": "FUTURE",
                "date": (pd.Timestamp(as_of) - pd.Timedelta(days=k)).date(),
                "TTL_TRD_QNTY": 100_000.0,
                "DELIV_QTY": (60_000.0 if k == 0 else 20_000.0),
                "DELIV_PER": 50.0,
            }
            for k in range(8, -1, -1)
        ]
    )
    future_idx = pd.bdate_range(
        start=pd.Timestamp(as_of) + pd.Timedelta(days=5), periods=3
    )
    bars = pd.DataFrame({"close": [1.0] * 3, "volume": [1.0] * 3}, index=future_idx)
    out = delivery.quiet_accumulation_events(
        {"FUTURE": bars}, panel, as_of, min_rvol_skip=2.0
    )
    assert out == []


def test_fetch_fii_dii_today(monkeypatch):
    monkeypatch.setattr(
        fii_dii, "nse_cached_json", lambda *a, **k: [{"category": "FII"}]
    )
    assert fii_dii.fetch_fii_dii_today() == [{"category": "FII"}]
    monkeypatch.setattr(fii_dii, "nse_cached_json", lambda *a, **k: {"not": "list"})
    assert fii_dii.fetch_fii_dii_today() is None


def test_as_float_variants():
    assert fii_dii._as_float("1,234.5") == 1234.5
    assert fii_dii._as_float(None) is None
    assert fii_dii._as_float("abc") is None


def test_parse_fii_dii_buy_sell_and_none():
    assert fii_dii.parse_fii_dii([], date(2026, 5, 15)) is None
    raw = [
        "notadict",
        {"category": "FII/FPI", "buyValue": "100", "sellValue": "40"},
        {"category": "DII", "netValue": "25"},
    ]
    rec = fii_dii.parse_fii_dii(raw, date(2026, 5, 15))
    assert rec["fii_net"] == 60.0 and rec["dii_net"] == 25.0
    # all-None nets → None
    assert (
        fii_dii.parse_fii_dii(
            [{"category": "FII", "netValue": None}], date(2026, 5, 15)
        )
        is None
    )


def test_fii_dii_metric_series_empty():
    assert fii_dii.fii_dii_metric_series(None).empty
    assert fii_dii.fii_dii_metric_series(pd.DataFrame()).empty


def test_fii_dii_metric_series_all_nan_dates():
    panel = pd.DataFrame([{"date": "not-a-date", "fii_net": 1.0, "dii_net": 2.0}])
    assert fii_dii.fii_dii_metric_series(panel).empty


def test_fii_dii_metric_series_trend_and_zero_baseline():
    base = date(2026, 4, 1)
    rows = [
        {
            "date": (pd.Timestamp(base) + pd.Timedelta(days=i)).date(),
            "fii_net": 100.0,
            "dii_net": 50.0,
        }
        for i in range(6)
    ]
    out = fii_dii.fii_dii_metric_series(pd.DataFrame(rows))
    assert out.iloc[-1]["fii_trend"] == pytest.approx(1.0)
    # zero baseline → trend stays None
    rows_zero = [
        {
            "date": (pd.Timestamp(base) + pd.Timedelta(days=i)).date(),
            "fii_net": (10.0 if i == 5 else -2.0),
            "dii_net": 1.0,
        }
        for i in range(6)
    ]
    # craft so mean of tail(20) is exactly 0
    rows_zero = [
        {
            "date": (pd.Timestamp(base) + pd.Timedelta(days=i)).date(),
            "fii_net": v,
            "dii_net": 1.0,
        }
        for i, v in enumerate([5.0, -5.0, 5.0, -5.0, 10.0, -10.0])
    ]
    out2 = fii_dii.fii_dii_metric_series(pd.DataFrame(rows_zero))
    assert pd.isna(out2.iloc[-1]["fii_trend"])


def test_compute_fii_dii_metrics_empty_and_cutoff():
    assert (
        fii_dii.compute_fii_dii_metrics(pd.DataFrame(), date(2026, 5, 15))["fii_5d_net"]
        is None
    )
    # all rows after cutoff → empty after filter
    base = date(2026, 6, 1)
    rows = [
        {
            "date": (pd.Timestamp(base) + pd.Timedelta(days=i)).date(),
            "fii_net": 1.0,
            "dii_net": 1.0,
        }
        for i in range(6)
    ]
    m = fii_dii.compute_fii_dii_metrics(pd.DataFrame(rows), date(2026, 1, 1))
    assert m["fii_5d_net"] is None


def test_overlay_fii_dii_with_record(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    monkeypatch.setattr(
        fii_dii,
        "fetch_fii_dii_today",
        lambda refresh=False: [
            {"category": "FII/FPI", "netValue": "123.45"},
            {"category": "DII", "netValue": "67.89"},
        ],
    )
    evs = [_event("A"), _event("B")]
    m = fii_dii.overlay_fii_dii(evs, date(2026, 5, 15))
    assert m is not None
    assert evs[0].fii_5d_net == evs[1].fii_5d_net


def test_overlay_fii_dii_no_record_reads_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    monkeypatch.setattr(fii_dii, "fetch_fii_dii_today", lambda refresh=False: None)
    monkeypatch.setattr(fii_dii, "read_frame", lambda path: None)
    evs = [_event("A")]
    m = fii_dii.overlay_fii_dii(evs, date(2026, 5, 15))
    assert m["fii_5d_net"] is None
    assert evs[0].fii_5d_net is None


def test_overlay_fii_dii_no_record_existing_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    monkeypatch.setattr(fii_dii, "fetch_fii_dii_today", lambda refresh=False: None)
    base = date(2026, 5, 1)
    existing = pd.DataFrame(
        [
            {
                "date": (pd.Timestamp(base) + pd.Timedelta(days=i)).date(),
                "fii_net": 10.0,
                "dii_net": 5.0,
            }
            for i in range(6)
        ]
    )
    monkeypatch.setattr(fii_dii, "read_frame", lambda path: existing)
    evs = [_event("A")]
    m = fii_dii.overlay_fii_dii(evs, date(2026, 5, 15))
    assert m["fii_5d_net"] is not None


def test_fetch_option_chain_dict_and_none(monkeypatch):
    monkeypatch.setattr(option_chain, "nse_cached_json", lambda *a, **k: {"ok": 1})
    assert option_chain.fetch_option_chain("tcs") == {"ok": 1}
    monkeypatch.setattr(option_chain, "nse_cached_json", lambda *a, **k: ["list"])
    assert option_chain.fetch_option_chain("tcs") is None


def test_oc_as_float_branches():
    assert option_chain._as_float(None) is None
    assert option_chain._as_float("nope") is None
    assert option_chain._as_float("5") == 5.0


def test_compute_oc_metrics_records_fallback():
    raw = {
        "records": {
            "data": [
                {"CE": {"openInterest": 100}, "PE": {"openInterest": 50}},
                {"CE": {}, "PE": {}},
            ]
        }
    }
    m = option_chain.compute_oc_metrics(raw)
    assert m["pcr"] == 0.5


def test_compute_oc_metrics_empty_raw():
    m = option_chain.compute_oc_metrics({})
    assert m["ce_oi"] is None and m["pcr"] is None


def test_overlay_option_chain_empty():
    assert option_chain.overlay_option_chain([]) == {}


def test_overlay_option_chain_some_none(monkeypatch):
    def fake_fetch(sym, refresh=False):
        return (
            None
            if sym == "BAD"
            else {"filtered": {"CE": {"totOI": 1000}, "PE": {"totOI": 2000}}}
        )

    monkeypatch.setattr(option_chain, "fetch_option_chain", fake_fetch)
    evs = [_event("GOOD"), _event("BAD")]
    out = option_chain.overlay_option_chain(evs, max_workers=2)
    assert "GOOD" in out and "BAD" not in out
    good = next(e for e in evs if e.symbol == "GOOD")
    assert good.pcr == 2.0
