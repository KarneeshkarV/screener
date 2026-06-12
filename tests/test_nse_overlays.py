"""Tests for the 9 Indian micro-structure overlays (no network)."""

from __future__ import annotations

from datetime import date
import io
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

from screener import cache, pledge
from screener import rs_breakout
from screener.unusual_volume import service
from screener.unusual_volume import fii_dii, nse_client, option_chain
from screener.unusual_volume.delivery import compute_delivery_metrics, overlay_events
from screener.unusual_volume.detector import Event


def _event(symbol: str = "RELIANCE", d: date = date(2026, 5, 15)) -> Event:
    return Event(
        symbol=symbol,
        date=d,
        close=2500.0,
        pct_change=1.0,
        volume=150_000,
        avg_volume_20d=50_000,
        rvol=3.0,
        rvol_5d=3.0,
        rvol_50d=3.0,
        rvol_90d=3.0,
        z_score=2.5,
        pct_rank_252d=0.9,
        direction="BUYING",
        strength="HIGH",
    )


def _delivery_panel(sym: str, as_of: date, n: int, last_per: float) -> pd.DataFrame:
    rows = []
    for offset in range(n, 0, -1):
        d = (pd.Timestamp(as_of) - pd.Timedelta(days=offset - 1)).date()
        per = 30.0 if offset > 1 else last_per
        rows.append(
            {
                "SYMBOL": sym,
                "date": d,
                "TTL_TRD_QNTY": 100_000.0,
                "DELIV_QTY": per * 1000.0,
                "DELIV_PER": per,
            }
        )
    return pd.DataFrame(rows)


# ── delivery_pct_last / delivery_trend / delivery_spike ────────────────────


def test_delivery_metrics_add_trend_and_spike():
    panel = _delivery_panel("RELIANCE", date(2026, 5, 15), n=30, last_per=90.0)
    out = compute_delivery_metrics(panel)
    assert "delivery_trend" in out.columns
    assert "delivery_spike" in out.columns
    last = out.sort_values("date").iloc[-1]
    # trend = DELIV_PER / 20d mean; last bar (90) well above the ~30 baseline.
    assert last["delivery_trend"] > 1.5
    assert last["delivery_spike"] > 0  # positive z-score on the jump


def test_overlay_sets_delivery_last_trend_spike():
    as_of = date(2026, 5, 15)
    panel = _delivery_panel("RELIANCE", as_of, n=30, last_per=90.0)
    ev = _event()
    overlay_events([ev], panel)
    assert ev.delivery_pct_last == ev.delivery_pct == 90.0
    assert ev.delivery_trend is not None and ev.delivery_trend > 1.5
    assert ev.delivery_spike is not None and ev.delivery_spike > 0


def test_compute_delivery_metrics_empty_has_new_columns():
    out = compute_delivery_metrics(pd.DataFrame())
    assert {"delivery_trend", "delivery_spike"} <= set(out.columns)


# ── option chain (pcr / call_put_oi_ratio) ─────────────────────────────────


def test_compute_oc_metrics_prefers_filtered_totals():
    raw = {"filtered": {"CE": {"totOI": 1000}, "PE": {"totOI": 2000}}}
    m = option_chain.compute_oc_metrics(raw)
    assert m["call_put_oi_ratio"] == 0.5
    assert m["pcr"] == 2.0


def test_compute_oc_metrics_sums_records_when_no_filtered():
    raw = {
        "records": {
            "data": [
                {"CE": {"openInterest": 100}, "PE": {"openInterest": 50}},
                {"CE": {"openInterest": 300}, "PE": {"openInterest": 150}},
            ]
        }
    }
    m = option_chain.compute_oc_metrics(raw)
    assert m["call_put_oi_ratio"] == 2.0
    assert m["pcr"] == 0.5


def test_compute_oc_metrics_zero_leg_is_none():
    m = option_chain.compute_oc_metrics(
        {"filtered": {"CE": {"totOI": 0}, "PE": {"totOI": 100}}}
    )
    assert m["call_put_oi_ratio"] is None
    assert m["pcr"] is None


def test_overlay_option_chain_mutates_and_returns_map(monkeypatch):
    monkeypatch.setattr(
        option_chain,
        "fetch_option_chain",
        lambda sym, refresh=False: {
            "filtered": {"CE": {"totOI": 1000}, "PE": {"totOI": 2000}}
        },
    )
    ev = _event("TCS")
    out = option_chain.overlay_option_chain([ev], max_workers=2)
    assert ev.pcr == 2.0
    assert ev.call_put_oi_ratio == 0.5
    assert out["TCS"]["pcr"] == 2.0


def test_fetch_option_chain_rewarms_page_after_nse_reprime(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    for name in ("session", "primed", "primed_pages"):
        if hasattr(nse_client._tls, name):
            delattr(nse_client._tls, name)

    class Resp:
        def __init__(self, status_code: int, payload: dict | None = None) -> None:
            self.status_code = status_code
            self.payload = payload or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

        def json(self) -> dict:
            return self.payload

    class Session:
        def __init__(self, name: str) -> None:
            self.name = name
            self.headers = {}
            self.option_page_hits = 0

        def get(self, url: str, timeout: float = 10.0) -> Resp:
            del timeout
            if url == option_chain._OC_PAGE:
                self.option_page_hits += 1
                events.append(("option-page", self.name))
                return Resp(200)
            if url.endswith("/"):
                events.append(("home", self.name))
                return Resp(200)
            events.append(("api", self.name, self.option_page_hits))
            if self.name == "old":
                return Resp(403)
            return Resp(
                200,
                {"filtered": {"CE": {"totOI": 1000}, "PE": {"totOI": 2000}}},
            )

    events: list[tuple] = []
    sessions = iter([Session("old"), Session("new")])
    monkeypatch.setattr(nse_client, "_new_session", lambda: next(sessions))
    monkeypatch.setattr(
        nse_client, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )

    raw = option_chain.fetch_option_chain("TCS", refresh=True)

    assert raw == {"filtered": {"CE": {"totOI": 1000}, "PE": {"totOI": 2000}}}
    assert events == [
        ("home", "old"),
        ("option-page", "old"),
        ("api", "old", 1),
        ("home", "new"),
        ("option-page", "new"),
        ("api", "new", 1),
    ]


def test_extra_prime_page_happens_once_per_thread(monkeypatch):
    """The extra page is warmed once per thread/session, not per API call."""
    for name in ("session", "primed", "primed_pages"):
        if hasattr(nse_client._tls, name):
            delattr(nse_client._tls, name)

    page_hits = {"n": 0}

    class Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"ok": True}

    class Session:
        headers: dict = {}

        def get(self, url: str, timeout: float = 10.0):
            del timeout
            if url == "https://www.nseindia.com/page":
                page_hits["n"] += 1
            return Resp()

    monkeypatch.setattr(nse_client, "_new_session", lambda: Session())
    monkeypatch.setattr(
        nse_client, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )

    for _ in range(3):
        out = nse_client.fetch_nse_json(
            "https://www.nseindia.com/api/x",
            "x",
            extra_prime_page="https://www.nseindia.com/page",
        )
        assert out == {"ok": True}
    assert page_hits["n"] == 1  # primed once, reused across all 3 calls


def test_fetch_fno_ban_list_uses_primed_session(monkeypatch):
    from screener.unusual_volume import filters

    captured = {}

    def _fake_fetch_text(url, operation, *, timeout=8.0):
        captured["url"] = url
        captured["operation"] = operation
        return "Securities in Ban For Trade Date 01-JAN-2026:\n1,SAIL\n2,IDEA\n"

    monkeypatch.setattr(filters, "fetch_nse_text", _fake_fetch_text)
    out = filters.fetch_fno_ban_list()
    assert out == {"SAIL", "IDEA"}
    assert captured["url"] == filters.FNO_BAN_URL


def test_fetch_fno_ban_list_empty_on_failure(monkeypatch):
    from screener.unusual_volume import filters

    monkeypatch.setattr(filters, "fetch_nse_text", lambda url, op, timeout=8.0: None)
    assert filters.fetch_fno_ban_list() == set()


# ── trading calendar ───────────────────────────────────────────────────────


def test_calendar_walk_back_skips_stubbed_holiday():
    cal = nse_client.TradingCalendar()
    # Thu 01-01 + Fri 01-02 are stubbed holidays, Sat 01-03/Sun 01-04 weekends,
    # so on-or-before Sat 2026-01-03 walks back to Wed 2025-12-31.
    cal._holidays = {date(2026, 1, 1), date(2026, 1, 2)}
    assert cal.is_trading_day(date(2026, 1, 1)) is False
    assert cal.is_trading_day(date(2026, 1, 2)) is False
    assert cal.is_trading_day(date(2025, 12, 31)) is True
    got = cal.last_trading_day_on_or_before(date(2026, 1, 3))
    assert got == date(2025, 12, 31)


def test_calendar_degrades_to_weekday_only_when_holiday_fetch_fails(monkeypatch):
    cal = nse_client.TradingCalendar()
    monkeypatch.setattr(nse_client, "nse_cached_json", lambda *a, **k: None)
    # No holidays known -> weekday-only. 2026-01-03 is a Saturday; walk back to
    # Friday 2026-01-02 (which, holiday-unaware, is treated as a trading day).
    assert cal._holiday_set() == set()
    assert cal.is_trading_day(date(2026, 1, 3)) is False  # Saturday
    assert cal.last_trading_day_on_or_before(date(2026, 1, 3)) == date(2026, 1, 2)


# ── FII/DII derivation + broadcast ─────────────────────────────────────────


def _fii_panel(n: int) -> pd.DataFrame:
    base = date(2026, 4, 1)
    return pd.DataFrame(
        [
            {
                "date": (pd.Timestamp(base) + pd.Timedelta(days=i)).date(),
                "fii_net": 100.0 + i,
                "dii_net": 50.0 + i,
            }
            for i in range(n)
        ]
    )


def _append_panel_worker(root: str, name: str, rows: list[dict]) -> None:
    from screener import cache as worker_cache

    worker_cache.PANEL_ROOT = Path(root)
    worker_cache.append_panel_snapshot(
        name, pd.DataFrame(rows), dedupe_keys=["date", "symbol"]
    )


def test_compute_fii_dii_metrics_5d_and_trend():
    panel = _fii_panel(25)
    as_of = panel["date"].max()
    m = fii_dii.compute_fii_dii_metrics(panel, as_of)
    fii = panel["fii_net"]
    assert m["fii_5d_net"] == pytest.approx(fii.tail(5).sum())
    assert m["dii_5d_net"] == pytest.approx(panel["dii_net"].tail(5).sum())
    assert m["fii_trend"] == pytest.approx(round(fii.iloc[-1] / fii.tail(20).mean(), 4))


def test_compute_fii_dii_metrics_cold_start():
    panel = _fii_panel(3)
    m = fii_dii.compute_fii_dii_metrics(panel, panel["date"].max())
    assert m["fii_5d_net"] == pytest.approx(panel["fii_net"].sum())
    assert m["fii_trend"] is None  # < 5 rows


def test_join_microstructure_panels_normalizes_and_shifts(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    prepared = {
        "RELIANCE": pd.DataFrame(
            {"close": [10.0, 11.0, 12.0]},
            index=pd.to_datetime(
                ["2026-05-14 09:15", "2026-05-15 09:15", "2026-05-18 09:15"]
            ),
        )
    }
    cache.append_panel_snapshot(
        "option_chain",
        pd.DataFrame(
            [
                {
                    "SYMBOL": "RELIANCE",
                    "as_of": date(2026, 5, 14),
                    "call_put_oi_ratio": 1.2,
                    "pcr": 0.8,
                },
                {
                    "SYMBOL": "RELIANCE",
                    "as_of": date(2026, 5, 15),
                    "call_put_oi_ratio": 1.5,
                    "pcr": 0.6,
                },
            ]
        ),
        dedupe_keys=["SYMBOL", "as_of"],
    )
    cache.append_panel_snapshot(
        "fii_dii",
        pd.DataFrame(
            [
                {
                    "date": (pd.Timestamp("2026-05-10") + pd.Timedelta(days=i)).date(),
                    "fii_net": float(i + 1),
                    "dii_net": float(10 + i),
                }
                for i in range(6)
            ]
        ),
        dedupe_keys=["date"],
    )

    rs_breakout._join_microstructure_panels(prepared)
    frame = prepared["RELIANCE"]

    assert pd.isna(frame.iloc[0]["pcr"])
    assert frame.iloc[1]["pcr"] == pytest.approx(0.8)
    assert frame.iloc[1]["fii_5d_net"] == pytest.approx(15.0)
    assert frame.iloc[2]["pcr"] == pytest.approx(0.6)
    assert frame.iloc[2]["fii_5d_net"] == pytest.approx(20.0)


def test_overlay_fii_dii_broadcasts_identical(monkeypatch, tmp_path):
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
    assert evs[0].dii_5d_net == evs[1].dii_5d_net


def test_live_nse_overlays_preserve_historical_events(monkeypatch, tmp_path):
    historical_as_of = date(2026, 5, 1)
    live_snapshot_date = date(2026, 5, 15)
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    monkeypatch.setattr(service, "_live_nse_snapshot_date", lambda: live_snapshot_date)

    def overlay_option_chain(events, refresh=False):
        for ev in events:
            ev.call_put_oi_ratio = 0.5
            ev.pcr = 2.0
        return {
            "RELIANCE": {
                "ce_oi": 100.0,
                "pe_oi": 200.0,
                "call_put_oi_ratio": 0.5,
                "pcr": 2.0,
            }
        }

    monkeypatch.setattr(
        option_chain,
        "overlay_option_chain",
        overlay_option_chain,
    )
    monkeypatch.setattr(
        fii_dii,
        "fetch_fii_dii_today",
        lambda refresh=False: [
            {"category": "FII/FPI", "netValue": "123.45"},
            {"category": "DII", "netValue": "67.89"},
        ],
    )
    request = service.UnusualVolumeRequest(
        market="india",
        as_of=historical_as_of,
        universe=["NSE:RELIANCE"],
        min_rvol=0.0,
        min_z=0.0,
        strength_floor="MODERATE",
        min_avg_volume=0.0,
        include_fno_ban=False,
        deep_india=False,
        buildup_enabled=False,
        buildup_window=20,
        buildup_min_score=0.0,
        option_chain=True,
        fii_dii=True,
        pledge=False,
    )

    ev = _event("RELIANCE", historical_as_of)
    service._overlay_india_microstructure(request, [ev], Console(file=io.StringIO()))

    oc = cache.read_frame(cache.panel_path("option_chain"))
    fd = cache.read_frame(cache.panel_path("fii_dii"))
    assert ev.call_put_oi_ratio is None
    assert ev.pcr is None
    assert ev.fii_5d_net is None
    assert ev.dii_5d_net is None
    assert ev.fii_trend is None
    assert oc is not None
    assert fd is not None
    assert oc.iloc[0]["as_of"].date() == live_snapshot_date
    assert fd.iloc[0]["date"].date() == live_snapshot_date
    assert oc.iloc[0]["as_of"].date() != historical_as_of
    assert fd.iloc[0]["date"].date() != historical_as_of


def test_india_microstructure_runs_after_buildup_adds_events(monkeypatch):
    as_of = date(2026, 5, 15)
    idx = pd.date_range(end=as_of, periods=25, freq="D")
    bars = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [100_000.0] * len(idx),
        },
        index=idx,
    )
    seen_event_counts: list[int] = []

    def add_buildup_event(_request, _liquid, _panel, events, _console):
        events.append(_event("RELIANCE", as_of))

    def overlay_microstructure(_request, events, _console):
        seen_event_counts.append(len(events))
        for ev in events:
            ev.pcr = 2.0

    monkeypatch.setattr(
        service, "fetch_bars", lambda *args, **kwargs: {"NSE:RELIANCE": bars}
    )
    monkeypatch.setattr(service, "detect_market", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        service, "_overlay_india_delivery", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(service, "_apply_buildup_overlay", add_buildup_event)
    monkeypatch.setattr(
        service, "_overlay_india_microstructure", overlay_microstructure
    )
    monkeypatch.setattr(service, "fetch_sector_map", lambda *args, **kwargs: {})

    request = service.UnusualVolumeRequest(
        market="india",
        as_of=as_of,
        universe=["NSE:RELIANCE"],
        min_rvol=0.0,
        min_z=0.0,
        strength_floor="MODERATE",
        min_avg_volume=0.0,
        min_market_cap=0.0,
        include_fno_ban=True,
        deep_india=False,
        buildup_enabled=True,
        buildup_window=20,
        buildup_min_score=0.0,
        option_chain=True,
        fii_dii=False,
        pledge=False,
    )

    result = service.run_unusual_volume_scan(request, Console(file=io.StringIO()))

    assert seen_event_counts == [1]
    assert result.events[0].pcr == 2.0


# ── panel snapshot dedupe ──────────────────────────────────────────────────


def test_append_panel_snapshot_dedupes_keep_last(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    r1 = pd.DataFrame([{"date": date(2026, 5, 15), "fii_net": 1.0}])
    cache.append_panel_snapshot("t", r1, dedupe_keys=["date"])
    r2 = pd.DataFrame([{"date": date(2026, 5, 15), "fii_net": 9.0}])
    merged = cache.append_panel_snapshot("t", r2, dedupe_keys=["date"])
    assert len(merged) == 1
    assert merged.iloc[0]["fii_net"] == 9.0
    r3 = pd.DataFrame([{"date": date(2026, 5, 16), "fii_net": 5.0}])
    merged = cache.append_panel_snapshot("t", r3, dedupe_keys=["date"])
    assert len(merged) == 2


def test_append_panel_snapshot_dedupes_date_after_parquet_roundtrip(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    r1 = pd.DataFrame([{"date": date(2026, 5, 15), "fii_net": 1.0}])
    cache.append_panel_snapshot("typed_dates", r1, dedupe_keys=["date"])
    assert str(
        cache.read_frame(cache.panel_path("typed_dates"))["date"].dtype
    ).startswith("datetime64")
    r2 = pd.DataFrame([{"date": date(2026, 5, 15), "fii_net": 9.0}])
    merged = cache.append_panel_snapshot("typed_dates", r2, dedupe_keys=["date"])

    assert len(merged) == 1
    assert merged.iloc[0]["fii_net"] == 9.0


def test_append_panel_snapshot_concurrent_processes_keep_all_rows(tmp_path):
    name = "concurrent_panel"
    workers = []
    for idx in range(4):
        rows = [
            {
                "date": date(2026, 5, 15).isoformat(),
                "symbol": f"S{idx}",
                "value": float(idx),
            },
            {
                "date": date(2026, 5, 15).isoformat(),
                "symbol": f"S{idx}",
                "value": float(idx) + 0.5,
            },
            {
                "date": date(2026, 5, 16).isoformat(),
                "symbol": f"S{idx}",
                "value": float(idx) + 1.0,
            },
        ]
        workers.append(
            mp.Process(target=_append_panel_worker, args=(str(tmp_path), name, rows))
        )

    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join(timeout=10)
        assert proc.exitcode == 0

    frame = pd.read_parquet(tmp_path / f"{name}.parquet")
    assert len(frame) == 8
    assert not frame.duplicated(subset=["date", "symbol"]).any()


# ── pledge dual-source ─────────────────────────────────────────────────────


def test_resolve_pledge_prefers_nse(monkeypatch):
    calls = {"osc": 0}
    monkeypatch.setattr(pledge, "fetch_nse_pledge", lambda s, refresh=False: 12.5)

    def _osc(name, refresh=False):
        calls["osc"] += 1
        return 99.0

    monkeypatch.setattr(pledge, "fetch_openscreener_pledge", _osc)
    assert pledge.resolve_pledge_pct("RELIANCE", "RELIANCE") == 12.5
    assert calls["osc"] == 0  # fallback not invoked


def test_resolve_pledge_falls_back_to_openscreener(monkeypatch):
    monkeypatch.setattr(pledge, "fetch_nse_pledge", lambda s, refresh=False: None)
    monkeypatch.setattr(
        pledge, "fetch_openscreener_pledge", lambda n, refresh=False: 7.0
    )
    assert pledge.resolve_pledge_pct("X", "X") == 7.0


def test_resolve_pledge_both_none(monkeypatch):
    monkeypatch.setattr(pledge, "fetch_nse_pledge", lambda s, refresh=False: None)
    monkeypatch.setattr(
        pledge, "fetch_openscreener_pledge", lambda n, refresh=False: None
    )
    assert pledge.resolve_pledge_pct("X", "X") is None


def test_openscreener_pledge_regex(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    html = "<div>... Pledged percentage</span> <span>13.37%</span> ...</div>"

    class _S:
        def fetch_page(self, name):
            return html

    monkeypatch.setattr(pledge, "_HttpScraper", _S)
    val = pledge.fetch_openscreener_pledge("ZZZ", refresh=True)
    assert val == 13.37


# ── US regression: new fields stay None ────────────────────────────────────


def test_new_fields_default_none_for_us_event():
    ev = _event("AAPL")
    for field in (
        "delivery_pct_last",
        "delivery_trend",
        "delivery_spike",
        "call_put_oi_ratio",
        "pcr",
        "fii_5d_net",
        "fii_trend",
        "dii_5d_net",
        "pledge_pct",
    ):
        assert getattr(ev, field) is None
