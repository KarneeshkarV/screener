"""Tests for the 9 Indian micro-structure overlays (no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener import cache, pledge
from screener.unusual_volume import fii_dii, option_chain
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
