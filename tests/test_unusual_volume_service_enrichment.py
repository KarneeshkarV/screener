from __future__ import annotations

import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from rich.console import Console

from screener.unusual_volume import (
    Event,
)
from screener.unusual_volume import buildup as uv_buildup
from screener.unusual_volume import enrich as uv_enrich
from screener.unusual_volume import service as uv_service
from screener.unusual_volume.buildup import BuildupScore
from screener.unusual_volume.enrich import deep_enrich_india
from screener.unusual_volume.enrichment import Enrichment
from tests.conftest import make_bars


def _make_delivery_panel(symbols, n_days, as_of: date, deliv_qty_fn) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for offset in range(n_days, 0, -1):
            d = pd.Timestamp(as_of) - pd.Timedelta(days=offset - 1)
            qty = deliv_qty_fn(sym, offset)
            rows.append(
                {
                    "SYMBOL": sym,
                    "date": d.date(),
                    "TTL_TRD_QNTY": 100_000.0,
                    "DELIV_QTY": qty,
                    "DELIV_PER": (qty / 100_000.0) * 100.0,
                }
            )
    return pd.DataFrame(rows)


def _buildup_bars() -> pd.DataFrame:
    dates = pd.date_range("2026-02-01", periods=90)
    base = np.linspace(90.0, 120.0, len(dates))
    wave = np.sin(np.linspace(0, 8 * np.pi, len(dates)))
    close = base + wave
    open_ = close - 0.4
    high = close + 1.0
    low = close - 1.0 + np.linspace(0, 2.0, len(dates))
    volume = np.where(close > open_, 2_000.0, 800.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def _event_for_output(
    symbol: str,
    as_of: date,
    *,
    direction: str,
    strength: str,
    rvol: float = 3.0,
    sector: str | None = None,
    buildup_score: float | None = None,
    buildup_flags: list[str] | None = None,
    fii_5d_net: float | None = None,
    dii_5d_net: float | None = None,
    fii_trend: float | None = None,
) -> Event:
    return Event(
        symbol=symbol,
        date=as_of,
        close=100.0,
        pct_change=2.5,
        volume=150_000.0,
        avg_volume_20d=50_000.0,
        rvol=rvol,
        rvol_5d=rvol,
        rvol_50d=rvol,
        rvol_90d=rvol,
        z_score=2.5,
        pct_rank_252d=0.9,
        direction=direction,
        strength=strength,
        delivery_qty=60_000.0,
        delivery_pct=60.0,
        delivery_rvol=2.0,
        conviction_score=1.8,
        sector=sector,
        market_cap=10_000_000_000.0,
        notes="note",
        buildup_score=buildup_score,
        buildup_flags=buildup_flags or [],
        delivery_pct_last=60.0,
        delivery_trend=1.5,
        delivery_spike=2.0,
        call_put_oi_ratio=0.5,
        pcr=2.0,
        fii_5d_net=fii_5d_net,
        fii_trend=fii_trend,
        dii_5d_net=dii_5d_net,
        pledge_pct=1.0,
    )


def test_buildup_leaf_scoring_branches():
    as_of = date(2026, 4, 24)
    bars = _buildup_bars()
    short = bars.tail(5)

    with pytest.raises(ValidationError, match="symbol must not be empty"):
        BuildupScore(
            symbol=" ",
            as_of=as_of,
            window=20,
            range_compression=None,
            updown_volume=None,
            higher_lows=None,
            sustained_delivery=None,
            close_near_high=None,
            composite=0.0,
        )

    assert uv_buildup._score_range_compression(short, 20) == (None, None, None)
    zero_atr = bars.assign(high=1.0, low=1.0, close=1.0)
    assert uv_buildup._score_range_compression(zero_atr, 20) == (None, None, None)
    zero_basis = bars.assign(high=1.0, low=-1.0, close=0.0)
    zero_basis_score, _, zero_basis_bb = uv_buildup._score_range_compression(
        zero_basis, 20
    )
    assert zero_basis_score is not None
    assert zero_basis_bb is None
    range_score, atr_ratio, bb_ratio = uv_buildup._score_range_compression(bars, 20)
    assert range_score is not None
    assert atr_ratio is not None
    assert bb_ratio is not None

    assert uv_buildup._score_updown_volume(short, 20) == (None, None)
    assert uv_buildup._score_updown_volume(
        pd.DataFrame({"open": [1, 1], "close": [1, 1], "volume": [0, 0]}), 2
    ) == (None, None)
    assert uv_buildup._score_updown_volume(
        pd.DataFrame({"open": [1, 1], "close": [2, 2], "volume": [10, 20]}), 2
    ) == (1.0, None)
    mixed = bars.copy()
    mixed.iloc[-10:, mixed.columns.get_loc("open")] = (
        mixed.iloc[-10:]["close"].to_numpy() + 0.5
    )
    updown_score, updown_ratio = uv_buildup._score_updown_volume(mixed, 20)
    assert updown_score is not None
    assert updown_ratio is not None

    assert uv_buildup._score_higher_lows(short, 20) == (None, None)
    assert uv_buildup._score_higher_lows(bars.assign(low=0.0), 20) == (None, None)
    flat_score, flat_slope = uv_buildup._score_higher_lows(bars.assign(low=100.0), 20)
    assert flat_score == 0.0
    assert flat_slope == 0.0
    higher_score, higher_slope = uv_buildup._score_higher_lows(bars, 20)
    assert higher_score is not None
    assert higher_slope is not None
    falling = bars.copy()
    falling["low"] = np.linspace(120.0, 90.0, len(falling))
    falling_score, falling_slope = uv_buildup._score_higher_lows(falling, 20)
    assert falling_score == 0.0
    assert falling_slope is not None and falling_slope < 0
    assert uv_buildup._swing_lows(np.array([5, 4, 3, 4, 5, 2, 3]), k=1) == [3.0, 2.0]

    assert uv_buildup._score_close_near_high(short, 20) == (None, None)
    assert uv_buildup._score_close_near_high(bars.assign(high=1.0, low=1.0), 20) == (
        None,
        None,
    )
    close_score, absorption = uv_buildup._score_close_near_high(bars, 20)
    assert close_score is not None
    assert absorption is not None

    assert uv_buildup._score_sustained_delivery(None, "AAA", as_of, 20) == (
        None,
        None,
        None,
    )
    empty_panel = pd.DataFrame()
    assert uv_buildup._score_sustained_delivery(empty_panel, "AAA", as_of, 20) == (
        None,
        None,
        None,
    )
    short_panel = _make_delivery_panel(["AAA"], 2, as_of, lambda sym, offset: 50_000)
    assert uv_buildup._score_sustained_delivery(short_panel, "AAA", as_of, 20) == (
        None,
        None,
        None,
    )
    missing_panel = _make_delivery_panel(["BBB"], 20, as_of, lambda sym, offset: 50_000)
    assert uv_buildup._score_sustained_delivery(missing_panel, "AAA", as_of, 20) == (
        None,
        None,
        None,
    )
    nan_panel = _make_delivery_panel(["AAA"], 20, as_of, lambda sym, offset: 50_000)
    nan_panel["DELIV_PER"] = float("nan")
    assert uv_buildup._score_sustained_delivery(nan_panel, "AAA", as_of, 20) == (
        None,
        None,
        None,
    )
    panel = _make_delivery_panel(["AAA"], 20, as_of, lambda sym, offset: 60_000)
    delivery_score, delivery_mean, delivery_hit = uv_buildup._score_sustained_delivery(
        panel, "aaa", as_of, 20
    )
    assert delivery_score is not None
    assert delivery_mean == 60.0
    assert delivery_hit == 1.0


def test_buildup_compute_and_scan_paths():
    as_of = date(2026, 4, 24)
    bars = _buildup_bars()
    bars_with_date = bars.reset_index(names="date")
    panel = _make_delivery_panel(["AAA"], 20, as_of, lambda sym, offset: 60_000)

    assert uv_buildup.compute_buildup_score("AAA", None, as_of) is None
    assert uv_buildup.compute_buildup_score("AAA", pd.DataFrame(), as_of) is None
    assert (
        uv_buildup.compute_buildup_score("AAA", pd.DataFrame({"close": [1.0]}), as_of)
        is None
    )
    assert uv_buildup.compute_buildup_score("AAA", bars.tail(10), as_of) is None

    original = (
        uv_buildup._score_range_compression,
        uv_buildup._score_updown_volume,
        uv_buildup._score_higher_lows,
        uv_buildup._score_sustained_delivery,
        uv_buildup._score_close_near_high,
    )
    try:
        uv_buildup._score_range_compression = lambda *args, **kwargs: (None, None, None)
        uv_buildup._score_updown_volume = lambda *args, **kwargs: (None, None)
        uv_buildup._score_higher_lows = lambda *args, **kwargs: (None, None)
        uv_buildup._score_sustained_delivery = lambda *args, **kwargs: (
            None,
            None,
            None,
        )
        uv_buildup._score_close_near_high = lambda *args, **kwargs: (None, None)
        assert uv_buildup.compute_buildup_score("AAA", bars, as_of) is None

        uv_buildup._score_range_compression = lambda *args, **kwargs: (0.7, 0.1, 0.2)
        uv_buildup._score_updown_volume = lambda *args, **kwargs: (0.6, 2.0)
        uv_buildup._score_higher_lows = lambda *args, **kwargs: (0.6, 0.2)
        uv_buildup._score_sustained_delivery = lambda *args, **kwargs: (0.6, 60.0, 1.0)
        uv_buildup._score_close_near_high = lambda *args, **kwargs: (0.6, 0.8)
        flagged = uv_buildup.compute_buildup_score("AAA", bars, as_of)
        assert flagged is not None
        assert flagged.flags == [
            "compression",
            "up_vol_dominant",
            "higher_lows",
            "sustained_delivery",
            "close_near_high",
        ]
    finally:
        (
            uv_buildup._score_range_compression,
            uv_buildup._score_updown_volume,
            uv_buildup._score_higher_lows,
            uv_buildup._score_sustained_delivery,
            uv_buildup._score_close_near_high,
        ) = original

    score = uv_buildup.compute_buildup_score(
        "aaa", bars_with_date, as_of, delivery_panel=panel, window=20
    )
    assert score is not None
    assert score.symbol == "AAA"
    assert 0.0 <= score.composite <= 1.0
    assert score.to_dict()["symbol"] == "AAA"

    scores = uv_buildup.scan_buildups(
        {"LOW": bars.tail(10), "AAA": bars, "BBB": bars * 1.01},
        as_of,
        delivery_panel=panel,
        window=20,
        min_score=0.0,
    )
    assert [score.symbol for score in scores] == ["AAA", "BBB"]


def test_fetch_bars_maps_yfinance_symbols_and_handles_fetch_errors(monkeypatch):
    bars = make_bars(n=30, seed=21)

    class Fetcher:
        def fetch(self, symbols, start, end):
            assert symbols == ["AAA.NS", "BBB.NS"]
            return {"AAA.NS": bars, "BBB.NS": pd.DataFrame()}

    monkeypatch.setattr(
        uv_service, "build_price_fetcher", lambda refresh=False: Fetcher()
    )
    monkeypatch.setattr(uv_service, "tv_to_yf", lambda ticker, market: f"{ticker}.NS")
    console = Console(record=True)

    out = uv_service.fetch_bars(
        ["AAA", "BBB"], "india", date(2026, 1, 31), console, refresh=True
    )

    assert out == {"AAA": bars}

    class FailingFetcher:
        def fetch(self, symbols, start, end):
            raise ValueError("bad provider")

    monkeypatch.setattr(
        uv_service, "build_price_fetcher", lambda refresh=False: FailingFetcher()
    )
    assert uv_service.fetch_bars(["AAA"], "us", date(2026, 1, 31), console) == {}


def test_service_delivery_buildup_microstructure_and_scan(monkeypatch):
    as_of = date(2026, 4, 24)
    bars = make_bars(start="2026-01-01", n=120, seed=22)
    event = _event_for_output("NSE:AAA", as_of, direction="BUYING", strength="HIGH")
    quiet = _event_for_output(
        "AAA", as_of, direction="QUIET_ACCUMULATION", strength="MODERATE"
    )
    panel = _make_delivery_panel(
        ["AAA"],
        n_days=30,
        as_of=as_of,
        deliv_qty_fn=lambda sym, offset: 20_000.0 if offset > 1 else 60_000.0,
    )
    console = Console(record=True)

    monkeypatch.setattr(
        uv_service,
        "fetch_bars",
        lambda universe, market, as_of, console, refresh=False: {"NSE:AAA": bars},
    )
    monkeypatch.setattr(uv_service, "fetch_fno_ban_list", lambda: {"BANNED"})
    monkeypatch.setattr(uv_service, "passes_volume_floor", lambda *args, **kwargs: True)
    monkeypatch.setattr(uv_service, "detect_market", lambda *args, **kwargs: [event])
    monkeypatch.setattr(
        uv_service, "load_delivery_panel", lambda *args, **kwargs: panel
    )
    monkeypatch.setattr(uv_service, "overlay_events", lambda events, panel: None)
    monkeypatch.setattr(
        uv_service,
        "quiet_accumulation_events",
        lambda *args, **kwargs: [quiet],
    )
    monkeypatch.setattr(
        uv_service,
        "compute_buildup_score",
        lambda *args, **kwargs: BuildupScore(
            symbol="AAA",
            as_of=as_of,
            window=20,
            range_compression=0.5,
            updown_volume=0.6,
            higher_lows=0.7,
            sustained_delivery=0.8,
            close_near_high=0.9,
            composite=0.75,
            flags=["compression"],
        ),
    )
    monkeypatch.setattr(uv_service, "scan_buildups", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        uv_service, "fetch_sector_map", lambda *args, **kwargs: {"AAA": "IT"}
    )
    monkeypatch.setattr(
        uv_service,
        "attach_sector",
        lambda events, sectors: [
            setattr(e, "sector", sectors.get(e.symbol)) for e in events
        ],
    )
    monkeypatch.setattr(uv_service, "passes_market_cap", lambda market_cap, floor: True)
    monkeypatch.setattr(
        uv_service,
        "deep_enrich_india",
        lambda events: [setattr(e, "notes", "deep") for e in events],
    )
    monkeypatch.setattr(uv_service, "_live_nse_snapshot_date", lambda: as_of)

    option_mod = types.SimpleNamespace(
        overlay_option_chain=lambda events, refresh=False: {
            "AAA": {"ce_oi": 10, "pe_oi": 20, "call_put_oi_ratio": 0.5, "pcr": 2.0}
        }
    )
    fii_mod = types.SimpleNamespace(
        overlay_fii_dii=lambda events, snap_date, refresh=False: {
            "fii_5d_net": 1,
            "dii_5d_net": 2,
            "fii_trend": 3,
        }
    )
    pledge_mod = types.SimpleNamespace(
        overlay_pledge=lambda events, refresh=False: [
            setattr(e, "pledge_pct", 1.25) for e in events
        ]
    )
    monkeypatch.setitem(sys.modules, "screener.unusual_volume.option_chain", option_mod)
    monkeypatch.setitem(sys.modules, "screener.unusual_volume.fii_dii", fii_mod)
    monkeypatch.setitem(sys.modules, "screener.pledge", pledge_mod)
    monkeypatch.setattr(
        "screener.cache.append_panel_snapshot", lambda *args, **kwargs: None
    )

    req = uv_service.UnusualVolumeRequest(
        market="india",
        as_of=as_of,
        universe=["NSE:AAA"],
        min_rvol=2.0,
        min_z=2.0,
        strength_floor="MODERATE",
        min_avg_volume=0,
        min_market_cap=1,
        include_fno_ban=False,
        buildup_window=20,
        buildup_min_score=0.5,
        enrichments=frozenset(
            {
                Enrichment.DEEP_INDIA,
                Enrichment.BUILDUP,
                Enrichment.OPTION_CHAIN,
                Enrichment.FII_DII,
                Enrichment.PLEDGE,
            }
        ),
        refresh=True,
    )

    result = uv_service.run_unusual_volume_scan(req, console)

    assert result.fetched_count == 1
    assert result.liquid_count == 1
    assert [e.symbol for e in result.events] == ["AAA", "AAA"]
    assert all(e.notes == "deep" for e in result.events)


def test_service_scan_empty_paths_and_overlay_failures(monkeypatch):
    as_of = date(2026, 4, 24)
    console = Console(record=True)
    req = uv_service.UnusualVolumeRequest(
        market="us",
        as_of=as_of,
        universe=["AAA"],
        min_rvol=2.0,
        min_z=2.0,
        strength_floor="HIGH",
        min_avg_volume=0,
        min_market_cap=None,
        include_fno_ban=True,
        buildup_window=20,
        buildup_min_score=0.5,
    )

    monkeypatch.setattr(uv_service, "fetch_bars", lambda *args, **kwargs: {})
    empty = uv_service.run_unusual_volume_scan(req, console)
    assert empty.events == []
    assert empty.fetched_count == 0

    bars = make_bars(n=60, seed=23)
    monkeypatch.setattr(uv_service, "fetch_bars", lambda *args, **kwargs: {"AAA": bars})
    monkeypatch.setattr(
        uv_service, "passes_volume_floor", lambda *args, **kwargs: False
    )
    no_liquid = uv_service.run_unusual_volume_scan(req, console)
    assert no_liquid.fetched_count == 1
    assert no_liquid.liquid_count == 0

    india_req = req.model_copy(
        update={
            "market": "india",
            "include_fno_ban": True,
            "enrichments": frozenset(
                {Enrichment.OPTION_CHAIN, Enrichment.FII_DII, Enrichment.PLEDGE}
            ),
        }
    )
    ev = _event_for_output("AAA", as_of, direction="BUYING", strength="HIGH")
    monkeypatch.setattr(uv_service, "passes_volume_floor", lambda *args, **kwargs: True)
    monkeypatch.setattr(uv_service, "detect_market", lambda *args, **kwargs: [ev])
    monkeypatch.setattr(
        uv_service,
        "load_delivery_panel",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("delivery down")),
    )
    monkeypatch.setattr(uv_service, "fetch_sector_map", lambda *args, **kwargs: {})
    monkeypatch.setattr(uv_service, "passes_market_cap", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        uv_service, "_live_nse_snapshot_date", lambda: date(2026, 4, 25)
    )
    monkeypatch.setitem(
        sys.modules,
        "screener.unusual_volume.option_chain",
        types.SimpleNamespace(
            overlay_option_chain=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("option down")
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "screener.unusual_volume.fii_dii",
        types.SimpleNamespace(
            overlay_fii_dii=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("fii down")
            )
        ),
    )

    out = uv_service.run_unusual_volume_scan(india_req, console)

    assert len(out.events) == 1
    rendered = console.export_text()
    assert "Delivery overlay failed" in rendered
    assert "Option-chain overlay failed" in rendered
    assert "FII/DII overlay failed" in rendered
    assert "Pledge overlay skipped" in rendered


def test_deep_enrich_india_handles_section_based_openscreener(monkeypatch):
    class FakeStock:
        def __init__(self, symbol: str, **kwargs) -> None:
            self.symbol = symbol

        def fetch(self, sections: str):
            assert sections == "shareholding"
            return {"shareholding": [{"date": "Mar 2026", "promoters": "51.25"}]}

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    ev = Event(
        symbol="SUYOG",
        date=date(2026, 5, 28),
        close=100.0,
        pct_change=1.0,
        volume=10_000,
        avg_volume_20d=5_000,
        rvol=2.0,
        rvol_5d=2.0,
        rvol_50d=2.0,
        rvol_90d=2.0,
        z_score=2.0,
        pct_rank_252d=0.9,
        direction="BUYING",
        strength="MODERATE",
    )

    deep_enrich_india([ev])

    assert ev.notes == "promoter holding 51.2%"


def test_enrich_sector_map_and_attach(monkeypatch):
    rows = pd.DataFrame(
        [
            {"name": "AAA", "sector": "Technology", "market_cap_basic": 1_000_000.0},
            {"name": "BBB", "sector": None, "market_cap_basic": float("nan")},
            {"name": "", "sector": "Ignored", "market_cap_basic": 1.0},
        ]
    )
    captured = {}

    def fake_fetch(key, loader, **kwargs):
        captured["key"] = key
        captured["kwargs"] = kwargs
        return rows

    monkeypatch.setattr(uv_enrich._TV_SECTOR_PROVIDER, "fetch", fake_fetch)

    assert uv_enrich.fetch_sector_map("bad", ["AAA"]) == {}
    assert uv_enrich.fetch_sector_map("us", []) == {}
    sector_map = uv_enrich.fetch_sector_map(
        "us", ["aaa", "AAA", "bbb"], cache_ttl=60, refresh=True
    )

    assert captured["key"] == ("sector_enrichment", "us", ["AAA", "BBB"])
    assert captured["kwargs"]["refresh"] is True
    assert captured["kwargs"]["ttl_seconds"] == 60
    assert sector_map == {
        "AAA": {"sector": "Technology", "market_cap": 1_000_000.0},
        "BBB": {"sector": None, "market_cap": None},
    }

    events = [
        _event_for_output("AAA", date(2026, 1, 1), direction="BUYING", strength="HIGH"),
        _event_for_output("CCC", date(2026, 1, 1), direction="BUYING", strength="HIGH"),
    ]
    uv_enrich.attach_sector(events, sector_map)
    assert events[0].sector == "Technology"
    assert events[0].market_cap == 1_000_000.0
    assert events[1].sector is None


def test_enrich_sector_map_empty_provider(monkeypatch):
    monkeypatch.setattr(
        uv_enrich._TV_SECTOR_PROVIDER,
        "fetch",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    assert uv_enrich.fetch_sector_map("india", ["AAA"]) == {}


def test_deep_enrich_india_fetch_variants_and_failures(monkeypatch):
    ev = _event_for_output("AAA", date(2026, 1, 1), direction="BUYING", strength="HIGH")
    existing = _event_for_output(
        "BBB", date(2026, 1, 1), direction="BUYING", strength="HIGH"
    )
    existing.notes = "existing"

    class FetchNoArgs:
        def __init__(self, symbol: str, **kwargs) -> None:
            self.symbol = symbol

        def fetch(self):
            return {
                "shareholding": pd.DataFrame(
                    {"Mar 2026": ["51.0"]}, index=["Promoters"]
                )
            }

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FetchNoArgs)
    )
    uv_enrich.deep_enrich_india([ev])
    assert ev.notes == "note; promoter holding 51.0%"

    class PropertyOnly:
        def __init__(self, symbol: str, **kwargs) -> None:
            self.shareholding_quarterly = pd.DataFrame(
                {"Mar 2026": ["55.5"]}, index=["Promoters"]
            )

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=PropertyOnly)
    )
    uv_enrich.deep_enrich_india([existing])
    assert existing.notes == "existing; promoter holding 55.5%"

    class RaisingStock:
        def __init__(self, symbol: str, **kwargs) -> None:
            raise RuntimeError("scrape failed")

    untouched = _event_for_output(
        "CCC", date(2026, 1, 1), direction="BUYING", strength="HIGH"
    )
    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=RaisingStock)
    )
    uv_enrich.deep_enrich_india([untouched])
    assert untouched.notes == "note"

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "openscreener":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "openscreener", raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)
    uv_enrich.deep_enrich_india([untouched])
    assert untouched.notes == "note"


def test_deep_enrich_india_empty_and_promoter_failure_paths(monkeypatch):
    as_of = date(2026, 1, 1)
    empty_event = _event_for_output("EMPTY", as_of, direction="BUYING", strength="HIGH")
    none_event = _event_for_output("NONE", as_of, direction="BUYING", strength="HIGH")
    callable_event = _event_for_output(
        "CALLABLE", as_of, direction="BUYING", strength="HIGH"
    )
    raising_event = _event_for_output(
        "RAISING", as_of, direction="BUYING", strength="HIGH"
    )

    class StockVariants:
        def __init__(self, symbol: str, **kwargs) -> None:
            self.symbol = symbol

        def fetch(self, section: str):
            if self.symbol == "EMPTY":
                return {"shareholding": pd.DataFrame()}
            if self.symbol == "NONE":
                return {"shareholding": [{"public": "10"}]}
            if self.symbol == "RAISING":
                return {"shareholding": [{"Promoters": "55"}]}
            return {}

        def shareholding_quarterly(self):
            return [{"promoters": "57"}]

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=StockVariants)
    )
    original_extract = uv_enrich._extract_promoter_pct

    def fake_extract(df):
        if isinstance(df, list) and df and df[0].get("Promoters") == "55":
            raise ValueError("bad promoter table")
        return original_extract(df)

    monkeypatch.setattr(uv_enrich, "_extract_promoter_pct", fake_extract)

    uv_enrich.deep_enrich_india(
        [empty_event, none_event, callable_event, raising_event]
    )

    assert empty_event.notes == "note"
    assert none_event.notes == "note"
    assert callable_event.notes == "note; promoter holding 57.0%"
    assert raising_event.notes == "note"


def test_enrich_extract_promoter_pct_shapes():
    assert uv_enrich._extract_promoter_pct(None) is None
    assert uv_enrich._extract_promoter_pct([]) is None
    assert uv_enrich._extract_promoter_pct(["bad"]) is None
    assert uv_enrich._extract_promoter_pct([{"Promoters": "52.25%"}]) == 52.25
    assert uv_enrich._extract_promoter_pct([{"public": "10"}]) is None
    assert (
        uv_enrich._extract_promoter_pct(
            pd.DataFrame({"Mar 2026": [pd.NA]}, index=["Promoters"])
        )
        is None
    )
    assert (
        uv_enrich._extract_promoter_pct(
            pd.DataFrame({"Mar 2026": ["bad"]}, index=["Promoters"])
        )
        is None
    )
    assert (
        uv_enrich._extract_promoter_pct(pd.DataFrame({"x": [1]}, index=["Public"]))
        is None
    )
