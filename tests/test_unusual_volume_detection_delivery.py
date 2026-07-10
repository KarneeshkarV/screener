from __future__ import annotations


import json


import sys


import types


from datetime import date


import numpy as np


import pandas as pd


import pytest


from pydantic import ValidationError


from rich.console import Console


from screener.unusual_volume import (
    DEFAULT_MIN_RVOL,
    Event,
    detect_market,
    detect_ticker,
)


from screener.unusual_volume import service as uv_service


from screener.unusual_volume.buildup import BuildupScore


from screener.unusual_volume.classify import classify_direction, classify_strength


from screener.unusual_volume.delivery import (
    compute_delivery_metrics,
    overlay_events,
    quiet_accumulation_events,
)


from screener.unusual_volume.filters import _parse_ban_csv, passes_volume_floor


from screener.unusual_volume.output import (
    _color_direction,
    _color_strength,
    _fii_dii_footer,
    _json_safe,
    _sort_by_buildup,
    render_rich,
    sort_events,
    write_json,
    write_markdown,
)


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


def test_direction_buying():
    # close > open AND close in upper third of range
    assert (
        classify_direction(open_px=100, high=110, low=99, close=109, prev_close=100)
        == "BUYING"
    )


def test_direction_selling():
    assert (
        classify_direction(open_px=100, high=101, low=90, close=91, prev_close=100)
        == "SELLING"
    )


def test_direction_churn_small_change():
    assert (
        classify_direction(open_px=100, high=102, low=99, close=100.3, prev_close=100)
        == "CHURN"
    )


def test_direction_reversal_gap_up_close_down():
    # gap up 3%, but bar closes below prev_close → reversal
    assert (
        classify_direction(open_px=103, high=104, low=98, close=99, prev_close=100)
        == "REVERSAL"
    )


def test_direction_reversal_gap_down_close_up():
    assert (
        classify_direction(open_px=97, high=104, low=96.5, close=103, prev_close=100)
        == "REVERSAL"
    )


def test_direction_defaults_to_churn_for_midrange_bar():
    assert (
        classify_direction(open_px=100, high=110, low=90, close=101, prev_close=0)
        == "CHURN"
    )


def test_strength_tiers():
    assert classify_strength(rvol=1.5, z=1.5) == "MODERATE"
    assert classify_strength(rvol=3.5, z=2.0) == "HIGH"
    assert classify_strength(rvol=2.0, z=2.7) == "HIGH"
    assert classify_strength(rvol=6.0, z=2.0) == "EXTREME"
    assert classify_strength(rvol=2.0, z=4.0) == "EXTREME"


def test_detector_emits_extreme_event_on_volume_spike():
    bars = make_bars(start="2024-01-01", n=300, seed=1)
    spike_idx = 299
    avg = float(bars["volume"].iloc[200:299].mean())
    bars.iat[spike_idx, bars.columns.get_loc("volume")] = avg * 8.0
    bars.iat[spike_idx, bars.columns.get_loc("open")] = 100.0
    bars.iat[spike_idx, bars.columns.get_loc("low")] = 99.0
    bars.iat[spike_idx, bars.columns.get_loc("high")] = 110.0
    bars.iat[spike_idx, bars.columns.get_loc("close")] = 109.5
    # Set prior close so pct_change is positive and direction = BUYING
    bars.iat[spike_idx - 1, bars.columns.get_loc("close")] = 100.0
    spike_date = bars.index[spike_idx].date()

    ev = detect_ticker("AAPL", bars, spike_date)
    assert ev is not None
    assert ev.symbol == "AAPL"
    assert ev.strength == "EXTREME"
    assert ev.direction == "BUYING"
    assert ev.rvol > 5.0
    assert ev.volume == avg * 8.0


def test_detector_drops_normal_volume_bars():
    bars = make_bars(n=300, seed=2)
    last_date = bars.index[-1].date()
    ev = detect_ticker("MSFT", bars, last_date)
    assert ev is None


def test_detect_market_runs_per_ticker():
    quiet = make_bars(n=300, seed=3)
    spiked = make_bars(n=300, seed=4)
    spike_idx = 299
    avg = float(spiked["volume"].iloc[200:299].mean())
    spiked.iat[spike_idx, spiked.columns.get_loc("volume")] = avg * 4.0
    as_of = spiked.index[-1].date()

    events = detect_market({"QUIET": quiet, "SPIKE": spiked}, as_of)
    syms = {e.symbol for e in events}
    assert "SPIKE" in syms
    assert "QUIET" not in syms


def test_detector_handles_short_history():
    bars = make_bars(n=10, seed=5)
    ev = detect_ticker("X", bars, bars.index[-1].date())
    assert ev is None  # not enough history for SMA20


def test_passes_volume_floor_drops_thin_names():
    bars = make_bars(n=60, seed=6)
    # Force volumes well below 1M
    assert (
        passes_volume_floor(bars, min_avg_volume=1_000_000, as_of=bars.index[-1].date())
        is False
    )
    assert (
        passes_volume_floor(bars, min_avg_volume=1_000, as_of=bars.index[-1].date())
        is True
    )


def test_passes_volume_floor_rejects_nan_rolling_average():
    bars = make_bars(n=60, seed=6)
    # A NaN volume inside the trailing 20-day window leaves the rolling mean
    # undefined; the ticker must be ineligible, not compared against NaN.
    bars.iat[-5, bars.columns.get_loc("volume")] = float("nan")
    assert (
        passes_volume_floor(bars, min_avg_volume=1_000, as_of=bars.index[-1].date())
        is False
    )


def test_parse_ban_csv():
    text = "Securities in Ban For Trade Date 27-APR-2026:\n1,SAIL\n2,FOO\n"
    assert _parse_ban_csv(text) == {"SAIL", "FOO"}


def test_filter_helpers_cover_fetch_and_market_cap_branches(monkeypatch):
    from screener.unusual_volume import filters

    monkeypatch.setattr(filters, "fetch_nse_text", lambda *args, **kwargs: None)
    assert filters.fetch_fno_ban_list(timeout=1.0) == set()
    monkeypatch.setattr(filters, "fetch_nse_text", lambda *args, **kwargs: "IDEA\n")
    assert filters.fetch_fno_ban_list(timeout=1.0) == {"IDEA"}
    assert _parse_ban_csv(" lone \n1,\n2,SAIL\n") == {"LONE", "SAIL"}

    assert passes_volume_floor(None, min_avg_volume=0, as_of=date(2026, 1, 1)) is False
    assert (
        passes_volume_floor(
            pd.DataFrame({"close": [1.0]}),
            min_avg_volume=0,
            as_of=date(2026, 1, 1),
        )
        is False
    )
    short_with_date = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=5), "volume": [1, 2, 3, 4, 5]}
    )
    assert (
        passes_volume_floor(short_with_date, min_avg_volume=0, as_of=date(2026, 1, 5))
        is False
    )

    assert filters.passes_market_cap(1, min_market_cap=0) is True
    assert filters.passes_market_cap(None, min_market_cap=100) is True
    assert filters.passes_market_cap(float("nan"), min_market_cap=100) is True
    assert filters.passes_market_cap(99, min_market_cap=100) is False
    assert filters.passes_market_cap(100, min_market_cap=100) is True


def test_overlay_events_adds_delivery_fields():
    as_of = date(2026, 4, 24)
    panel = _make_delivery_panel(
        ["RELIANCE"],
        n_days=30,
        as_of=as_of,
        deliv_qty_fn=lambda sym, offset: 30_000.0 if offset > 1 else 60_000.0,
    )
    ev = Event(
        symbol="RELIANCE",
        date=as_of,
        close=2500.0,
        pct_change=2.5,
        volume=150_000,
        avg_volume_20d=50_000,
        rvol=3.0,
        rvol_5d=3.1,
        rvol_50d=2.9,
        rvol_90d=2.8,
        z_score=2.7,
        pct_rank_252d=0.97,
        direction="BUYING",
        strength="HIGH",
    )
    overlay_events([ev], panel)
    assert ev.delivery_qty == 60_000.0
    assert ev.delivery_pct == 60.0
    # Delivery RVOL ≈ 60_000 / 30_000 = 2.0
    assert ev.delivery_rvol is not None and abs(ev.delivery_rvol - 2.0) < 1e-6
    # Conviction = rvol * delivery_pct / 100 = 3.0 * 0.6 = 1.8
    assert abs(ev.conviction_score - 1.8) < 1e-6
    assert "strong institutional footprint" in ev.notes


def test_overlay_long_holder_distribution_note():
    as_of = date(2026, 4, 24)
    panel = _make_delivery_panel(
        ["INFY"],
        n_days=30,
        as_of=as_of,
        deliv_qty_fn=lambda sym, offset: 20_000.0 if offset > 1 else 70_000.0,
    )
    ev = Event(
        symbol="INFY",
        date=as_of,
        close=1500.0,
        pct_change=-3.2,
        volume=200_000,
        avg_volume_20d=50_000,
        rvol=4.0,
        rvol_5d=3.5,
        rvol_50d=3.0,
        rvol_90d=2.8,
        z_score=3.0,
        pct_rank_252d=0.99,
        direction="SELLING",
        strength="HIGH",
    )
    overlay_events([ev], panel)
    assert ev.delivery_pct is not None and ev.delivery_pct > 60.0
    assert "long-holder distribution" in ev.notes


def test_quiet_accumulation_event():
    """Delivery RVOL >= 2 even when raw volume RVOL is below threshold."""
    bars = make_bars(start="2024-01-01", n=300, seed=7)
    # Map by index so the as-of date matches the last bar.
    bars_by_symbol = {"RELIANCE": bars}
    # Synthesize a delivery panel with an as-of-day spike.
    last_date = bars.index[-1].date()
    panel = _make_delivery_panel(
        ["RELIANCE"],
        n_days=30,
        as_of=last_date,
        deliv_qty_fn=lambda sym, offset: 20_000.0 if offset > 1 else 60_000.0,
    )
    quiet = quiet_accumulation_events(
        bars_by_symbol, panel, last_date, min_rvol_skip=DEFAULT_MIN_RVOL
    )
    assert len(quiet) == 1
    ev = quiet[0]
    assert ev.symbol == "RELIANCE"
    assert ev.direction == "QUIET_ACCUMULATION"
    assert ev.delivery_pct == 60.0
    assert ev.delivery_rvol is not None and ev.delivery_rvol >= 2.0
    assert "quiet accumulation" in ev.notes.lower()


def test_quiet_accumulation_skips_existing_detector_event():
    bars = make_bars(start="2024-01-01", n=300, seed=8)
    bars["volume"] = 100_000.0
    last_date = bars.index[-1].date()
    panel = _make_delivery_panel(
        ["RELIANCE"],
        n_days=30,
        as_of=last_date,
        deliv_qty_fn=lambda sym, offset: 20_000.0 if offset > 1 else 60_000.0,
    )
    existing = Event(
        symbol="RELIANCE",
        date=last_date,
        close=100.0,
        pct_change=0.0,
        volume=100_000.0,
        avg_volume_20d=100_000.0,
        rvol=1.0,
        rvol_5d=1.0,
        rvol_50d=1.0,
        rvol_90d=1.0,
        z_score=2.5,
        pct_rank_252d=0.99,
        direction="BUYING",
        strength="HIGH",
    )
    quiet = quiet_accumulation_events(
        {"RELIANCE": bars},
        panel,
        last_date,
        min_rvol_skip=DEFAULT_MIN_RVOL,
        existing_events=[existing],
    )
    assert quiet == []


def test_compute_delivery_metrics_handles_empty():
    out = compute_delivery_metrics(pd.DataFrame())
    assert out.empty
    assert "delivery_rvol" in out.columns


def test_standalone_buildup_event_uses_as_of_bar():
    bars = make_bars(start="2024-01-01", n=8, seed=9)
    as_of = bars.index[4].date()
    bars.iat[3, bars.columns.get_loc("close")] = 90.0
    bars.iat[4, bars.columns.get_loc("close")] = 100.0
    bars.iat[4, bars.columns.get_loc("volume")] = 1_000.0
    bars.iat[5, bars.columns.get_loc("close")] = 500.0
    bars.iat[5, bars.columns.get_loc("volume")] = 9_000.0
    score = BuildupScore(
        symbol="AAA",
        as_of=as_of,
        window=20,
        range_compression=0.7,
        updown_volume=0.6,
        higher_lows=0.6,
        sustained_delivery=None,
        close_near_high=0.7,
        composite=0.65,
        flags=["compression"],
    )
    ev = uv_service.standalone_buildup_event(score, bars, as_of)
    assert ev is not None
    assert ev.close == 100.0
    assert ev.volume == 1_000.0
    assert ev.pct_change == 11.1111


def test_write_json_sanitizes_nonfinite_metrics(tmp_path):
    ev = Event(
        symbol="AAA",
        date=date(2026, 4, 24),
        close=100.0,
        pct_change=0.0,
        volume=1_000.0,
        avg_volume_20d=0.0,
        rvol=float("nan"),
        rvol_5d=float("nan"),
        rvol_50d=float("nan"),
        rvol_90d=float("nan"),
        z_score=float("inf"),
        pct_rank_252d=float("-inf"),
        direction="BUILDUP",
        strength="MODERATE",
        market_cap=float("nan"),
    )
    path = tmp_path / "events.json"
    write_json([ev], path)
    text = path.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    payload = json.loads(
        text,
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError(f"non-strict JSON token: {token}")
        ),
    )
    assert payload[0]["rvol"] is None
    assert payload[0]["z_score"] is None
    assert payload[0]["pct_rank_252d"] is None
    assert payload[0]["market_cap"] is None


def test_output_sort_render_markdown_and_json_helpers(tmp_path):
    as_of = date(2026, 4, 24)
    buying = _event_for_output(
        "AAA",
        as_of,
        direction="BUYING",
        strength="EXTREME",
        rvol=4.0,
        sector="Tech",
        fii_5d_net=1.5,
        dii_5d_net=-0.5,
        fii_trend=0.2,
    )
    selling = _event_for_output(
        "BBB", as_of, direction="SELLING", strength="HIGH", rvol=float("nan")
    )
    buildup = _event_for_output(
        "CCC",
        as_of,
        direction="BUILDUP",
        strength="MODERATE",
        buildup_score=0.9,
        buildup_flags=["tight range"],
    )

    assert [e.symbol for e in sort_events([selling, buying, buildup])] == [
        "AAA",
        "BBB",
        "CCC",
    ]
    assert _sort_by_buildup([buying, buildup])[0].symbol == "CCC"
    assert "FII 5d net" in _fii_dii_footer([buying])
    assert _fii_dii_footer([selling]) == ""
    assert _color_direction("QUIET_ACCUMULATION") == "[cyan]QUIET ACC[/cyan]"
    assert _color_direction("UNKNOWN") == "UNKNOWN"
    assert _color_strength("EXTREME") == "[bold red]EXTREME[/bold red]"
    assert _color_strength("LOW") == "LOW"
    assert _json_safe({"a": [float("nan"), pd.NA], "b": (1, 2)}) == {
        "a": [None, None],
        "b": [1, 2],
    }
    ambiguous = pd.Series([1])
    assert _json_safe(ambiguous) is ambiguous

    console = Console(record=True, width=180)
    render_rich([], "us", as_of, console)
    render_rich([buying, selling, buildup], "india", as_of, console)
    rendered = console.export_text()
    assert "No unusual-volume events" in rendered
    assert "Unusual Volume" in rendered
    assert "Market-wide FII/DII" in rendered

    md_path = tmp_path / "uv.md"
    write_markdown([buying, selling, buildup], md_path, "india", as_of)
    md = md_path.read_text()
    assert "## BUYING (1)" in md
    assert "## SELLING (1)" in md
    assert "## BUILDUP (1)" in md
    assert "tight range" in md

    us_path = tmp_path / "uv_us.md"
    write_markdown([buying], us_path, "us", as_of)
    assert "Volume" in us_path.read_text()


def test_service_models_and_small_helpers():
    with pytest.raises(ValidationError, match="value must not be empty"):
        uv_service.UnusualVolumeRequest(
            market=" ",
            as_of=date(2026, 4, 24),
            universe=["AAA"],
            min_rvol=1,
            min_z=1,
            strength_floor="HIGH",
            min_avg_volume=0,
            include_fno_ban=False,
            deep_india=False,
            buildup_enabled=False,
            buildup_window=20,
            buildup_min_score=0.5,
        )
    with pytest.raises(ValidationError, match="universe must include"):
        uv_service.UnusualVolumeRequest(
            market="us",
            as_of=date(2026, 4, 24),
            universe=[" ", ""],
            min_rvol=1,
            min_z=1,
            strength_floor="HIGH",
            min_avg_volume=0,
            include_fno_ban=False,
            deep_india=False,
            buildup_enabled=False,
            buildup_window=20,
            buildup_min_score=0.5,
        )

    assert uv_service.india_symbol("NSE:reliance") == "RELIANCE"
    assert uv_service.india_symbol("tcs") == "TCS"
    assert uv_service._human_mcap(2_500_000_000) == "$2.5B"
    assert uv_service._human_mcap(250_000_000) == "$250M"
    assert uv_service._human_mcap(25_000) == "$25,000"

    no_date = pd.DataFrame({"close": [1.0]})
    assert uv_service.bars_on_or_before_as_of(None, date(2026, 1, 1)).empty
    assert uv_service.bars_on_or_before_as_of(no_date, date(2026, 1, 1)).empty
    dated = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
            "close": [1.0, 3.0],
        }
    )
    filtered = uv_service.bars_on_or_before_as_of(dated, date(2026, 1, 2))
    assert filtered["close"].tolist() == [1.0]

    one_bar = pd.DataFrame(
        {"date": [pd.Timestamp("2026-01-01")], "close": [0.0], "volume": [5.0]}
    )
    score = BuildupScore(
        symbol="AAA",
        as_of=date(2026, 1, 1),
        window=20,
        range_compression=0.1,
        updown_volume=0.2,
        higher_lows=0.3,
        sustained_delivery=None,
        close_near_high=0.4,
        composite=0.5,
        flags=[],
    )
    standalone = uv_service.standalone_buildup_event(score, one_bar, date(2026, 1, 1))
    assert standalone is not None
    assert standalone.pct_change == 0.0
    assert (
        uv_service.standalone_buildup_event(score, pd.DataFrame(), date(2026, 1, 1))
        is None
    )


def test_service_private_overlay_helpers_cover_fallbacks(monkeypatch):
    as_of = date(2026, 4, 24)
    console = Console(record=True)
    req_us = uv_service.UnusualVolumeRequest(
        market="us",
        as_of=as_of,
        universe=["AAA"],
        min_rvol=2.0,
        min_z=2.0,
        strength_floor="HIGH",
        min_avg_volume=0,
        min_market_cap=None,
        include_fno_ban=True,
        deep_india=False,
        buildup_enabled=True,
        buildup_window=20,
        buildup_min_score=0.5,
    )
    assert uv_service._overlay_india_delivery(
        req_us, {"AAA": make_bars()}, [], console
    ).empty
    uv_service._overlay_india_microstructure(req_us, [], console)

    req_india = req_us.model_copy(update={"market": "india"})
    monkeypatch.setattr(
        uv_service, "load_delivery_panel", lambda *args, **kwargs: pd.DataFrame()
    )
    assert uv_service._overlay_india_delivery(
        req_india, {"NSE:AAA": make_bars()}, [], console
    ).empty

    ev = _event_for_output("AAA", as_of, direction="BUYING", strength="HIGH")
    bars = make_bars(start="2026-01-01", n=40, seed=24)
    extra_score = BuildupScore(
        symbol="BBB",
        as_of=as_of,
        window=20,
        range_compression=0.1,
        updown_volume=0.2,
        higher_lows=0.3,
        sustained_delivery=None,
        close_near_high=0.4,
        composite=0.6,
        flags=["extra"],
    )
    duplicate_score = extra_score.model_copy(update={"symbol": "AAA"})
    missing_score = extra_score.model_copy(update={"symbol": "MISSING"})
    empty_score = extra_score.model_copy(update={"symbol": "EMPTY"})

    monkeypatch.setattr(
        uv_service, "compute_buildup_score", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        uv_service,
        "scan_buildups",
        lambda *args, **kwargs: [
            duplicate_score,
            missing_score,
            empty_score,
            extra_score,
        ],
    )
    uv_service._apply_buildup_overlay(
        req_us,
        {"AAA": bars, "EMPTY": pd.DataFrame(), "BBB": bars},
        pd.DataFrame(),
        [ev],
        console,
    )

    assert [e.symbol for e in [ev] if e.direction == "BUYING"] == ["AAA"]

    events = [ev]
    monkeypatch.setattr(
        uv_service,
        "scan_buildups",
        lambda *args, **kwargs: [extra_score],
    )
    monkeypatch.setattr(uv_service, "standalone_buildup_event", lambda *args: None)
    uv_service._apply_buildup_overlay(
        req_us,
        {"BBB": bars},
        pd.DataFrame(),
        events,
        console,
    )
    assert events == [ev]

    pledge_req = req_us.model_copy(update={"market": "india", "pledge": True})
    monkeypatch.setattr(uv_service, "_live_nse_snapshot_date", lambda: as_of)
    monkeypatch.setitem(
        sys.modules,
        "screener.pledge",
        types.SimpleNamespace(
            overlay_pledge=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("pledge down")
            )
        ),
    )
    uv_service._overlay_india_microstructure(pledge_req, [ev], console)
    assert "Pledge overlay failed" in console.export_text()


def test_live_snapshot_date_success_and_fallback(monkeypatch):
    import screener.operator.fetch as operator_fetch

    monkeypatch.setattr(
        operator_fetch, "latest_trading_day", lambda today: date(2026, 4, 23)
    )
    assert uv_service._live_nse_snapshot_date() == date(2026, 4, 23)

    monkeypatch.setattr(
        operator_fetch,
        "latest_trading_day",
        lambda today: (_ for _ in ()).throw(RuntimeError("calendar down")),
    )
    assert uv_service._live_nse_snapshot_date() == date.today()
