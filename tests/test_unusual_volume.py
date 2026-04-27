from __future__ import annotations

from datetime import date

import pandas as pd

from screener.unusual_volume import (
    DEFAULT_MIN_RVOL,
    Event,
    detect_market,
    detect_ticker,
)
from screener.unusual_volume.classify import classify_direction, classify_strength
from screener.unusual_volume.delivery import (
    compute_delivery_metrics,
    overlay_events,
    quiet_accumulation_events,
)
from screener.unusual_volume.filters import _parse_ban_csv, passes_volume_floor
from tests.conftest import make_bars


# ─────────────────────────── classify ───────────────────────────


def test_direction_buying():
    # close > open AND close in upper third of range
    assert classify_direction(open_px=100, high=110, low=99, close=109, prev_close=100) == "BUYING"


def test_direction_selling():
    assert classify_direction(open_px=100, high=101, low=90, close=91, prev_close=100) == "SELLING"


def test_direction_churn_small_change():
    assert classify_direction(open_px=100, high=102, low=99, close=100.3, prev_close=100) == "CHURN"


def test_direction_reversal_gap_up_close_down():
    # gap up 3%, but bar closes below prev_close → reversal
    assert classify_direction(open_px=103, high=104, low=98, close=99, prev_close=100) == "REVERSAL"


def test_direction_reversal_gap_down_close_up():
    assert classify_direction(open_px=97, high=104, low=96.5, close=103, prev_close=100) == "REVERSAL"


def test_strength_tiers():
    assert classify_strength(rvol=1.5, z=1.5) == "MODERATE"
    assert classify_strength(rvol=3.5, z=2.0) == "HIGH"
    assert classify_strength(rvol=2.0, z=2.7) == "HIGH"
    assert classify_strength(rvol=6.0, z=2.0) == "EXTREME"
    assert classify_strength(rvol=2.0, z=4.0) == "EXTREME"


# ─────────────────────────── detector ───────────────────────────


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


# ─────────────────────────── filters ───────────────────────────


def test_passes_volume_floor_drops_thin_names():
    bars = make_bars(n=60, seed=6)
    # Force volumes well below 1M
    assert passes_volume_floor(bars, min_avg_volume=1_000_000, as_of=bars.index[-1].date()) is False
    assert passes_volume_floor(bars, min_avg_volume=1_000, as_of=bars.index[-1].date()) is True


def test_parse_ban_csv():
    text = (
        "Securities in Ban For Trade Date 27-APR-2026:\n"
        "1,SAIL\n"
        "2,FOO\n"
    )
    assert _parse_ban_csv(text) == {"SAIL", "FOO"}


# ─────────────────────────── delivery ───────────────────────────


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
    as_of = date(2026, 4, 24)
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


def test_compute_delivery_metrics_handles_empty():
    out = compute_delivery_metrics(pd.DataFrame())
    assert out.empty
    assert "delivery_rvol" in out.columns
