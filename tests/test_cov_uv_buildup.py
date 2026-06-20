"""Offline coverage tests for the unusual_volume build-up / output / enrich /
detector / filters / classify modules.

All tests are deterministic and never touch the network: every NSE / provider /
openscreener seam is monkeypatched.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from screener.unusual_volume import buildup as B
from screener.unusual_volume import detector as D
from screener.unusual_volume import enrich as E
from screener.unusual_volume import filters as F
from screener.unusual_volume import output as O
from screener.unusual_volume.classify import classify_direction, classify_strength
from screener.unusual_volume.detector import Event
from tests.conftest import make_bars


# ─────────────────────────── helpers ───────────────────────────


def _event(**overrides) -> Event:
    base = dict(
        symbol="AAA",
        date=date(2026, 4, 24),
        close=100.0,
        pct_change=1.0,
        volume=1_000.0,
        avg_volume_20d=500.0,
        rvol=2.0,
        rvol_5d=2.0,
        rvol_50d=2.0,
        rvol_90d=2.0,
        z_score=2.0,
        pct_rank_252d=0.9,
        direction="BUYING",
        strength="MODERATE",
    )
    base.update(overrides)
    return Event(**base)


def _compression_bars(n: int = 80, seed: int = 11) -> pd.DataFrame:
    """A long, tightly-compressed, gently rising panel that lights every
    build-up fingerprint."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    # Gently rising price with shrinking range toward the end.
    base = 100.0 + np.linspace(0, 4.0, n)
    rng = np.linspace(2.0, 0.2, n)
    openp = base - rng * 0.1
    close = base + rng * 0.1  # close near high, up days
    high = base + rng * 0.5
    low = base - rng * 0.5
    # Up-day volume heavy.
    volume = np.full(n, 20_000.0)
    df = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    return df


# ─────────────────────────── classify ───────────────────────────


def test_classify_direction_neutral_mid_range_close():
    # change >= 1%, no gap reversal, close == open, mid-range -> CHURN (final)
    assert (
        classify_direction(open_px=100, high=105, low=95, close=100, prev_close=98)
        == "CHURN"
    )


def test_classify_direction_prev_close_zero_path():
    # prev_close <= 0 skips the gap/churn block; falls through to BUYING.
    assert (
        classify_direction(open_px=100, high=110, low=99, close=109, prev_close=0)
        == "BUYING"
    )


def test_classify_strength_tiers():
    assert classify_strength(rvol=1.0, z=0.5) == "MODERATE"
    assert classify_strength(rvol=3.0, z=1.0) == "HIGH"  # rvol >= 3
    assert classify_strength(rvol=1.0, z=2.5) == "HIGH"  # z >= 2.5
    assert classify_strength(rvol=5.0, z=1.0) == "EXTREME"
    assert classify_strength(rvol=1.0, z=3.5) == "EXTREME"


def test_classify_direction_reversal():
    # gap up > 2% but closes below prev_close -> REVERSAL (line 35).
    assert (
        classify_direction(open_px=103, high=104, low=98, close=99, prev_close=100)
        == "REVERSAL"
    )


def test_classify_direction_selling():
    # close < open and close in lower third -> SELLING (line 43).
    assert (
        classify_direction(open_px=100, high=101, low=90, close=91, prev_close=100)
        == "SELLING"
    )


# ─────────────────────────── filters ───────────────────────────


def test_passes_volume_floor_empty_and_none():
    assert F.passes_volume_floor(pd.DataFrame(), 1_000, date(2026, 1, 1)) is False
    assert F.passes_volume_floor(None, 1_000, date(2026, 1, 1)) is False


def test_passes_volume_floor_date_column_index():
    bars = make_bars(n=60, seed=6).reset_index(names="date")
    assert (
        F.passes_volume_floor(bars, min_avg_volume=1_000, as_of=date(2024, 4, 1))
        is True
    )


def test_passes_volume_floor_no_datetime_no_date_column():
    df = pd.DataFrame({"volume": [1, 2, 3]})  # RangeIndex, no "date" column
    assert F.passes_volume_floor(df, 1_000, date(2026, 1, 1)) is False


def test_passes_volume_floor_short_history():
    bars = make_bars(n=10, seed=6)
    assert F.passes_volume_floor(bars, 1_000, bars.index[-1].date()) is False


def test_passes_volume_floor_nan_rolling_average():
    bars = make_bars(n=60, seed=6)
    # NaN inside the trailing 20d window -> rolling mean undefined -> False.
    bars.iat[-5, bars.columns.get_loc("volume")] = float("nan")
    assert F.passes_volume_floor(bars, 1_000, bars.index[-1].date()) is False


def test_passes_market_cap_paths():
    assert F.passes_market_cap(None, 0) is True  # floor <= 0 short-circuit
    assert F.passes_market_cap(None, 1_000) is True  # unknown cap passes
    assert F.passes_market_cap(float("nan"), 1_000) is True
    assert F.passes_market_cap(2_000.0, 1_000) is True
    assert F.passes_market_cap(500.0, 1_000) is False


def test_parse_ban_csv_single_alpha_token():
    # A line with a single alphabetic token is treated as a symbol.
    text = "Securities in Ban For Trade Date 01-JAN-2026:\nSAIL\n3,FOO\n,\n"
    assert F._parse_ban_csv(text) == {"SAIL", "FOO"}


def test_fetch_fno_ban_list_success(monkeypatch):
    monkeypatch.setattr(
        F, "fetch_nse_text", lambda url, label, timeout=8.0: "hdr\n1,SAIL\n2,FOO\n"
    )
    # First line "hdr" is not a ban header sentence -> single alpha token symbol.
    assert F.fetch_fno_ban_list() == {"HDR", "SAIL", "FOO"}


def test_fetch_fno_ban_list_none(monkeypatch):
    monkeypatch.setattr(F, "fetch_nse_text", lambda url, label, timeout=8.0: None)
    assert F.fetch_fno_ban_list() == set()


# ─────────────────────────── detector ───────────────────────────


def test_detect_ticker_none_and_empty():
    assert D.detect_ticker("X", None, date(2026, 1, 1)) is None
    assert D.detect_ticker("X", pd.DataFrame(), date(2026, 1, 1)) is None


def test_detect_ticker_no_datetime_index_no_date_column():
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    assert D.detect_ticker("X", df, date(2026, 1, 1)) is None


def test_detect_ticker_date_column_reindex():
    bars = make_bars(n=300, seed=1)
    spike_idx = 299
    avg = float(bars["volume"].iloc[200:299].mean())
    bars.iat[spike_idx, bars.columns.get_loc("volume")] = avg * 8.0
    as_of = bars.index[spike_idx].date()
    df = bars.reset_index(names="date")  # no DatetimeIndex; has "date" column
    ev = D.detect_ticker("AAA", df, as_of)
    assert ev is not None and ev.symbol == "AAA"


def test_detect_ticker_all_bars_after_as_of_empty():
    bars = make_bars(n=300, seed=1)
    # as_of before the first bar -> df empties after the index filter.
    assert D.detect_ticker("X", bars, date(2000, 1, 1)) is None


def test_detect_ticker_stale_last_bar_skipped():
    bars = make_bars(n=300, seed=1)
    # as_of more than 7 days after the last bar -> skipped.
    far = bars.index[-1].date() + pd.Timedelta(days=30)
    assert (
        D.detect_ticker("X", bars, far.date() if hasattr(far, "date") else far) is None
    )


def test_detect_ticker_nan_avg20_returns_none():
    bars = make_bars(n=25, seed=2)
    # Inject a NaN into the trailing 20d window so sma_20_prev is NaN.
    bars.iat[-3, bars.columns.get_loc("volume")] = float("nan")
    assert D.detect_ticker("X", bars, bars.index[-1].date()) is None


def test_event_symbol_validator_rejects_empty():
    with pytest.raises(ValueError):
        _event(symbol="   ")


def test_detect_ticker_short_history_safe_ratio_nan():
    # Just over 21 bars: sma_50/sma_90 priors are NaN -> _safe_ratio nan branch,
    # with a volume spike large enough to still emit on rvol_20.
    bars = make_bars(n=25, seed=2)
    avg = float(bars["volume"].iloc[:24].mean())
    bars.iat[24, bars.columns.get_loc("volume")] = avg * 6.0
    ev = D.detect_ticker("X", bars, bars.index[-1].date())
    assert ev is not None
    # 50d/90d RVOL undefined on short history.
    assert ev.rvol_50d != ev.rvol_50d  # NaN
    assert ev.rvol_90d != ev.rvol_90d  # NaN


def test_detect_ticker_below_threshold_returns_none():
    # Normal volume, no spike -> rvol/z below floor -> None (line 183).
    bars = make_bars(n=120, seed=2)
    assert D.detect_ticker("X", bars, bars.index[-1].date()) is None


def test_detect_market_skips_empty_and_none():
    spiked = make_bars(n=300, seed=4)
    avg = float(spiked["volume"].iloc[200:299].mean())
    spiked.iat[299, spiked.columns.get_loc("volume")] = avg * 6.0
    as_of = spiked.index[-1].date()
    events = D.detect_market(
        {"EMPTY": pd.DataFrame(), "NONE": None, "SPIKE": spiked}, as_of
    )
    assert {e.symbol for e in events} == {"SPIKE"}


# ─────────────────────────── buildup ───────────────────────────


def test_buildup_score_none_when_no_bars():
    assert B.compute_buildup_score("X", None, date(2026, 1, 1)) is None
    assert B.compute_buildup_score("X", pd.DataFrame(), date(2026, 1, 1)) is None


def test_buildup_score_no_index_no_date_column():
    df = pd.DataFrame({"close": [1.0], "high": [1.0], "low": [1.0]})
    assert B.compute_buildup_score("X", df, date(2026, 1, 1)) is None


def test_buildup_score_date_column_reindex_short_history():
    df = make_bars(n=10, seed=3).reset_index(names="date")
    # Reindexes via the "date" column, then too-short history -> None.
    assert B.compute_buildup_score("X", df, date(2024, 4, 1)) is None


def test_buildup_score_full_fingerprints():
    df = _compression_bars()
    score = B.compute_buildup_score("aaa", df, df.index[-1].date(), window=20)
    assert score is not None
    assert score.symbol == "AAA"
    # All four price fingerprints populated (delivery skipped -> None).
    assert score.range_compression is not None
    assert score.updown_volume is not None
    assert score.higher_lows is not None
    assert score.sustained_delivery is None
    assert score.close_near_high is not None
    assert 0.0 <= score.composite <= 1.0
    # to_dict round-trips.
    assert score.to_dict()["symbol"] == "AAA"


def test_buildup_score_with_delivery_panel():
    df = _compression_bars()
    as_of = df.index[-1].date()
    rows = []
    for off in range(40):
        d = (pd.Timestamp(as_of) - pd.Timedelta(days=off)).date()
        rows.append({"SYMBOL": "AAA", "date": d, "DELIV_PER": 70.0})
    panel = pd.DataFrame(rows)
    score = B.compute_buildup_score("AAA", df, as_of, delivery_panel=panel, window=20)
    assert score is not None
    assert score.sustained_delivery is not None
    assert "sustained_delivery" in score.flags
    assert score.delivery_mean is not None and score.delivery_hit_rate is not None


def test_buildup_score_higher_lows_and_close_near_high_flags():
    n = 45
    idx = pd.bdate_range("2024-01-01", periods=n)
    low = np.array([100 + i * 0.6 + (0.8 if i % 4 == 0 else 0.0) for i in range(n)])
    high = low + 1.0
    close = high - 0.05  # close right at the high
    openp = low + 0.1
    vol = np.where(close > openp, 30_000.0, 5_000.0)
    df = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
    score = B.compute_buildup_score("AAA", df, df.index[-1].date(), window=20)
    assert score is not None
    assert "higher_lows" in score.flags
    assert "close_near_high" in score.flags


def test_buildup_symbol_validator_rejects_empty():
    with pytest.raises(ValueError):
        B.BuildupScore(
            symbol="   ",
            as_of=date(2026, 1, 1),
            window=20,
            range_compression=None,
            updown_volume=None,
            higher_lows=None,
            sustained_delivery=None,
            close_near_high=None,
            composite=0.0,
        )


def test_score_range_compression_short_returns_none():
    df = make_bars(n=10, seed=3)
    assert B._score_range_compression(df, 20) == (None, None, None)


def test_score_range_compression_bb_none_branch():
    # Flat close -> Bollinger basis std == 0 -> bb_ratio None path.
    n = 80
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    sub, atr_ratio, bb_ratio = B._score_range_compression(df, 20)
    assert bb_ratio is None
    assert sub is not None and atr_ratio is not None


def test_score_range_compression_zero_atr_window_returns_none():
    # Perfectly flat bars -> ATR is 0 across the window -> win.max() <= 0 None path.
    n = 40
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    assert B._score_range_compression(df, 20) == (None, None, None)


def test_score_updown_volume_short_returns_none():
    df = make_bars(n=5, seed=1)
    assert B._score_updown_volume(df, 20) == (None, None)


def test_score_updown_volume_all_flat_returns_none():
    n = 25
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),  # close == open: no up/down days
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    assert B._score_updown_volume(df, 20) == (None, None)


def test_score_updown_volume_infinite_ratio():
    # All up days, no down days -> down_vol == 0 -> ratio inf -> sub 1.0.
    n = 25
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = np.linspace(100, 110, n)
    openp = close - 0.5  # close > open every bar
    df = pd.DataFrame(
        {
            "open": openp,
            "high": close + 0.5,
            "low": openp - 0.5,
            "close": close,
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    sub, ratio = B._score_updown_volume(df, 20)
    assert sub == 1.0
    assert ratio is None  # non-finite ratio reported as None


def test_score_higher_lows_short_returns_none():
    df = make_bars(n=5, seed=1)
    assert B._score_higher_lows(df, 20) == (None, None)


def test_score_higher_lows_nonpositive_lows_returns_none():
    n = 25
    idx = pd.bdate_range("2024-01-01", periods=n)
    low = np.full(n, 100.0)
    low[15] = -1.0  # non-positive low inside the trailing window
    df = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": low,
            "close": np.full(n, 100.0),
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    assert B._score_higher_lows(df, 20) == (None, None)


def test_score_higher_lows_flat_lows_returns_zero():
    n = 25
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 100.0),  # all lows identical
            "close": np.full(n, 100.0),
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    assert B._score_higher_lows(df, 20) == (0.0, 0.0)


def test_score_higher_lows_negative_slope_returns_zero():
    n = 25
    idx = pd.bdate_range("2024-01-01", periods=n)
    low = np.linspace(110, 100, n)  # descending lows -> negative slope
    df = pd.DataFrame(
        {
            "open": low,
            "high": low + 1,
            "low": low,
            "close": low,
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    sub, slope = B._score_higher_lows(df, 20)
    assert sub == 0.0
    assert slope <= 0.0


def test_score_higher_lows_ascending_swings_full_score():
    # Rising staircase lows with clean local minima -> last3_ok True.
    n = 25
    idx = pd.bdate_range("2024-01-01", periods=n)
    low = np.array([100 + i * 0.5 + (1.0 if i % 4 == 0 else 0.0) for i in range(n)])
    df = pd.DataFrame(
        {
            "open": low,
            "high": low + 2,
            "low": low,
            "close": low + 1,
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    sub, slope = B._score_higher_lows(df, 20)
    assert sub is not None and sub > 0.0
    assert slope > 0.0


def test_score_sustained_delivery_none_panel():
    assert B._score_sustained_delivery(None, "AAA", date(2026, 1, 1), 20) == (
        None,
        None,
        None,
    )
    assert B._score_sustained_delivery(pd.DataFrame(), "AAA", date(2026, 1, 1), 20) == (
        None,
        None,
        None,
    )


def test_score_sustained_delivery_symbol_absent():
    panel = pd.DataFrame(
        {"SYMBOL": ["XXX"], "date": [date(2026, 1, 1)], "DELIV_PER": [50.0]}
    )
    assert B._score_sustained_delivery(panel, "AAA", date(2026, 1, 1), 20) == (
        None,
        None,
        None,
    )


def test_score_sustained_delivery_too_few_rows():
    as_of = date(2026, 1, 31)
    panel = pd.DataFrame(
        {
            "SYMBOL": ["AAA", "AAA"],
            "date": [date(2026, 1, 1), date(2026, 1, 2)],
            "DELIV_PER": [50.0, 60.0],
        }
    )
    assert B._score_sustained_delivery(panel, "AAA", as_of, 20) == (None, None, None)


def test_score_sustained_delivery_all_nan_pct():
    as_of = date(2026, 1, 31)
    rows = [
        {"SYMBOL": "AAA", "date": date(2026, 1, d), "DELIV_PER": float("nan")}
        for d in range(1, 16)
    ]
    panel = pd.DataFrame(rows)
    assert B._score_sustained_delivery(panel, "AAA", as_of, 20) == (None, None, None)


def test_score_close_near_high_short_returns_none():
    df = make_bars(n=5, seed=1)
    assert B._score_close_near_high(df, 20) == (None, None)


def test_score_close_near_high_all_zero_range():
    n = 25
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),  # high == low -> range 0 -> all NaN dropped
            "close": np.full(n, 100.0),
            "volume": np.full(n, 10_000.0),
        },
        index=idx,
    )
    assert B._score_close_near_high(df, 20) == (None, None)


def test_compute_buildup_score_all_subs_none_returns_none():
    # Long enough to pass the length gate, but every fingerprint degenerate.
    n = 40
    idx = pd.bdate_range("2024-01-01", periods=n)
    # Flat -> range_compression None (zero ATR), close-near-high None (zero range);
    # volume 0 -> updown None; non-positive low -> higher_lows None; no panel ->
    # delivery None. All five sub-scores None -> compute returns None.
    df = pd.DataFrame(
        {
            "open": np.full(n, -1.0),
            "high": np.full(n, -1.0),
            "low": np.full(n, -1.0),  # high==low -> zero range; <=0 -> higher_lows None
            "close": np.full(n, -1.0),
            "volume": np.full(n, 0.0),
        },
        index=idx,
    )
    assert B.compute_buildup_score("X", df, df.index[-1].date(), window=20) is None


def test_scan_buildups_filters_and_sorts():
    strong = _compression_bars()
    weak = make_bars(n=80, seed=42)
    as_of = strong.index[-1].date()
    out = B.scan_buildups(
        {"STRONG": strong, "WEAK": weak, "SHORT": make_bars(n=5)},
        as_of,
        min_score=0.0,
    )
    syms = [s.symbol for s in out]
    assert "STRONG" in syms
    # SHORT has too little history -> skipped.
    assert "SHORT" not in syms
    # Sorted descending by composite.
    comps = [s.composite for s in out]
    assert comps == sorted(comps, reverse=True)


def test_scan_buildups_min_score_drops_low():
    weak = make_bars(n=80, seed=42)
    as_of = weak.index[-1].date()
    out = B.scan_buildups({"WEAK": weak}, as_of, min_score=0.99)
    assert out == []


# ─────────────────────────── enrich ───────────────────────────


def test_attach_sector_updates_and_skips_missing():
    ev1 = _event(symbol="AAA", sector="Old", market_cap=10.0)
    ev2 = _event(symbol="BBB")
    sector_map = {"AAA": {"sector": "Tech", "market_cap": 999.0}}
    E.attach_sector([ev1, ev2], sector_map)
    assert ev1.sector == "Tech"
    assert ev1.market_cap == 999.0
    # No entry for BBB -> untouched.
    assert ev2.sector is None


def test_attach_sector_falls_back_to_existing_when_meta_empty():
    ev = _event(symbol="AAA", sector="Keep", market_cap=42.0)
    E.attach_sector([ev], {"AAA": {"sector": None, "market_cap": None}})
    assert ev.sector == "Keep"
    assert ev.market_cap == 42.0


def test_fetch_sector_map_empty_inputs():
    assert E.fetch_sector_map("us", []) == {}
    assert E.fetch_sector_map("nasdaq_only_unknown", ["AAA"]) == {}


def test_fetch_sector_map_parses_rows(monkeypatch):
    df = pd.DataFrame(
        [
            {"name": "aaa", "sector": "Technology", "market_cap_basic": 1234.0},
            {"name": "bbb", "sector": None, "market_cap_basic": None},
            {"name": "", "sector": "X", "market_cap_basic": 1.0},  # skipped
        ]
    )

    def fake_fetch(key, fn, **kwargs):
        return df

    monkeypatch.setattr(E._TV_SECTOR_PROVIDER, "fetch", fake_fetch)
    out = E.fetch_sector_map("us", ["AAA", "BBB"])
    assert out["AAA"] == {"sector": "Technology", "market_cap": 1234.0}
    assert out["BBB"] == {"sector": None, "market_cap": None}
    assert "" not in out


def test_fetch_sector_map_empty_frame(monkeypatch):
    monkeypatch.setattr(
        E._TV_SECTOR_PROVIDER, "fetch", lambda key, fn, **kw: pd.DataFrame()
    )
    assert E.fetch_sector_map("india", ["AAA"]) == {}


def test_fetch_sector_map_none_frame(monkeypatch):
    monkeypatch.setattr(E._TV_SECTOR_PROVIDER, "fetch", lambda key, fn, **kw: None)
    assert E.fetch_sector_map("india", ["AAA"]) == {}


def test_deep_enrich_india_no_openscreener(monkeypatch):
    # Simulate openscreener not installed -> early return.
    monkeypatch.setitem(sys.modules, "openscreener", None)
    ev = _event(notes="")
    E.deep_enrich_india([ev])  # ImportError swallowed
    assert ev.notes == ""


def test_deep_enrich_india_appends_to_existing_notes(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, **kwargs):
            self.symbol = symbol

        def fetch(self, sections):
            return {"shareholding": [{"Promoters": "62.5%"}]}

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    # _HttpScraper import target must exist; patch screener.insiders attribute.
    import screener.insiders as ins

    monkeypatch.setattr(ins, "_HttpScraper", lambda *a, **k: object(), raising=False)
    ev = _event(notes="prior note")
    E.deep_enrich_india([ev])
    assert ev.notes == "prior note; promoter holding 62.5%"


def test_deep_enrich_india_stock_raises_is_skipped(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    ev = _event(notes="keep")
    E.deep_enrich_india([ev])
    assert ev.notes == "keep"


def test_deep_enrich_india_empty_df_skipped(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, **kwargs):
            pass

        def fetch(self, sections):
            return {"shareholding": pd.DataFrame()}

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    import screener.insiders as ins

    monkeypatch.setattr(ins, "_HttpScraper", lambda *a, **k: object(), raising=False)
    ev = _event(notes="")
    E.deep_enrich_india([ev])
    assert ev.notes == ""


def test_deep_enrich_india_promoter_none_skipped(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, **kwargs):
            pass

        def fetch(self, sections):
            return {"shareholding": [{"other": "1"}]}  # no promoter key

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    import screener.insiders as ins

    monkeypatch.setattr(ins, "_HttpScraper", lambda *a, **k: object(), raising=False)
    ev = _event(notes="")
    E.deep_enrich_india([ev])
    assert ev.notes == ""


def test_deep_enrich_india_extract_raises_typeerror_skipped(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, **kwargs):
            pass

        def fetch(self, sections):
            return {"shareholding": [{"promoters": object()}]}  # float() -> TypeError

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    import screener.insiders as ins

    monkeypatch.setattr(ins, "_HttpScraper", lambda *a, **k: object(), raising=False)
    ev = _event(notes="")
    E.deep_enrich_india([ev])
    assert ev.notes == ""


def test_deep_enrich_india_extract_exception_continues(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, **kwargs):
            pass

        def fetch(self, sections):
            return {"shareholding": [{"promoters": "10"}]}

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    import screener.insiders as ins

    monkeypatch.setattr(ins, "_HttpScraper", lambda *a, **k: object(), raising=False)

    def boom(_df):
        raise ValueError("explode")

    monkeypatch.setattr(E, "_extract_promoter_pct", boom)
    ev = _event(notes="orig")
    E.deep_enrich_india([ev])
    assert ev.notes == "orig"  # exception swallowed, note untouched


# ── enrich internal helpers ──


def test_fetch_shareholding_fetch_typeerror_falls_back_no_arg():
    class Stock:
        def fetch(self, sections=None):
            if sections is not None:
                raise TypeError("no positional")
            return {"shareholding": [{"promoters": "10"}]}

    out = E._fetch_shareholding_quarterly(Stock())
    assert out == [{"promoters": "10"}]


def test_fetch_shareholding_payload_without_shareholding_uses_attr():
    class Stock:
        def fetch(self, sections):
            return {"something_else": 1}  # no "shareholding" key

        def shareholding_quarterly(self):
            return ["data"]

    assert E._fetch_shareholding_quarterly(Stock()) == ["data"]


def test_fetch_shareholding_non_callable_attr():
    class Stock:
        fetch = None  # not callable
        shareholding_quarterly = ["frozen"]  # not callable -> returned directly

    assert E._fetch_shareholding_quarterly(Stock()) == ["frozen"]


def test_extract_promoter_pct_none():
    assert E._extract_promoter_pct(None) is None


def test_extract_promoter_pct_empty_list():
    assert E._extract_promoter_pct([]) is None


def test_extract_promoter_pct_list_not_dict():
    assert E._extract_promoter_pct(["not a dict"]) is None


def test_extract_promoter_pct_list_no_promoter_key():
    assert E._extract_promoter_pct([{"foo": "1"}]) is None


def test_extract_promoter_pct_list_value():
    assert E._extract_promoter_pct([{"promoters": "55.5%"}]) == 55.5


def test_extract_promoter_pct_dataframe_index():
    df = pd.DataFrame({"Mar 2026": ["48.2%"]}, index=["Promoters"])
    assert E._extract_promoter_pct(df) == 48.2


def test_extract_promoter_pct_dataframe_empty_row():
    df = pd.DataFrame(index=["Promoters"])  # zero columns -> len(row)==0
    assert E._extract_promoter_pct(df) is None


def test_extract_promoter_pct_dataframe_nan_value():
    df = pd.DataFrame({"q1": [float("nan")]}, index=["Promoters"])
    assert E._extract_promoter_pct(df) is None


def test_extract_promoter_pct_no_promoter_row():
    df = pd.DataFrame({"q1": ["10%"]}, index=["Public"])
    assert E._extract_promoter_pct(df) is None


def test_extract_promoter_pct_value_error_swallowed():
    df = pd.DataFrame({"q1": ["not-a-number"]}, index=["Promoters"])
    assert E._extract_promoter_pct(df) is None


# ─────────────────────────── output ───────────────────────────


def test_sort_events_nan_rvol():
    a = _event(symbol="A", strength="HIGH", rvol=float("nan"))
    b = _event(symbol="B", strength="HIGH", rvol=5.0)
    out = O.sort_events([a, b])
    assert out[0].symbol == "B"  # nan rvol sorts low


def test_render_rich_empty():
    console = Console(record=True)
    O.render_rich([], "us", date(2026, 1, 1), console)
    assert "No unusual-volume events" in console.export_text()


def test_render_rich_us_table():
    console = Console(record=True)
    ev = _event(symbol="AAA", sector="Tech", notes="hi", buildup_score=0.5)
    O.render_rich([ev], "us", date(2026, 1, 1), console)
    txt = console.export_text()
    assert "AAA" in txt and "Unusual Volume" in txt


def test_render_rich_india_with_fii_footer():
    console = Console(record=True)
    ev = _event(
        symbol="REL",
        direction="QUIET_ACCUMULATION",
        strength="EXTREME",
        delivery_pct=60.0,
        delivery_rvol=2.0,
        conviction_score=1.5,
        pcr=0.8,
        call_put_oi_ratio=1.2,
        pledge_pct=5.0,
        fii_5d_net=1000.0,
        dii_5d_net=-500.0,
        fii_trend=0.3,
    )
    O.render_rich([ev], "india", date(2026, 1, 1), console)
    txt = console.export_text()
    assert "REL" in txt
    assert "FII" in txt


def test_render_rich_india_no_fii_footer():
    console = Console(record=True)
    ev = _event(symbol="REL")  # all FII/DII None -> footer empty
    O.render_rich([ev], "india", date(2026, 1, 1), console)
    assert "REL" in console.export_text()


def test_color_helpers_unknown_passthrough():
    assert O._color_direction("WEIRD") == "WEIRD"
    assert O._color_strength("WEIRD") == "WEIRD"
    assert "green" in O._color_direction("BUYING")
    assert "red" in O._color_strength("EXTREME")


def test_json_safe_variants():
    assert O._json_safe(None) is None
    assert O._json_safe({"a": 1, "b": float("nan")}) == {"a": 1, "b": None}
    assert O._json_safe([1, 2.0, float("inf")]) == [1, 2.0, None]
    assert O._json_safe((1,)) == [1]
    assert O._json_safe(True) is True  # bool not coerced to int
    assert O._json_safe(np.int64(5)) == 5
    assert O._json_safe(3.5) == 3.5
    assert O._json_safe("text") == "text"


def test_json_safe_isna_raises_falls_through():
    # pd.isna on a multi-element array raises -> except branch (lines 158-159),
    # then the value (a non-Real container) is returned unchanged.
    arr = np.array([1, 2, 3])
    out = O._json_safe(arr)
    assert isinstance(out, np.ndarray)


def test_write_json_roundtrip(tmp_path):
    ev = _event(symbol="AAA")
    path = tmp_path / "events.json"
    O.write_json([ev], path)
    payload = json.loads(path.read_text())
    assert payload[0]["symbol"] == "AAA"


def test_write_markdown_us(tmp_path):
    evs = [
        _event(symbol="A", direction="BUYING"),
        _event(symbol="B", direction="SELLING"),
        _event(symbol="R", direction="REVERSAL"),
        _event(symbol="C", direction="CHURN"),
        _event(
            symbol="BU",
            direction="BUILDUP",
            buildup_score=0.7,
            buildup_flags=["compression"],
        ),
    ]
    path = tmp_path / "out.md"
    O.write_markdown(evs, path, "us", date(2026, 1, 1))
    text = path.read_text()
    assert "# Unusual Volume — US" in text
    assert "## BUYING" in text
    assert "## BUILDUP" in text
    assert "compression" in text


def test_write_markdown_india_with_quiet_and_fii(tmp_path):
    evs = [
        _event(
            symbol="Q",
            direction="QUIET_ACCUMULATION",
            delivery_pct=60.0,
            fii_5d_net=1.0,
            dii_5d_net=2.0,
            fii_trend=0.1,
        ),
    ]
    path = tmp_path / "out_india.md"
    O.write_markdown(evs, path, "india", date(2026, 1, 1))
    text = path.read_text()
    assert "# Unusual Volume — INDIA" in text
    assert "## QUIET_ACCUMULATION" in text
    assert "Market-wide FII/DII" in text


def test_write_markdown_buildup_none_score(tmp_path):
    ev = _event(symbol="BU", direction="BUILDUP", buildup_score=None, buildup_flags=[])
    path = tmp_path / "bu.md"
    O.write_markdown([ev], path, "us", date(2026, 1, 1))
    text = path.read_text()
    assert "## BUILDUP" in text


def test_sort_by_buildup_handles_none():
    a = _event(symbol="A", buildup_score=None)
    b = _event(symbol="B", buildup_score=0.9)
    out = O._sort_by_buildup([a, b])
    assert out[0].symbol == "B"
