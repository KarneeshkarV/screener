"""Local bar-store scanner: feature parity with the backtester and screen path.

Synthetic 1m sessions are pushed through the local scanner and its picks are
checked against the *same* criteria filter expressions evaluated directly on the
computed features — the offline-stub equivalent of "the local scanner produces
the same picks as the equivalent expressions the backtester uses". Also covers
the ``--source local`` workflow branch and the unsupported-field guard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from screener.backtester.bar_store import save_bars, stored_symbols
from screener.criteria import resolve_criteria
from screener.local_scanner import (
    LocalScanUnsupported,
    compute_features,
    local_scan,
    passes_all,
)
from screener.screen_workflow import (
    ScreenMode,
    ScreenRequest,
    ScreenSource,
    run_screen_workflow,
)

US_TZ = "America/New_York"


def _two_session_index(bars_per: int = 30) -> pd.DatetimeIndex:
    """Two consecutive US regular sessions of 1m stamps (naive UTC)."""
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp("2026-07-20 14:30:00")  # Monday 09:30 ET
    for _ in range(2):
        stamps.extend(day + pd.Timedelta(minutes=b) for b in range(bars_per))
        day = day + pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def _frame(closes: np.ndarray, volume: float, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": np.full(len(index), volume),
        },
        index=index,
    )


def _rising_frame(index: pd.DatetimeIndex, *, start: float, step: float, volume: float):
    closes = start + step * np.arange(len(index), dtype=float)
    return _frame(closes, volume, index)


def _write(root: Path, symbol: str, frame: pd.DataFrame) -> None:
    save_bars(symbol, frame, market="us", interval="1m", root=root)


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #
def test_compute_features_session_semantics() -> None:
    index = _two_session_index(bars_per=10)
    # Session 1 flat at 100 (vol 1000/bar), session 2 rising from 110 (vol 2000/bar).
    s1 = _frame(np.full(10, 100.0), 1000.0, index[:10])
    s2 = _rising_frame(index[10:], start=110.0, step=1.0, volume=2000.0)
    frame = pd.concat([s1, s2])

    features = compute_features(frame, US_TZ)

    # volume = current (2nd) session cumulative = 10 * 2000.
    assert features["volume"] == pytest.approx(20_000.0)
    # average_volume_10d_calc = prior session total = 10 * 1000.
    assert features["average_volume_10d_calc"] == pytest.approx(10_000.0)
    assert features["relative_volume_10d_calc"] == pytest.approx(2.0)
    # change = last close (119) vs previous session close (100).
    assert features["close"] == pytest.approx(119.0)
    assert features["change"] == pytest.approx(19.0)
    # price_52_week_high = max high over both sessions (119 + 0.5).
    assert features["price_52_week_high"] == pytest.approx(119.5)


def test_compute_features_single_session_has_nan_change() -> None:
    index = _two_session_index(bars_per=5)[:5]
    frame = _rising_frame(index, start=50.0, step=0.5, volume=500.0)

    features = compute_features(frame, US_TZ)

    assert np.isnan(features["change"])
    assert np.isnan(features["average_volume_10d_calc"])
    assert np.isnan(features["relative_volume_10d_calc"])


# --------------------------------------------------------------------------- #
# Filter-expression parity: scanner picks == direct evaluation of the criteria
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("criterion", ["intraday_momentum", "intraday_breakout", "ema"])
def test_local_scan_matches_direct_filter_evaluation(
    tmp_path: Path, criterion: str
) -> None:
    index = _two_session_index(bars_per=40)
    # A basket with varied trend/volume so the criterion splits the field.
    symbols = {
        "STRONG": _rising_frame(index, start=100.0, step=0.3, volume=50_000.0),
        "FLAT": _frame(np.full(len(index), 20.0), 5_000.0, index),
        "WEAK": _rising_frame(index, start=200.0, step=-0.2, volume=1_000.0),
    }
    for symbol, frame in symbols.items():
        _write(tmp_path, symbol, frame)

    selection = resolve_criteria((criterion,))

    # Ground truth: evaluate the identical filter expressions on the features.
    expected = sorted(
        symbol
        for symbol, frame in symbols.items()
        if passes_all(compute_features(frame, US_TZ), selection.filters)
    )

    total, df = local_scan(
        market="us",
        filters=selection.filters,
        interval="5m",
        limit=50,
        root=tmp_path,
    )

    assert total == len(expected)
    assert sorted(df["name"].tolist()) == expected


def test_local_scan_discovers_stored_symbols(tmp_path: Path) -> None:
    index = _two_session_index(bars_per=20)
    _write(tmp_path, "AAA", _rising_frame(index, start=10.0, step=0.1, volume=9_000.0))
    _write(tmp_path, "BBB", _rising_frame(index, start=30.0, step=0.2, volume=9_000.0))

    assert stored_symbols("us", "1m", root=tmp_path) == ["AAA", "BBB"]

    # No explicit symbol list — the scanner enumerates the store.
    total, df = local_scan(
        market="us",
        filters=[],  # empty filter list matches every stored symbol
        interval="15m",
        root=tmp_path,
    )
    assert total == 2
    assert sorted(df["name"].tolist()) == ["AAA", "BBB"]


def test_local_scan_setup_score_orders_and_drops_ema_columns(tmp_path: Path) -> None:
    index = _two_session_index(bars_per=40)
    _write(
        tmp_path, "AAA", _rising_frame(index, start=100.0, step=0.3, volume=80_000.0)
    )
    _write(tmp_path, "BBB", _rising_frame(index, start=50.0, step=0.1, volume=9_000.0))

    total, df = local_scan(
        market="us",
        filters=[],
        interval="5m",
        order_by="setup_score",
        root=tmp_path,
    )

    assert total == 2
    assert "setup_score" in df.columns
    # EMA helper columns are dropped from setup_score output, matching scanner.
    assert not any(col.startswith("EMA") for col in df.columns)
    # Sorted by setup_score descending.
    scores = df["setup_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_local_scan_skips_missing_and_empty(tmp_path: Path) -> None:
    index = _two_session_index(bars_per=20)
    _write(tmp_path, "AAA", _rising_frame(index, start=10.0, step=0.1, volume=9_000.0))

    total, df = local_scan(
        market="us",
        filters=[],
        interval="5m",
        symbols=["AAA", "GHOST"],  # GHOST has no stored file
        root=tmp_path,
    )
    assert total == 1
    assert df["name"].tolist() == ["AAA"]


# --------------------------------------------------------------------------- #
# Filter interpreter maps the registry criteria exactly (non-vacuous semantics)
# --------------------------------------------------------------------------- #
def test_intraday_momentum_filter_interpreter() -> None:
    filters = resolve_criteria(("intraday_momentum",)).filters
    passing = {
        "relative_volume_10d_calc": 1.6,
        "volume": 300_000.0,
        "close": 105.0,
        "EMA20": 100.0,
        "EMA200": 90.0,
        "RSI": 65.0,
        "change": 2.0,
    }
    assert passes_all(passing, filters) is True

    # Each threshold is load-bearing: nudging one field past it fails the scan.
    assert passes_all({**passing, "RSI": 85.0}, filters) is False  # RSI <= 80
    assert passes_all({**passing, "change": 0.5}, filters) is False  # change >= 1
    assert passes_all({**passing, "close": 95.0}, filters) is False  # close >= EMA20
    assert passes_all({**passing, "relative_volume_10d_calc": 1.0}, filters) is False


def test_intraday_breakout_filter_interpreter() -> None:
    filters = resolve_criteria(("intraday_breakout",)).filters
    passing = {
        "close": 99.0,
        "price_52_week_high": 100.0,  # close > 0.97 * high
        "relative_volume_10d_calc": 2.5,
        "change": 2.0,
        "EMA5": 101.0,
        "EMA20": 100.0,
    }
    assert passes_all(passing, filters) is True
    # 97%-of-high band (above%): 96 is below 0.97 * 100 = 97 → fails.
    assert passes_all({**passing, "close": 96.0}, filters) is False


def test_nan_feature_fails_comparison() -> None:
    filters = resolve_criteria(("intraday_momentum",)).filters
    passing = {
        "relative_volume_10d_calc": float("nan"),  # missing → excluded
        "volume": 300_000.0,
        "close": 105.0,
        "EMA20": 100.0,
        "EMA200": 90.0,
        "RSI": 65.0,
        "change": 2.0,
    }
    assert passes_all(passing, filters) is False


# --------------------------------------------------------------------------- #
# Unsupported (fundamental) fields
# --------------------------------------------------------------------------- #
def test_local_scan_raises_on_fundamental_field(tmp_path: Path) -> None:
    index = _two_session_index(bars_per=20)
    _write(tmp_path, "AAA", _rising_frame(index, start=10.0, step=0.1, volume=9_000.0))

    selection = resolve_criteria(("momentum_value",))  # uses price_earnings_ttm
    with pytest.raises(LocalScanUnsupported):
        local_scan(
            market="us",
            filters=selection.filters,
            interval="5m",
            root=tmp_path,
        )


# --------------------------------------------------------------------------- #
# Workflow --source local branch (default TradingView path is untouched)
# --------------------------------------------------------------------------- #
def test_workflow_local_source_uses_bar_store(monkeypatch, tmp_path: Path) -> None:
    index = _two_session_index(bars_per=40)
    _write(
        tmp_path, "STRONG", _rising_frame(index, start=100.0, step=0.3, volume=50_000.0)
    )

    def _boom(**kwargs):  # the TV scanner must not be reached in local mode
        raise AssertionError("TradingView scan must not run for --source local")

    import screener.screen_workflow as sw

    monkeypatch.setattr(sw, "scan", _boom)

    request = ScreenRequest(
        market="us",
        criteria_names=("ema",),  # rising series satisfies the bullish EMA stack
        limit=10,
        order_by="volume",
        output_csv=True,
        detail=False,
        refresh=False,
        cache_ttl="15m",
        report_path=None,
        source=ScreenSource.LOCAL,
        interval="5m",
        bar_store_root=tmp_path,
    )
    outcome = run_screen_workflow(request)

    assert outcome.mode is ScreenMode.CSV
    assert outcome.df["name"].tolist() == ["STRONG"]
