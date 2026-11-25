"""A cache read must return exactly the frame ``to_pandas`` would have."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from screener.backtester.price_cache import (
    _frame_from_table,
    _has_missing,
    coverage_path,
    empty_history_path,
    load_cached_frame,
    record_coverage,
    record_empty_history,
    record_scale_verified,
    save_cached_frame,
    scale_verified_path,
)


def _ohlcv(bars: int = 8) -> pd.DataFrame:
    close = 100.0 + np.arange(bars, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(bars, dtype=float) * 100.0,
        },
        index=pd.DatetimeIndex(
            pd.bdate_range("2024-01-01", periods=bars).to_numpy(), name="Date"
        ),
    )


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(_ohlcv(), id="ohlcv"),
        pytest.param(_ohlcv().assign(adj_close=1.0, split_factor=1.0), id="adjusted"),
        pytest.param(_ohlcv().assign(volume=np.arange(8)), id="integer-column"),
        pytest.param(_ohlcv().rename_axis("Datetime"), id="other-index-name"),
        pytest.param(_ohlcv().rename_axis(None), id="unnamed-index"),
        pytest.param(_ohlcv().assign(note="a"), id="string-column-falls-back"),
        pytest.param(_ohlcv().reset_index(drop=True), id="range-index-falls-back"),
        pytest.param(
            _ohlcv().set_index(pd.MultiIndex.from_arrays([range(8), range(8)])),
            id="multi-index-falls-back",
        ),
        pytest.param(_ohlcv().iloc[:0], id="empty"),
        pytest.param(_ohlcv().tz_localize("Asia/Kolkata"), id="timezone-falls-back"),
        pytest.param(
            _ohlcv().astype({"volume": "Int64"}), id="nullable-integer-falls-back"
        ),
        pytest.param(
            _ohlcv().astype({"close": "Float64"}), id="nullable-float-falls-back"
        ),
        pytest.param(
            _ohlcv().set_axis(range(5), axis="columns"), id="numeric-labels-fall-back"
        ),
        pytest.param(
            _ohlcv().rename_axis("__index_level_custom__"), id="placeholder-like-name"
        ),
        pytest.param(
            _ohlcv().rename_axis("__index_level_0__"), id="reserved-placeholder-name"
        ),
    ],
)
def test_shortcut_matches_to_pandas(frame, tmp_path):
    path = tmp_path / "entry.parquet"
    frame.to_parquet(path)
    table = pq.read_table(path)
    pd.testing.assert_frame_equal(_frame_from_table(table), table.to_pandas())


def test_table_without_pandas_metadata_falls_back():
    table = pa.table({"close": pa.array([1.0, 2.0])})
    pd.testing.assert_frame_equal(_frame_from_table(table), table.to_pandas())


def test_round_trip_through_the_cache(tmp_path):
    frame = _ohlcv()
    save_cached_frame("TEST", frame, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(load_cached_frame("TEST", cache_dir=tmp_path), frame)


@pytest.mark.parametrize("interval", ["1d", "15m"])
def test_timezone_cache_read_uses_canonical_dates(tmp_path, interval):
    """Daily dates use local midnight; intraday stamps use naive UTC."""
    frame = _ohlcv().tz_localize("Asia/Kolkata")
    save_cached_frame("TZ", frame, cache_dir=tmp_path)
    expected = frame.copy()
    if interval == "1d":
        expected.index = frame.index.tz_localize(None)
    else:
        expected.index = frame.index.tz_convert("UTC").tz_localize(None)
    pd.testing.assert_frame_equal(
        load_cached_frame("TZ", cache_dir=tmp_path, interval=interval), expected
    )


def test_null_rows_are_dropped_on_read(tmp_path):
    frame = _ohlcv()
    frame.iloc[3, frame.columns.get_loc("close")] = np.nan
    frame.to_parquet(tmp_path / "NULLS.parquet")
    loaded = load_cached_frame("NULLS", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(loaded, frame.dropna(subset=["close"]))


def test_nan_stored_as_a_value_is_still_dropped(tmp_path):
    """Parquet can hold NaN as a value, not a null; the guard must see it."""
    frame = _ohlcv()
    table = pa.Table.from_pandas(frame)
    close = pa.array(
        [np.nan if row == 3 else value for row, value in enumerate(frame["close"])],
        type=pa.float64(),
    )
    table = table.set_column(table.column_names.index("close"), "close", close)
    assert table.column("close").null_count == 0
    pq.write_table(table, tmp_path / "NANS.parquet")
    loaded = load_cached_frame("NANS", cache_dir=tmp_path)
    assert len(loaded) == len(frame) - 1


def test_a_clean_frame_is_not_copied(tmp_path):
    """The guard must not drop rows, or pay for a copy, when nothing is missing."""
    frame = _ohlcv()
    table = pa.Table.from_pandas(frame)
    assert not _has_missing(table, ["open", "high", "low", "close", "volume"])


# ── JSON sidecars are written atomically ─────────────────────────────


def _record_empty(ticker: str, cache_dir: Path) -> None:
    record_empty_history(
        ticker,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
        cache_dir,
    )


def _record_coverage(ticker: str, cache_dir: Path) -> None:
    record_coverage(
        ticker,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-01-31"),
        cache_dir,
    )


def _record_scale(ticker: str, cache_dir: Path) -> None:
    record_scale_verified(ticker, cache_dir)


SIDECAR_RECORDERS = [
    pytest.param(_record_empty, empty_history_path, id="empty-history"),
    pytest.param(_record_coverage, coverage_path, id="coverage"),
    pytest.param(_record_scale, scale_verified_path, id="scale-verified"),
]


@pytest.mark.parametrize(("recorder", "path_for"), SIDECAR_RECORDERS)
def test_sidecar_recorders_commit_through_an_atomic_replace(
    tmp_path, monkeypatch, recorder, path_for
):
    """A reader must see the old file or the new one, never a partial write."""
    destination = path_for("TEST", tmp_path)
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source, target, *args, **kwargs):
        replacements.append((Path(source), Path(target)))
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr("screener.cache.os.replace", recording_replace)
    recorder("TEST", tmp_path)

    assert len(replacements) == 1
    source, target = replacements[0]
    assert target == destination
    assert source.parent == tmp_path
    assert source.name.startswith(f".{destination.name}.")
    assert source.suffix == ".tmp"
    # The temporary file was renamed away, not left behind.
    assert not source.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(destination.read_text())


@pytest.mark.parametrize(("recorder", "path_for"), SIDECAR_RECORDERS)
def test_sidecar_recorders_preserve_the_old_marker_when_a_write_fails(
    tmp_path, monkeypatch, recorder, path_for
):
    """A failed write must leave the previous marker readable, not truncate it."""
    destination = path_for("TEST", tmp_path)
    recorder("TEST", tmp_path)
    original_bytes = destination.read_bytes()
    original_payload = json.loads(original_bytes)

    real_write_text = Path.write_text

    def fail_mid_write(self, data, *args, **kwargs):
        # Write a truncated prefix to whatever path the writer opens, then fail.
        # Non-atomic writers corrupt the live marker; atomic writers only touch
        # a temp file that gets unlinked before replace.
        real_write_text(self, data[: max(1, len(data) // 2)], *args, **kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail_mid_write)
    recorder("TEST", tmp_path)  # swallows the OSError by design

    assert destination.read_bytes() == original_bytes
    assert json.loads(destination.read_text()) == original_payload
    assert list(tmp_path.glob("*.tmp")) == []
