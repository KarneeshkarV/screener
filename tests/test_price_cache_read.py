"""A cache read must return exactly the frame ``to_pandas`` would have."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from screener.backtester.price_cache import (
    _frame_from_table,
    load_cached_frame,
    save_cached_frame,
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
