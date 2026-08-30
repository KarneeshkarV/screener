"""``range_slice`` must return exactly what the boolean mask returned."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.price_frames import frame_has_range, range_slice

START = pd.Timestamp("2024-02-01")
END = pd.Timestamp("2024-03-01")


def _mask_slice(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[(frame.index >= START) & (frame.index <= END)]


def _frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({"close": np.arange(len(index), dtype=float)}, index=index)


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(
            _frame(pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60))),
            id="spans-the-window",
        ),
        pytest.param(
            _frame(pd.DatetimeIndex(pd.bdate_range("2024-02-01", periods=21))),
            id="starts-on-the-bound",
        ),
        pytest.param(
            _frame(pd.DatetimeIndex(pd.bdate_range("2024-04-01", periods=20))),
            id="entirely-after",
        ),
        pytest.param(
            _frame(pd.DatetimeIndex(pd.bdate_range("2023-01-01", periods=20))),
            id="entirely-before",
        ),
        pytest.param(
            _frame(pd.DatetimeIndex(["2024-02-05", "2024-02-05", "2024-02-06"])),
            id="duplicate-labels",
        ),
        pytest.param(
            _frame(pd.DatetimeIndex(["2024-03-01", "2024-01-01", "2024-02-15"])),
            id="unsorted-falls-back",
        ),
        pytest.param(_frame(pd.DatetimeIndex([])), id="empty"),
    ],
)
def test_range_slice_matches_the_mask(frame):
    pd.testing.assert_frame_equal(range_slice(frame, START, END), _mask_slice(frame))


@pytest.mark.parametrize(
    "frame",
    [
        _frame(pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60))),
        _frame(pd.DatetimeIndex(pd.bdate_range("2024-02-20", periods=60))),
        _frame(pd.DatetimeIndex(pd.bdate_range("2024-04-01", periods=20))),
        _frame(pd.DatetimeIndex(["2024-03-01", "2024-01-01", "2024-02-15"])),
        _frame(pd.DatetimeIndex([])),
    ],
)
def test_frame_has_range_matches_the_mask(frame):
    in_range = _mask_slice(frame)
    expected = (
        not in_range.empty
        and in_range.index.min() <= START + pd.Timedelta(days=3)
        and in_range.index.max() >= END - pd.Timedelta(days=3)
    )
    assert frame_has_range(frame, START, END) is expected


def test_the_slice_does_not_write_through_to_its_frame():
    frame = _frame(pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60)))
    sliced = range_slice(frame, START, END)
    sliced.iloc[0, 0] = -1.0
    assert frame.loc[sliced.index[0], "close"] != -1.0


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_a_sub_unit_bound_keeps_the_same_rows(unit):
    """An intraday end bound is ``midnight - 1ns``, finer than a stored index."""
    index = pd.DatetimeIndex(
        pd.date_range("2024-01-01 09:15", periods=100, freq="5min")
    ).as_unit(unit)
    frame = _frame(index)
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-01-02") - pd.Timedelta(1, "ns")
    expected = frame.loc[(frame.index >= start) & (frame.index <= end)]
    pd.testing.assert_frame_equal(range_slice(frame, start, end), expected)
    assert frame_has_range(frame, start, end) is (not expected.empty)
