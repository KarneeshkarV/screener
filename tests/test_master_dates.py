"""``_master_dates`` must give the dates the full boolean mask gives.

It slices a sorted index with two binary searches instead, and falls back to
the mask for any bound it cannot search exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester import price_panel


def _masked(bars_by_tv, start_ts, end_ts):
    days = [
        b.index[(b.index >= start_ts) & (b.index <= end_ts)].to_numpy()
        for b in bars_by_tv.values()
        if b is not None and not b.empty
    ]
    days = [d for d in days if len(d)]
    if not days:
        return []
    return list(pd.DatetimeIndex(np.unique(np.concatenate(days))))


def _frame_on(index):
    return pd.DataFrame({"close": np.arange(len(index), dtype=float)}, index=index)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-03", "2024-01-10"),
        ("2023-12-01", "2024-01-05"),
        ("2024-01-09", "2024-03-01"),
        ("2024-02-01", "2024-03-01"),
        ("2024-01-05", "2024-01-05"),
        ("2024-01-06", "2024-01-06"),
    ],
)
def test_master_dates_matches_the_mask_on_every_index_shape(start, end):
    sorted_days = pd.bdate_range("2024-01-01", periods=10)
    bars = {
        "sorted": _frame_on(sorted_days),
        "duplicates": _frame_on(sorted_days.repeat(2)),
        "unsorted": _frame_on(sorted_days[::-1]),
        "intraday": _frame_on(pd.date_range("2024-01-04 14:30", periods=20, freq="1h")),
        "with_nat": _frame_on(
            pd.DatetimeIndex([sorted_days[0], pd.NaT, sorted_days[4]])
        ),
        "empty": _frame_on(pd.DatetimeIndex([])),
        "none": None,
    }
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    got = price_panel._master_dates(bars, start_ts, end_ts)

    assert got == _masked(bars, start_ts, end_ts)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_master_dates_keeps_bounds_finer_than_the_index_unit(unit):
    days = pd.bdate_range("2024-01-01", periods=10).as_unit(unit)
    bars = {"a": _frame_on(days), "b": _frame_on(days[2:])}
    start_ts = pd.Timestamp("2024-01-03") - pd.Timedelta(1, "ns")
    end_ts = pd.Timestamp("2024-01-10") + pd.Timedelta(1, "D") - pd.Timedelta(1, "ns")

    got = price_panel._master_dates(bars, start_ts, end_ts)

    assert got == _masked(bars, start_ts, end_ts)
    assert got[0] == pd.Timestamp("2024-01-03")
    assert got[-1] == pd.Timestamp("2024-01-10")


def test_master_dates_matches_the_mask_for_tz_aware_bars():
    days = pd.date_range("2024-01-04 09:30", periods=12, freq="1h", tz="UTC")
    bars = {"a": _frame_on(days)}
    start_ts = pd.Timestamp("2024-01-04 12:00", tz="UTC")
    end_ts = pd.Timestamp("2024-01-04 16:00", tz="UTC")

    got = price_panel._master_dates(bars, start_ts, end_ts)

    assert got == _masked(bars, start_ts, end_ts)
    assert len(got) == 4


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2500-01-01")),
        (pd.Timestamp("1600-01-01"), pd.Timestamp("2024-01-02")),
        (pd.Timestamp("2024-01-01"), pd.NaT),
        (pd.NaT, pd.Timestamp("2024-01-02")),
    ],
)
def test_master_dates_matches_the_mask_for_unconvertible_bounds(start, end):
    days = pd.date_range("2024-01-01", periods=3).as_unit("ns")
    bars = {"a": _frame_on(days)}

    assert price_panel._master_dates(bars, start, end) == _masked(bars, start, end)
