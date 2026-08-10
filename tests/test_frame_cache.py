"""Guards for ``_build_frame_cache``'s NumPy construction.

The cache is built with ``Series.to_numpy(dtype=float)`` rather than
``Series.astype(float).to_numpy()``. That is a real speedup, but it changes an
invisible property: for an already-``float64`` column ``to_numpy`` hands back a
*view* into the caller's DataFrame where ``astype`` produced a private copy.
Every consumer reads and none write, which is what makes it sound — these tests
exist so that stops being an accident.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from screener.backtester.core import _build_frame_cache


def _bars(n: int = 10, close: np.ndarray | None = None, **kwargs) -> pd.DataFrame:
    values = np.linspace(10.0, 20.0, n) if close is None else close
    index = kwargs.get("index", pd.date_range("2024-01-01", periods=n))
    return pd.DataFrame(
        {
            "open": values * 1.01,
            "high": values * 1.02,
            "low": values * 0.98,
            "close": values,
            "volume": np.arange(n, dtype=float) + 1.0,
        },
        index=index,
    )


def _reference(bars: pd.DataFrame) -> dict:
    """The pre-NumPy implementation, kept as the parity oracle."""
    close_f = bars["close"].astype(float)
    return {
        "open_arr": bars["open"].astype(float).to_numpy(),
        "high_arr": bars["high"].astype(float).to_numpy(),
        "low_arr": bars["low"].astype(float).to_numpy(),
        "close_arr": close_f.to_numpy(),
        "volume_f": bars["volume"].astype(float),
        "rets_f": close_f / close_f.shift(1) - 1,
    }


_NAN_CLOSE = np.linspace(10.0, 20.0, 10)
_NAN_CLOSE[3] = np.nan
_ZERO_CLOSE = np.linspace(10.0, 20.0, 10)
_ZERO_CLOSE[4] = 0.0

CASES = {
    "normal": _bars(10),
    "nan_close": _bars(10, close=_NAN_CLOSE),
    # 0.0 denominator: pandas is silent, NumPy warns -- the result must still
    # be inf and the warning must not escape.
    "zero_close": _bars(10, close=_ZERO_CLOSE),
    "all_nan_close": _bars(6, close=np.full(6, np.nan)),
    "length_one": _bars(1),
    "length_two": _bars(2),
    "empty": _bars(0),
    "int_columns": _bars(10).astype({"open": "int64", "volume": "int64"}),
    "range_index": _bars(5, index=pd.RangeIndex(5)),
    "tz_aware_index": _bars(5, index=pd.date_range("2024-01-01", periods=5, tz="UTC")),
}


@pytest.mark.parametrize("label", sorted(CASES))
def test_matches_the_pandas_reference(label):
    bars = CASES[label]
    with warnings.catch_warnings():
        # A RuntimeWarning escaping to a caller would itself be a behaviour change.
        warnings.simplefilter("error", RuntimeWarning)
        cache = _build_frame_cache(bars)
    expected = _reference(bars)

    for field in ("open_arr", "high_arr", "low_arr", "close_arr"):
        actual = getattr(cache, field)
        assert actual.dtype == expected[field].dtype, field
        assert np.array_equal(actual, expected[field], equal_nan=True), field

    # These two must stay Series -- _cached_trailing_liquidity calls
    # .iloc[...].mean() and .iloc[...].dropna().std() on them.
    for field in ("volume_f", "rets_f"):
        actual = getattr(cache, field)
        assert isinstance(actual, pd.Series), field
        assert actual.dtype == expected[field].dtype, field
        assert actual.index.equals(expected[field].index), field
        assert np.array_equal(
            actual.to_numpy(), expected[field].to_numpy(), equal_nan=True
        ), field


def test_empty_frame_does_not_raise():
    """``rets[0] = nan`` on a zero-length array is an IndexError."""
    cache = _build_frame_cache(_bars(0))
    assert len(cache.close_arr) == 0
    assert len(cache.rets_f) == 0


def test_building_the_cache_does_not_mutate_the_source_frame():
    bars = _bars(10)
    before = bars.copy(deep=True)
    _build_frame_cache(bars)
    pd.testing.assert_frame_equal(bars, before)


def test_no_consumer_writes_through_the_cached_arrays():
    """``to_numpy`` may alias the caller's frame; every consumer must read only.

    This is the invariant the NumPy construction relies on. If a future caller
    starts writing into ``open_arr``/``close_arr``/etc., it would silently
    corrupt the bars frame the cache was built from -- so pin it here rather
    than leaving it to be discovered in a backtest result.
    """
    bars = _bars(10)
    cache = _build_frame_cache(bars)
    for field in ("open_arr", "high_arr", "low_arr", "close_arr"):
        getattr(cache, field).flags.writeable = False

    from screener.backtester.core import _cached_trailing_liquidity

    # Exercise the read paths that consume the arrays; any write now raises.
    _cached_trailing_liquidity(cache, bars, 5)
    assert float(cache.close_arr[3]) == pytest.approx(bars["close"].iloc[3])
    assert float(cache.open_arr[0]) == pytest.approx(bars["open"].iloc[0])


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_index_i8_matches_timestamp_value_for_every_naive_unit(unit):
    """day_loop searchsorteds against ``Timestamp.value`` (always nanoseconds).

    Pandas 3 builds many daily indexes as ``datetime64[us]``. Caching a raw us
    i8 view would make every lookup miss and fall back incorrectly.
    """
    idx = pd.date_range("2024-01-01", periods=8, freq="D").as_unit(unit)
    cache = _build_frame_cache(_bars(8, index=idx))
    assert cache.index_i8 is not None
    for position, stamp in enumerate(idx):
        assert cache.index_i8[position] == stamp.value
        found = int(np.searchsorted(cache.index_i8, stamp.value))
        assert found == position
        assert cache.index_i8[found] == stamp.value


def test_index_i8_disabled_for_tz_aware_and_duplicates():
    aware = _build_frame_cache(
        _bars(5, index=pd.date_range("2024-01-01", periods=5, tz="UTC"))
    )
    assert aware.index_i8 is None

    dup_idx = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]
    )
    dup = _build_frame_cache(_bars(5, index=dup_idx))
    assert dup.index_i8 is None
