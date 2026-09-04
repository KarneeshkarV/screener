"""Exact-label alignment in the rolling entry/exit signal masks.

:func:`~screener.backtester.rolling_candidates._signal_mask_matrix` used to
build one pandas Series per ticker and reindex it onto the master calendar.
It now writes positions straight into the block. These tests pin that the two
agree on every calendar shape the panel actually produces, because the cheap
wrong version of this change (``searchsorted``) carries a signal forward into
a session the ticker never traded, which silently invents entries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.rolling_candidates import _signal_mask_matrix


def _reindex_reference(
    signals_by_tv: dict[str, pd.Series | np.ndarray],
    bars_by_tv: dict[str, pd.DataFrame],
    master_ix: pd.DatetimeIndex,
    valid_tickers: list[str],
) -> pd.DataFrame:
    """The per-ticker reindex the fast path replaced, kept as the oracle."""
    block = np.zeros((len(master_ix), len(valid_tickers)), dtype=bool)
    for column, tv in enumerate(valid_tickers):
        signal = signals_by_tv.get(tv)
        if signal is None:
            continue
        if isinstance(signal, np.ndarray):
            values = signal if signal.dtype == bool else np.asarray(signal, dtype=bool)
            series = pd.Series(values, index=bars_by_tv[tv].index, copy=False)
        else:
            series = signal
        block[:, column] = (
            series.reindex(master_ix).fillna(False).astype(bool).to_numpy(dtype=bool)
        )
    return pd.DataFrame(block, index=master_ix, columns=valid_tickers, copy=False)


MASTER = pd.bdate_range("2026-01-05", periods=30)


def _bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"close": np.ones(len(index))}, index=index)


@pytest.mark.parametrize(
    ("label", "index"),
    [
        ("aligned", MASTER),
        ("late_listing", MASTER[7:]),
        ("delisted_early", MASTER[:22]),
        ("missing_sessions", MASTER.delete([3, 4, 11, 19])),
        ("both_ends_and_gaps", MASTER[2:25].delete([1, 6, 14])),
        ("single_bar", MASTER[13:14]),
        ("no_overlap", pd.bdate_range("2020-01-06", periods=9)),
    ],
)
@pytest.mark.parametrize("as_series", [False, True])
def test_mask_matches_the_reindex_it_replaced(
    label: str, index: pd.DatetimeIndex, as_series: bool
) -> None:
    rng = np.random.default_rng(len(label))
    values = rng.random(len(index)) > 0.5
    signal: pd.Series | np.ndarray = (
        pd.Series(values, index=index) if as_series else values
    )
    bars = {"T": _bars(index)}
    result = _signal_mask_matrix({"T": signal}, bars, MASTER, ["T"])
    expected = _reindex_reference({"T": signal}, bars, MASTER, ["T"])
    pd.testing.assert_frame_equal(result, expected)


def test_a_signal_never_carries_into_a_session_the_ticker_missed() -> None:
    """The searchsorted failure mode, pinned directly."""
    index = MASTER.delete([10, 11])
    values = np.zeros(len(index), dtype=bool)
    values[9] = True  # fires on the bar immediately before the two-day gap
    result = _signal_mask_matrix({"T": values}, {"T": _bars(index)}, MASTER, ["T"])
    assert bool(result["T"].iloc[9]) is True
    assert not result["T"].iloc[10:12].any()


def test_nan_in_a_float_signal_series_reads_as_no_signal() -> None:
    series = pd.Series([1.0, np.nan, 0.0, 1.0], index=MASTER[:4])
    result = _signal_mask_matrix({"T": series}, {"T": _bars(MASTER[:4])}, MASTER, ["T"])
    assert list(result["T"].iloc[:4]) == [True, False, False, True]


def test_a_duplicated_bar_label_still_raises_the_pandas_message() -> None:
    """get_indexer cannot align a duplicated index, so that case keeps reindex."""
    index = pd.DatetimeIndex(list(MASTER[:3]) + [MASTER[2]])
    values = np.array([True, False, True, False])
    with pytest.raises(ValueError, match="duplicate"):
        _signal_mask_matrix({"T": values}, {"T": _bars(index)}, MASTER, ["T"])


def test_a_missing_ticker_signal_is_all_false() -> None:
    result = _signal_mask_matrix({}, {"T": _bars(MASTER)}, MASTER, ["T"])
    assert not result["T"].any()
