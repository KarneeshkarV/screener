"""The panel form of a bar column must equal the per-ticker form exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.indicators.plugins.sar import sar
from screener.indicators.plugins.supertrend import supertrend_dir
from screener.strategies import bar_column_recipes as recipes
from screener.strategies.spec import apply_bar_columns, apply_bar_columns_to_panel

BAR_COLUMNS = {"st_dir": recipes.supertrend_direction, "sar": recipes.parabolic_sar}


def _frame(seed: int, bars: int, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.0, 1.0, bars))
    index = pd.bdate_range(start, periods=bars)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + rng.random(bars),
            "low": close - rng.random(bars),
            "close": close,
            "volume": rng.integers(1_000, 10_000, bars).astype(float),
        },
        index=index,
    )


@pytest.mark.parametrize("indicator", [supertrend_dir, sar])
def test_panel_indicator_matches_per_series(indicator):
    """A stacked call gives each column exactly what the 1-D call gives."""
    frames = [_frame(seed, 120) for seed in range(4)]
    high, low, close = (
        np.column_stack([frame[field].to_numpy(dtype=float) for frame in frames])
        for field in ("high", "low", "close")
    )
    panel = indicator(high, low, close)
    for position, frame in enumerate(frames):
        expected = indicator(
            frame["high"].to_numpy(dtype=float),
            frame["low"].to_numpy(dtype=float),
            frame["close"].to_numpy(dtype=float),
        )
        assert np.array_equal(panel[:, position], expected, equal_nan=True)


def test_grouped_frames_match_per_ticker_application():
    bars_by_tv = {f"T{seed}": _frame(seed, 300) for seed in range(5)}
    prepared = apply_bar_columns_to_panel(BAR_COLUMNS, bars_by_tv)
    for tv, bars in bars_by_tv.items():
        pd.testing.assert_frame_equal(prepared[tv], apply_bar_columns(BAR_COLUMNS, bars))


def test_frames_with_unlike_indexes_stay_on_the_per_ticker_path():
    """Only frames that share an index group; the rest must still be built."""
    bars_by_tv = {
        "SAME_A": _frame(1, 200),
        "SAME_B": _frame(2, 200),
        "SHORTER": _frame(3, 150),
        "SHIFTED": _frame(4, 200, start="2021-01-01"),
        "EMPTY": _frame(5, 200).iloc[:0],
    }
    prepared = apply_bar_columns_to_panel(BAR_COLUMNS, bars_by_tv)
    assert set(prepared) == set(bars_by_tv)
    for tv, bars in bars_by_tv.items():
        pd.testing.assert_frame_equal(prepared[tv], apply_bar_columns(BAR_COLUMNS, bars))


def test_duplicate_index_is_not_stacked():
    """A non-unique index cannot be stacked positionally, so it must not be."""
    duplicated = _frame(6, 100)
    duplicated.index = duplicated.index[:1].repeat(len(duplicated))
    bars_by_tv = {"DUP_A": duplicated, "DUP_B": duplicated.copy()}
    prepared = apply_bar_columns_to_panel(BAR_COLUMNS, bars_by_tv)
    for tv, bars in bars_by_tv.items():
        pd.testing.assert_frame_equal(prepared[tv], apply_bar_columns(BAR_COLUMNS, bars))


def test_interior_gap_is_not_padded_into_a_panel():
    """Padding can express a late start, not a hole in the middle of a history."""
    full = _frame(7, 200)
    gapped = full.drop(full.index[80:90])
    bars_by_tv = {"FULL": full, "GAPPED": gapped}
    prepared = apply_bar_columns_to_panel(BAR_COLUMNS, bars_by_tv)
    for tv, bars in bars_by_tv.items():
        pd.testing.assert_frame_equal(prepared[tv], apply_bar_columns(BAR_COLUMNS, bars))


def test_late_starting_history_is_padded_and_still_exact():
    """A short history stacked against a long one must not see the padding."""
    full = _frame(8, 400)
    bars_by_tv = {"FULL": full, "LATE": _frame(9, 400).iloc[260:]}
    prepared = apply_bar_columns_to_panel(BAR_COLUMNS, bars_by_tv)
    for tv, bars in bars_by_tv.items():
        pd.testing.assert_frame_equal(prepared[tv], apply_bar_columns(BAR_COLUMNS, bars))
