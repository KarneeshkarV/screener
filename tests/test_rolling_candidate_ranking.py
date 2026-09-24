"""The per-day candidate scan ranks in NumPy; pandas is kept here as the oracle.

``_candidate_rows_for_day`` runs on every scanned day, so it no longer builds
a pandas Series to take the percentile or a DataFrame to sort by factor
score. These tests pin that the NumPy spellings return what the pandas ones
did, ties included, since a tie broken differently changes which name fills a
slot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.rolling_candidates import (
    _average_rank_pct,
    _last_positions,
    _master_ticks,
    _signal_mask_matrix,
)


@pytest.mark.parametrize("seed", range(40))
def test_average_rank_matches_pandas_percentile_rank(seed: int) -> None:
    rng = np.random.default_rng(seed)
    size = int(rng.integers(1, 60))
    # Few distinct values so most draws carry ties; one seed in four is
    # continuous, plus infinities, which pandas ranks like any other value.
    values = rng.integers(0, 5, size).astype(float)
    if seed % 4 == 0:
        values = rng.normal(size=size)
        values[rng.integers(0, size)] = np.inf
    expected = pd.Series(values).rank(pct=True).to_numpy(dtype=float)

    assert np.array_equal(_average_rank_pct(values), expected)


@pytest.mark.parametrize("seed", range(40))
def test_lexsort_ranking_matches_the_pandas_multi_key_sort(seed: int) -> None:
    rng = np.random.default_rng(seed)
    size = int(rng.integers(1, 40))
    score = rng.integers(0, 4, size).astype(float)
    dollar_vol = rng.integers(0, 3, size).astype(float)
    expected = (
        pd.DataFrame({"rank_score": score, "dollar_vol": dollar_vol})
        .sort_values(["rank_score", "dollar_vol"], ascending=False, kind="mergesort")
        .index.to_numpy()
    )

    assert np.array_equal(np.lexsort((-dollar_vol, -score)), expected)


MASTER = pd.bdate_range("2026-01-05", periods=20)


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(MASTER[3:15].as_unit("us"), id="other-unit"),
        pytest.param(MASTER[3:15].tz_localize("UTC"), id="tz-aware"),
        pytest.param(MASTER[3:15][::-1], id="unsorted"),
        pytest.param(MASTER[3:15].delete([2, 5]), id="gaps"),
        pytest.param(MASTER[:0], id="empty"),
    ],
)
def test_alignment_falls_back_to_pandas_whenever_ticks_do_not_compare(index):
    master = MASTER.as_unit("ns")
    values = np.arange(len(index)) % 3 == 0
    series = pd.Series(values, index=index)
    bars = {"AAA": pd.DataFrame({"close": np.ones(len(index))}, index=index)}

    # Whatever pandas makes of the pair - including a tz-aware frame against a
    # naive calendar, whose instants must not be matched by raw ticks - the
    # mask has to agree with it.
    mask = _signal_mask_matrix({"AAA": series}, bars, master, ["AAA"])
    expected = series.reindex(master).fillna(False).astype(bool)
    assert mask["AAA"].to_numpy().tolist() == expected.to_numpy().tolist()

    if not index.is_monotonic_increasing:
        return  # searchsorted has no meaning on an unsorted index either way
    try:
        expected_last = index.searchsorted(master, side="right") - 1
    except TypeError:
        # pandas refuses to compare naive and tz-aware stamps; so must we.
        with pytest.raises(TypeError):
            _last_positions(index, master, _master_ticks(master))
        return
    last = _last_positions(index, master, _master_ticks(master))
    assert last.tolist() == expected_last.tolist()
