"""Parity tests for the panel form of :func:`_precompute_filter_signals`.

The panel path exists purely as a speedup, so the bar it has to clear is that
every ticker gets *exactly* the boolean Series the per-ticker loop produced —
including for tickers that cannot be batched (duplicate index labels, mixed
timezones) and for filter configurations that select nothing or everything.

``_reference`` below is the implementation this replaced, kept verbatim as the
oracle; if it and the panel path ever disagree, the panel path is wrong.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from screener.backtester.core import _precompute_filter_signals


@dataclass
class _Cfg:
    """Minimal stand-in for the three BacktestConfig fields this function reads."""

    min_price: float | None = 1.0
    min_avg_dollar_volume: float | None = 1000.0
    avg_dollar_volume_window: int = 20


def _reference(bars_by_ticker: dict, cfg: _Cfg) -> dict[str, pd.Series]:
    """The per-ticker loop the panel path replaced."""
    if cfg.min_price is None and cfg.min_avg_dollar_volume is None:
        return {}
    window = max(int(cfg.avg_dollar_volume_window), 1)
    out: dict[str, pd.Series] = {}
    for ticker, bars in bars_by_ticker.items():
        if bars is None or bars.empty:
            continue
        close = bars["close"].astype(float)
        passes = pd.Series(True, index=bars.index)
        if cfg.min_price is not None:
            passes &= close >= float(cfg.min_price)
        if cfg.min_avg_dollar_volume is not None:
            volume = bars["volume"].astype(float)
            dollar_vol = close * volume
            adv = dollar_vol.rolling(window=window, min_periods=1).mean()
            adv_ok = np.isfinite(adv.values) & (
                adv.values >= float(cfg.min_avg_dollar_volume)
            )
            passes &= pd.Series(adv_ok, index=bars.index)
        out[ticker] = passes.astype(bool)
    return out


def _bars(n: int = 60, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=pd.date_range(start, periods=n, freq="D"),
    )


def _assert_matches_reference(bars_by_ticker: dict, cfg: _Cfg) -> dict:
    # pandas is silent on 0.0 denominators and inf arithmetic; numpy is not, so
    # promote RuntimeWarning to catch any the panel path introduces.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        expected = _reference(bars_by_ticker, cfg)
        actual = _precompute_filter_signals(bars_by_ticker, cfg)
    assert list(actual) == list(expected)
    for ticker, want in expected.items():
        pd.testing.assert_series_equal(actual[ticker], want, check_names=True)
    return actual


# ── configurations ───────────────────────────────────────────────────

CFGS = {
    "both_filters": _Cfg(1.0, 1000.0, 20),
    "adv_only": _Cfg(None, 1000.0, 20),
    "price_only": _Cfg(1.0, None, 20),
    "nothing_passes": _Cfg(1e9, 1e15, 20),
    "everything_passes": _Cfg(-1e9, -1e9, 20),
    "window_one": _Cfg(105.0, 500_000.0, 1),
    "window_clamped_to_one": _Cfg(105.0, 500_000.0, 0),
    "window_longer_than_history": _Cfg(100.0, 400_000.0, 500),
}


@pytest.mark.parametrize("cfg_name", list(CFGS))
def test_matches_reference_for_aligned_bars(cfg_name):
    bars_by_ticker = {f"T{i}": _bars(seed=i) for i in range(8)}
    _assert_matches_reference(bars_by_ticker, CFGS[cfg_name])


@pytest.mark.parametrize("cfg_name", list(CFGS))
def test_matches_reference_for_ragged_bars(cfg_name):
    """Mixed calendars: some tickers batch, the rest go solo."""
    bars_by_ticker = {
        "A": _bars(60, seed=1),
        "B": _bars(60, seed=2),
        "SHORT": _bars(25, seed=3, start="2024-02-01"),
        "OFFSET": _bars(60, seed=4, start="2023-06-01"),
        "GAPPED": _bars(60, seed=5).drop(
            index=pd.date_range("2024-01-20", periods=7, freq="D")
        ),
    }
    _assert_matches_reference(bars_by_ticker, CFGS[cfg_name])


def test_no_filters_configured_returns_the_empty_sentinel():
    bars_by_ticker = {f"T{i}": _bars(seed=i) for i in range(3)}
    assert _precompute_filter_signals(bars_by_ticker, _Cfg(None, None, 20)) == {}


# ── tickers that must not be batched ─────────────────────────────────


def test_ragged_ticker_is_not_contaminated_by_its_neighbours():
    """Exact-index grouping: no NaN padding may leak into a short history.

    A union-index panel would pad SHORT's missing rows and change its rolling
    ADV; grouping on exact index equality must leave it bit-identical.
    """
    short = _bars(25, seed=3, start="2024-02-01")
    bars_by_ticker = {"A": _bars(60, seed=1), "B": _bars(60, seed=2), "SHORT": short}
    result = _precompute_filter_signals(bars_by_ticker, _Cfg(1.0, 1000.0, 20))
    assert result["SHORT"].index.equals(short.index)
    pd.testing.assert_series_equal(
        result["SHORT"], _reference({"SHORT": short}, _Cfg(1.0, 1000.0, 20))["SHORT"]
    )


def test_alone_matches_grouped_for_the_same_ticker():
    """A ticker's result must not depend on who else is in the run."""
    cfg = _Cfg(100.0, 400_000.0, 10)
    target = _bars(60, seed=1)
    alone = _precompute_filter_signals({"T": target}, cfg)["T"]
    crowded = _precompute_filter_signals(
        {"T": target, "U": _bars(60, seed=2), "V": _bars(60, seed=3)}, cfg
    )["T"]
    pd.testing.assert_series_equal(alone, crowded)


def test_duplicate_index_labels_fall_back_to_solo():
    dup = _bars(20, seed=9)
    dup = dup.set_axis(dup.index[:19].append(dup.index[18:19]))
    bars_by_ticker = {"A": _bars(20, seed=1), "B": _bars(20, seed=2), "DUP": dup}
    _assert_matches_reference(bars_by_ticker, _Cfg(100.0, 400_000.0, 5))


@pytest.mark.parametrize("variant", ["tz_convert", "tz_naive", "resolution"])
def test_equal_instants_with_different_index_dtypes_do_not_group(variant):
    """Timestamps compare and hash by instant, not by label.

    Two indexes over the same moments in different timezones (or at a different
    resolution) are equal as tuples, so without the dtype in the group key they
    would stack together — and every member is handed the first member's index
    back, silently relabelling the rest into that timezone. This is the same
    trap ``pine._group_key`` guards, via the shared ``panel_index_key``.
    """
    aware = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    other = {
        "tz_convert": aware.tz_convert("Asia/Kolkata"),
        "tz_naive": aware.tz_localize(None),
        "resolution": aware.tz_localize(None).astype("datetime64[us]"),
    }[variant]
    bars_by_ticker = {
        "AWARE": _bars(40, seed=1).set_axis(aware),
        "OTHER": _bars(40, seed=2).set_axis(other),
    }
    result = _assert_matches_reference(bars_by_ticker, _Cfg(100.0, 400_000.0, 5))
    # Specifically: the index must survive, not just the values.
    assert result["AWARE"].index.equals(aware)
    assert result["OTHER"].index.equals(other)


def test_non_datetime_index_still_matches():
    bars_by_ticker = {
        f"T{i}": _bars(20, seed=i).set_axis(pd.RangeIndex(20)) for i in range(3)
    }
    _assert_matches_reference(bars_by_ticker, _Cfg(100.0, 400_000.0, 5))


# ── degenerate inputs ────────────────────────────────────────────────


def test_empty_and_missing_frames_are_skipped():
    bars_by_ticker = {
        "REAL": _bars(seed=1),
        "EMPTY": pd.DataFrame(),
        "NONE": None,
        "REAL2": _bars(seed=2),
    }
    result = _precompute_filter_signals(bars_by_ticker, _Cfg(1.0, 1000.0, 20))
    assert list(result) == ["REAL", "REAL2"]


def test_result_order_follows_input_order_not_grouping_order():
    """Callers build DataFrames straight from this dict, so order is observable."""
    bars_by_ticker = {
        "SOLO_FIRST": _bars(30, seed=1, start="2023-01-01"),
        "GROUP_A": _bars(60, seed=2),
        "SOLO_SECOND": _bars(45, seed=3, start="2022-06-01"),
        "GROUP_B": _bars(60, seed=4),
    }
    result = _precompute_filter_signals(bars_by_ticker, _Cfg(1.0, 1000.0, 20))
    assert list(result) == list(bars_by_ticker)


def test_single_bar_history():
    _assert_matches_reference(
        {"ONE": _bars(1, seed=1), "TWO": _bars(1, seed=2)}, _Cfg(1.0, 1000.0, 20)
    )


def test_nan_and_inf_fail_the_filter_like_the_per_ticker_path():
    dirty = _bars(30, seed=4)
    dirty.loc[dirty.index[3], "close"] = np.nan
    dirty.loc[dirty.index[7], "volume"] = np.nan
    dirty.loc[dirty.index[11], "close"] = np.inf
    dirty.loc[dirty.index[15], "volume"] = np.inf
    dirty.loc[dirty.index[19], "volume"] = 0.0
    result = _assert_matches_reference(
        {"CLEAN": _bars(30, seed=5), "DIRTY": dirty}, _Cfg(1.0, 1000.0, 5)
    )
    # The NaN close bar must fail regardless of what the ADV window says.
    assert not bool(result["DIRTY"].iloc[3])


def test_all_nan_volume_column():
    nan_vol = _bars(20, seed=6)
    nan_vol["volume"] = np.nan
    result = _assert_matches_reference(
        {"OK": _bars(20, seed=7), "NANVOL": nan_vol}, _Cfg(1.0, 1000.0, 5)
    )
    assert not result["NANVOL"].any()


@pytest.mark.parametrize(
    "close_dtype,volume_dtype",
    [("int64", "int64"), (object, object), ("Float64", "Int64")],
)
def test_non_float_input_dtypes(close_dtype, volume_dtype):
    """int, object and pandas-nullable columns must coerce as they did before."""
    odd = _bars(30, seed=8)
    odd["close"] = odd["close"].round().astype(close_dtype)
    odd["volume"] = odd["volume"].astype(volume_dtype)
    _assert_matches_reference(
        {"ODD": odd, "NORMAL": _bars(30, seed=9)}, _Cfg(1.0, 1000.0, 10)
    )


def test_missing_close_column_still_raises():
    """The panel path must not swallow the KeyError the loop raised."""
    broken = _bars(20, seed=1).drop(columns=["close"])
    with pytest.raises(KeyError):
        _precompute_filter_signals({"BROKEN": broken}, _Cfg(1.0, 1000.0, 5))


def test_missing_volume_column_only_matters_when_adv_is_configured():
    no_volume = _bars(20, seed=1).drop(columns=["volume"])
    ok = _precompute_filter_signals({"NOVOL": no_volume}, _Cfg(1.0, None, 5))
    assert list(ok) == ["NOVOL"]
    with pytest.raises(KeyError):
        _precompute_filter_signals({"NOVOL": no_volume}, _Cfg(1.0, 1000.0, 5))


def test_input_frames_are_not_mutated():
    bars_by_ticker = {f"T{i}": _bars(seed=i) for i in range(4)}
    before = {t: b.copy(deep=True) for t, b in bars_by_ticker.items()}
    _precompute_filter_signals(bars_by_ticker, _Cfg(1.0, 1000.0, 20))
    for ticker, original in before.items():
        pd.testing.assert_frame_equal(bars_by_ticker[ticker], original)


def test_returned_series_do_not_alias_each_other():
    """Each ticker's Series is sliced out of one shared block; writes must not bleed."""
    bars_by_ticker = {f"T{i}": _bars(30, seed=i) for i in range(4)}
    result = _precompute_filter_signals(bars_by_ticker, _Cfg(-1e9, -1e9, 5))
    result["T0"].iloc[:] = False
    assert result["T1"].all()
    assert result["T2"].all()
    assert result["T3"].all()
