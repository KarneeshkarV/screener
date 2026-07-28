"""Parity tests for :func:`evaluate_panel` against the per-ticker evaluator.

The panel path exists purely as a speedup, so the bar it has to clear is that
every ticker gets *exactly* the Series ``evaluate`` would have produced for it
on its own — including for tickers that cannot be batched (ragged indexes,
missing columns) and for expressions that fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.pine import (
    PineError,
    PineNameError,
    _group_key,
    _panel_column_names,
    evaluate,
    evaluate_panel,
    evaluate_panel_many,
    parse,
)

EXPRESSIONS = [
    "close",
    "close > sma(close, 5)",
    "crossover(close, sma(close, 10)) and close > sma(close, 5)",
    "crossunder(close, sma(close, 10))",
    "ema(close, 8) > sma(close, 12)",
    "rsi(close, 14) < 70",
    "atr(14) > 0.5",
    "highest(high, 5) - lowest(low, 5)",
    "not (close < open)",
    "adj_close * 2 - 1 > 0",
    "(high + low) / 2 >= close",
    "volume > 0 or close > 0",
    "-close < 0",
    "1 > 0",
]


def _bars(n: int = 60, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    openp = close + rng.normal(0, 0.3, n)
    vol = rng.integers(1_000, 10_000, n).astype(float)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _assert_matches_per_ticker(expr: str, bars_by_ticker: dict) -> dict:
    """Assert evaluate_panel(expr) equals evaluate(expr) for every ticker."""
    node = parse(expr)
    panel = evaluate_panel(node, bars_by_ticker)
    for ticker, bars in bars_by_ticker.items():
        if bars is None or bars.empty:
            assert ticker not in panel
            continue
        try:
            expected = evaluate(node, bars)
        except PineError as exc:
            assert isinstance(panel[ticker], PineError)
            assert str(panel[ticker]) == str(exc)
            assert type(panel[ticker]) is type(exc)
            continue
        actual = panel[ticker]
        assert not isinstance(actual, PineError), actual
        pd.testing.assert_series_equal(
            actual, expected, check_names=False, check_dtype=True
        )
    return panel


# ── parity on a cleanly batchable universe ───────────────────────────


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_panel_matches_per_ticker_for_aligned_bars(expr):
    bars_by_ticker = {f"T{i}": _bars(seed=i) for i in range(6)}
    _assert_matches_per_ticker(expr, bars_by_ticker)


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_panel_matches_per_ticker_for_ragged_bars(expr):
    """Mixed index lengths/offsets: some tickers batch, the rest go solo."""
    bars_by_ticker = {
        "ALIGNED_A": _bars(60, seed=1),
        "ALIGNED_B": _bars(60, seed=2),
        "ALIGNED_C": _bars(60, seed=3),
        # Shorter history — an IPO mid-window.
        "SHORT": _bars(25, seed=4, start="2024-02-01"),
        # Same length, different calendar — cannot share a panel.
        "OFFSET": _bars(60, seed=5, start="2023-06-01"),
        # A gap, as if the name were halted for a stretch.
        "GAPPED": _bars(60, seed=6).drop(
            index=pd.date_range("2024-01-20", periods=7, freq="D")
        ),
    }
    _assert_matches_per_ticker(expr, bars_by_ticker)


def test_panel_many_matches_repeated_panel_evaluation():
    bars_by_ticker = {f"T{i}": _bars(seed=i) for i in range(6)}
    nodes = [
        parse("close > sma(close, 5)"),
        parse("crossunder(close, sma(close, 5))"),
    ]

    together = evaluate_panel_many(nodes, bars_by_ticker)

    assert len(together) == len(nodes)
    for position, node in enumerate(nodes):
        separate = evaluate_panel(node, bars_by_ticker)
        assert list(together[position]) == list(separate)
        for ticker, expected in separate.items():
            actual = together[position][ticker]
            assert not isinstance(actual, PineError)
            assert not isinstance(expected, PineError)
            pd.testing.assert_series_equal(actual, expected)


def test_ragged_ticker_is_not_contaminated_by_its_neighbours():
    """The whole point of exact-index grouping: no NaN padding leaks in.

    A union-index panel would pad SHORT's early rows and change its rolling
    mean; grouping on exact index equality must leave it bit-identical.
    """
    short = _bars(25, seed=4, start="2024-02-01")
    bars_by_ticker = {
        "LONG_A": _bars(60, seed=1),
        "LONG_B": _bars(60, seed=2),
        "SHORT": short,
    }
    node = parse("sma(close, 10)")
    panel = evaluate_panel(node, bars_by_ticker)
    pd.testing.assert_series_equal(
        panel["SHORT"],
        short["close"].rolling(10, min_periods=10).mean(),
        check_names=False,
    )
    assert panel["SHORT"].index.equals(short.index)


def test_alone_matches_grouped_for_the_same_ticker():
    """A ticker's result must not depend on who else is in the run."""
    node = parse("crossover(close, sma(close, 10))")
    target = _bars(60, seed=1)
    alone = evaluate_panel(node, {"T": target})["T"]
    crowded = evaluate_panel(
        node, {"T": target, "U": _bars(60, seed=2), "V": _bars(60, seed=3)}
    )["T"]
    pd.testing.assert_series_equal(alone, crowded, check_names=False)


# ── error attribution ────────────────────────────────────────────────


def test_unknown_identifier_reported_per_ticker():
    bars_by_ticker = {f"T{i}": _bars(seed=i) for i in range(3)}
    result = evaluate_panel(parse("nonexistent > 0"), bars_by_ticker)
    assert set(result) == set(bars_by_ticker)
    for value in result.values():
        assert isinstance(value, PineNameError)


def test_missing_column_isolated_to_the_affected_ticker():
    """A ticker missing OHLCV fails alone; its neighbours still evaluate."""
    bars_by_ticker = {
        "GOOD_A": _bars(seed=1),
        "GOOD_B": _bars(seed=2),
        "BROKEN": _bars(seed=3).drop(columns=["volume"]),
    }
    result = evaluate_panel(parse("close > sma(close, 5)"), bars_by_ticker)
    assert isinstance(result["BROKEN"], PineError)
    assert isinstance(result["GOOD_A"], pd.Series)
    assert isinstance(result["GOOD_B"], pd.Series)


def test_column_present_for_some_tickers_only():
    """Fundamentals/options joins add columns to a subset — parity must hold."""
    bars_by_ticker = {}
    for i in range(4):
        bars = _bars(seed=i)
        if i % 2 == 0:
            bars = bars.assign(pcr=np.linspace(0.5, 1.5, len(bars)))
        bars_by_ticker[f"T{i}"] = bars
    _assert_matches_per_ticker("pcr > 1.0", bars_by_ticker)


def test_non_numeric_extra_column_is_coerced_like_the_solo_path():
    bars_by_ticker = {}
    for i in range(3):
        bars = _bars(20, seed=i)
        values = ["1.5"] * 20
        values[3] = "not-a-number"
        bars_by_ticker[f"T{i}"] = bars.assign(score=values)
    _assert_matches_per_ticker("score > 1.0", bars_by_ticker)


def test_empty_frames_are_skipped():
    bars_by_ticker = {
        "REAL": _bars(seed=1),
        "EMPTY": pd.DataFrame(),
        "NONE": None,
    }
    result = evaluate_panel(parse("close > 0"), bars_by_ticker)
    assert set(result) == {"REAL"}


def test_duplicate_index_labels_fall_back_to_solo():
    """Duplicate labels would make a positional stack unsafe."""
    dup = _bars(20, seed=9)
    dup = dup.set_axis(dup.index[:19].append(dup.index[18:19]))
    bars_by_ticker = {"A": _bars(20, seed=1), "B": _bars(20, seed=2), "DUP": dup}
    _assert_matches_per_ticker("sma(close, 3) > 0", bars_by_ticker)


@pytest.mark.parametrize(
    "expr", ["sma(close, 5)", "crossover(close, sma(close, 3))", "atr(5) > 0"]
)
@pytest.mark.parametrize(
    "variant",
    ["tz_convert", "tz_naive", "resolution"],
)
def test_equal_instants_with_different_index_dtypes_do_not_group(expr, variant):
    """Timestamps compare and hash by instant, not by label.

    Two indexes over the same moments in different timezones (or at a different
    resolution) are equal as tuples, so without the dtype in the group key they
    would stack together — and every member is handed ``frames[0].index`` back,
    silently relabelling the rest into the first member's timezone.
    """
    aware = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    other = {
        "tz_convert": aware.tz_convert("Asia/Kolkata"),
        "tz_naive": aware.tz_localize(None),
        "resolution": aware.tz_localize(None).astype("datetime64[us]"),
    }[variant]
    base = _bars(40, seed=1)
    bars_by_ticker = {
        "AWARE": base.set_axis(aware),
        "OTHER": _bars(40, seed=2).set_axis(other),
    }
    _assert_matches_per_ticker(expr, bars_by_ticker)
    # Specifically: the index must survive, not just the values.
    panel = evaluate_panel(parse(expr), bars_by_ticker)
    assert panel["OTHER"].index.equals(other)
    assert panel["AWARE"].index.equals(aware)


def test_uniform_timezone_still_groups():
    """The dtype guard must not stop a normal single-timezone universe."""
    idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    names = _panel_column_names(parse("sma(close, 5)"))
    keys = {_group_key(_bars(40, seed=i).set_axis(idx), names) for i in range(4)}
    assert len(keys) == 1 and None not in keys


def test_non_datetime_index_still_matches():
    bars_by_ticker = {}
    for i in range(3):
        bars = _bars(20, seed=i)
        bars_by_ticker[f"T{i}"] = bars.set_axis(pd.RangeIndex(len(bars)))
    _assert_matches_per_ticker("close > sma(close, 4)", bars_by_ticker)


# ── the aliasing guard (evaluate returns the caller's column) ─────────


def test_evaluate_does_not_mutate_or_alias_into_caller_state():
    """``_as_float`` skips the defensive copy for already-float columns.

    That is only sound while nothing downstream writes into an evaluate()
    result. This test fails loudly if a caller ever starts doing so, which is
    the trade accepted when the redundant astype was removed.
    """
    bars = _bars(30, seed=3)
    before = bars.copy(deep=True)
    for expr in EXPRESSIONS:
        evaluate(parse(expr), bars)
    pd.testing.assert_frame_equal(bars, before)


def test_panel_does_not_mutate_input_frames():
    bars_by_ticker = {f"T{i}": _bars(seed=i) for i in range(4)}
    before = {t: b.copy(deep=True) for t, b in bars_by_ticker.items()}
    for expr in EXPRESSIONS:
        evaluate_panel(parse(expr), bars_by_ticker)
    for ticker, original in before.items():
        pd.testing.assert_frame_equal(bars_by_ticker[ticker], original)
