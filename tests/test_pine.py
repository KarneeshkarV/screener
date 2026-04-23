from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.pine import (
    PineNameError,
    PineSyntaxError,
    evaluate,
    parse,
    required_lookback,
)


def _bars(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    openp = close + rng.normal(0, 0.3, n)
    vol = rng.integers(1_000, 10_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ── indicator correctness ──────────────────────────────────────────


def test_sma_matches_manual_mean():
    bars = _bars(10)
    node = parse("sma(close, 3)")
    out = evaluate(node, bars)
    expected = bars["close"].rolling(3, min_periods=3).mean()
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_ema_matches_pandas_ewm():
    bars = _bars(20)
    out = evaluate(parse("ema(close, 5)"), bars)
    expected = bars["close"].ewm(span=5, adjust=False, min_periods=5).mean()
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_rsi_bounds_and_known_vector():
    # Pure monotone up = RSI 100 after warmup; pure down = RSI 0.
    up = pd.DataFrame(
        {
            "open": np.arange(1, 21, dtype=float),
            "high": np.arange(1, 21, dtype=float) + 0.5,
            "low": np.arange(1, 21, dtype=float) - 0.5,
            "close": np.arange(1, 21, dtype=float),
            "volume": np.ones(20),
        }
    )
    r = evaluate(parse("rsi(close, 14)"), up)
    assert r.iloc[-1] == pytest.approx(100.0, rel=1e-6)
    assert (r.dropna() >= 0).all() and (r.dropna() <= 100).all()

    down = up.assign(close=up["close"].iloc[::-1].values)
    r2 = evaluate(parse("rsi(close, 14)"), down)
    # strict down → all losses, no gains → RSI 0
    assert r2.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_highest_lowest_window():
    bars = _bars(15)
    hi = evaluate(parse("highest(high, 5)"), bars)
    lo = evaluate(parse("lowest(low, 5)"), bars)
    pd.testing.assert_series_equal(
        hi, bars["high"].rolling(5, min_periods=5).max(), check_names=False
    )
    pd.testing.assert_series_equal(
        lo, bars["low"].rolling(5, min_periods=5).min(), check_names=False
    )


def test_atr_length_14():
    bars = _bars(30, seed=3)
    out = evaluate(parse("atr(14)"), bars)
    # first 13 are NaN (min_periods=14); values are positive after warmup
    assert out.iloc[:13].isna().all()
    assert (out.iloc[14:] > 0).all()


# ── no-lookahead properties ────────────────────────────────────────


def test_rolling_is_causal():
    bars = _bars(20, seed=1)
    out = evaluate(parse("sma(close, 5)"), bars)
    # Mutating a future bar must NOT change earlier values
    bars2 = bars.copy()
    bars2.iloc[15:, bars2.columns.get_loc("close")] += 1000
    out2 = evaluate(parse("sma(close, 5)"), bars2)
    pd.testing.assert_series_equal(out.iloc[:15], out2.iloc[:15])


def test_crossover_only_uses_i_and_iminus1():
    # Craft two series: fast and slow. A change at bar i-2 should not flip the
    # crossover result at bar i.
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    fast = pd.Series([1, 1, 1, 1, 1, 2, 3, 4, 5, 6], index=idx, dtype=float)
    slow = pd.Series([2, 2, 2, 2, 3, 3, 3, 3, 3, 3], index=idx, dtype=float)
    bars = pd.DataFrame(
        {
            "open": fast,
            "high": fast + 0.1,
            "low": fast - 0.1,
            "close": fast,
            "volume": 1.0,
            "_slow": slow,
        }
    )
    # Use close as fast, manually-built slow via sma(close, 1) equivalence for test
    # Instead, evaluate crossover directly by injecting two DataFrames: build one
    # with close=fast and another with close=slow, then evaluate both externally.
    a = evaluate(parse("close"), bars)
    b = bars["_slow"]
    from screener.backtester.pine import _crossover  # noqa: E402
    out = _crossover(a, b)
    # First cross happens at index 6 (bar i=6): fast[6]=3>slow[6]=3? actually
    # fast[6]=3 vs slow[6]=3 -> not strictly greater. Try at index 7:
    # fast[7]=4 > slow[7]=3 True; prev fast[6]=3 <= slow[6]=3 True => crossover
    assert out.iloc[7] is np.True_ or out.iloc[7] == True  # noqa: E712
    # Mutate bar i-2 (index 5) and confirm crossover at index 7 unchanged
    bars2 = bars.copy()
    bars2.iloc[5, bars2.columns.get_loc("close")] = 999.0
    a2 = evaluate(parse("close"), bars2)
    out2 = _crossover(a2, b)
    assert bool(out2.iloc[7]) == bool(out.iloc[7])


def test_crossunder_symmetric():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    a = pd.Series([5, 5, 5, 3, 2, 1], index=idx, dtype=float)
    b = pd.Series([4, 4, 4, 4, 4, 4], index=idx, dtype=float)
    bars = pd.DataFrame(
        {"open": a, "high": a, "low": a, "close": a, "volume": 1.0, "_b": b}
    )
    from screener.backtester.pine import _crossunder
    out = _crossunder(evaluate(parse("close"), bars), bars["_b"])
    # bar 3: a=3<b=4 and prev a=5>=b=4 → True
    assert bool(out.iloc[3])
    assert not bool(out.iloc[2])


# ── error handling ─────────────────────────────────────────────────


def test_unknown_identifier_raises():
    bars = _bars(5)
    with pytest.raises(PineNameError) as exc:
        evaluate(parse("foo + 1"), bars)
    assert "foo" in str(exc.value)


def test_unknown_function_raises():
    bars = _bars(5)
    with pytest.raises(PineNameError) as exc:
        evaluate(parse("bollinger(close, 20)"), bars)
    assert "bollinger" in str(exc.value)


def test_malformed_parens_raises_with_column():
    with pytest.raises(PineSyntaxError) as exc:
        parse("sma(close, 5")
    assert "column" in str(exc.value).lower() or "expected" in str(exc.value).lower()


def test_empty_expression_raises():
    with pytest.raises(PineSyntaxError):
        parse("   ")


def test_trailing_junk_raises():
    with pytest.raises(PineSyntaxError):
        parse("close > 1 junk")


def test_no_python_eval_injection():
    with pytest.raises((PineSyntaxError, PineNameError)):
        node = parse("__import__")
        evaluate(node, _bars(5))


def test_length_must_be_positive_int_literal():
    with pytest.raises(PineSyntaxError):
        evaluate(parse("sma(close, 0)"), _bars(5))
    with pytest.raises(PineSyntaxError):
        evaluate(parse("sma(close, 2.5)"), _bars(5))


def test_adj_close_alias_when_missing():
    bars = _bars(5)  # no adj_close column
    out_adj = evaluate(parse("adj_close"), bars)
    out_close = evaluate(parse("close"), bars)
    pd.testing.assert_series_equal(out_adj, out_close, check_names=False)


# ── composition / end-to-end expressions ───────────────────────────


def test_composite_expression():
    bars = _bars(50)
    node = parse(
        "close > ema(close, 20) and ema(close, 20) > ema(close, 50)"
    )
    out = evaluate(node, bars)
    assert out.dtype == bool
    assert len(out) == len(bars)


def test_breakout_expression():
    bars = _bars(300, seed=5)
    node = parse("close >= highest(close, 252) * 0.9 and volume > sma(volume, 10)")
    out = evaluate(node, bars)
    assert out.dtype == bool


def test_required_lookback():
    node = parse(
        "close > sma(close, 20) and ema(close, 50) > rsi(close, 14)"
    )
    assert required_lookback(node) == 50
    node2 = parse("crossover(close, sma(close, 10))")
    assert required_lookback(node2) == 10
    node3 = parse("atr(7) > 1")
    assert required_lookback(node3) == 7
