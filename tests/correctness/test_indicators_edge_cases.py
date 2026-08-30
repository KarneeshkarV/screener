"""Edge-case contracts for indicators: warm-up shape, short/flat/short series, NaN.

These pin behaviour at the boundaries where bugs hide (off-by-one warm-up,
divide-by-zero on flat markets, empty windows). They are property/contract
assertions, independent of any reference library.
"""

from __future__ import annotations

import numpy as np

from screener.indicators.plugins.atr import atr
from screener.indicators.plugins.bollinger_bands import bollinger_bands
from screener.indicators.plugins.ema import ema
from screener.indicators.plugins.rma import rma
from screener.indicators.plugins.rsi import rsi
from screener.indicators.plugins.sar import sar
from screener.indicators.plugins.sma import sma
from screener.indicators.plugins.stdev import stdev


# --------------------------------------------------------------------------- #
# Warm-up NaN shape
# --------------------------------------------------------------------------- #
def test_sma_warmup_nan_shape():
    x = np.arange(1, 21, dtype=float)
    got = sma(x, 5)
    assert np.isnan(got[:4]).all()
    assert np.isfinite(got[4:]).all()


def test_stdev_warmup_nan_shape():
    x = np.arange(1, 21, dtype=float)
    got = stdev(x, 5)
    assert np.isnan(got[:4]).all()
    assert np.isfinite(got[4:]).all()


def test_rma_warmup_nan_shape():
    x = np.arange(1, 21, dtype=float)
    got = rma(x, 5)
    assert np.isnan(got[:4]).all()
    assert np.isfinite(got[4:]).all()


def test_atr_warmup_nan_shape():
    n = 14
    high = np.arange(2, 32, dtype=float)
    low = np.arange(0, 30, dtype=float)
    close = np.arange(1, 31, dtype=float)
    got = atr(high, low, close, n)
    assert np.isnan(got[: n - 1]).all()
    assert np.isfinite(got[n - 1 :]).all()


def test_ema_has_no_warmup_nan():
    """EMA seeds at out[0]=x[0] and emits a value from bar 0 — no NaN warm-up."""
    x = np.arange(1, 21, dtype=float)
    got = ema(x, 5)
    assert np.isfinite(got).all()


# --------------------------------------------------------------------------- #
# Series shorter than window
# --------------------------------------------------------------------------- #
def test_short_series_returns_all_nan_no_exception():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isnan(sma(x, 5)).all()
    assert np.isnan(stdev(x, 5)).all()
    assert np.isnan(rma(x, 5)).all()
    # shape preserved
    assert sma(x, 5).shape == x.shape


# --------------------------------------------------------------------------- #
# Flat (zero-variance) series
# --------------------------------------------------------------------------- #
def test_flat_series_zero_std_and_collapsed_bands():
    x = np.full(30, 5.0)
    s = stdev(x, 10)
    np.testing.assert_allclose(s[9:], 0.0, atol=1e-12)
    lower, middle, upper = bollinger_bands(x, 10, 2.0)
    np.testing.assert_allclose(lower[9:], middle[9:], atol=1e-12)
    np.testing.assert_allclose(upper[9:], middle[9:], atol=1e-12)


def test_flat_series_rsi_warmup_nan_then_100():
    """Flat market: the first n bars are NaN, then RSI is pinned at 100.

    n, not n-1: bar 0 has no prior close, so the smoother needs bars 1..n to
    collect n real changes. Past that there is zero downside, so RSI is 100.
    """
    n = 14
    x = np.full(30, 5.0)
    got = rsi(x, n)
    assert np.isnan(got[:n]).all()
    assert np.all(got[n:] == 100.0)


# --------------------------------------------------------------------------- #
# Single-element series
# --------------------------------------------------------------------------- #
def test_single_element_series():
    x = np.array([5.0])
    np.testing.assert_array_equal(ema(x, 3), [5.0])
    assert np.isnan(rma(x, 3)).all()
    # rsi must not raise on a length-1 input
    assert rsi(x, 14).shape == (1,)
    # sar short-circuits when there are fewer than two bars.
    np.testing.assert_array_equal(sar(x, x, x), [0.0])


# --------------------------------------------------------------------------- #
# NaN in the middle (behavioural pinning)
# --------------------------------------------------------------------------- #
def test_ema_propagates_nan_forward():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    got = ema(x, 3)
    assert np.isfinite(got[:2]).all()
    assert np.isnan(got[2:]).all()


def test_sma_window_with_nan_is_nan():
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0])
    got = sma(x, 3)
    # any window covering the NaN (indices 3,4,5) is NaN; window at idx 6 is clean.
    assert np.isnan(got[3]) and np.isnan(got[4]) and np.isnan(got[5])
    assert np.isfinite(got[6])
