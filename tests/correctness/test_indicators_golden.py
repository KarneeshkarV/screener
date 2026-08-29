"""Hand-derived golden values for indicator warm-up seeds and edge contracts.

These pin the exact behaviour no external library reproduces: the screener's
specific seeding (RMA seed = mean of first n; EMA seed = first value), the
ATR first-bar true range, and the RSI all-up == 100 edge. Expected values are
re-derived here with plain arithmetic (NOT by calling the function under test),
so a mismatch is a real defect, not a transcription of the implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.indicators.frames import wilder_rsi
from screener.indicators.plugins.atr import atr
from screener.indicators.plugins.ema import ema
from screener.indicators.plugins.rma import rma
from screener.indicators.plugins.rsi import rsi
from screener.indicators.plugins.supertrend import supertrend_dir


def test_rma_seed_and_recursion_golden():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = rma(x, 3)
    # Independent re-derivation: seed at index n-1 = mean(first n); then alpha=1/n.
    e2 = (1 + 2 + 3) / 3
    e3 = (1 / 3) * 4 + (2 / 3) * e2
    e4 = (1 / 3) * 5 + (2 / 3) * e3
    assert np.isnan(got[0]) and np.isnan(got[1])
    np.testing.assert_allclose(got[2:], [e2, e3, e4], atol=1e-12, rtol=0)


def test_ema_seed_golden():
    x = np.array([10.0, 20.0, 30.0])
    got = ema(x, 2)  # alpha = 2/3, seed out[0] = x[0]
    e0 = 10.0
    e1 = (2 / 3) * 20 + (1 / 3) * e0
    e2 = (2 / 3) * 30 + (1 / 3) * e1
    np.testing.assert_allclose(got, [e0, e1, e2], atol=1e-12, rtol=0)


def test_atr_first_bar_true_range_golden():
    high = np.array([10.0, 12.0, 11.0])
    low = np.array([8.0, 9.0, 9.0])
    close = np.array([9.0, 11.0, 10.0])
    got = atr(high, low, close, 2)
    # prev_close = [9, 9, 11]; TR[0] uses close[0] as its own prev close.
    tr0 = max(10 - 8, abs(10 - 9), abs(8 - 9))  # = 2 (high-low dominates)
    tr1 = max(12 - 9, abs(12 - 9), abs(9 - 9))  # = 3
    a1 = (tr0 + tr1) / 2  # rma seed = mean of first 2 TRs = 2.5
    tr2 = max(11 - 9, abs(11 - 11), abs(9 - 11))  # = 2
    a2 = (1 / 2) * tr2 + (1 / 2) * a1  # = 2.25
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], [a1, a2], atol=1e-12, rtol=0)


def test_rsi_all_up_warmup_nan_then_100():
    """A strictly increasing series: the n-1 warm-up bars are NaN (RMA warm-up
    convention); post-warm-up bars have zero downside → RSI pinned at 100.

    (Previously the warm-up region was a spurious 100 because rma_dn=NaN made
    rs=inf — the NaN-warmup fix removes that false overbought signal.)
    """
    n = 3
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    got = rsi(close, n)
    assert np.isnan(got[: n - 1]).all()
    assert np.all(got[n - 1 :] == 100.0)


def test_rsi_mixed_golden():
    close = np.array([10.0, 11.0, 10.5, 11.5, 11.0])
    got = rsi(close, 3)
    # diff (prepend close[0]) = [0, 1, -0.5, 1, -0.5]
    # up=[0,1,0,1,0], dn=[0,0,.5,0,.5]; rma(n=3) seeds at idx2.
    rma_up2 = (0 + 1 + 0) / 3
    rma_dn2 = (0 + 0 + 0.5) / 3
    rsi2 = 100 - 100 / (1 + rma_up2 / rma_dn2)  # = 66.666...
    rma_up3 = (1 / 3) * 1 + (2 / 3) * rma_up2
    rma_dn3 = (1 / 3) * 0 + (2 / 3) * rma_dn2
    rsi3 = 100 - 100 / (1 + rma_up3 / rma_dn3)  # = 83.333...
    rma_up4 = (1 / 3) * 0 + (2 / 3) * rma_up3
    rma_dn4 = (1 / 3) * 0.5 + (2 / 3) * rma_dn3
    rsi4 = 100 - 100 / (1 + rma_up4 / rma_dn4)  # = 60.606...
    np.testing.assert_allclose(got[2:], [rsi2, rsi3, rsi4], atol=1e-9, rtol=0)


def test_supertrend_direction_trend_contract():
    """Independent property: a strong uptrend ends uptrend (<0), downtrend ends >0."""
    up = np.linspace(100, 200, 60)
    high_up, low_up, close_up = up + 1, up - 1, up
    dir_up = supertrend_dir(high_up, low_up, close_up, 10, 3.0)
    assert dir_up[-1] < 0  # uptrend convention

    down = np.linspace(200, 100, 60)
    high_dn, low_dn, close_dn = down + 1, down - 1, down
    dir_dn = supertrend_dir(high_dn, low_dn, close_dn, 10, 3.0)
    assert dir_dn[-1] > 0  # downtrend convention


def test_wilder_rsi_seeds_from_the_sma_like_pine_ta_rma():
    """``wilder_rsi`` must seed each smoother from the mean of the first n
    changes, the way Pine ``ta.rma`` does - not from the first change alone.

    ``ewm(adjust=False)`` seeds from the first observation, and that error only
    decays by ``(1 - 1/n)`` per bar: at n=14 it is still worth more than a full
    RSI point around bar 50, which is enough to flip an ``rsi > 50`` crossing.
    """
    close = pd.Series([10.0, 11.0, 10.5, 11.5, 11.0, 12.0])
    got = wilder_rsi(close, 3, min_periods=3)

    # Independent re-derivation. close.diff() is NaN on bar 0, so the three
    # seeding observations are bars 1..3 and the first value lands on bar 3.
    # gains  = [nan, 1, 0, 1, 0, 1]; losses = [nan, 0, 0.5, 0, 0.5, 0]
    g3 = (1 + 0 + 1) / 3
    l3 = (0 + 0.5 + 0) / 3
    g4 = (1 / 3) * 0 + (2 / 3) * g3
    l4 = (1 / 3) * 0.5 + (2 / 3) * l3
    g5 = (1 / 3) * 1 + (2 / 3) * g4
    l5 = (1 / 3) * 0 + (2 / 3) * l4
    expected = [100 - 100 / (1 + g / lo) for g, lo in ((g3, l3), (g4, l4), (g5, l5))]

    assert got.iloc[:3].isna().all()
    np.testing.assert_allclose(got.iloc[3:], expected, atol=1e-12, rtol=0)


def test_wilder_rsi_min_periods_masks_without_moving_the_seed():
    """``min_periods`` hides early bars; it must not change what was seeded.

    Conflating the two is the defect this pins: passing ``min_periods=period``
    was assumed to give Pine's warm-up *and* Pine's seed, and it gave only the
    first.
    """
    close = pd.Series([10.0, 11.0, 10.5, 11.5, 11.0, 12.0, 11.5, 12.5])

    unmasked = wilder_rsi(close, 3)
    masked = wilder_rsi(close, 3, min_periods=3)

    both = unmasked.notna() & masked.notna()
    assert both.sum() > 0
    pd.testing.assert_series_equal(unmasked[both], masked[both])
