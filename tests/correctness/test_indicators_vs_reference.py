"""Indicators vs independent reference libraries (pandas_ta_classic / TA-Lib).

These are the *independent* checks the existing suite lacks: the screener's
numpy indicators are compared against separately-implemented references, with
every convention difference reconciled in ``reference_adapters``. Recursive
indicators (EMA/RSI/ATR) are compared on their converged tail; exact-formula
indicators (SMA/BBANDS/STDEV) are compared everywhere they are defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as pta
import pytest

from screener.indicators.plugins.atr import atr
from screener.indicators.plugins.bollinger_bands import bollinger_bands
from screener.indicators.plugins.ema import ema
from screener.indicators.plugins.rsi import rsi
from screener.indicators.plugins.sma import sma
from screener.indicators.plugins.stdev import stdev
from screener.indicators.plugins.supertrend import supertrend_dir
from screener.backtester.vbt_sweep import _obv

from tests.correctness.reference_adapters import finite_tail_mask, require_talib


@pytest.fixture(scope="module")
def series():
    """Deterministic 250-bar OHLCV sample with positive prices."""
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 250))
    high = close + rng.uniform(0.2, 1.2, 250)
    low = close - rng.uniform(0.2, 1.2, 250)
    volume = rng.integers(100_000, 500_000, 250).astype(float)
    return {"close": close, "high": high, "low": low, "volume": volume}


# --------------------------------------------------------------------------- #
# Exact-formula indicators (agree everywhere they are defined)
# --------------------------------------------------------------------------- #
def test_sma_matches_pandas_ta_exact(series):
    got = sma(series["close"], 20)
    ref = pta.sma(pd.Series(series["close"]), length=20).to_numpy()
    mask = np.isfinite(got) & np.isfinite(ref)
    assert mask.sum() > 100
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-12, rtol=0)


def test_stdev_matches_numpy_population_exact(series):
    """Independent witness: manual numpy population std per rolling window."""
    x, n = series["close"], 20
    got = stdev(x, n)
    ref = np.full_like(got, np.nan)
    for i in range(n - 1, len(x)):
        ref[i] = np.std(x[i - n + 1 : i + 1], ddof=0)
    mask = np.isfinite(got) & np.isfinite(ref)
    assert mask.sum() > 100
    # 1e-10, not 1e-12: pandas' streaming rolling-variance differs from numpy's
    # batch std only by float-accumulation noise (~2e-12 here), not formula.
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-10, rtol=0)


def test_bbands_matches_pandas_ta_population_std(series):
    """Screener and pandas_ta_classic both use population std (ddof=0)."""
    lower, middle, upper = bollinger_bands(series["close"], 20, 2.0)
    bb = pta.bbands(pd.Series(series["close"]), length=20, std=2.0)
    ref_lo = bb["BBL_20_2.0"].to_numpy()
    ref_mid = bb["BBM_20_2.0"].to_numpy()
    ref_up = bb["BBU_20_2.0"].to_numpy()
    mask = np.isfinite(upper) & np.isfinite(ref_up)
    assert mask.sum() > 100
    np.testing.assert_allclose(middle[mask], ref_mid[mask], atol=1e-9, rtol=0)
    np.testing.assert_allclose(lower[mask], ref_lo[mask], atol=1e-9, rtol=0)
    np.testing.assert_allclose(upper[mask], ref_up[mask], atol=1e-9, rtol=0)


# --------------------------------------------------------------------------- #
# Recursive indicators (agree on the converged tail; warm-up seed differs)
# --------------------------------------------------------------------------- #
def test_ema_matches_pandas_ta_tail(series):
    got = ema(series["close"], 20)
    ref = pta.ema(pd.Series(series["close"]), length=20, presma=False).to_numpy()
    # EMA seed differs (out[0]=x[0] vs SMA seed); converges as (1-2/21)^k. Need
    # ~200 bars for an n=20 EMA to agree to 1e-6 (residual ~5e-9 by then).
    mask = finite_tail_mask(got, ref, start=200)
    assert mask.sum() > 40
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-6, rtol=0)


def test_rsi_matches_pandas_ta_tail(series):
    got = rsi(series["close"], 14)
    ref = pta.rsi(pd.Series(series["close"]), length=14).to_numpy()
    mask = finite_tail_mask(got, ref, start=100)
    assert mask.sum() > 50
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-3, rtol=0)


def test_atr_matches_pandas_ta_tail(series):
    got = atr(series["high"], series["low"], series["close"], 14)
    ref = pta.atr(
        pd.Series(series["high"]),
        pd.Series(series["low"]),
        pd.Series(series["close"]),
        length=14,
    ).to_numpy()
    mask = finite_tail_mask(got, ref, start=100)
    assert mask.sum() > 50
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-2, rtol=0)


def test_supertrend_dir_matches_pandas_ta_signflip_tail(series):
    """Screener direction<0==uptrend is the inverse of pandas_ta's +1==uptrend."""
    got = supertrend_dir(series["high"], series["low"], series["close"], 10, 3.0)
    st = pta.supertrend(
        pd.Series(series["high"]),
        pd.Series(series["low"]),
        pd.Series(series["close"]),
        length=10,
        multiplier=3.0,
    )
    dcol = next(c for c in st.columns if c.startswith("SUPERTd"))
    ref_dir = st[dcol].to_numpy()
    tail = slice(50, None)
    # Categorical agreement on the sign (uptrend/downtrend), reconciled by flip.
    assert np.all(np.sign(got[tail]) == -np.sign(ref_dir[tail]))


def test_obv_matches_pandas_ta_after_offset_removal(series):
    """TA-Lib/pandas_ta seed OBV at volume[0]; screener starts at 0. Compare diffs."""
    close = pd.DataFrame({"a": series["close"]})
    volume = pd.DataFrame({"a": series["volume"]})
    got = _obv(close, volume)[:, 0]
    ref = pta.obv(pd.Series(series["close"]), pd.Series(series["volume"])).to_numpy()
    # First-differencing removes the constant seed offset.
    np.testing.assert_allclose(np.diff(got), np.diff(ref), atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# TA-Lib as a second independent witness (skipped if the C lib is absent)
# --------------------------------------------------------------------------- #
def test_bbands_matches_talib_exact(series):
    talib = require_talib()
    lower, middle, upper = bollinger_bands(series["close"], 20, 2.0)
    tu, tm, tl = talib.BBANDS(
        series["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    mask = np.isfinite(upper) & np.isfinite(tu)
    np.testing.assert_allclose(upper[mask], tu[mask], atol=1e-9, rtol=0)
    np.testing.assert_allclose(lower[mask], tl[mask], atol=1e-9, rtol=0)
    np.testing.assert_allclose(middle[mask], tm[mask], atol=1e-9, rtol=0)


def test_rsi_matches_talib_tail(series):
    talib = require_talib()
    got = rsi(series["close"], 14)
    ref = talib.RSI(series["close"], 14)
    mask = finite_tail_mask(got, ref, start=100)
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-3, rtol=0)


def test_atr_matches_talib_tail(series):
    talib = require_talib()
    got = atr(series["high"], series["low"], series["close"], 14)
    ref = talib.ATR(series["high"], series["low"], series["close"], 14)
    mask = finite_tail_mask(got, ref, start=100)
    np.testing.assert_allclose(got[mask], ref[mask], atol=1e-2, rtol=0)


def test_obv_seed_offset_is_volume0(series):
    """Pin the documented OBV seed difference: TA-Lib starts at volume[0], we start at 0."""
    talib = require_talib()
    close = pd.DataFrame({"a": series["close"]})
    volume = pd.DataFrame({"a": series["volume"]})
    got = _obv(close, volume)[:, 0]
    ref = talib.OBV(series["close"], series["volume"])
    np.testing.assert_allclose(got, ref - series["volume"][0], atol=1e-6, rtol=0)
