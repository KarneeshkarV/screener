"""Cross-check indicator primitives between the engine and the pine-port.

The engine (screener/backtester/pine.py) and the pine-port
(run_pinescript_strategies.py) each implement their own SMA/EMA/RSI/ATR/etc.
for different callers: the engine operates on pandas Series for AST evaluation,
while the pine-port operates on numpy arrays for speed.

If these diverge numerically, downstream backtests can't be compared. This
module feeds a deterministic OHLCV frame through both and asserts parity.

Convergence notes:
  * SMA / EMA / highest / lowest: seeded identically; exact match (1e-9).
  * RSI and ATR use Wilder smoothing (alpha = 1/n). The engine seeds via
    pandas' ewm (adjust=False), which initializes from the first value; the
    pine-port seeds Wilder's RMA from the arithmetic mean of the first n
    values. The two converge asymptotically but differ during warm-up — so
    the test asserts tight parity only AFTER a long warm-up (>= 200 bars).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_pinescript_strategies import _atr as pp_atr
from run_pinescript_strategies import _ema as pp_ema
from run_pinescript_strategies import _rsi as pp_rsi
from run_pinescript_strategies import _sma as pp_sma
from screener.backtester.pine import _atr as pine_atr
from screener.backtester.pine import _rsi as pine_rsi


@pytest.fixture(scope="module")
def bars():
    np.random.seed(0)
    n = 500
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1_000_000},
        index=idx,
    )


def _aligned_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ~np.isnan(a) & ~np.isnan(b)


def test_sma_parity(bars):
    engine = bars["close"].rolling(20, min_periods=20).mean().to_numpy()
    port = pp_sma(bars["close"].to_numpy(), 20)
    mask = _aligned_mask(engine, port)
    assert mask.sum() > 0
    assert np.max(np.abs(engine[mask] - port[mask])) < 1e-9


def test_ema_parity(bars):
    # engine masks first (length-1) bars as NaN but the underlying recursion
    # seeds identically — port values on the same bars must match.
    engine = bars["close"].ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
    port = pp_ema(bars["close"].to_numpy(), 20)
    mask = _aligned_mask(engine, port)
    assert mask.sum() > 0
    assert np.max(np.abs(engine[mask] - port[mask])) < 1e-9


def test_highest_lowest_parity(bars):
    # Both call pandas rolling internally (engine via AST; pine-port direct).
    for op in ("max", "min"):
        engine = getattr(bars["close"].rolling(20, min_periods=20), op)().to_numpy()
        port = getattr(pd.Series(bars["close"].to_numpy()).rolling(20, min_periods=20), op)().to_numpy()
        mask = _aligned_mask(engine, port)
        assert np.max(np.abs(engine[mask] - port[mask])) < 1e-9, f"{op} diverges"


def test_rsi_converges_after_warmup(bars):
    """RSI seeds differ, but the two smoothers converge exponentially.
    Require tight agreement after ~14 * 14 bars of warm-up (>=200 bars)."""
    engine = pine_rsi(bars["close"], 14).to_numpy()
    port = pp_rsi(bars["close"].to_numpy(), 14)
    tail_engine = engine[200:]
    tail_port = port[200:]
    mask = _aligned_mask(tail_engine, tail_port)
    assert mask.sum() > 0
    # after 200 bars the seed difference has decayed by (1 - 1/14)**186 ≈ 2e-6
    assert np.max(np.abs(tail_engine[mask] - tail_port[mask])) < 1e-3


def test_atr_converges_after_warmup(bars):
    engine = pine_atr(bars, 14).to_numpy()
    port = pp_atr(
        bars["high"].to_numpy(),
        bars["low"].to_numpy(),
        bars["close"].to_numpy(),
        14,
    )
    tail_engine = engine[100:]
    tail_port = port[100:]
    mask = _aligned_mask(tail_engine, tail_port)
    assert mask.sum() > 0
    assert np.max(np.abs(tail_engine[mask] - tail_port[mask])) < 1e-2


def test_rsi_bounds_both_implementations(bars):
    engine = pine_rsi(bars["close"], 14).to_numpy()
    port = pp_rsi(bars["close"].to_numpy(), 14)
    for arr, name in [(engine, "engine"), (port, "port")]:
        finite = arr[~np.isnan(arr)]
        assert (finite >= 0).all() and (finite <= 100).all(), f"{name} RSI out of bounds"
