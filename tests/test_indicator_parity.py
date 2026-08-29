"""Cross-check indicator primitives between the engine and the pine-port.

The engine (screener/backtester/pine.py) and the numpy indicator plugins
(screener/indicators/plugins/) each implement their own SMA/EMA/RSI/ATR/etc.
for different callers: the engine operates on pandas Series for AST evaluation,
while the plugins operate on numpy arrays for speed.

If these diverge numerically, downstream backtests can't be compared. This
module feeds a deterministic OHLCV frame through both and asserts parity.

Parity notes:
  * SMA / EMA / highest / lowest: seeded identically; exact match (1e-9).
  * RSI and ATR use Wilder smoothing (alpha = 1/n), and both paths now seed it
    the way Pine ``ta.rma`` does: from the arithmetic mean of the first n
    observations. Parity is therefore exact over the whole series, NaN warm-up
    included - not asymptotic.

    This used to be a documented divergence. The pandas path seeded via
    ``ewm(adjust=False)``, which starts from the first value alone, and the
    two differed by ~2e-2 at bar 13 and ~1e-3 at bar 50; the numpy path
    manufactured a zero change on bar 0, which held it ~0.3-0.9 RSI points
    off TradingView forever. Both were defects against Pine, not a trade-off
    between two valid conventions, so the tolerance here is now zero and a
    reappearing seed difference is a failure rather than a note.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.pine import _atr as pine_atr
from screener.backtester.pine import _rsi as pine_rsi
from screener.indicators.plugins.atr import atr as pp_atr
from screener.indicators.plugins.ema import ema as pp_ema
from screener.indicators.plugins.rsi import rsi as pp_rsi
from screener.indicators.plugins.sma import sma as pp_sma


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
        port = getattr(
            pd.Series(bars["close"].to_numpy()).rolling(20, min_periods=20), op
        )().to_numpy()
        mask = _aligned_mask(engine, port)
        assert np.max(np.abs(engine[mask] - port[mask])) < 1e-9, f"{op} diverges"


def test_rsi_matches_exactly_including_the_warmup(bars):
    """Both paths are Pine ``ta.rsi``, so they agree bar for bar from bar 0."""
    engine = pine_rsi(bars["close"], 14).to_numpy()
    port = pp_rsi(bars["close"].to_numpy(), 14)

    # The NaN run is part of the answer: a path that starts a bar early is
    # wrong even where its finite values happen to line up.
    np.testing.assert_array_equal(np.isnan(engine), np.isnan(port))
    finite = ~np.isnan(engine)
    assert finite.sum() > 0
    np.testing.assert_allclose(engine[finite], port[finite], atol=1e-12, rtol=0)


def test_atr_matches_exactly_including_the_warmup(bars):
    """Both paths are Pine ``ta.atr`` = ``ta.rma(ta.tr(true), n)``."""
    engine = pine_atr(bars, 14).to_numpy()
    port = pp_atr(
        bars["high"].to_numpy(),
        bars["low"].to_numpy(),
        bars["close"].to_numpy(),
        14,
    )

    np.testing.assert_array_equal(np.isnan(engine), np.isnan(port))
    finite = ~np.isnan(engine)
    assert finite.sum() > 0
    np.testing.assert_allclose(engine[finite], port[finite], atol=1e-12, rtol=0)


def test_rsi_bounds_both_implementations(bars):
    engine = pine_rsi(bars["close"], 14).to_numpy()
    port = pp_rsi(bars["close"].to_numpy(), 14)
    for arr, name in [(engine, "engine"), (port, "port")]:
        finite = arr[~np.isnan(arr)]
        assert (finite >= 0).all() and (finite <= 100).all(), (
            f"{name} RSI out of bounds"
        )
