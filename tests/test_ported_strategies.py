from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.registry import STRATEGIES

PORTED_STRATEGIES = [
    "awesome_oscillator",
    "bb_pattern",
    "heikin_ashi",
    "macd_oscillator",
    "parabolic_sar",
    "rsi_pattern",
    "shooting_star",
]

def _ohlcv(n: int = 700) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    x = np.linspace(0, 18, n)
    close = 100 + np.linspace(0, 80, n) + np.sin(x) * 8
    high = close + 1.5
    low = close - 1.5
    open_ = close + np.sin(x / 2) * 0.5
    volume = np.full(n, 10_000.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )

def test_ported_strategies_registered():
    for name in PORTED_STRATEGIES:
        assert name in STRATEGIES

def test_ported_strategies_smoke():
    df = _ohlcv()

    for name in PORTED_STRATEGIES:
        strategy_fn = STRATEGIES[name]
        trades = strategy_fn(df)

        assert isinstance(trades, list), f"{name} returned {type(trades)}"
        for trade in trades:
            assert trade.entry_idx <= trade.exit_idx, f"{name}: entry > exit"


def _mkdf(open_, high, low, close) -> pd.DataFrame:
    """Build an OHLCV frame from explicit price arrays."""
    n = len(close)
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=n, freq="D"),
            "open": np.asarray(open_, dtype=float),
            "high": np.asarray(high, dtype=float),
            "low": np.asarray(low, dtype=float),
            "close": np.asarray(close, dtype=float),
            "adj_close": np.asarray(close, dtype=float),
            "volume": np.full(n, 10_000.0),
        }
    )


def test_shooting_star_short_series_returns_no_trades():
    # Fewer than 4 bars short-circuits before the pattern scan.
    df = _mkdf([1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert STRATEGIES["shooting_star"](df) == []


def test_shooting_star_pattern_entry_and_exit():
    # A shooting-star candle at index 5 (small bearish body, long upper shadow,
    # close on the low) confirmed at index 6, then a >5% move that exits.
    close = [98, 98.5, 99, 99, 99.5, 100.0, 99.5, 106, 106.5, 107, 107.5, 108]
    open_ = [c - 2 for c in close]
    high = [c + 1 for c in close]
    low = [c - 3 for c in close]
    open_[5], high[5], low[5] = 100.4, 101.4, 100.0  # the shooting star
    open_[6], high[6], low[6] = 97.5, 100.5, 96.5  # confirmation bar
    trades = STRATEGIES["shooting_star"](_mkdf(open_, high, low, close))
    assert len(trades) == 1
    assert trades[0].entry_idx <= trades[0].exit_idx


def test_heikin_ashi_entry_and_exit():
    # A clean HA downtrend (open == high, falling) builds long entries; the
    # following HA uptrend (open == low, rising) closes the position.
    open_, high, low, close = [], [], [], []
    price = 200.0
    for _ in range(10):  # strong downtrend
        top, bot = price, price - 5
        open_.append(top); high.append(top); low.append(bot); close.append(bot)
        price = bot
    for _ in range(10):  # strong uptrend
        bot, top = price, price + 5
        open_.append(bot); low.append(bot); high.append(top); close.append(top)
        price = top
    trades = STRATEGIES["heikin_ashi"](_mkdf(open_, high, low, close))
    assert len(trades) >= 1


def test_rsi_pattern_head_and_shoulders_entry_and_exit():
    # Left shoulder / head / right shoulder around a flat baseline drives the
    # nested pattern search to an entry, then the holding-period exit fires.
    base = 100.0
    close = [base] * 56
    close[25] = 101.0  # left shoulder
    close[33] = 103.0  # head (window max)
    close[38] = 101.0  # right shoulder
    high = [c + 0.05 for c in close]
    low = [c - 0.05 for c in close]
    trades = STRATEGIES["rsi_pattern"](_mkdf(close, high, low, close))
    assert len(trades) >= 1


def test_bb_pattern_double_bottom_entry_and_exit():
    # Bollinger double-bottom: a mid-band plateau tuned to equal the breakout
    # bar's upper band (satisfying the tight alpha tolerances), a rise for the
    # node-L check, a slight decline near the lower band, then a breakout above
    # the upper band, followed by a flat region that contracts and exits.
    from screener.indicators.numpy import _bb

    n, v = 185, 101.0
    close = np.full(n, 100.0)
    for t in range(90, 105):  # ramp into the plateau (node-L material)
        close[t] = 100.0 + (v - 100.0) * (t - 89) / 15.0
    close[105:141] = v
    for t in range(141, 160):  # slight decline near the lower band
        close[t] = v - 0.05
    close[160] = v + 0.5  # breakout above the upper band
    close[161:] = v  # flat contraction region for the exit
    _, _, upper = _bb(close, 20, 2.0)
    close[105:141] = upper[160]  # plateau == breakout's upper band
    trades = STRATEGIES["bb_pattern"](_mkdf(close, close, close, close))
    assert len(trades) == 1


def test_sar_short_series_returns_zeros():
    from screener.indicators.registry import get_indicator

    sar = get_indicator("sar")
    out = sar(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    assert out.tolist() == [0.0]
