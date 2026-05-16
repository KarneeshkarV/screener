"""Tests for market regime detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.regime import RegimeDetector, Regime, stress_position_multiplier


def _trending_series(n: int = 200) -> pd.Series:
    """Strongly trending prices (Hurst > 0.5)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def _mr_series(n: int = 200) -> pd.Series:
    """Mean-reverting prices (oscillating sine wave)."""
    t = np.linspace(0, 10 * np.pi, n)
    prices = 100 + 10 * np.sin(t)
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_hurst_trending() -> None:
    prices = _trending_series(300)
    h = RegimeDetector.hurst_exponent(prices, max_lag=80)
    assert h > 0.45, f"Expected trending Hurst > 0.45, got {h}"


def test_hurst_mean_reverting() -> None:
    prices = _mr_series(300)
    h = RegimeDetector.hurst_exponent(prices, max_lag=80)
    assert h < 0.5, f"Expected mean-reverting Hurst < 0.5, got {h}"


def test_hurst_too_short() -> None:
    prices = pd.Series([100, 101, 102])
    h = RegimeDetector.hurst_exponent(prices)
    assert h == 0.5


def test_volatility_regime() -> None:
    rng = np.random.default_rng(42)
    low_vol = rng.normal(0, 0.005, 100)
    high_vol = rng.normal(0, 0.05, 50)
    returns = pd.Series(np.concatenate([low_vol, high_vol]))
    regime = RegimeDetector.volatility_regime(returns, window=10, lookback=100)
    # Last 50 bars are high vol, so regime should be HIGH or EXTREME
    assert regime in ("HIGH", "EXTREME"), f"Expected HIGH/EXTREME, got {regime}"


def test_volatility_regime_short_history() -> None:
    returns = pd.Series([0.01, -0.01])
    assert RegimeDetector.volatility_regime(returns) == "UNKNOWN"


def test_trend_regime_uptrend() -> None:
    prices = pd.Series(np.linspace(100, 200, 250))
    idx = pd.date_range("2024-01-01", periods=250, freq="B")
    prices.index = idx
    regime = RegimeDetector.trend_regime(prices)
    assert regime == "UPTREND"


def test_trend_regime_downtrend() -> None:
    prices = pd.Series(np.linspace(200, 100, 250))
    idx = pd.date_range("2024-01-01", periods=250, freq="B")
    prices.index = idx
    regime = RegimeDetector.trend_regime(prices)
    assert regime == "DOWNTREND"


def test_trend_regime_chop() -> None:
    prices = pd.Series(100 + np.sin(np.linspace(0, 4 * np.pi, 250)) * 5)
    idx = pd.date_range("2024-01-01", periods=250, freq="B")
    prices.index = idx
    regime = RegimeDetector.trend_regime(prices)
    assert regime == "CHOP"


def test_classify_returns_regime() -> None:
    prices = _trending_series(200)
    bars = pd.DataFrame({"close": prices})
    regime = RegimeDetector.classify(bars, bars)
    assert isinstance(regime, Regime)
    assert regime.trend_regime == "UPTREND"
    assert regime.is_tradeable is True


def test_classify_empty() -> None:
    regime = RegimeDetector.classify(pd.DataFrame(), pd.DataFrame())
    assert regime.is_tradeable is False


def test_stress_position_multiplier() -> None:
    assert stress_position_multiplier(0.0) == 1.0
    assert stress_position_multiplier(0.5) == 0.75
    assert stress_position_multiplier(0.7) == 0.5
    assert stress_position_multiplier(0.9) == 0.25
    assert stress_position_multiplier(1.0) == 0.25


def test_classify_series() -> None:
    prices = _trending_series(100)
    bars = pd.DataFrame({"close": prices})
    history = RegimeDetector.classify_series(bars, bars)
    assert isinstance(history, pd.DataFrame)
    assert "hurst" in history.columns
    assert "is_tradeable" in history.columns
    assert len(history) > 0
