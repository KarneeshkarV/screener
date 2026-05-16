"""Tests for the ML Signal Confidence Layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.ml_signal import BreakoutFeatureExtractor


def _make_bars(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_feature_extractor_columns() -> None:
    bars = _make_bars(200)
    extractor = BreakoutFeatureExtractor()
    features = extractor.extract(bars)
    assert list(features.columns) == BreakoutFeatureExtractor.FEATURE_COLUMNS
    assert len(features) == len(bars)


def test_feature_extractor_with_benchmark() -> None:
    bars = _make_bars(200)
    bench = _make_bars(200, seed=43)
    extractor = BreakoutFeatureExtractor()
    features = extractor.extract(bars, benchmark_bars=bench)
    assert "benchmark_return_20d" in features.columns
    # benchmark_return_20d should have some non-zero values after bar 20
    assert features["benchmark_return_20d"].iloc[25] != 0


def test_feature_extractor_empty_bars() -> None:
    extractor = BreakoutFeatureExtractor()
    features = extractor.extract(pd.DataFrame())
    assert features.empty


def test_feature_extractor_missing_columns() -> None:
    extractor = BreakoutFeatureExtractor()
    bad = pd.DataFrame({"close": [1, 2, 3]})
    features = extractor.extract(bad)
    assert features.empty


def test_ensemble_predict() -> None:
    from screener.ml_signal import EnsembleConfidence, SignalConfidenceModel

    # Create two dummy models with known predictions
    m1 = SignalConfidenceModel()
    m1.model = _DummyModel([0.3, 0.7])
    m2 = SignalConfidenceModel()
    m2.model = _DummyModel([0.5, 0.5])

    ensemble = EnsembleConfidence([m1, m2])
    df = pd.DataFrame({k: [0.0] for k in BreakoutFeatureExtractor.FEATURE_COLUMNS})
    probs = ensemble.predict(df)
    assert len(probs) == 1
    assert pytest.approx(probs[0], 0.01) == 0.4


class _DummyModel:
    """Fake XGBoost-like model for unit tests."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = np.array(probs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        out = np.zeros((n, 2))
        for i in range(n):
            p = self._probs[i % len(self._probs)]
            out[i] = [1 - p, p]
        return out

    @property
    def feature_importances_(self):
        return np.ones(len(BreakoutFeatureExtractor.FEATURE_COLUMNS)) / len(
            BreakoutFeatureExtractor.FEATURE_COLUMNS
        )
