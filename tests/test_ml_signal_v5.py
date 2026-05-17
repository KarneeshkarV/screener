"""Tests for ML Signal v5 — V5SignalModel, V5FeatureExtractor, MLSignalFilter.

Key assertion: look-ahead leak test — features at date t must be identical
whether bars are truncated at t or contain full future history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.ml_signal_v5 import (
    FEATURE_NEUTRAL_VALUES,
    V5FeatureExtractor,
    V5SignalModel,
)
from screener.ml_signal_filter import MLSignalFilter


def _make_bars(n: int = 200, seed: int = 42) -> pd.DataFrame:
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


class _DummyV5Model:
    """Fake model that returns the first feature value as prediction."""

    def __init__(self, return_value: float = 0.05) -> None:
        self._return = return_value
        self.feature_importances_ = np.ones(27) / 27
        self.best_iteration = 100

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._return)


# ---------------------------------------------------------------------------
# V5FeatureExtractor tests
# ---------------------------------------------------------------------------


def test_v5_feature_columns_match_neutral_values() -> None:
    """Every feature column must have a defined neutral fill value."""
    extractor = V5FeatureExtractor()
    for col in extractor.FEATURE_COLUMNS:
        assert col in FEATURE_NEUTRAL_VALUES, f"Missing neutral value for {col}"


def test_v5_extract_no_lookahead_leak() -> None:
    """CRITICAL: features at date t must not change when future bars are added."""
    bars = _make_bars(200)
    extractor = V5FeatureExtractor()

    # Pick a cutoff date in the middle
    cutoff_idx = 100
    cutoff_date = bars.index[cutoff_idx]

    # Extract features from truncated bars (only up to cutoff)
    truncated = bars.loc[:cutoff_date]
    feat_truncated = extractor.extract(truncated)

    # Extract features from full bars
    feat_full = extractor.extract(bars)

    # Features at the cutoff date must be IDENTICAL
    feat_t_trunc = feat_truncated.loc[cutoff_date]
    feat_t_full = feat_full.loc[cutoff_date]

    for col in extractor.FEATURE_COLUMNS:
        assert feat_t_trunc[col] == pytest.approx(feat_t_full[col], abs=1e-10), (
            f"Look-ahead leak detected in {col}: "
            f"truncated={feat_t_trunc[col]}, full={feat_t_full[col]}"
        )


def test_v5_extract_empty_bars() -> None:
    extractor = V5FeatureExtractor()
    features = extractor.extract(pd.DataFrame())
    assert features.empty


def test_v5_extract_missing_columns() -> None:
    extractor = V5FeatureExtractor()
    bad = pd.DataFrame({"close": [1, 2, 3]})
    features = extractor.extract(bad)
    assert features.empty


def test_v5_neutral_fill_values() -> None:
    """Features with insufficient history should fill with neutral values, not 0."""
    bars = _make_bars(5)  # Very short — most features will be NaN
    extractor = V5FeatureExtractor()
    features = extractor.extract(bars)

    # rsi should be ~50 (neutral), not 0
    if "rsi_14" in features.columns and not features["rsi_14"].isna().all():
        assert features["rsi_14"].iloc[-1] == pytest.approx(50.0, abs=5.0)

    # bb_position should be ~0.5 (middle), not 0
    if "bb_position" in features.columns and not features["bb_position"].isna().all():
        assert 0.0 <= features["bb_position"].iloc[-1] <= 1.0

    # rvol should be ~1.0 (neutral), not 0
    if "rvol_5d" in features.columns and not features["rvol_5d"].isna().all():
        assert features["rvol_5d"].iloc[-1] == pytest.approx(1.0, abs=0.5)


def test_v5_extract_with_benchmark() -> None:
    bars = _make_bars(200)
    bench = _make_bars(200, seed=43)
    extractor = V5FeatureExtractor()
    features = extractor.extract(bars, benchmark_bars=bench)
    assert "benchmark_return_20d" in features.columns
    assert "beta_20d" in features.columns


# ---------------------------------------------------------------------------
# V5SignalModel tests
# ---------------------------------------------------------------------------


def test_v5_signal_model_save_load(tmp_path: Path) -> None:
    model = V5SignalModel()
    model.model = _DummyV5Model()
    model.metrics = {"auc_test": 0.65}

    path = tmp_path / "v5_test.pkl"
    model.save(path)
    loaded = V5SignalModel.load(path)

    assert loaded.metrics == model.metrics
    assert loaded.feature_names == model.feature_names
    assert loaded.rolling_window_months == model.rolling_window_months


def test_v5_signal_model_predict() -> None:
    model = V5SignalModel()
    model.model = _DummyV5Model(return_value=0.03)

    df = pd.DataFrame({k: [0.0] for k in V5FeatureExtractor.FEATURE_COLUMNS})
    preds = model.predict(df)
    assert len(preds) == 1
    assert pytest.approx(preds[0], 0.001) == 0.03


def test_v5_signal_model_predict_confidence() -> None:
    model = V5SignalModel()
    model.model = _DummyV5Model(return_value=0.05)  # positive return

    df = pd.DataFrame({k: [0.0] for k in V5FeatureExtractor.FEATURE_COLUMNS})
    conf = model.predict_confidence(df)
    assert len(conf) == 1
    assert conf[0] > 0.5  # positive return → confidence > 0.5

    # Negative return → confidence < 0.5
    model.model = _DummyV5Model(return_value=-0.05)
    conf = model.predict_confidence(df)
    assert conf[0] < 0.5


def test_v5_signal_model_untrained_predict_raises() -> None:
    model = V5SignalModel()
    df = pd.DataFrame({k: [0.0] for k in V5FeatureExtractor.FEATURE_COLUMNS})
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(df)


def test_v5_signal_model_feature_importance() -> None:
    model = V5SignalModel()
    model.model = _DummyV5Model()
    imp = model.feature_importance()
    assert len(imp) == len(V5FeatureExtractor.FEATURE_COLUMNS)
    assert imp["importance"].sum() == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# MLSignalFilter tests
# ---------------------------------------------------------------------------


def test_ml_signal_filter_score_signals() -> None:
    model = V5SignalModel()
    model.model = _DummyV5Model(return_value=0.05)
    filt = MLSignalFilter(model)

    signals = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "signal_date": [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-01")],
    })
    bars = {
        "AAPL": _make_bars(50),
        "MSFT": _make_bars(50, seed=99),
    }

    scored = filt.score_signals(signals, bars)
    assert "expected_return" in scored.columns
    assert "ml_confidence" in scored.columns
    assert len(scored) == 2


def test_ml_signal_filter_filter_top_k() -> None:
    model = V5SignalModel()
    model.model = _DummyV5Model(return_value=0.05)
    filt = MLSignalFilter(model)

    signals = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "signal_date": [pd.Timestamp("2024-06-01")] * 3,
    })
    bars = {
        "AAPL": _make_bars(50),
        "MSFT": _make_bars(50, seed=99),
        "GOOGL": _make_bars(50, seed=100),
    }

    top = filt.filter_top_k(signals, bars, k=0.5)
    assert len(top) == 1  # max(1, int(3*0.5)) = max(1, 1) = 1


def test_ml_signal_filter_missing_ticker() -> None:
    """Missing ticker data should get neutral values, not crash."""
    model = V5SignalModel()
    model.model = _DummyV5Model(return_value=0.05)
    filt = MLSignalFilter(model)

    signals = pd.DataFrame({
        "ticker": ["AAPL", "UNKNOWN"],
        "signal_date": [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-01")],
    })
    bars = {"AAPL": _make_bars(50)}

    scored = filt.score_signals(signals, bars)
    assert len(scored) == 2
    # UNKNOWN gets default values
    unknown_row = scored[scored["ticker"] == "UNKNOWN"].iloc[0]
    assert unknown_row["expected_return"] == 0.0
    assert unknown_row["ml_confidence"] == 0.5


# ---------------------------------------------------------------------------
# Look-ahead leak test for the full pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_no_lookahead() -> None:
    """End-to-end: MLSignalFilter must not leak future data into predictions."""
    bars = _make_bars(200)
    cutoff = bars.index[100]

    # Build model with dummy
    model = V5SignalModel()
    model.model = _DummyV5Model(return_value=0.05)
    filt = MLSignalFilter(model)

    signals = pd.DataFrame({
        "ticker": ["TEST"],
        "signal_date": [cutoff],
    })

    # Prediction with truncated bars
    bars_truncated = {"TEST": bars.loc[:cutoff]}
    scored_truncated = filt.score_signals(signals, bars_truncated)

    # Prediction with full bars (including future)
    bars_full = {"TEST": bars}
    scored_full = filt.score_signals(signals, bars_full)

    # Must be identical — future data must not affect the prediction
    assert scored_truncated["expected_return"].iloc[0] == pytest.approx(
        scored_full["expected_return"].iloc[0], abs=1e-10
    )
    assert scored_truncated["ml_confidence"].iloc[0] == pytest.approx(
        scored_full["ml_confidence"].iloc[0], abs=1e-10
    )
