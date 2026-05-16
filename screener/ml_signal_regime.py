"""Regime-aware ML Signal Confidence.

Trains separate models for different market regimes (bull, bear, chop)
and selects the appropriate model based on current regime detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    _HAS_ML = True
except ImportError:
    _HAS_ML = False

from screener.backtester.models import Trade
from screener.ml_signal_v3 import SimpleFeatureExtractor, SimpleSignalModel
from screener.regime import RegimeDetector, TrendRegime


def _require_ml() -> None:
    if not _HAS_ML:
        raise RuntimeError("xgboost and scikit-learn required")


@dataclass
class RegimeAwareModel:
    """Collection of regime-specific models + a regime detector."""

    bull_model: SimpleSignalModel | None = None
    bear_model: SimpleSignalModel | None = None
    chop_model: SimpleSignalModel | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def train(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame,
    ) -> RegimeAwareModel:
        """Train separate models for each regime using signal-date regime."""
        _require_ml()

        # Classify regime for each trade's signal date
        bull_trades: list[Trade] = []
        bear_trades: list[Trade] = []
        chop_trades: list[Trade] = []

        print(f"Classifying {len(trades)} trades by regime...")
        for trade in trades:
            sig_ts = pd.Timestamp(trade.signal_date)
            bench_sub = benchmark_bars[benchmark_bars.index <= sig_ts]
            if len(bench_sub) < 30:
                continue
            trend = RegimeDetector.trend_regime(
                pd.to_numeric(bench_sub["close"], errors="coerce").dropna(),
                fast=50, slow=200,
            )
            if trend == "UPTREND":
                bull_trades.append(trade)
            elif trend == "DOWNTREND":
                bear_trades.append(trade)
            else:
                chop_trades.append(trade)

        print(f"  Bull (UPTREND): {len(bull_trades)} trades")
        print(f"  Bear (DOWNTREND): {len(bear_trades)} trades")
        print(f"  Chop: {len(chop_trades)} trades")

        self.metrics["bull_count"] = len(bull_trades)
        self.metrics["bear_count"] = len(bear_trades)
        self.metrics["chop_count"] = len(chop_trades)

        min_trades = 30

        if len(bull_trades) >= min_trades:
            print("\nTraining bull model...")
            self.bull_model = SimpleSignalModel()
            self.bull_model.train(bull_trades, bars_by_symbol, benchmark_bars)
            self.metrics["bull_auc"] = self.bull_model.metrics.get("auc")
        else:
            print(f"  Skipping bull model (need {min_trades}, got {len(bull_trades)})")

        if len(bear_trades) >= min_trades:
            print("\nTraining bear model...")
            self.bear_model = SimpleSignalModel()
            self.bear_model.train(bear_trades, bars_by_symbol, benchmark_bars)
            self.metrics["bear_auc"] = self.bear_model.metrics.get("auc")
        else:
            print(f"  Skipping bear model (need {min_trades}, got {len(bear_trades)})")

        if len(chop_trades) >= min_trades:
            print("\nTraining chop model...")
            self.chop_model = SimpleSignalModel()
            self.chop_model.train(chop_trades, bars_by_symbol, benchmark_bars)
            self.metrics["chop_auc"] = self.chop_model.metrics.get("auc")
        else:
            print(f"  Skipping chop model (need {min_trades}, got {len(chop_trades)})")

        return self

    def predict(
        self,
        features_df: pd.DataFrame,
        regime: TrendRegime,
    ) -> np.ndarray:
        """Predict using the model matching the current regime."""
        _require_ml()
        if regime == "UPTREND" and self.bull_model is not None:
            return self.bull_model.predict(features_df)
        elif regime == "DOWNTREND" and self.bear_model is not None:
            return self.bear_model.predict(features_df)
        elif self.chop_model is not None:
            return self.chop_model.predict(features_df)
        # Fallback: any available model
        for m in [self.bull_model, self.bear_model, self.chop_model]:
            if m is not None:
                return m.predict(features_df)
        raise RuntimeError("No models trained")

    def save(self, path: str | Path) -> None:
        _require_ml()
        import joblib
        payload = {
            "bull_model": self.bull_model,
            "bear_model": self.bear_model,
            "chop_model": self.chop_model,
            "metrics": self.metrics,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> RegimeAwareModel:
        _require_ml()
        import joblib
        payload = joblib.load(Path(path))
        instance = cls()
        instance.bull_model = payload.get("bull_model")
        instance.bear_model = payload.get("bear_model")
        instance.chop_model = payload.get("chop_model")
        instance.metrics = payload.get("metrics", {})
        return instance
