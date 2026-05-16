"""Filter scanner signals using the v5 ML confidence model.

Usage inside scanner/backtest pipeline:
    from screener.ml_signal_filter import MLSignalFilter
    filter = MLSignalFilter.from_path("model_v5_us_production.pkl")
    scored = filter.score_signals(signals_df, bars_by_symbol)
    top_signals = scored[scored["ml_confidence"] >= 0.6]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screener.ml_signal_v5 import V5FeatureExtractor, V5SignalModel


class MLSignalFilter:
    """Wraps a v5 model to score scanner signals."""

    def __init__(self, model: V5SignalModel) -> None:
        self.model = model
        self.extractor = V5FeatureExtractor()

    @classmethod
    def from_path(cls, path: str | Path) -> MLSignalFilter:
        model = V5SignalModel.load(path)
        return cls(model)

    def score_signals(
        self,
        signals: pd.DataFrame,
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Add `expected_return` and `ml_confidence` columns to a signals DataFrame.

        ``signals`` must contain at least a ``ticker`` column and a
        ``signal_date`` (or index-level date).
        """
        df = signals.copy()
        if "signal_date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["signal_date"] = df.index
            else:
                raise ValueError("signals must have 'signal_date' column or DatetimeIndex")

        expected_returns = []
        confidences = []

        for _, row in df.iterrows():
            ticker = row.get("ticker") or row.get("symbol") or row.get("name")
            sig_date = pd.Timestamp(row["signal_date"])
            bars = bars_by_symbol.get(ticker)
            if bars is None or bars.empty:
                expected_returns.append(0.0)
                confidences.append(0.5)
                continue

            features = self.extractor.extract(bars, benchmark_bars=benchmark_bars)
            mask = features.index <= sig_date
            if not mask.any():
                expected_returns.append(0.0)
                confidences.append(0.5)
                continue

            row_features = features.loc[mask].iloc[[-1]]
            pred = self.model.predict(row_features)[0]
            conf = self.model.predict_confidence(row_features)[0]
            expected_returns.append(pred)
            confidences.append(conf)

        df["expected_return"] = expected_returns
        df["ml_confidence"] = confidences
        return df

    def filter_top_k(
        self,
        signals: pd.DataFrame,
        bars_by_symbol: dict[str, pd.DataFrame],
        k: float = 0.2,
        min_confidence: float = 0.0,
        benchmark_bars: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return the top K fraction of signals by expected return."""
        scored = self.score_signals(signals, bars_by_symbol, benchmark_bars=benchmark_bars)
        scored = scored[scored["ml_confidence"] >= min_confidence]
        if scored.empty:
            return scored
        n = max(1, int(len(scored) * k))
        return scored.nlargest(n, "expected_return")
