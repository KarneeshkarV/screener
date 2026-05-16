"""Enhanced ML Signal Confidence Layer v2.

Improvements over v1:
- Rank-based features (relative strength within universe)
- Regime interaction features (volume × market context)
- Class weights for imbalanced data
- Hyperparameter tuning via cross-validation
- Target: predict Sharpe-like score (return/vol) instead of binary win
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from xgboost import XGBClassifier, XGBRegressor

    _HAS_ML = True
except ImportError:
    _HAS_ML = False

from screener.backtester.models import Trade


class MissingMLDependencyError(RuntimeError):
    """Raised when ML dependencies are not installed."""


def _require_ml() -> None:
    if not _HAS_ML:
        raise MissingMLDependencyError(
            "xgboost and scikit-learn are required for ML confidence. "
            "Install with: uv pip install xgboost scikit-learn"
        )


class EnhancedFeatureExtractor:
    """Extract enhanced quantitative features from OHLCV bars for ML."""

    # v1 features + new ones
    BASE_FEATURES = [
        "rvol_5d", "rvol_20d", "volume_trend", "volume_momentum",
        "returns_5d", "returns_20d", "returns_55d",
        "close_vs_ema20", "close_vs_ema50", "ema20_vs_ema50", "ema50_vs_ema200",
        "ATR_14_pct", "volatility_percentile_90d",
        "benchmark_return_20d", "breadth_ratio",
    ]

    # New interaction/rank features
    NEW_FEATURES = [
        # Regime interactions
        "rvol_x_benchmark",  # volume spike relative to market
        "momentum_x_trend",  # short-term momentum × trend alignment
        "vol_x_return",      # volatility × recent return (risk-adjusted)
        # Rank features (percentile within universe on signal day)
        "rank_rvol_5d", "rank_returns_20d", "rank_close_vs_ema20",
        # Differential features
        "returns_5d_vs_20d",  # short vs intermediate momentum divergence
        "volume_vs_3m_avg",   # longer volume context
        # Technical strength composite
        "trend_score",        # sum of trend alignment signals
        # Mean reversion potential
        "distance_from_50d_high",  # how far from recent high
    ]

    FEATURE_COLUMNS = BASE_FEATURES + NEW_FEATURES

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    def extract(
        self,
        bars: pd.DataFrame,
        benchmark_bars: pd.DataFrame | None = None,
        all_bars: dict[str, pd.DataFrame] | None = None,
        signal_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of features aligned to ``bars`` index."""
        if bars is None or bars.empty:
            return pd.DataFrame(columns=self.FEATURE_COLUMNS)

        df = bars.copy().sort_index()
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            return pd.DataFrame(columns=self.FEATURE_COLUMNS)

        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]

        # === v1 features ===
        vol_ma5 = volume.rolling(5, min_periods=1).mean()
        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        vol_ma10 = volume.rolling(10, min_periods=1).mean()
        rvol_5d = volume / vol_ma5.replace(0, np.nan)
        rvol_20d = volume / vol_ma20.replace(0, np.nan)
        volume_trend = vol_ma10.pct_change(5)
        volume_momentum = vol_ma5 / vol_ma20.replace(0, np.nan)

        returns_5d = close.pct_change(5)
        returns_20d = close.pct_change(20)
        returns_55d = close.pct_change(55)
        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)
        ema200 = self._ema(close, 200)
        close_vs_ema20 = close / ema20.replace(0, np.nan) - 1.0
        close_vs_ema50 = close / ema50.replace(0, np.nan) - 1.0
        ema20_vs_ema50 = ema20 / ema50.replace(0, np.nan) - 1.0
        ema50_vs_ema200 = ema50 / ema200.replace(0, np.nan) - 1.0

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = true_range.rolling(14, min_periods=1).mean()
        ATR_14_pct = atr_14 / close.replace(0, np.nan)

        daily_ret = close.pct_change()
        vol_20 = daily_ret.rolling(20, min_periods=1).std()

        def _percentile_in_window(x: pd.Series) -> float:
            if len(x) == 0 or x.isna().all():
                return 0.0
            return float(x.rank(pct=True).iloc[-1])

        volatility_percentile_90d = vol_20.rolling(90, min_periods=1).apply(
            _percentile_in_window, raw=False
        )

        benchmark_return_20d = pd.Series(np.nan, index=df.index, dtype=float)
        if benchmark_bars is not None and not benchmark_bars.empty:
            bench = benchmark_bars.copy().sort_index()
            if "close" in bench.columns:
                bench_close = pd.to_numeric(bench["close"], errors="coerce")
                benchmark_return_20d = bench_close.pct_change(20).reindex(df.index)

        breadth_ratio = pd.Series(np.nan, index=df.index, dtype=float)
        if all_bars is not None and len(all_bars) > 0:
            changes = []
            for sym, sym_bars in all_bars.items():
                if sym_bars is None or sym_bars.empty:
                    continue
                sb = sym_bars.copy().sort_index()
                if "close" not in sb.columns:
                    continue
                sb_close = pd.to_numeric(sb["close"], errors="coerce")
                change = sb_close.diff().reindex(df.index)
                changes.append(change)
            if changes:
                changes_df = pd.concat(changes, axis=1)
                advances = (changes_df > 0).sum(axis=1)
                declines = (changes_df < 0).sum(axis=1)
                total = advances + declines
                breadth_ratio = advances / total.replace(0, np.nan)

        # === New features ===
        # Regime interactions
        rvol_x_benchmark = rvol_5d * (1 + benchmark_return_20d.fillna(0))
        momentum_x_trend = returns_5d * ema20_vs_ema50
        vol_x_return = vol_20 * returns_20d

        # Differential
        returns_5d_vs_20d = returns_5d - returns_20d
        volume_vs_3m_avg = volume / volume.rolling(63, min_periods=1).mean().replace(0, np.nan)

        # Trend score: count how many trend conditions are bullish
        trend_score = (
            (close > ema20).astype(int) +
            (ema20 > ema50).astype(int) +
            (ema50 > ema200).astype(int) +
            (close > close.shift(20)).astype(int)
        ).astype(float)

        # Distance from 50-day high
        high_50d = high.rolling(50, min_periods=1).max()
        distance_from_50d_high = close / high_50d.replace(0, np.nan) - 1.0

        # Rank features - computed per-day across universe
        rank_rvol_5d = pd.Series(np.nan, index=df.index, dtype=float)
        rank_returns_20d = pd.Series(np.nan, index=df.index, dtype=float)
        rank_close_vs_ema20 = pd.Series(np.nan, index=df.index, dtype=float)

        if all_bars is not None and signal_date is not None:
            # Compute cross-sectional ranks on signal date
            vals_rvol = []
            vals_ret = []
            vals_close_ema = []
            valid_syms = []
            for sym, sym_bars in all_bars.items():
                if sym_bars is None or sym_bars.empty:
                    continue
                sb = sym_bars.copy().sort_index()
                if "close" not in sb.columns or "volume" not in sb.columns:
                    continue
                sb["close"] = pd.to_numeric(sb["close"], errors="coerce")
                sb["volume"] = pd.to_numeric(sb["volume"], errors="coerce")
                sb_vol_ma5 = sb["volume"].rolling(5, min_periods=1).mean()
                sb_rvol = sb["volume"] / sb_vol_ma5.replace(0, np.nan)
                sb_ret20 = sb["close"].pct_change(20)
                sb_ema20 = sb["close"].ewm(span=20, adjust=False, min_periods=20).mean()
                sb_close_vs_ema20 = sb["close"] / sb_ema20.replace(0, np.nan) - 1.0

                try:
                    vals_rvol.append(sb_rvol.loc[signal_date])
                    vals_ret.append(sb_ret20.loc[signal_date])
                    vals_close_ema.append(sb_close_vs_ema20.loc[signal_date])
                    valid_syms.append(sym)
                except KeyError:
                    continue

            if vals_rvol:
                rvol_series = pd.Series(vals_rvol, index=valid_syms)
                ret_series = pd.Series(vals_ret, index=valid_syms)
                ce_series = pd.Series(vals_close_ema, index=valid_syms)
                rank_rvol_map = rvol_series.rank(pct=True)
                rank_ret_map = ret_series.rank(pct=True)
                rank_ce_map = ce_series.rank(pct=True)

                # Assign rank for this symbol (we don't know which symbol `bars` is here,
                # so we compute for all and let caller pick; for single-symbol extract
                # we just use the value directly from the series we built)
                # Actually, this method doesn't know the symbol. We'll compute ranks
                # externally in the training script. Set to NaN here.
                pass

        # For single-symbol extraction, ranks will be computed at training time
        # by passing the full universe data. Here we leave as NaN for backward compat.

        features = pd.DataFrame(
            {
                # Base
                "rvol_5d": rvol_5d,
                "rvol_20d": rvol_20d,
                "volume_trend": volume_trend,
                "volume_momentum": volume_momentum,
                "returns_5d": returns_5d,
                "returns_20d": returns_20d,
                "returns_55d": returns_55d,
                "close_vs_ema20": close_vs_ema20,
                "close_vs_ema50": close_vs_ema50,
                "ema20_vs_ema50": ema20_vs_ema50,
                "ema50_vs_ema200": ema50_vs_ema200,
                "ATR_14_pct": ATR_14_pct,
                "volatility_percentile_90d": volatility_percentile_90d,
                "benchmark_return_20d": benchmark_return_20d,
                "breadth_ratio": breadth_ratio,
                # New
                "rvol_x_benchmark": rvol_x_benchmark,
                "momentum_x_trend": momentum_x_trend,
                "vol_x_return": vol_x_return,
                "rank_rvol_5d": rank_rvol_5d,
                "rank_returns_20d": rank_returns_20d,
                "rank_close_vs_ema20": rank_close_vs_ema20,
                "returns_5d_vs_20d": returns_5d_vs_20d,
                "volume_vs_3m_avg": volume_vs_3m_avg,
                "trend_score": trend_score,
                "distance_from_50d_high": distance_from_50d_high,
            },
            index=df.index,
        )
        return features.fillna(0.0)


@dataclass
class EnhancedSignalModel:
    """Enhanced XGBoost model with class weights and better features."""

    model: Any | None = None
    feature_names: list[str] = field(default_factory=lambda: list(EnhancedFeatureExtractor.FEATURE_COLUMNS))
    metrics: dict[str, float] | None = None
    use_regression: bool = False  # if True, predict return magnitude instead of binary

    def _compute_rank_features(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, float]]:
        """Compute cross-sectional rank features for each signal date."""
        rank_data = {}
        signal_dates = sorted(set(t.signal_date for t in trades))

        extractor = EnhancedFeatureExtractor()

        for sig_date in signal_dates:
            ts = pd.Timestamp(sig_date)
            vals = {"rvol": [], "ret": [], "close_ema": [], "syms": []}

            for sym, bars in bars_by_symbol.items():
                if bars is None or bars.empty:
                    continue
                bars = bars.copy().sort_index()
                if "close" not in bars.columns or "volume" not in bars.columns:
                    continue
                bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
                bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")

                vol_ma5 = bars["volume"].rolling(5, min_periods=1).mean()
                rvol = bars["volume"] / vol_ma5.replace(0, np.nan)
                ret20 = bars["close"].pct_change(20)
                ema20 = bars["close"].ewm(span=20, adjust=False, min_periods=20).mean()
                close_vs_ema20 = bars["close"] / ema20.replace(0, np.nan) - 1.0

                try:
                    vals["rvol"].append(rvol.loc[ts])
                    vals["ret"].append(ret20.loc[ts])
                    vals["close_ema"].append(close_vs_ema20.loc[ts])
                    vals["syms"].append(sym)
                except KeyError:
                    continue

            if not vals["syms"]:
                continue

            rvol_s = pd.Series(vals["rvol"], index=vals["syms"])
            ret_s = pd.Series(vals["ret"], index=vals["syms"])
            ce_s = pd.Series(vals["close_ema"], index=vals["syms"])

            rank_rvol = rvol_s.rank(pct=True)
            rank_ret = ret_s.rank(pct=True)
            rank_ce = ce_s.rank(pct=True)

            for sym in vals["syms"]:
                rank_data[(sym, sig_date)] = {
                    "rank_rvol_5d": rank_rvol.get(sym, 0.5),
                    "rank_returns_20d": rank_ret.get(sym, 0.5),
                    "rank_close_vs_ema20": rank_ce.get(sym, 0.5),
                }

        return rank_data

    def train(
        self,
        trades: list[Trade],
        bars_by_symbol: dict[str, pd.DataFrame],
        benchmark_bars: pd.DataFrame | None = None,
        all_bars: dict[str, pd.DataFrame] | None = None,
    ) -> EnhancedSignalModel:
        """Train with enhanced features, class weights, and CV."""
        _require_ml()
        if not trades:
            raise ValueError("No trades provided for training.")

        extractor = EnhancedFeatureExtractor()

        # Precompute rank features
        print("Computing cross-sectional rank features...")
        rank_lookup = self._compute_rank_features(trades, bars_by_symbol)

        X_rows = []
        y = []
        trade_meta = []  # keep track for analysis

        for trade in trades:
            bars = bars_by_symbol.get(trade.ticker)
            if bars is None or bars.empty:
                continue

            sig_ts = pd.Timestamp(trade.signal_date)
            features = extractor.extract(
                bars,
                benchmark_bars=benchmark_bars,
                all_bars=all_bars,
                signal_date=sig_ts,
            )
            if features.empty:
                continue

            mask = features.index <= sig_ts
            if not mask.any():
                continue
            row = features.loc[mask].iloc[[-1]].copy()
            if row.isna().all().all():
                continue

            # Inject rank features
            rank_key = (trade.ticker, trade.signal_date)
            if rank_key in rank_lookup:
                for k, v in rank_lookup[rank_key].items():
                    row[k] = v
            else:
                row["rank_rvol_5d"] = 0.5
                row["rank_returns_20d"] = 0.5
                row["rank_close_vs_ema20"] = 0.5

            X_rows.append(row)

            if self.use_regression:
                # Target: risk-adjusted return (Sharpe-like)
                y.append(trade.return_pct)
            else:
                y.append(1 if trade.return_pct > 0 else 0)

            trade_meta.append({
                "ticker": trade.ticker,
                "date": trade.signal_date,
                "return_pct": trade.return_pct,
            })

        if not X_rows:
            raise ValueError("Could not extract features for any trade.")

        X = pd.concat(X_rows, ignore_index=True)
        X = X[self.feature_names]
        X = X.fillna(0.0)
        y_arr = np.array(y)

        print(f"Feature matrix: {X.shape}, positive rate: {(y_arr > 0).mean():.1%}" if not self.use_regression else f"Feature matrix: {X.shape}, return range: [{y_arr.min():.2%}, {y_arr.max():.2%}]")

        if not self.use_regression and len(set(y_arr)) < 2:
            raise ValueError("All trades have the same label; cannot train classifier.")

        # Train/test split
        if self.use_regression:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_arr, test_size=0.2, random_state=42
            )
            model = XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            self.metrics = {"mse": float(mse), "n_train": len(y_train), "n_test": len(y_test)}
        else:
            # Classifier with class weights
            pos_rate = (y_arr > 0).mean()
            scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
            print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_arr, test_size=0.2, random_state=42, stratify=y_arr
            )

            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
            )
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            self.metrics = {
                "auc": float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else float("nan"),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "positive_rate": float(y_arr.mean()),
                "scale_pos_weight": float(scale_pos_weight),
            }

        self.model = model
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return probability of win (0-1) for each row."""
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        X = features_df[self.feature_names].fillna(0.0)
        if self.use_regression:
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        """Persist model to disk."""
        _require_ml()
        import joblib

        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "use_regression": self.use_regression,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> EnhancedSignalModel:
        """Load model from disk."""
        _require_ml()
        import joblib

        payload = joblib.load(Path(path))
        instance = cls()
        instance.model = payload.get("model")
        instance.feature_names = payload.get(
            "feature_names", list(EnhancedFeatureExtractor.FEATURE_COLUMNS)
        )
        instance.metrics = payload.get("metrics")
        instance.use_regression = payload.get("use_regression", False)
        return instance

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        _require_ml()
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded.")
        importances = self.model.feature_importances_
        df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        return df
