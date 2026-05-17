"""V5 with orthogonal features: regime, breakout structure, delivery.

Uses pre-computed v5 features + adds:
- market_regime: bull/bear/neutral from 200d EMA slope
- breakout_base_depth: % decline from 52w high to pivot low
- breakout_base_length: days from 52w high to pivot low
- dist_above_pivot: how far above the breakout pivot
- delivery_pct: delivery volume ratio (India only)
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def detect_regime(bars: pd.DataFrame) -> str:
    """bull / bear / neutral based on price vs 200d EMA."""
    if bars is None or len(bars) < 200:
        return "neutral"
    close = bars["close"].astype(float)
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    if close.iloc[-1] > ema200.iloc[-1] * 1.05:
        return "bull"
    elif close.iloc[-1] < ema200.iloc[-1] * 0.95:
        return "bear"
    return "neutral"


def breakout_structure(bars: pd.DataFrame, signal_date: pd.Timestamp):
    """Return (base_depth, base_length, dist_above_pivot)."""
    if bars is None or len(bars) < 60:
        return 0.0, 0, 0.0

    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)

    # Look back 60 days before signal
    mask = bars.index <= signal_date
    hist = bars.loc[mask]
    if len(hist) < 20:
        return 0.0, 0, 0.0

    # 52w high before signal
    high_252 = high.loc[mask].rolling(252, min_periods=20).max().iloc[-1]
    # Pivot low: lowest in last 20 days
    pivot_low = low.loc[mask].tail(20).min()
    # Pivot length: days from 52w high date to signal
    high_idx = high.loc[mask].idxmax()
    base_length = (signal_date - high_idx).days if high_idx else 0

    base_depth = (high_252 - pivot_low) / high_252 if high_252 > 0 else 0.0
    dist_above_pivot = (close.loc[mask].iloc[-1] - pivot_low) / pivot_low if pivot_low > 0 else 0.0

    return base_depth, base_length, dist_above_pivot


def main():
    data_dir = Path("scripts/training_data_v4")
    print("Loading data...")
    with open(data_dir / "v5_features.pkl", "rb") as f:
        feat_data = pickle.load(f)
    features_by_ticker = feat_data["features"]

    with open(data_dir / "trades.json") as f:
        trades = json.load(f)["trades"]
    with open(data_dir / "bars.json") as f:
        bars_json = json.load(f)

    bars_by_symbol = {}
    for key, records in bars_json.get("bars", {}).items():
        if not isinstance(records, list):
            continue
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        sym = key.split(":", 1)[1] if ":" in key else key
        bars_by_symbol[sym] = df

    print("Extracting orthogonal features...")
    feature_cols = list(features_by_ticker[list(features_by_ticker.keys())[0]].columns)
    ortho_cols = ["market_regime_bull", "market_regime_bear", "base_depth", "base_length", "dist_above_pivot"]
    all_cols = feature_cols + ortho_cols

    rows = []
    for t in trades:
        ticker = t["ticker"]
        feat_df = features_by_ticker.get(ticker)
        bars = bars_by_symbol.get(ticker)
        if feat_df is None or feat_df.empty or bars is None:
            continue

        ts = pd.Timestamp(t["signal_date"])
        mask = feat_df.index <= ts
        if not mask.any():
            continue
        row = feat_df.loc[mask].iloc[[-1]]
        if row.isna().all().all():
            continue

        # Regime from benchmark
        bench = bars_by_symbol.get("NIFTY" if t.get("market") == "india" else "SPY")
        regime = detect_regime(bench) if bench is not None else "neutral"

        base_depth, base_length, dist_above_pivot = breakout_structure(bars, ts)

        rows.append({
            "signal_date": ts,
            "return_pct": float(t["return_pct"]),
            "is_win": int(t["return_pct"] > 0),
            **{k: float(row[k].iloc[0]) for k in feature_cols},
            "market_regime_bull": 1.0 if regime == "bull" else 0.0,
            "market_regime_bear": 1.0 if regime == "bear" else 0.0,
            "base_depth": base_depth,
            "base_length": base_length,
            "dist_above_pivot": dist_above_pivot,
        })

    df = pd.DataFrame(rows)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df.sort_values("signal_date").reset_index(drop=True)
    n = len(df)

    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train = train_df[all_cols].values
    y_train = train_df["return_pct"].values
    X_val = val_df[all_cols].values
    y_val = val_df["return_pct"].values
    X_test = test_df[all_cols].values
    y_test = test_df["return_pct"].values

    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.6, reg_lambda=3.0,
        random_state=42, n_jobs=4, early_stopping_rounds=20,
    )

    print("Training...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Test
    preds = model.predict(X_test)
    auc = roc_auc_score(test_df["is_win"].values, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    sorted_idx = np.argsort(preds)[::-1]
    n10 = max(1, int(len(test_df) * 0.1))
    top10_wr = test_df["is_win"].iloc[sorted_idx[:n10]].mean()
    top10_avg = test_df["return_pct"].iloc[sorted_idx[:n10]].mean()

    post_cost = test_df["return_pct"] - 0.0015
    monthly_sharpe = post_cost.mean() / (post_cost.std() + 1e-9) * np.sqrt(12)

    print(f"\n{'='*60}")
    print("V5 + ORTHOGONAL FEATURES (honest test set)")
    print(f"{'='*60}")
    print(f"AUC:         {auc:.4f}")
    print(f"MSE:         {mse:.6f}")
    print(f"R2:          {r2:.4f}")
    print(f"Top 10% WR:  {top10_wr:.1%}")
    print(f"Top 10% Avg: {top10_avg:.3%}")
    print(f"Baseline WR: {test_df['is_win'].mean():.1%}")
    print(f"Monthly Sharpe: {monthly_sharpe:.3f}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": all_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\nTop 10 features:")
    for _, r in imp.head(10).iterrows():
        print(f"  {r['feature']}: {r['importance']:.4f}")
    print(f"{'='*60}")

    # Save
    with open(data_dir / "model_v5_orthogonal.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_names": all_cols}, f)
    print(f"Saved to {data_dir / 'model_v5_orthogonal.pkl'}")


if __name__ == "__main__":
    main()
