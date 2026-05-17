"""Simple sklearn MLP neural net for v5 signal confidence.
Fast CPU-only baseline for NN approach.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_data():
    data_dir = Path(__file__).parent / "training_data_v4"

    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)
    with open(data_dir / "v5_features.pkl", "rb") as f:
        cache = pickle.load(f)

    trades = trades_data["trades"]
    features_cache = cache["features"]

    feature_names = [
        "rvol_5d", "rvol_20d", "volume_trend_10d",
        "returns_5d", "returns_20d", "returns_60d",
        "momentum_5d_vs_20d",
        "close_vs_ema20", "close_vs_ema50", "ema20_vs_ema50", "ema50_vs_ema200",
        "ATR_14_pct", "volatility_percentile_90d", "bb_position",
        "rsi_14", "macd_hist", "adx_14",
        "dist_from_52w_high", "dist_from_52w_low",
        "benchmark_return_20d", "beta_20d",
        "max_dd_20d", "range_pct", "gap_pct",
        "consecutive_up_days", "volume_price_corr_20d",
        "sharpe_20d",
    ]

    X_rows, y, markets = [], [], []
    for t in trades:
        feat = features_cache.get(t["ticker"])
        if feat is None or feat.empty:
            continue
        ts = pd.Timestamp(t["signal_date"])
        mask = feat.index <= ts
        if not mask.any():
            continue
        row = feat.loc[mask].iloc[[-1]].copy()
        if row.isna().all().all():
            continue
        X_rows.append(row)
        y.append(t["return_pct"])
        markets.append(t.get("market", "us"))

    X = pd.concat(X_rows, ignore_index=True)[feature_names].fillna(0.0)
    y = np.array(y)
    markets = np.array(markets)

    return X.values, y, markets, feature_names


def evaluate(y_true, y_pred, markets):
    baseline_wr = (y_true > 0).mean()
    auc = roc_auc_score((y_true > 0).astype(int), y_pred)

    sorted_idx = np.argsort(y_pred)[::-1]

    results = {"auc": float(auc), "baseline_wr": float(baseline_wr)}

    for pct in [0.1, 0.2]:
        n = max(1, int(len(y_true) * pct))
        sel = sorted_idx[:n]
        wr = (y_true[sel] > 0).mean()
        avg = y_true[sel].mean()
        results[f"top{pct:.0%}_wr"] = float(wr)
        results[f"top{pct:.0%}_avg"] = float(avg)
        results[f"top{pct:.0%}_delta_wr"] = float(wr - baseline_wr)

    for mkt in ["us", "india"]:
        mask = markets == mkt
        if mask.sum() < 20 or len(set((y_true[mask] > 0).astype(int))) < 2:
            continue
        auc_m = roc_auc_score((y_true[mask] > 0).astype(int), y_pred[mask])
        results[f"{mkt}_auc"] = float(auc_m)

    return results


def main():
    print("=" * 70)
    print("SKLEARN MLP NEURAL NET FOR V5")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    X, y, markets, feature_names = load_data()
    print(f"  Samples: {len(y)} | Features: {X.shape[1]}")
    print(f"  Baseline WR: {(y > 0).mean():.1%}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified CV
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[2/3] Training MLP with 5-fold CV...")
    oof_preds = np.full(len(y), np.nan)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, stratify)):
        print(f"  Fold {fold + 1}/5...", end=" ")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42 + fold,
        )

        mlp.fit(X_train, y_train)
        preds = mlp.predict(X_val)
        oof_preds[val_idx] = preds

        try:
            auc = roc_auc_score((y_val > 0).astype(int), preds)
        except ValueError:
            auc = 0.5
        fold_aucs.append(auc)
        print(f"AUC={auc:.4f} (iters={mlp.n_iter_})")

    print("\n[3/3] Evaluating OOF predictions...")
    metrics = evaluate(y, oof_preds, markets)
    print(f"  OOF AUC:        {metrics['auc']:.4f}")
    print(f"  Top 10% WR:     {metrics['top10%_wr']:.1%} (delta: {metrics['top10%_delta_wr']:+.1%})")
    print(f"  Top 10% Avg:    {metrics['top10%_avg']:.3%}")
    print(f"  Top 20% WR:     {metrics['top20%_wr']:.1%} (delta: {metrics['top20%_delta_wr']:+.1%})")
    print(f"  Top 20% Avg:    {metrics['top20%_avg']:.3%}")
    if "us_auc" in metrics:
        print(f"  US AUC:         {metrics['us_auc']:.4f}")
    if "india_auc" in metrics:
        print(f"  India AUC:      {metrics['india_auc']:.4f}")

    print(f"\n  Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")

    # Save results
    data_dir = Path(__file__).parent / "training_data_v4"
    with open(data_dir / "mlp_v5_results.json", "w") as f:
        json.dump({
            "model": "MLPRegressor",
            "params": {"hidden_layer_sizes": [64, 32, 16], "alpha": 0.01},
            "metrics": metrics,
            "fold_aucs": fold_aucs,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {data_dir / 'mlp_v5_results.json'}")

    print("\n" + "=" * 70)
    print("MLP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
