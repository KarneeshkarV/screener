"""TabNet for v5 signal confidence -- tabular deep learning approach.

TabNet uses sequential attention for feature selection.
Target: AUC > 0.650, top10% WR > 65%
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
from pytorch_tabnet.tab_model import TabNetRegressor

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


def make_optimizer(params, **kwargs):
    lr = kwargs.get("lr", 0.02)
    return __import__("torch").optim.Adam(params, lr=lr)


def main():
    print("=" * 70)
    print("TABNET FOR V5 SIGNAL CONFIDENCE")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    X, y, markets, feature_names = load_data()
    print(f"  Samples: {len(y)} | Features: {X.shape[1]}")
    print(f"  US: {(markets=='us').sum()} | India: {(markets=='india').sum()}")
    print(f"  Baseline WR: {(y > 0).mean():.1%}")

    # Stratified CV
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[2/3] Training TabNet with 5-fold CV...")
    oof_preds = np.full(len(y), np.nan)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify)):
        print(f"  Fold {fold + 1}/5...", end=" ")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # TabNet expects 2D arrays
        y_train_2d = y_train.reshape(-1, 1)
        y_val_2d = y_val.reshape(-1, 1)

        clf = TabNetRegressor(
            n_d=16,
            n_a=16,
            n_steps=3,
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=make_optimizer,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=__import__("torch").optim.lr_scheduler.StepLR,
            mask_type="sparsemax",
            verbose=0,
            seed=42 + fold,
            device_name="cpu",
        )

        clf.fit(
            X_train=X_train,
            y_train=y_train_2d,
            eval_set=[(X_val, y_val_2d)],
            eval_name=["valid"],
            eval_metric=["mse"],
            max_epochs=200,
            patience=20,
            batch_size=512,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )

        preds = clf.predict(X_val).flatten()
        oof_preds[val_idx] = preds

        try:
            auc = roc_auc_score((y_val > 0).astype(int), preds)
        except ValueError:
            auc = 0.5
        fold_aucs.append(auc)
        print(f"AUC={auc:.4f}")

    print(f"\n[3/3] Evaluating OOF predictions...")
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
    with open(data_dir / "tabnet_v5_results.json", "w") as f:
        json.dump({
            "model": "TabNetRegressor",
            "params": {"n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.5},
            "metrics": metrics,
            "fold_aucs": fold_aucs,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {data_dir / 'tabnet_v5_results.json'}")

    # Train full model for production
    print("\n  Training production TabNet on full data...")
    full_clf = TabNetRegressor(
        n_d=16, n_a=16, n_steps=3, gamma=1.5, lambda_sparse=1e-4,
        optimizer_fn=make_optimizer,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=__import__("torch").optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        verbose=0,
        seed=42,
        device_name="cpu",
    )
    full_clf.fit(
        X_train=X,
        y_train=y.reshape(-1, 1),
        max_epochs=200,
        patience=30,
        batch_size=512,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    with open(data_dir / "model_v5_tabnet.pkl", "wb") as f:
        pickle.dump({"model": full_clf, "feature_names": feature_names, "metrics": metrics}, f)
    print(f"  Saved production model to {data_dir / 'model_v5_tabnet.pkl'}")

    print("\n" + "=" * 70)
    print("TABNET COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
