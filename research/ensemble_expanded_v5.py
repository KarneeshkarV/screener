"""Expanded ensemble: 5 XGBoost + LightGBM + CatBoost.

Target: AUC > 0.650, top10% WR > 65%
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

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

    return X, y, markets, feature_names


def evaluate(y_true, y_pred, markets):
    baseline_wr = (y_true > 0).mean()
    baseline_avg = y_true.mean()
    auc = roc_auc_score((y_true > 0).astype(int), y_pred)

    sorted_idx = np.argsort(y_pred)[::-1]

    results = {
        "auc": float(auc),
        "baseline_wr": float(baseline_wr),
        "baseline_avg": float(baseline_avg),
    }

    for pct in [0.1, 0.2, 0.3]:
        n = max(1, int(len(y_true) * pct))
        sel = sorted_idx[:n]
        wr = (y_true[sel] > 0).mean()
        avg = y_true[sel].mean()
        results[f"top{pct:.0%}_wr"] = float(wr)
        results[f"top{pct:.0%}_avg"] = float(avg)
        results[f"top{pct:.0%}_delta_wr"] = float(wr - baseline_wr)
        results[f"top{pct:.0%}_delta_avg"] = float(avg - baseline_avg)

    for mkt in ["us", "india"]:
        mask = markets == mkt
        if mask.sum() < 20 or len(set((y_true[mask] > 0).astype(int))) < 2:
            continue
        auc_m = roc_auc_score((y_true[mask] > 0).astype(int), y_pred[mask])
        results[f"{mkt}_auc"] = float(auc_m)

    return results


def train_model_cv(X, y, markets, model_fn, model_name, n_splits=5):
    """Generic CV trainer."""
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.full(len(y), np.nan)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_fn(fold)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        try:
            auc = roc_auc_score((y_val > 0).astype(int), preds)
        except ValueError:
            auc = 0.5
        fold_aucs.append(auc)

    oof_auc = roc_auc_score((y_true > 0).astype(int), oof_preds)
    return oof_preds, float(oof_auc), fold_aucs


def main():
    print("=" * 70)
    print("EXPANDED ENSEMBLE: 5 XGB + LightGBM + CatBoost")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    X, y, markets, feature_names = load_data()
    print(f"  Samples: {len(y)} | Features: {len(feature_names)}")
    print(f"  Baseline WR: {(y > 0).mean():.1%}")

    # Load optimization results for top 5 XGB configs
    results_path = Path(__file__).parent / "training_data_v4" / "optimization_results_v5.json"
    with open(results_path) as f:
        opt_results = json.load(f)
    top5 = opt_results["top_10_configs"][:5]

    # Store all OOF predictions
    all_preds = []
    all_names = []
    all_aucs = []

    # --- 5 XGBoost models ---
    print("\n[2/5] Training top 5 XGBoost models...")
    for i, cfg_result in enumerate(top5):
        cfg = cfg_result["hyperparams"].copy()
        cfg["feature_subset"] = cfg_result["feature_names"]
        cfg["seed"] = 42 + i

        def make_xgb(fold):
            if cfg.get("feature_subset"):
                use_cols = [c for c in feature_names if c in cfg["feature_subset"]]
            else:
                use_cols = feature_names
            # Store for later
            return XGBRegressor(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                colsample_bylevel=cfg.get("colsample_bylevel", 1.0),
                min_child_weight=cfg.get("min_child_weight", 1),
                reg_alpha=cfg.get("reg_alpha", 0.0),
                reg_lambda=cfg.get("reg_lambda", 1.0),
                gamma=cfg.get("gamma", 0.0),
                random_state=cfg.get("seed", 42) + fold,
                n_jobs=4,
            )

        # Need to handle feature subset manually
        labels = (y > 0).astype(int)
        market_codes = pd.Categorical(markets).codes
        stratify = labels * 10 + market_codes
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        oof_preds = np.full(len(y), np.nan)
        fold_aucs = []

        if cfg.get("feature_subset"):
            use_cols = [c for c in feature_names if c in cfg["feature_subset"]]
        else:
            use_cols = feature_names
        X_sub = X[use_cols].copy()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_sub, stratify)):
            X_train, X_val = X_sub.iloc[train_idx], X_sub.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = XGBRegressor(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                colsample_bylevel=cfg.get("colsample_bylevel", 1.0),
                min_child_weight=cfg.get("min_child_weight", 1),
                reg_alpha=cfg.get("reg_alpha", 0.0),
                reg_lambda=cfg.get("reg_lambda", 1.0),
                gamma=cfg.get("gamma", 0.0),
                random_state=cfg.get("seed", 42) + fold,
                n_jobs=4,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            oof_preds[val_idx] = preds

            try:
                auc = roc_auc_score((y_val > 0).astype(int), preds)
            except ValueError:
                auc = 0.5
            fold_aucs.append(auc)

        oof_auc = roc_auc_score((y > 0).astype(int), oof_preds)
        all_preds.append(oof_preds)
        all_names.append(f"XGB_{i+1}")
        all_aucs.append(oof_auc)
        print(f"  XGB {i+1}: OOF AUC={oof_auc:.4f} (folds: {[f'{a:.3f}' for a in fold_aucs]})")

    # --- LightGBM ---
    print("\n[3/5] Training LightGBM...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes

    oof_preds = np.full(len(y), np.nan)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.0,
            reg_lambda=3.0,
            min_child_samples=10,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=4,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        try:
            auc = roc_auc_score((y_val > 0).astype(int), preds)
        except ValueError:
            auc = 0.5
        fold_aucs.append(auc)

    oof_auc = roc_auc_score((y > 0).astype(int), oof_preds)
    all_preds.append(oof_preds)
    all_names.append("LGB")
    all_aucs.append(oof_auc)
    print(f"  LightGBM: OOF AUC={oof_auc:.4f} (folds: {[f'{a:.3f}' for a in fold_aucs]})")

    # --- CatBoost ---
    print("\n[4/5] Training CatBoost...")
    oof_preds = np.full(len(y), np.nan)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostRegressor(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            subsample=0.7,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=False,
            thread_count=4,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        try:
            auc = roc_auc_score((y_val > 0).astype(int), preds)
        except ValueError:
            auc = 0.5
        fold_aucs.append(auc)

    oof_auc = roc_auc_score((y > 0).astype(int), oof_preds)
    all_preds.append(oof_preds)
    all_names.append("CB")
    all_aucs.append(oof_auc)
    print(f"  CatBoost: OOF AUC={oof_auc:.4f} (folds: {[f'{a:.3f}' for a in fold_aucs]})")

    # --- Build ensembles ---
    print(f"\n[5/5] Building expanded ensembles...")
    ensemble_results = []

    # Simple mean of all 7
    ens_mean = np.mean(all_preds, axis=0)
    metrics = evaluate(y, ens_mean, markets)
    metrics["ensemble"] = "mean_7models"
    metrics["models"] = all_names
    ensemble_results.append(metrics)
    print(f"\n  Mean (7 models):     AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} top10Avg={metrics['top10%_avg']:.3%}")

    # Weighted mean by OOF AUC
    weights = np.array(all_aucs)
    weights = weights / weights.sum()
    ens_weighted = np.average(all_preds, axis=0, weights=weights)
    metrics = evaluate(y, ens_weighted, markets)
    metrics["ensemble"] = "weighted_7models"
    metrics["models"] = all_names
    metrics["weights"] = weights.tolist()
    ensemble_results.append(metrics)
    print(f"  Weighted (7 models): AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} top10Avg={metrics['top10%_avg']:.3%}")

    # Stacked Ridge on all 7
    meta_X = np.column_stack(all_preds)
    oof_meta = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, stratify)):
        meta_train, meta_val = meta_X[train_idx], meta_X[val_idx]
        y_train = y[train_idx]

        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_train, y_train)
        oof_meta[val_idx] = ridge.predict(meta_val)

    metrics = evaluate(y, oof_meta, markets)
    metrics["ensemble"] = "stacked_ridge_7models"
    metrics["models"] = all_names
    ensemble_results.append(metrics)
    print(f"  Stacked Ridge (7):   AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} top10Avg={metrics['top10%_avg']:.3%}")

    # Stacked Ridge on best subset (XGB1 + LGB + CB)
    subset_preds = [all_preds[0], all_preds[5], all_preds[6]]  # Best XGB, LGB, CB
    meta_X_sub = np.column_stack(subset_preds)
    oof_meta_sub = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X_sub, stratify)):
        meta_train, meta_val = meta_X_sub[train_idx], meta_X_sub[val_idx]
        y_train = y[train_idx]

        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_train, y_train)
        oof_meta_sub[val_idx] = ridge.predict(meta_val)

    metrics = evaluate(y, oof_meta_sub, markets)
    metrics["ensemble"] = "stacked_ridge_best3"
    metrics["models"] = [all_names[0], all_names[5], all_names[6]]
    ensemble_results.append(metrics)
    print(f"  Stacked Ridge (best3): AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} top10Avg={metrics['top10%_avg']:.3%}")

    # Find best
    best = max(ensemble_results, key=lambda r: r["auc"])
    print(f"\n" + "=" * 70)
    print(f"BEST ENSEMBLE: {best['ensemble']}")
    print(f"=" * 70)
    print(f"  AUC:        {best['auc']:.4f}")
    print(f"  Top 10% WR: {best['top10%_wr']:.1%} (delta: {best['top10%_delta_wr']:+.1%})")
    print(f"  Top 10% Avg: {best['top10%_avg']:.3%}")
    print(f"  Top 20% WR: {best['top20%_wr']:.1%} (delta: {best['top20%_delta_wr']:+.1%})")
    print(f"  Top 20% Avg: {best['top20%_avg']:.3%}")
    if "us_auc" in best:
        print(f"  US AUC:     {best['us_auc']:.4f}")
    if "india_auc" in best:
        print(f"  India AUC:  {best['india_auc']:.4f}")

    # Target check
    print(f"\n" + "=" * 70)
    print("TARGET CHECK")
    print("=" * 70)
    target_auc = 0.650
    target_wr = 0.65
    print(f"  AUC target:    {target_auc:.4f} | Achieved: {best['auc']:.4f} | {'✅ HIT' if best['auc'] > target_auc else '❌ MISS'} ({best['auc'] - target_auc:+.4f})")
    print(f"  Top10% WR target: {target_wr:.1%} | Achieved: {best['top10%_wr']:.1%} | {'✅ HIT' if best['top10%_wr'] > target_wr else '❌ MISS'}")

    # Save results
    data_dir = Path(__file__).parent / "training_data_v4"
    with open(data_dir / "ensemble_expanded_results.json", "w") as f:
        json.dump({
            "individual_models": [{"name": n, "auc": a} for n, a in zip(all_names, all_aucs)],
            "ensemble_methods": ensemble_results,
            "best_ensemble": best["ensemble"],
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {data_dir / 'ensemble_expanded_results.json'}")

    print("\n" + "=" * 70)
    print("EXPANDED ENSEMBLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
