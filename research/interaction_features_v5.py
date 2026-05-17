"""Feature interactions for v5 -- add non-linear combinations.

Hypothesis: domain-specific interactions capture effects that
individual features miss, especially for tree models with limited depth.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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


def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific interaction features."""
    X = X.copy()

    # Volume-confirmed momentum
    X["rvol_x_returns5d"] = X["rvol_5d"] * X["returns_5d"]

    # Trend strength × distance from trend
    X["close_vs_ema20_x_adx"] = X["close_vs_ema20"] * X["adx_14"]

    # Risk-adjusted return × drawdown
    X["sharpe_x_maxdd"] = X["sharpe_20d"] * X["max_dd_20d"]

    # Mean-reversion × short momentum
    X["rsi_x_returns5d"] = X["rsi_14"] * X["returns_5d"]

    # Momentum × accumulation
    X["macd_x_volprice_corr"] = X["macd_hist"] * X["volume_price_corr_20d"]

    # Intraday conviction × overnight sentiment
    X["range_x_gap"] = X["range_pct"] * X["gap_pct"]

    # Market context × sensitivity
    X["bench_x_beta"] = X["benchmark_return_20d"] * X["beta_20d"]

    # Streak × momentum
    X["streak_x_returns5d"] = X["consecutive_up_days"] * X["returns_5d"]

    # Volatility regime × momentum
    X["volpct_x_returns20d"] = X["volatility_percentile_90d"] * X["returns_20d"]

    # Distance from 52w high × momentum
    X["dist52h_x_returns20d"] = X["dist_from_52w_high"] * X["returns_20d"]

    return X


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


def train_xgb_cv(X, y, markets, cfg, n_splits=5):
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, stratify)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
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
        oof_preds[val_idx] = model.predict(X_val)

    oof_auc = roc_auc_score((y > 0).astype(int), oof_preds)
    return oof_preds, oof_auc


def main():
    print("=" * 70)
    print("FEATURE INTERACTIONS FOR V5")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    X, y, markets, feature_names = load_data()
    print(f"  Base features: {len(feature_names)}")

    print("\n[2/4] Adding interaction features...")
    X_inter = add_interactions(X)
    new_features = [c for c in X_inter.columns if c not in feature_names]
    print(f"  Added {len(new_features)} interactions: {new_features}")
    print(f"  Total features: {X_inter.shape[1]}")

    # Load top 5 configs
    results_path = Path(__file__).parent / "training_data_v4" / "optimization_results_v5.json"
    with open(results_path) as f:
        opt_results = json.load(f)
    top5 = opt_results["top_10_configs"][:5]

    # --- Baseline (no interactions) ---
    print("\n[3/4] Baseline (no interactions) -- best single model...")
    best_cfg = top5[0]["hyperparams"].copy()
    best_cfg["seed"] = 42
    oof_base, auc_base = train_xgb_cv(X, y, markets, best_cfg, n_splits=5)
    metrics_base = evaluate(y, oof_base, markets)
    print(f"  Baseline: AUC={metrics_base['auc']:.4f} top10WR={metrics_base['top10%_wr']:.1%}")

    # --- With interactions ---
    print("\n  With interactions -- best single model...")
    oof_inter, auc_inter = train_xgb_cv(X_inter, y, markets, best_cfg, n_splits=5)
    metrics_inter = evaluate(y, oof_inter, markets)
    print(f"  Interact: AUC={metrics_inter['auc']:.4f} top10WR={metrics_inter['top10%_wr']:.1%}")

    # --- Ensemble with interactions ---
    print("\n[4/4] 5-model ensemble WITH interactions...")
    all_preds = []
    all_aucs = []

    for i, cfg_result in enumerate(top5):
        cfg = cfg_result["hyperparams"].copy()
        cfg["seed"] = 42 + i
        oof, auc = train_xgb_cv(X_inter, y, markets, cfg, n_splits=5)
        all_preds.append(oof)
        all_aucs.append(auc)
        print(f"  Model {i+1}: OOF AUC={auc:.4f}")

    # Simple mean
    ens_mean = np.mean(all_preds, axis=0)
    metrics_mean = evaluate(y, ens_mean, markets)
    print(f"\n  Mean (5):   AUC={metrics_mean['auc']:.4f} top10WR={metrics_mean['top10%_wr']:.1%} top10Avg={metrics_mean['top10%_avg']:.3%}")

    # Weighted mean
    weights = np.array(all_aucs)
    weights = weights / weights.sum()
    ens_weighted = np.average(all_preds, axis=0, weights=weights)
    metrics_w = evaluate(y, ens_weighted, markets)
    print(f"  Weighted:   AUC={metrics_w['auc']:.4f} top10WR={metrics_w['top10%_wr']:.1%} top10Avg={metrics_w['top10%_avg']:.3%}")

    # Stacked Ridge
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    meta_X = np.column_stack(all_preds)
    oof_meta = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_X, stratify)):
        meta_train, meta_val = meta_X[train_idx], meta_X[val_idx]
        y_train = y[train_idx]
        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_train, y_train)
        oof_meta[val_idx] = ridge.predict(meta_val)

    metrics_ridge = evaluate(y, oof_meta, markets)
    print(f"  Stacked:    AUC={metrics_ridge['auc']:.4f} top10WR={metrics_ridge['top10%_wr']:.1%} top10Avg={metrics_ridge['top10%_avg']:.3%}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Approach':<30} {'AUC':>7} {'Top10WR':>9} {'Top10Avg':>10}")
    print("-" * 60)
    print(f"{'Baseline single':<30} {metrics_base['auc']:>7.4f} {metrics_base['top10%_wr']:>8.1%} {metrics_base['top10%_avg']:>9.3%}")
    print(f"{'+Interactions single':<30} {metrics_inter['auc']:>7.4f} {metrics_inter['top10%_wr']:>8.1%} {metrics_inter['top10%_avg']:>9.3%}")
    print(f"{'Baseline ensemble (prev)':<30} {'0.6438':>7} {'67.4%':>9} {'4.573%':>10}")
    print(f"{'+Interactions ensemble mean':<30} {metrics_mean['auc']:>7.4f} {metrics_mean['top10%_wr']:>8.1%} {metrics_mean['top10%_avg']:>9.3%}")
    print(f"{'+Interactions ensemble weighted':<30} {metrics_w['auc']:>7.4f} {metrics_w['top10%_wr']:>8.1%} {metrics_w['top10%_avg']:>9.3%}")
    print(f"{'+Interactions ensemble stacked':<30} {metrics_ridge['auc']:>7.4f} {metrics_ridge['top10%_wr']:>8.1%} {metrics_ridge['top10%_avg']:>9.3%}")

    # Target
    best_all = max([metrics_ridge, metrics_w, metrics_mean], key=lambda r: r["auc"])
    print(f"\n" + "=" * 70)
    print("TARGET CHECK")
    print("=" * 70)
    print(f"  AUC target: 0.6500 | Best achieved: {best_all['auc']:.4f} | {'✅ HIT' if best_all['auc'] > 0.650 else '❌ MISS'} ({best_all['auc'] - 0.650:+.4f})")
    print(f"  Top10% WR target: 65.0% | Achieved: {best_all['top10%_wr']:.1%} | {'✅ HIT' if best_all['top10%_wr'] > 0.65 else '❌ MISS'}")

    # Save results
    data_dir = Path(__file__).parent / "training_data_v4"
    with open(data_dir / "interaction_features_results.json", "w") as f:
        json.dump({
            "new_features": new_features,
            "baseline_single": metrics_base,
            "interactions_single": metrics_inter,
            "ensemble_mean": metrics_mean,
            "ensemble_weighted": metrics_w,
            "ensemble_stacked": metrics_ridge,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to {data_dir / 'interaction_features_results.json'}")

    print("\n" + "=" * 70)
    print("INTERACTION FEATURES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
