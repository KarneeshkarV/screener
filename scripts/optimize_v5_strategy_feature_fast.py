"""V5 with strategy feature -- FAST version (30 iterations, 3-fold)."""
from __future__ import annotations

import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

STRATEGIES = ["ema_trend", "ema_vol", "golden_cross", "golden_cross_vol", "rsi2_rev", "breakout"]
FEATURE_NAMES = [
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


def load_data():
    data_dir = Path(__file__).parent / "training_data_v4"
    with open(data_dir / "trades.json") as f:
        trades = json.load(f)["trades"]
    with open(data_dir / "v5_features.pkl", "rb") as f:
        cache = pickle.load(f)["features"]

    # Compute strategy win rates for target encoding
    strat_wr = {}
    for s in STRATEGIES:
        subset = [t for t in trades if t["strategy"] == s]
        if subset:
            wr = sum(1 for t in subset if t["return_pct"] > 0) / len(subset)
            strat_wr[s] = wr
    print(f"Strategy win rates: {strat_wr}")

    X_rows, y, markets, strategies = [], [], [], []
    for t in trades:
        feat = cache.get(t["ticker"])
        if feat is None or feat.empty:
            continue
        ts = pd.Timestamp(t["signal_date"])
        mask = feat.index <= ts
        if not mask.any():
            continue
        row = feat.loc[mask].iloc[[-1]][FEATURE_NAMES]
        if row.isna().all().all():
            continue
        X_rows.append(row)
        y.append(t["return_pct"])
        markets.append(t.get("market", "us"))
        strategies.append(t["strategy"])

    X_base = pd.concat(X_rows, ignore_index=True).fillna(0.0)
    y = np.array(y)
    markets = np.array(markets)
    strategies = np.array(strategies)

    # One-hot encode strategies
    for s in STRATEGIES:
        X_base[f"is_{s}"] = (strategies == s).astype(int)

    # Target encode: historical win rate of the strategy
    X_base["strat_wr"] = [strat_wr.get(s, 0.35) for s in strategies]

    return X_base, y, markets


def evaluate(X, y, markets, cfg, n_splits=3):
    use_cols = [c for c in X.columns if c in cfg.get("feature_subset", X.columns)]
    X_sub = X[use_cols].copy()

    labels = (y > 0).astype(int)
    mkt_codes = pd.Categorical(markets).codes
    strat = [int(s in ["rsi2_rev"]) for s in ["ema_trend"] * len(y)]  # simplified
    strat_vals = np.array([1 if "rsi" in str(s) else 0 for s in markets])
    stratify = labels * 10 + mkt_codes

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get("seed", 42))

    aucs, top10_wr, top10_avg = [], [], []
    for train_idx, test_idx in skf.split(X_sub, stratify):
        X_tr, X_te = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = XGBRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_lambda=cfg.get("reg_lambda", 1.0),
            reg_alpha=cfg.get("reg_alpha", 0.0),
            min_child_weight=cfg.get("min_child_weight", 1),
            gamma=cfg.get("gamma", 0.0),
            random_state=cfg.get("seed", 42),
            n_jobs=4,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        try:
            auc = roc_auc_score((y_te > 0).astype(int), y_pred)
        except ValueError:
            auc = 0.5
        aucs.append(auc)

        sorted_idx = np.argsort(y_pred)[::-1]
        n10 = max(1, int(len(y_te) * 0.1))
        sel = sorted_idx[:n10]
        top10_wr.append((y_te[sel] > 0).mean())
        top10_avg.append(y_te[sel].mean())

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "top10_wr_mean": float(np.mean(top10_wr)),
        "top10_avg_mean": float(np.mean(top10_avg)),
    }


def main():
    print("Loading data with strategy features...")
    X, y, markets = load_data()
    print(f"Trades: {len(y)} | Features: {len(X.columns)}")
    print(f"Columns: {list(X.columns)}")

    all_features = list(X.columns)
    base = FEATURE_NAMES
    strategy_oh = [c for c in all_features if c.startswith("is_")]
    strategy_te = ["strat_wr"]

    # Strategy feature subsets to test
    feature_sets = [
        ("base_only", base),
        ("base+onehot", base + strategy_oh),
        ("base+target_encode", base + strategy_te),
        ("base+both", base + strategy_oh + strategy_te),
    ]

    best_results = []
    for fs_name, fs_cols in feature_sets:
        print(f"\n=== Testing {fs_name} ({len(fs_cols)} features) ===")

        # Fixed best hyperparams from prior search
        cfg = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "reg_lambda": 3.0,
            "reg_alpha": 0.0,
            "min_child_weight": 1,
            "gamma": 0.0,
            "seed": 42,
            "feature_subset": fs_cols,
        }

        metrics = evaluate(X, y, markets, cfg, n_splits=3)
        print(f"  AUC={metrics['auc_mean']:.4f}  top10WR={metrics['top10_wr_mean']:.1%}  top10Avg={metrics['top10_avg_mean']:.3%}")
        best_results.append({"name": fs_name, "features": len(fs_cols), **metrics})

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for r in best_results:
        print(f"{r['name']:20s} AUC={r['auc_mean']:.4f}  top10WR={r['top10_wr_mean']:.1%}")

    # Save best
    best = max(best_results, key=lambda x: x["auc_mean"])
    print(f"\nBest: {best['name']} with AUC={best['auc_mean']:.4f}")

    out_dir = Path(__file__).parent / "training_data_v4"
    with open(out_dir / "strategy_feature_results.json", "w") as f:
        json.dump({"results": best_results, "best": best}, f, indent=2)

    # Train production model with best feature set
    best_cfg = {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.6, "colsample_bytree": 0.6,
        "reg_lambda": 3.0, "reg_alpha": 0.0,
        "min_child_weight": 1, "gamma": 0.0,
        "seed": 42, "feature_subset": best["features"],
    }
    # Reconstruct actual feature names
    if best["name"] == "base_only":
        use_cols = base
    elif best["name"] == "base+onehot":
        use_cols = base + strategy_oh
    elif best["name"] == "base+target_encode":
        use_cols = base + strategy_te
    else:
        use_cols = base + strategy_oh + strategy_te

    final_model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.6,
        reg_lambda=3.0, reg_alpha=0.0,
        min_child_weight=1, gamma=0.0,
        random_state=42, n_jobs=4,
    )
    final_model.fit(X[use_cols], y)

    with open(out_dir / "model_v5_strategy_feature.pkl", "wb") as f:
        pickle.dump({"model": final_model, "feature_names": use_cols, "results": best_results}, f)

    print(f"Saved model to {out_dir}/model_v5_strategy_feature.pkl")


if __name__ == "__main__":
    main()
