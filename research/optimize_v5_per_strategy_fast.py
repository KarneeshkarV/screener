"""V5 Per-Strategy Ensemble -- FAST version.

Trains separate XGBoost models for each strategy category using
best hyperparams from prior search. Combines via stacked Ridge.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

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

CATEGORIES = {
    "trend": ["ema_trend", "ema_vol", "golden_cross", "golden_cross_vol"],
    "mean_rev": ["rsi2_rev"],
    "breakout": ["breakout"],
}


def load_data():
    data_dir = Path(__file__).parent / "training_data_v4"
    with open(data_dir / "trades.json") as f:
        trades = json.load(f)["trades"]
    with open(data_dir / "v5_features.pkl", "rb") as f:
        cache = pickle.load(f)["features"]

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

    X = pd.concat(X_rows, ignore_index=True).fillna(0.0)
    y = np.array(y)
    markets = np.array(markets)
    strategies = np.array(strategies)
    return X, y, markets, strategies


def train_category_oof(X, y, markets, strategies, cat_name, cat_strats, n_splits=3):
    mask = np.isin(strategies, cat_strats)
    X_cat = X[mask].copy()
    y_cat = y[mask]
    markets_cat = markets[mask]

    n = len(y_cat)
    if n < 100:
        return None, f"{cat_name}: too few samples ({n})"

    labels = (y_cat > 0).astype(int)
    mkt_codes = pd.Categorical(markets_cat).codes
    stratify = labels * 10 + mkt_codes

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(n)
    fold_aucs = []

    for train_idx, test_idx in skf.split(X_cat, stratify):
        X_tr, X_te = X_cat.iloc[train_idx], X_cat.iloc[test_idx]
        y_tr, y_te = y_cat[train_idx], y_cat[test_idx]

        model = XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.6, colsample_bytree=0.6,
            reg_lambda=3.0, reg_alpha=0.0,
            min_child_weight=1, gamma=0.0,
            random_state=42, n_jobs=4,
        )
        model.fit(X_tr, y_tr)
        oof_preds[test_idx] = model.predict(X_te)

        try:
            auc = roc_auc_score((y_te > 0).astype(int), oof_preds[test_idx])
            fold_aucs.append(auc)
        except ValueError:
            pass

    auc_mean = np.mean(fold_aucs) if fold_aucs else 0.5

    # Top 10% metrics
    sorted_idx = np.argsort(oof_preds)[::-1]
    n10 = max(1, int(n * 0.1))
    sel10 = sorted_idx[:n10]
    top10_wr = (y_cat[sel10] > 0).mean()
    top10_avg = y_cat[sel10].mean()

    # Train full model
    full_model = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.6,
        reg_lambda=3.0, reg_alpha=0.0,
        min_child_weight=1, gamma=0.0,
        random_state=42, n_jobs=4,
    )
    full_model.fit(X_cat, y_cat)

    return {
        "model": full_model,
        "oof_preds": oof_preds,
        "mask": mask,
        "auc": auc_mean,
        "top10_wr": top10_wr,
        "top10_avg": top10_avg,
        "n": n,
    }, f"{cat_name}: n={n} AUC={auc_mean:.4f} top10WR={top10_wr:.1%}"


def main():
    print("Loading data...")
    X, y, markets, strategies = load_data()
    print(f"Total: {len(y)} trades")

    # Train per-category models
    print("\nTraining per-category models...")
    cat_results = {}
    cat_messages = []

    for cat_name, cat_strats in CATEGORIES.items():
        result, msg = train_category_oof(X, y, markets, strategies, cat_name, cat_strats, n_splits=3)
        print(f"  {msg}")
        if result:
            cat_results[cat_name] = result
            cat_messages.append(msg)

    # Build combined predictions
    print("\nBuilding ensemble...")
    combined_preds = np.zeros(len(y))
    combined_weights = np.zeros(len(y))

    for cat_name, result in cat_results.items():
        mask = result["mask"]
        combined_preds[mask] += result["oof_preds"] * result["auc"]
        combined_weights[mask] += result["auc"]

    # Normalize by category weights
    combined_preds = np.where(combined_weights > 0, combined_preds / combined_weights, 0)

    # Evaluate combined
    labels = (y > 0).astype(int)
    auc_all = roc_auc_score(labels, combined_preds)

    sorted_idx = np.argsort(combined_preds)[::-1]
    n10 = max(1, int(len(y) * 0.1))
    sel10 = sorted_idx[:n10]
    top10_wr = (y[sel10] > 0).mean()
    top10_avg = y[sel10].mean()

    n20 = max(1, int(len(y) * 0.2))
    sel20 = sorted_idx[:n20]
    top20_wr = (y[sel20] > 0).mean()
    top20_avg = y[sel20].mean()

    print(f"\n{'='*60}")
    print("PER-STRATEGY ENSEMBLE RESULTS")
    print(f"{'='*60}")
    for msg in cat_messages:
        print(f"  {msg}")
    print(f"\nCombined: AUC={auc_all:.4f} top10WR={top10_wr:.1%} top10Avg={top10_avg:.3%}")
    print(f"          top20WR={top20_wr:.1%} top20Avg={top20_avg:.3%}")

    # Stacked Ridge ensemble
    print("\nTraining stacked Ridge ensemble...")
    # Build meta-features from category OOF predictions
    meta_features = np.zeros((len(y), len(CATEGORIES)))
    for i, cat_name in enumerate(CATEGORIES.keys()):
        col = np.zeros(len(y))
        if cat_name in cat_results:
            mask = cat_results[cat_name]["mask"]
            col[mask] = cat_results[cat_name]["oof_preds"]
        meta_features[:, i] = col

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    labels = (y > 0).astype(int)
    mkt_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + mkt_codes

    oof_stack = np.zeros(len(y))
    for train_idx, test_idx in skf.split(meta_features, stratify):
        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_features[train_idx], y[train_idx])
        oof_stack[test_idx] = ridge.predict(meta_features[test_idx])

    auc_stack = roc_auc_score(labels, oof_stack)
    sorted_idx = np.argsort(oof_stack)[::-1]
    sel10 = sorted_idx[:n10]
    top10_wr_stack = (y[sel10] > 0).mean()
    top10_avg_stack = y[sel10].mean()

    print(f"Stacked Ridge: AUC={auc_stack:.4f} top10WR={top10_wr_stack:.1%} top10Avg={top10_avg_stack:.3%}")

    # Train final stacked model
    final_ridge = Ridge(alpha=1.0)
    final_ridge.fit(meta_features, y)

    out_dir = Path(__file__).parent / "training_data_v4"
    with open(out_dir / "model_v5_per_strategy.pkl", "wb") as f:
        pickle.dump({
            "category_models": {k: v["model"] for k, v in cat_results.items()},
            "stack_model": final_ridge,
            "categories": CATEGORIES,
            "feature_names": FEATURE_NAMES,
            "results": {
                "combined_auc": auc_all,
                "combined_top10_wr": top10_wr,
                "combined_top10_avg": top10_avg,
                "stacked_auc": auc_stack,
                "stacked_top10_wr": top10_wr_stack,
                "stacked_top10_avg": top10_avg_stack,
            },
        }, f)

    print(f"\nSaved to {out_dir}/model_v5_per_strategy.pkl")

    # Write report
    report = f"""# Per-Strategy Model Results

## Category Models

"""
    for msg in cat_messages:
        report += f"- {msg}\n"

    report += f"""
## Ensemble Results

| Method | AUC | Top 10% WR | Top 10% Avg |
|--------|-----|-----------|-------------|
| Weighted Average | {auc_all:.4f} | {top10_wr:.1%} | {top10_avg:.3%} |
| Stacked Ridge | {auc_stack:.4f} | {top10_wr_stack:.1%} | {top10_avg_stack:.3%} |

## Comparison to Baseline

| Model | AUC | Top 10% WR |
|-------|-----|-----------|
| v5 Single (best) | 0.6346 | 66.0% |
| v5 Ensemble | 0.6438 | 67.4% |
| Per-Strategy Weighted | {auc_all:.4f} | {top10_wr:.1%} |
| Per-Strategy Stacked | {auc_stack:.4f} | {top10_wr_stack:.1%} |
"""
    with open("PER_STRATEGY_RESULTS.md", "w") as f:
        f.write(report)
    print("Wrote PER_STRATEGY_RESULTS.md")


if __name__ == "__main__":
    main()
