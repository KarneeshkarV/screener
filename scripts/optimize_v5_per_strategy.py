"""V5 Per-Strategy Ensemble Model.

Trains separate XGBoost models for each strategy category:
- trend: ema_trend, ema_vol, golden_cross, golden_cross_vol
- mean_rev: rsi2_rev
- breakout: breakout

Then ensembles them with stacked Ridge regression on OOF predictions.
"""
from __future__ import annotations

import json
import pickle
import random
import warnings
from datetime import date
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def load_data():
    data_dir = Path(__file__).parent / "training_data_v4"
    cache_path = data_dir / "v5_features.pkl"

    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)
    with open(data_dir / "bars.json") as f:
        bars_json = json.load(f)

    trades_list = trades_data["trades"]
    bars_data = bars_json.get("bars", bars_json)
    bench_data = bars_json.get("benchmarks", {})

    bars_by_tv = {}
    for sym, records in bars_data.items():
        if not isinstance(records, list):
            continue
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        bars_by_tv[sym] = df

    benchmark_by_market = {}
    for market, records in bench_data.items():
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        benchmark_by_market[market] = df

    class ExitReason(Enum):
        HOLD = "hold"; STOP_LOSS = "stop"; TAKE_PROFIT = "target"
        TRAILING_STOP = "trail"; EXIT_SIGNAL = "exit_expr"; TIME = "time"; EOD = "eod"

    class SimpleTrade:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    trades = []
    for t in trades_list:
        trades.append(SimpleTrade(
            ticker=t["ticker"], market=t.get("market", "us"), strategy=t.get("strategy", ""),
            rank=t["rank"], signal_date=date.fromisoformat(t["signal_date"]),
            entry_date=date.fromisoformat(t["entry_date"]), entry_price=t["entry_price"],
            exit_date=date.fromisoformat(t["exit_date"]), exit_price=t["exit_price"],
            exit_reason=ExitReason(t["exit_reason"]), shares=t["shares"],
            entry_cost=t["entry_cost"], exit_value=t["exit_value"],
            pnl=t["pnl"], return_pct=t["return_pct"], dividend_income=t["dividend_income"],
        ))

    bars_lookup = {}
    for key, df in bars_by_tv.items():
        if ":" in key:
            _, sym = key.split(":", 1)
        else:
            sym = key
        bars_lookup[sym] = df

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    features_cache = cache["features"]

    return trades, features_cache, bars_lookup, benchmark_by_market


def build_matrix(trades, features_cache):
    """Build feature matrix and target vector."""
    X_rows = []
    y = []
    dates = []
    markets = []
    strategies = []
    valid_trades = []

    for trade in trades:
        features = features_cache.get(trade.ticker)
        if features is None or features.empty:
            continue
        signal_ts = pd.Timestamp(trade.signal_date)
        mask = features.index <= signal_ts
        if not mask.any():
            continue
        row = features.loc[mask].iloc[[-1]].copy()
        if row.isna().all().all():
            continue
        X_rows.append(row)
        y.append(trade.return_pct)
        dates.append(trade.signal_date)
        markets.append(trade.market)
        strategies.append(trade.strategy)
        valid_trades.append(trade)

    X = pd.concat(X_rows, ignore_index=True)
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
    X = X[feature_names].fillna(0.0)
    y_arr = np.array(y)
    dates_arr = pd.to_datetime(dates)
    markets_arr = np.array(markets)
    strategies_arr = np.array(strategies)
    return X, y_arr, dates_arr, markets_arr, strategies_arr, valid_trades, feature_names


STRATEGY_CATEGORIES = {
    "trend": ["ema_trend", "ema_vol", "golden_cross", "golden_cross_vol"],
    "mean_rev": ["rsi2_rev"],
    "breakout": ["breakout"],
}


def generate_configs(feature_names, n=25):
    """Generate diverse hyperparameter configurations."""
    configs = []
    base_features = [
        "rvol_5d", "returns_5d", "returns_20d", "close_vs_ema20",
        "ema20_vs_ema50", "ATR_14_pct", "volatility_percentile_90d",
        "benchmark_return_20d",
    ]
    extended_features = [
        "max_dd_20d", "range_pct", "gap_pct", "consecutive_up_days",
        "volume_price_corr_20d", "sharpe_20d",
    ]
    momentum_features = [
        "returns_60d", "momentum_5d_vs_20d", "close_vs_ema50", "ema50_vs_ema200",
        "rsi_14", "macd_hist", "adx_14",
    ]
    volume_features = ["rvol_20d", "volume_trend_10d"]
    structure_features = ["dist_from_52w_high", "dist_from_52w_low", "bb_position", "beta_20d"]

    all_groups = {
        "base": base_features,
        "extended": extended_features,
        "momentum": momentum_features,
        "volume": volume_features,
        "structure": structure_features,
    }

    n_estimators_opts = [50, 100, 150, 200, 300, 400]
    max_depth_opts = [2, 3, 4, 5, 6]
    learning_rate_opts = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15]
    subsample_opts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_opts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    reg_lambda_opts = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    reg_alpha_opts = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    min_child_opts = [1, 3, 5, 7, 10]
    gamma_opts = [0.0, 0.1, 0.5, 1.0]

    for i in range(n):
        subset = list(base_features)
        for group_name, group_feats in all_groups.items():
            if group_name == "base":
                continue
            if random.random() < 0.6:
                subset.extend(group_feats)
        subset = list(dict.fromkeys(subset))

        cfg = {
            "id": i + 1,
            "seed": 42 + i,
            "n_estimators": random.choice(n_estimators_opts),
            "max_depth": random.choice(max_depth_opts),
            "learning_rate": random.choice(learning_rate_opts),
            "subsample": random.choice(subsample_opts),
            "colsample_bytree": random.choice(colsample_opts),
            "colsample_bylevel": random.choice(colsample_opts),
            "reg_lambda": random.choice(reg_lambda_opts),
            "reg_alpha": random.choice(reg_alpha_opts),
            "min_child_weight": random.choice(min_child_opts),
            "gamma": random.choice(gamma_opts),
            "feature_subset": subset,
        }
        configs.append(cfg)
    return configs


def train_category_model(X_train, y_train, feature_names, config):
    """Train an XGBRegressor for a category."""
    if config.get("feature_subset"):
        use_cols = [c for c in feature_names if c in config["feature_subset"]]
    else:
        use_cols = feature_names
    X_sub = X_train[use_cols].copy()
    model = XGBRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        colsample_bylevel=config.get("colsample_bylevel", 1.0),
        min_child_weight=config.get("min_child_weight", 1),
        reg_alpha=config.get("reg_alpha", 0.0),
        reg_lambda=config.get("reg_lambda", 1.0),
        gamma=config.get("gamma", 0.0),
        random_state=config.get("seed", 42),
        n_jobs=4,
    )
    model.fit(X_sub, y_train)
    return model, use_cols


def evaluate_category_cv(X, y, markets, strategies, trades, feature_names, category_name, category_strategies, configs, n_splits=5):
    """Run stratified CV for a single category, testing multiple configs."""
    mask = np.isin(strategies, category_strategies)
    X_cat = X[mask].copy().reset_index(drop=True)
    y_cat = y[mask]
    markets_cat = markets[mask]
    trades_cat = [trades[i] for i in np.where(mask)[0]]

    if len(y_cat) < 50:
        print(f"  [{category_name}] Too few samples ({len(y_cat)}), skipping.")
        return None, None, None

    labels = (y_cat > 0).astype(int)
    market_codes = pd.Categorical(markets_cat).codes
    stratify = labels * 10 + market_codes

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score = -999
    best_cfg = None
    best_metrics = None
    best_oof = None

    print(f"\n  [{category_name}] {len(y_cat)} trades | Baseline WR: {(y_cat > 0).mean():.1%}")

    for cfg in configs:
        oof_preds = np.zeros(len(y_cat))
        fold_aucs = []
        fold_top10_wr = []
        fold_top10_avg = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_cat, stratify)):
            X_tr, X_te = X_cat.iloc[train_idx], X_cat.iloc[test_idx]
            y_tr, y_te = y_cat[train_idx], y_cat[test_idx]
            t_te = [trades_cat[i] for i in test_idx]

            model, use_cols = train_category_model(X_tr, y_tr, feature_names, cfg)
            preds = model.predict(X_te[use_cols])
            oof_preds[test_idx] = preds

            try:
                auc = roc_auc_score((y_te > 0).astype(int), preds)
            except ValueError:
                auc = 0.5
            fold_aucs.append(auc)

            sorted_idx = np.argsort(preds)[::-1]
            n10 = max(1, int(len(y_te) * 0.1))
            sel10 = sorted_idx[:n10]
            wr10 = (y_te[sel10] > 0).mean()
            avg10 = y_te[sel10].mean()
            fold_top10_wr.append(wr10)
            fold_top10_avg.append(avg10)

        auc_mean = float(np.mean(fold_aucs))
        top10_wr_mean = float(np.mean(fold_top10_wr))
        top10_avg_mean = float(np.mean(fold_top10_avg))
        score = auc_mean + (top10_wr_mean - (y_cat > 0).mean()) * 2

        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_metrics = {
                "auc_mean": auc_mean,
                "auc_std": float(np.std(fold_aucs)),
                "top10_wr_mean": top10_wr_mean,
                "top10_avg_mean": top10_avg_mean,
                "baseline_wr": float((y_cat > 0).mean()),
                "n_trades": len(y_cat),
            }
            best_oof = oof_preds.copy()

    print(f"  [{category_name}] BEST → AUC={best_metrics['auc_mean']:.4f} top10WR={best_metrics['top10_wr_mean']:.1%} score={best_score:.4f}")
    return best_cfg, best_metrics, best_oof


def build_full_oof_predictions(X, y, markets, strategies, trades, feature_names, category_results):
    """Build a full OOF prediction matrix for ALL samples using category-specific models."""
    # Create unified CV folds for all data
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n = len(y)
    oof_matrix = np.full((n, len(category_results)), np.nan)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, stratify)):
        X_train = X.iloc[train_idx].copy().reset_index(drop=True)
        y_train = y[train_idx]
        strategies_train = strategies[train_idx]
        X_test = X.iloc[test_idx].copy().reset_index(drop=True)
        # test_idx maps back to original positions

        for cat_idx, (cat_name, (best_cfg, _, _)) in enumerate(category_results.items()):
            if best_cfg is None:
                continue
            cat_mask_train = np.isin(strategies_train, STRATEGY_CATEGORIES[cat_name])
            if cat_mask_train.sum() < 10:
                continue
            X_cat_train = X_train[cat_mask_train].copy().reset_index(drop=True)
            y_cat_train = y_train[cat_mask_train]

            model, use_cols = train_category_model(X_cat_train, y_cat_train, feature_names, best_cfg)
            preds = model.predict(X_test[use_cols])
            oof_matrix[test_idx, cat_idx] = preds

    return oof_matrix


def evaluate_ensemble(oof_preds, y, trades, weights=None):
    """Evaluate ensemble predictions."""
    # oof_preds may contain NaNs where a model wasn't applicable; fill with column means
    col_means = np.nanmean(oof_preds, axis=0)
    filled = oof_preds.copy()
    for j in range(oof_preds.shape[1]):
        nan_mask = np.isnan(filled[:, j])
        filled[nan_mask, j] = col_means[j]

    if weights is not None:
        ens_pred = filled @ weights
    else:
        ens_pred = np.nanmean(oof_preds, axis=1)
        # for rows that are all nan, fallback to 0
        ens_pred = np.nan_to_num(ens_pred, nan=0.0)

    auc = roc_auc_score((y > 0).astype(int), ens_pred)

    sorted_idx = np.argsort(ens_pred)[::-1]
    n10 = max(1, int(len(y) * 0.1))
    sel10 = sorted_idx[:n10]
    wr10 = (y[sel10] > 0).mean()
    avg10 = y[sel10].mean()

    n20 = max(1, int(len(y) * 0.2))
    sel20 = sorted_idx[:n20]
    wr20 = (y[sel20] > 0).mean()
    avg20 = y[sel20].mean()

    pnl10 = sum(trades[i].pnl for i in sel10)
    pnl20 = sum(trades[i].pnl for i in sel20)

    return {
        "auc": float(auc),
        "top10_wr": float(wr10),
        "top10_avg": float(avg10),
        "top10_pnl": float(pnl10),
        "top20_wr": float(wr20),
        "top20_avg": float(avg20),
        "top20_pnl": float(pnl20),
    }


def main():
    print("=" * 70)
    print("V5 PER-STRATEGY ENSEMBLE MODEL")
    print("=" * 70)

    print("\n[1/6] Loading data...")
    trades, features_cache, bars_lookup, benchmarks = load_data()
    X, y, dates, markets, strategies, valid_trades, feature_names = build_matrix(trades, features_cache)
    print(f"  Total trades: {len(y)} | Features: {len(feature_names)}")
    print(f"  US: {(markets == 'us').sum()} | India: {(markets == 'india').sum()}")
    print(f"  Baseline WR: {(y > 0).mean():.1%} | Avg return: {y.mean():.3%}")

    for cat_name, cat_strats in STRATEGY_CATEGORIES.items():
        mask = np.isin(strategies, cat_strats)
        print(f"  {cat_name}: {mask.sum()} trades | WR: {(y[mask] > 0).mean():.1%}")

    print("\n[2/6] Generating hyperparameter configs...")
    random.seed(42)
    configs = generate_configs(feature_names, n=25)
    print(f"  Generated {len(configs)} configs per category")

    print("\n[3/6] Training per-category models with CV...")
    category_results = {}
    category_oofs = {}
    for cat_name, cat_strats in STRATEGY_CATEGORIES.items():
        best_cfg, best_metrics, best_oof = evaluate_category_cv(
            X, y, markets, strategies, valid_trades, feature_names,
            cat_name, cat_strats, configs, n_splits=5
        )
        category_results[cat_name] = (best_cfg, best_metrics, best_oof)
        if best_oof is not None:
            category_oofs[cat_name] = best_oof

    # Build OOF matrix on ALL data using a unified CV
    print("\n[4/6] Building unified OOF predictions for ensemble...")
    oof_matrix = build_full_oof_predictions(X, y, markets, strategies, valid_trades, feature_names, category_results)
    print(f"  OOF matrix shape: {oof_matrix.shape}")

    # Evaluate simple mean ensemble
    mean_metrics = evaluate_ensemble(oof_matrix, y, valid_trades)
    print(f"  Simple mean ensemble → AUC={mean_metrics['auc']:.4f} top10WR={mean_metrics['top10_wr']:.1%} top10Avg={mean_metrics['top10_avg']:.3%}")

    # Optimize weighted ensemble on OOF
    print("\n[5/6] Optimizing weighted ensemble...")
    col_means = np.nanmean(oof_matrix, axis=0)
    filled = oof_matrix.copy()
    for j in range(oof_matrix.shape[1]):
        nan_mask = np.isnan(filled[:, j])
        filled[nan_mask, j] = col_means[j]

    best_w_auc = 0
    best_weights = None
    best_w_metrics = None
    # Grid search weights
    cat_names = list(category_results.keys())
    n_cats = len(cat_names)
    for w0 in np.arange(0.1, 1.0, 0.1):
        for w1 in np.arange(0.1, 1.0, 0.1):
            for w2 in np.arange(0.1, 1.0, 0.1):
                w = np.array([w0, w1, w2])
                w = w / w.sum()
                preds = filled @ w
                auc = roc_auc_score((y > 0).astype(int), preds)
                if auc > best_w_auc:
                    best_w_auc = auc
                    best_weights = w.copy()

    # Also try Ridge regression stacking
    ridge = Ridge(alpha=1.0)
    ridge.fit(filled, (y > 0).astype(int))
    ridge_preds = ridge.predict(filled)
    ridge_auc = roc_auc_score((y > 0).astype(int), ridge_preds)
    print(f"  Best grid-search weights AUC: {best_w_auc:.4f} (weights: {best_weights.round(3)})")
    print(f"  Ridge stacking AUC: {ridge_auc:.4f} (coefs: {ridge.coef_.round(3)})")

    # Choose best ensemble
    if ridge_auc >= best_w_auc:
        ensemble_type = "ridge"
        final_weights = ridge.coef_.copy()
        final_intercept = float(ridge.intercept_)
        ens_metrics = evaluate_ensemble(oof_matrix, y, valid_trades, weights=final_weights)
        print(f"  → Selected Ridge ensemble")
    else:
        ensemble_type = "weighted_mean"
        final_weights = best_weights.copy()
        final_intercept = 0.0
        ens_metrics = evaluate_ensemble(oof_matrix, y, valid_trades, weights=final_weights)
        print(f"  → Selected weighted-mean ensemble")

    print(f"\n  FINAL ENSEMBLE → AUC={ens_metrics['auc']:.4f} top10WR={ens_metrics['top10_wr']:.1%} top10Avg={ens_metrics['top10_avg']:.3%} top10Pnl=${ens_metrics['top10_pnl']:,.0f}")

    # Train final production models on ALL data per category
    print("\n[6/6] Training final production models...")
    final_models = {}
    for cat_name, cat_strats in STRATEGY_CATEGORIES.items():
        best_cfg, best_metrics, _ = category_results[cat_name]
        if best_cfg is None:
            continue
        mask = np.isin(strategies, cat_strats)
        X_cat = X[mask].copy().reset_index(drop=True)
        y_cat = y[mask]
        model, use_cols = train_category_model(X_cat, y_cat, feature_names, best_cfg)
        final_models[cat_name] = {
            "model": model,
            "feature_names": use_cols,
            "config": best_cfg,
            "metrics": best_metrics,
        }
        print(f"  [{cat_name}] Trained on {len(y_cat)} trades → AUC={best_metrics['auc_mean']:.4f}")

    # Save ensemble
    data_dir = Path(__file__).parent / "training_data_v4"
    data_dir.mkdir(exist_ok=True)
    model_path = data_dir / "model_v5_per_strategy.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "category_models": final_models,
            "ensemble_type": ensemble_type,
            "ensemble_weights": final_weights.tolist(),
            "ensemble_intercept": final_intercept,
            "feature_names": feature_names,
            "category_strategies": STRATEGY_CATEGORIES,
            "ensemble_metrics": ens_metrics,
            "category_metrics": {k: v[1] for k, v in category_results.items()},
        }, f)
    print(f"  Saved ensemble to {model_path}")

    # Summary
    print("\n" + "=" * 70)
    print("PER-STRATEGY ENSEMBLE SUMMARY")
    print("=" * 70)
    print(f"Overall AUC:         {ens_metrics['auc']:.4f}")
    print(f"Top 10% Win Rate:    {ens_metrics['top10_wr']:.1%}")
    print(f"Top 10% Avg Return:  {ens_metrics['top10_avg']:.3%}")
    print(f"Top 10% Total PnL:   ${ens_metrics['top10_pnl']:,.0f}")
    print(f"Top 20% Win Rate:    {ens_metrics['top20_wr']:.1%}")
    print(f"Top 20% Avg Return:  {ens_metrics['top20_avg']:.3%}")
    print(f"Top 20% Total PnL:   ${ens_metrics['top20_pnl']:,.0f}")
    print(f"Baseline WR:         {(y > 0).mean():.1%}")
    print(f"Baseline Avg Return: {y.mean():.3%}")
    print("=" * 70)

    return ens_metrics, category_results


if __name__ == "__main__":
    main()
