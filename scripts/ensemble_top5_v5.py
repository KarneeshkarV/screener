"""Ensemble top 5 XGBoost configs from 100-iteration optimization.

Methods tested:
  1. Simple mean of predicted returns
  2. Weighted mean by validation AUC
  3. Rank averaging (average percentile ranks)
  4. Stacked ensemble (meta-learner: Ridge regression on top-5 predictions)

Target: AUC > 0.650, top10% WR > 65%
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold

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

    X_rows, y, markets, tickers, dates = [], [], [], [], []
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
        tickers.append(t["ticker"])
        dates.append(t["signal_date"])

    X = pd.concat(X_rows, ignore_index=True)[feature_names].fillna(0.0)
    y = np.array(y)
    markets = np.array(markets)
    tickers = np.array(tickers)
    dates = pd.to_datetime(dates)

    return X, y, markets, tickers, dates, feature_names


def evaluate_predictions(y_true, y_pred, markets, tickers, trades_dict=None):
    """Return dict of metrics."""
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

    # By market
    for mkt in ["us", "india"]:
        mask = markets == mkt
        if mask.sum() < 20 or len(set((y_true[mask] > 0).astype(int))) < 2:
            continue
        auc_m = roc_auc_score((y_true[mask] > 0).astype(int), y_pred[mask])
        results[f"{mkt}_auc"] = float(auc_m)

    return results


def train_model_cv(X, y, markets, cfg, n_splits=5):
    """Train model with stratified CV, return OOF predictions and CV AUC."""
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get("seed", 42))

    oof_preds = np.full(len(y), np.nan)
    fold_aucs = []

    # Determine feature subset
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
    return oof_preds, float(oof_auc), fold_aucs, use_cols


def train_model_full(X, y, cfg, feature_names):
    """Train model on full dataset."""
    if cfg.get("feature_subset"):
        use_cols = [c for c in feature_names if c in cfg["feature_subset"]]
    else:
        use_cols = feature_names
    X_sub = X[use_cols].copy()

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
        random_state=cfg.get("seed", 42),
        n_jobs=4,
    )
    model.fit(X_sub, y)
    return model, use_cols


def ensemble_simple_mean(predictions_list):
    """Simple mean of predictions."""
    return np.mean(predictions_list, axis=0)


def ensemble_weighted_mean(predictions_list, weights):
    """Weighted mean of predictions."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.average(predictions_list, axis=0, weights=weights)


def ensemble_rank_average(predictions_list):
    """Average percentile ranks instead of raw values."""
    ranks = np.array([stats.rankdata(p) for p in predictions_list])
    avg_ranks = np.mean(ranks, axis=0)
    # Convert back to 0-1 scale
    return avg_ranks / len(avg_ranks)


def ensemble_stacked(X, y, predictions_list, markets):
    """Train Ridge regression on top-5 predictions as meta-learner."""
    meta_X = np.column_stack(predictions_list)

    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_meta = np.full(len(y), np.nan)

    for train_idx, val_idx in skf.split(meta_X, stratify):
        meta_train, meta_val = meta_X[train_idx], meta_X[val_idx]
        y_train = y[train_idx]

        # Meta-learner: Ridge with small alpha
        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_train, y_train)
        oof_meta[val_idx] = ridge.predict(meta_val)

    return oof_meta


def main():
    print("=" * 70)
    print("ENSEMBLE TOP 5 XGBOOST CONFIGS")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    X, y, markets, tickers, dates, feature_names = load_data()
    print(f"  Samples: {len(y)} | US: {(markets=='us').sum()} | India: {(markets=='india').sum()}")
    print(f"  Baseline WR: {(y > 0).mean():.1%}")

    # Load optimization results
    results_path = Path(__file__).parent / "training_data_v4" / "optimization_results_v5.json"
    with open(results_path) as f:
        opt_results = json.load(f)

    top10 = opt_results["top_10_configs"]
    top5 = top10[:5]

    print(f"\n[2/5] Top 5 configs loaded:")
    for i, cfg in enumerate(top5):
        hp = cfg["hyperparams"]
        print(f"  [{i+1}] AUC={cfg['auc_mean']:.4f} top10WR={cfg['top10_wr_mean']:.1%} "
              f"depth={hp['max_depth']} lr={hp['learning_rate']:.2f} "
              f"lambda={hp['reg_lambda']:.1f} feats={cfg['n_features']}")

    # Train each model with CV to get OOF predictions
    print(f"\n[3/5] Training top 5 models with 5-fold CV...")
    oof_predictions = []
    cv_aucs = []
    models_full = []
    feature_cols_list = []

    for i, cfg_result in enumerate(top5):
        cfg = cfg_result["hyperparams"].copy()
        cfg["feature_subset"] = cfg_result["feature_names"]
        cfg["seed"] = 42 + i

        print(f"  Model {i+1}/5...", end=" ")
        oof_preds, oof_auc, fold_aucs, use_cols = train_model_cv(X, y, markets, cfg, n_splits=5)
        oof_predictions.append(oof_preds)
        cv_aucs.append(oof_auc)
        feature_cols_list.append(use_cols)
        print(f"OOF AUC={oof_auc:.4f} (folds: {[f'{a:.3f}' for a in fold_aucs]})")

    # Also train full models for production
    print(f"\n  Training full models on all data...")
    for i, cfg_result in enumerate(top5):
        cfg = cfg_result["hyperparams"].copy()
        cfg["feature_subset"] = cfg_result["feature_names"]
        cfg["seed"] = 42 + i
        model, use_cols = train_model_full(X, y, cfg, feature_names)
        models_full.append(model)

    # Evaluate individual models
    print(f"\n[4/5] Evaluating individual models (OOF predictions)...")
    individual_results = []
    for i, preds in enumerate(oof_predictions):
        metrics = evaluate_predictions(y, preds, markets, tickers)
        metrics["model"] = f"model_{i+1}"
        metrics["cv_auc"] = cv_aucs[i]
        individual_results.append(metrics)
        print(f"  Model {i+1}: AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} "
              f"top10Avg={metrics['top10%_avg']:.3%}")

    # Build ensembles
    print(f"\n[5/5] Building and evaluating ensembles...")

    ensemble_results = []

    # 1. Simple mean
    ens_mean = ensemble_simple_mean(oof_predictions)
    metrics = evaluate_predictions(y, ens_mean, markets, tickers)
    metrics["ensemble"] = "simple_mean"
    ensemble_results.append(metrics)
    print(f"\n  Simple Mean:      AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} "
          f"top10Avg={metrics['top10%_avg']:.3%}")

    # 2. Weighted mean by CV AUC
    ens_weighted = ensemble_weighted_mean(oof_predictions, cv_aucs)
    metrics = evaluate_predictions(y, ens_weighted, markets, tickers)
    metrics["ensemble"] = "weighted_mean"
    ensemble_results.append(metrics)
    print(f"  Weighted Mean:    AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} "
          f"top10Avg={metrics['top10%_avg']:.3%}")

    # 3. Rank average
    ens_rank = ensemble_rank_average(oof_predictions)
    metrics = evaluate_predictions(y, ens_rank, markets, tickers)
    metrics["ensemble"] = "rank_average"
    ensemble_results.append(metrics)
    print(f"  Rank Average:     AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} "
          f"top10Avg={metrics['top10%_avg']:.3%}")

    # 4. Stacked ensemble (Ridge meta-learner)
    ens_stacked = ensemble_stacked(X, y, oof_predictions, markets)
    metrics = evaluate_predictions(y, ens_stacked, markets, tickers)
    metrics["ensemble"] = "stacked_ridge"
    ensemble_results.append(metrics)
    print(f"  Stacked Ridge:    AUC={metrics['auc']:.4f} top10WR={metrics['top10%_wr']:.1%} "
          f"top10Avg={metrics['top10%_avg']:.3%}")

    # Find best ensemble
    best_ensemble = max(ensemble_results, key=lambda r: r["auc"])
    print(f"\n" + "=" * 70)
    print(f"BEST ENSEMBLE: {best_ensemble['ensemble']}")
    print(f"=" * 70)
    print(f"  AUC:        {best_ensemble['auc']:.4f}")
    print(f"  Top 10% WR: {best_ensemble['top10%_wr']:.1%} (+")
    print(f"  Top 10% Avg Return: {best_ensemble['top10%_avg']:.3%}")
    print(f"  Top 20% WR: {best_ensemble['top20%_wr']:.1%}")
    print(f"  Top 20% Avg Return: {best_ensemble['top20%_avg']:.3%}")
    if 'us_auc' in best_ensemble:
        print(f"  US AUC:     {best_ensemble['us_auc']:.4f}")
    if 'india_auc' in best_ensemble:
        print(f"  India AUC:  {best_ensemble['india_auc']:.4f}")

    # Compare to single best
    single_best = max(individual_results, key=lambda r: r["auc"])
    print(f"\nvs Single Best Model:")
    print(f"  Single Best AUC: {single_best['auc']:.4f}")
    print(f"  Ensemble AUC:    {best_ensemble['auc']:.4f}")
    print(f"  Improvement:     {best_ensemble['auc'] - single_best['auc']:+.4f}")

    # Save results
    data_dir = Path(__file__).parent / "training_data_v4"
    results = {
        "individual_models": individual_results,
        "ensemble_methods": ensemble_results,
        "best_ensemble": best_ensemble["ensemble"],
        "cv_aucs": cv_aucs,
        "configs": [c["hyperparams"] for c in top5],
    }

    with open(data_dir / "ensemble_top5_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {data_dir / 'ensemble_top5_results.json'}")

    # Save production ensemble (train best method on full data)
    print(f"\n[6/6] Training production ensemble on full data...")

    # Retrain full models and compute ensemble on full data
    full_predictions = []
    for i, cfg_result in enumerate(top5):
        cfg = cfg_result["hyperparams"].copy()
        cfg["feature_subset"] = cfg_result["feature_names"]
        cfg["seed"] = 42 + i

        if cfg.get("feature_subset"):
            use_cols = [c for c in feature_names if c in cfg["feature_subset"]]
        else:
            use_cols = feature_names
        X_sub = X[use_cols].copy()

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
            random_state=cfg.get("seed", 42),
            n_jobs=4,
        )
        model.fit(X_sub, y)
        preds = model.predict(X_sub)
        full_predictions.append(preds)

    # Compute best ensemble on full predictions
    if best_ensemble["ensemble"] == "simple_mean":
        final_preds = ensemble_simple_mean(full_predictions)
    elif best_ensemble["ensemble"] == "weighted_mean":
        final_preds = ensemble_weighted_mean(full_predictions, cv_aucs)
    elif best_ensemble["ensemble"] == "rank_average":
        final_preds = ensemble_rank_average(full_predictions)
    elif best_ensemble["ensemble"] == "stacked_ridge":
        meta_X = np.column_stack(full_predictions)
        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_X, y)
        final_preds = ridge.predict(meta_X)
    else:
        final_preds = ensemble_simple_mean(full_predictions)

    # Save production model
    with open(data_dir / "model_v5_ensemble.pkl", "wb") as f:
        pickle.dump({
            "models": models_full,
            "feature_cols": feature_cols_list,
            "cv_aucs": cv_aucs,
            "ensemble_method": best_ensemble["ensemble"],
            "metrics": best_ensemble,
        }, f)
    print(f"  Saved production ensemble to {data_dir / 'model_v5_ensemble.pkl'}")

    print("\n" + "=" * 70)
    print("ENSEMBLE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
