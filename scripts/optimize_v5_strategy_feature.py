"""V5 Hyperparameter Optimization with Strategy ID as categorical feature.

Adds one-hot encoded strategy columns and target-encoded strategy win rate
to the 27 base features. Runs 100-iteration hyperparameter optimization,
builds ensemble of top 5 configs, evaluates, and saves production model.

Target: AUC > 0.650
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
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


STRATEGIES = ["ema_trend", "ema_vol", "golden_cross", "golden_cross_vol", "rsi2_rev", "breakout"]


def load_data():
    data_dir = Path(__file__).parent / "training_data_v4"
    cache_path = data_dir / "v5_features.pkl"

    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)

    trades_list = trades_data["trades"]

    # Pre-compute target encodings (global historical win rate per strategy)
    strategy_stats = {}
    for s in STRATEGIES:
        rets = [t["return_pct"] for t in trades_list if t.get("strategy", "") == s]
        strategy_stats[s] = {
            "win_rate": float(np.mean(np.array(rets) > 0)) if rets else 0.5,
            "mean_return": float(np.mean(rets)) if rets else 0.0,
        }

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

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    features_cache = cache["features"]

    return trades, features_cache, strategy_stats


BASE_FEATURE_NAMES = [
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

# Strategy one-hot columns
OHE_COLS = [f"strat_{s}" for s in STRATEGIES]
# Target encoding columns
TE_COLS = ["strat_win_rate", "strat_mean_return"]

ALL_FEATURE_NAMES = BASE_FEATURE_NAMES + OHE_COLS + TE_COLS


def build_matrix(trades, features_cache, strategy_stats):
    """Build feature matrix and target vector with strategy features."""
    X_rows = []
    y = []
    dates = []
    markets = []
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

        # Add one-hot strategy columns
        for s in STRATEGIES:
            row[f"strat_{s}"] = 1.0 if trade.strategy == s else 0.0

        # Add target-encoded strategy columns
        stats = strategy_stats.get(trade.strategy, {"win_rate": 0.5, "mean_return": 0.0})
        row["strat_win_rate"] = stats["win_rate"]
        row["strat_mean_return"] = stats["mean_return"]

        X_rows.append(row)
        y.append(trade.return_pct)
        dates.append(trade.signal_date)
        markets.append(trade.market)
        valid_trades.append(trade)

    X = pd.concat(X_rows, ignore_index=True)
    X = X[ALL_FEATURE_NAMES].fillna(0.0)
    y_arr = np.array(y)
    dates_arr = pd.to_datetime(dates)
    markets_arr = np.array(markets)
    return X, y_arr, dates_arr, markets_arr, valid_trades, ALL_FEATURE_NAMES


def evaluate_config(X, y, dates, markets, trades, feature_names, config, n_splits=5):
    """Evaluate a single configuration with stratified random-split CV."""
    if config.get("feature_subset"):
        use_cols = [c for c in feature_names if c in config["feature_subset"]]
    else:
        use_cols = feature_names
    X_sub = X[use_cols].copy()

    # Stratified split by market and win/loss
    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.get("seed", 42))

    fold_aucs = []
    fold_top10_wr = []
    fold_top10_avg = []
    fold_top20_wr = []
    fold_top20_avg = []
    fold_top10_pnl = []
    fold_top20_pnl = []
    fold_us_auc = []
    fold_india_auc = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_sub, stratify)):
        X_train, X_test = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        m_test = markets[test_idx]
        t_test = [trades[i] for i in test_idx]

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
            random_state=config.get("seed", 42) + fold,
            n_jobs=4,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # AUC
        try:
            auc = roc_auc_score((y_test > 0).astype(int), y_pred)
        except ValueError:
            auc = 0.5
        fold_aucs.append(auc)

        # Top 10%
        sorted_idx = np.argsort(y_pred)[::-1]
        n10 = max(1, int(len(y_test) * 0.1))
        sel10 = sorted_idx[:n10]
        wr10 = (y_test[sel10] > 0).mean()
        avg10 = y_test[sel10].mean()
        pnl10 = sum(t_test[j].pnl for j in sel10)
        fold_top10_wr.append(wr10)
        fold_top10_avg.append(avg10)
        fold_top10_pnl.append(pnl10)

        # Top 20%
        n20 = max(1, int(len(y_test) * 0.2))
        sel20 = sorted_idx[:n20]
        wr20 = (y_test[sel20] > 0).mean()
        avg20 = y_test[sel20].mean()
        pnl20 = sum(t_test[j].pnl for j in sel20)
        fold_top20_wr.append(wr20)
        fold_top20_avg.append(avg20)
        fold_top20_pnl.append(pnl20)

        # By market
        for mkt, lst in [("us", fold_us_auc), ("india", fold_india_auc)]:
            mask = m_test == mkt
            if mask.sum() < 20 or len(set((y_test[mask] > 0).astype(int))) < 2:
                continue
            auc_m = roc_auc_score((y_test[mask] > 0).astype(int), y_pred[mask])
            lst.append(auc_m)

    return {
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "top10_wr_mean": float(np.mean(fold_top10_wr)),
        "top10_avg_mean": float(np.mean(fold_top10_avg)),
        "top10_pnl_mean": float(np.mean(fold_top10_pnl)),
        "top20_wr_mean": float(np.mean(fold_top20_wr)),
        "top20_avg_mean": float(np.mean(fold_top20_avg)),
        "top20_pnl_mean": float(np.mean(fold_top20_pnl)),
        "us_auc_mean": float(np.mean(fold_us_auc)) if fold_us_auc else None,
        "india_auc_mean": float(np.mean(fold_india_auc)) if fold_india_auc else None,
        "n_features": len(use_cols),
        "features": use_cols,
    }


def generate_configs(feature_names, n=100):
    """Generate 100 diverse configurations with strategy feature options."""
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

    # Hyperparameter grids
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
        # Pick feature subset
        subset = list(base_features)  # Always include base
        for group_name, group_feats in all_groups.items():
            if group_name == "base":
                continue
            if random.random() < 0.6:
                subset.extend(group_feats)

        # Strategy feature options
        strategy_mode = random.choice([
            "none",           # no strategy features
            "onehot",         # only one-hot
            "target",         # only target encoding
            "onehot_target",  # both
        ])

        if strategy_mode in ("onehot", "onehot_target"):
            subset.extend(OHE_COLS)
        if strategy_mode in ("target", "onehot_target"):
            subset.extend(TE_COLS)

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
            "strategy_mode": strategy_mode,
        }
        configs.append(cfg)

    return configs


def main():
    print("=" * 70)
    print("V5 STRATEGY FEATURE OPTIMIZATION — 100 ITERATIONS")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    trades, features_cache, strategy_stats = load_data()
    X, y, dates, markets, valid_trades, feature_names = build_matrix(trades, features_cache, strategy_stats)
    print(f"  Trades: {len(y)} | Features: {len(feature_names)}")
    print(f"  US: {(markets == 'us').sum()} | India: {(markets == 'india').sum()}")
    print(f"  Baseline WR: {(y > 0).mean():.1%} | Avg return: {y.mean():.3%}")

    print("\n[2/5] Generating 100 configurations...")
    random.seed(42)
    configs = generate_configs(feature_names, n=100)
    print(f"  Generated {len(configs)} configs")

    print("\n[3/5] Running optimization (this will take a few minutes)...")
    results = []

    for i, cfg in enumerate(configs):
        print(f"  [{i+1:3d}/100] n_est={cfg['n_estimators']:3d} depth={cfg['max_depth']} "
              f"lr={cfg['learning_rate']:.2f} subsample={cfg['subsample']:.1f} "
              f"cols={cfg['colsample_bytree']:.1f} lambda={cfg['reg_lambda']:.1f} "
              f"alpha={cfg['reg_alpha']:.1f} gamma={cfg['gamma']:.1f} "
              f"feats={len(cfg['feature_subset'])} mode={cfg['strategy_mode']}", end=" ")

        metrics = evaluate_config(X, y, dates, markets, valid_trades, feature_names, cfg, n_splits=5)

        score = metrics["auc_mean"] + (metrics["top10_wr_mean"] - (y > 0).mean()) * 2
        print(f"→ AUC={metrics['auc_mean']:.4f} top10WR={metrics['top10_wr_mean']:.1%} "
              f"top10Avg={metrics['top10_avg_mean']:.3%} score={score:.4f}")

        results.append({
            "config_id": cfg["id"],
            "hyperparams": {k: v for k, v in cfg.items() if k not in ("id", "seed", "feature_subset")},
            "n_features": metrics["n_features"],
            "feature_names": metrics["features"],
            "auc_mean": metrics["auc_mean"],
            "auc_std": metrics["auc_std"],
            "top10_wr_mean": metrics["top10_wr_mean"],
            "top10_avg_mean": metrics["top10_avg_mean"],
            "top10_pnl_mean": metrics["top10_pnl_mean"],
            "top20_wr_mean": metrics["top20_wr_mean"],
            "top20_avg_mean": metrics["top20_avg_mean"],
            "top20_pnl_mean": metrics["top20_pnl_mean"],
            "us_auc": metrics["us_auc_mean"],
            "india_auc": metrics["india_auc_mean"],
            "composite_score": score,
        })

    print("\n[4/5] Saving optimization results...")
    results_dir = Path(__file__).parent / "training_data_v4"
    results_dir.mkdir(exist_ok=True)

    results_sorted = sorted(results, key=lambda r: r["composite_score"], reverse=True)

    output_path = results_dir / "optimization_results_v5_strategy.json"
    with open(output_path, "w") as f:
        json.dump({
            "n_iterations": len(configs),
            "baseline_wr": float((y > 0).mean()),
            "baseline_avg_return": float(y.mean()),
            "best_config": results_sorted[0],
            "top_10_configs": results_sorted[:10],
            "all_results": results_sorted,
        }, f, indent=2, default=str)
    print(f"  Saved to {output_path}")

    # Print top 10 summary
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)
    print(f"{'Rank':>4} {'AUC':>6} {'Top10WR':>8} {'Top10Avg':>9} {'Top20WR':>8} {'Score':>7} {'Feats':>5} {'Depth':>5} {'LR':>5} {'Lambda':>6} {'Mode':>15}")
    print("-" * 95)
    for i, r in enumerate(results_sorted[:10]):
        hp = r["hyperparams"]
        print(f"{i+1:>4} {r['auc_mean']:>6.4f} {r['top10_wr_mean']:>7.1%} {r['top10_avg_mean']:>8.3%} "
              f"{r['top20_wr_mean']:>7.1%} {r['composite_score']:>7.4f} {r['n_features']:>5} "
              f"{hp['max_depth']:>5} {hp['learning_rate']:>5.2f} {hp['reg_lambda']:>6.1f} {hp.get('strategy_mode', '?'):>15}")

    # ---- ENSEMBLE TOP 5 ----
    print("\n[5/5] Building ensemble of top 5 configs...")
    top5 = results_sorted[:5]

    labels = (y > 0).astype(int)
    market_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + market_codes
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_all = []
    cv_aucs = []
    models_full = []
    feature_cols_list = []

    for i, cfg_result in enumerate(top5):
        cfg = cfg_result["hyperparams"].copy()
        cfg["feature_subset"] = cfg_result["feature_names"]
        cfg["seed"] = 42 + i

        if cfg.get("feature_subset"):
            use_cols = [c for c in feature_names if c in cfg["feature_subset"]]
        else:
            use_cols = feature_names
        X_sub = X[use_cols].copy()

        # OOF predictions
        oof_preds = np.full(len(y), np.nan)
        fold_aucs = []
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
        oof_preds_all.append(oof_preds)
        cv_aucs.append(oof_auc)
        print(f"  Model {i+1} OOF AUC={oof_auc:.4f} (folds: {[f'{a:.3f}' for a in fold_aucs]})")

        # Full model for production
        full_model = XGBRegressor(
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
        full_model.fit(X_sub, y)
        models_full.append(full_model)
        feature_cols_list.append(use_cols)

    # Evaluate ensemble methods
    def eval_preds(y_pred):
        auc = roc_auc_score((y > 0).astype(int), y_pred)
        sorted_idx = np.argsort(y_pred)[::-1]
        n10 = max(1, int(len(y) * 0.1))
        sel10 = sorted_idx[:n10]
        wr10 = (y[sel10] > 0).mean()
        avg10 = y[sel10].mean()
        n20 = max(1, int(len(y) * 0.2))
        sel20 = sorted_idx[:n20]
        wr20 = (y[sel20] > 0).mean()
        avg20 = y[sel20].mean()
        return auc, wr10, avg10, wr20, avg20

    # Simple mean
    ens_mean = np.mean(oof_preds_all, axis=0)
    auc_m, wr10_m, avg10_m, wr20_m, avg20_m = eval_preds(ens_mean)

    # Weighted mean by CV AUC
    weights = np.array(cv_aucs)
    weights = weights / weights.sum()
    ens_wgt = np.average(oof_preds_all, axis=0, weights=weights)
    auc_w, wr10_w, avg10_w, wr20_w, avg20_w = eval_preds(ens_wgt)

    from scipy import stats as sp_stats
    ranks = np.array([sp_stats.rankdata(p) for p in oof_preds_all])
    ens_rank = np.mean(ranks, axis=0) / len(y)
    auc_r, wr10_r, avg10_r, wr20_r, avg20_r = eval_preds(ens_rank)

    print(f"\n  Ensemble Evaluations (OOF):")
    print(f"    Simple Mean:   AUC={auc_m:.4f} top10WR={wr10_m:.1%} top10Avg={avg10_m:.3%}")
    print(f"    Weighted Mean: AUC={auc_w:.4f} top10WR={wr10_w:.1%} top10Avg={avg10_w:.3%}")
    print(f"    Rank Average:  AUC={auc_r:.4f} top10WR={wr10_r:.1%} top10Avg={avg10_r:.3%}")

    best_auc = max(auc_m, auc_w, auc_r)
    if best_auc == auc_m:
        best_method = "simple_mean"
        best_ens = ens_mean
    elif best_auc == auc_w:
        best_method = "weighted_mean"
        best_ens = ens_wgt
    else:
        best_method = "rank_average"
        best_ens = ens_rank

    print(f"\n  BEST ENSEMBLE: {best_method} AUC={best_auc:.4f}")

    # Train production ensemble on full data
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
        preds = models_full[i].predict(X_sub)
        full_predictions.append(preds)

    if best_method == "simple_mean":
        final_preds = np.mean(full_predictions, axis=0)
    elif best_method == "weighted_mean":
        final_preds = np.average(full_predictions, axis=0, weights=np.array(cv_aucs) / sum(cv_aucs))
    else:
        ranks = np.array([sp_stats.rankdata(p) for p in full_predictions])
        final_preds = np.mean(ranks, axis=0) / len(y)

    # Final metrics on full data (in-sample, for reference)
    final_auc = roc_auc_score((y > 0).astype(int), final_preds)
    sorted_idx = np.argsort(final_preds)[::-1]
    n10 = max(1, int(len(y) * 0.1))
    sel10 = sorted_idx[:n10]
    final_wr10 = (y[sel10] > 0).mean()
    final_avg10 = y[sel10].mean()
    n20 = max(1, int(len(y) * 0.2))
    sel20 = sorted_idx[:n20]
    final_wr20 = (y[sel20] > 0).mean()
    final_avg20 = y[sel20].mean()

    print(f"\n  Full-data ensemble metrics (in-sample):")
    print(f"    AUC={final_auc:.4f} top10WR={final_wr10:.1%} top10Avg={final_avg10:.3%}")

    # Save production model
    model_path = results_dir / "model_v5_strategy_feature.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "models": models_full,
            "feature_cols": feature_cols_list,
            "cv_aucs": cv_aucs,
            "ensemble_method": best_method,
            "oof_auc": best_auc,
            "full_auc": final_auc,
            "top10_wr": final_wr10,
            "top10_avg": final_avg10,
            "top20_wr": final_wr20,
            "top20_avg": final_avg20,
            "strategy_stats": strategy_stats,
            "all_feature_names": feature_names,
        }, f)
    print(f"\n  Saved production ensemble to {model_path}")

    # Save strategy feature results report data
    report_data = {
        "best_config": results_sorted[0],
        "top_10_configs": results_sorted[:10],
        "ensemble_method": best_method,
        "oof_auc": best_auc,
        "full_auc": final_auc,
        "top10_wr": final_wr10,
        "top10_avg": final_avg10,
        "top20_wr": final_wr20,
        "top20_avg": final_avg20,
        "cv_aucs": cv_aucs,
        "strategy_win_rates": {s: strategy_stats[s]["win_rate"] for s in STRATEGIES},
        "strategy_mean_returns": {s: strategy_stats[s]["mean_return"] for s in STRATEGIES},
    }

    with open(results_dir / "strategy_feature_results.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("STRATEGY FEATURE OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
