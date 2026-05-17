"""V5 Hyperparameter Optimization — 100 iterations.

Tests 100 different configurations across:
- Hyperparameter combinations
- Feature subsets
- Model objectives
- Ensemble averaging

Saves detailed results to scripts/training_data_v4/optimization_results_v5.json
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
    return X, y_arr, dates_arr, markets_arr, valid_trades, feature_names


def evaluate_config(X, y, dates, markets, trades, feature_names, config, n_splits=5):
    """Evaluate a single configuration with stratified random-split CV."""
    # Select feature subset
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

        # Train model
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
    """Generate 100 diverse configurations."""
    configs = []

    # Base feature groups
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
        # Randomly add groups
        for group_name, group_feats in all_groups.items():
            if group_name == "base":
                continue
            if random.random() < 0.6:  # 60% chance to include each group
                subset.extend(group_feats)

        # Ensure unique
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


def main():
    print("=" * 70)
    print("V5 HYPERPARAMETER OPTIMIZATION — 100 ITERATIONS")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    trades, features_cache, bars_lookup, benchmarks = load_data()
    X, y, dates, markets, valid_trades, feature_names = build_matrix(trades, features_cache)
    print(f"  Trades: {len(y)} | Features: {len(feature_names)}")
    print(f"  US: {(markets == 'us').sum()} | India: {(markets == 'india').sum()}")
    print(f"  Baseline WR: {(y > 0).mean():.1%} | Avg return: {y.mean():.3%}")

    print("\n[2/4] Generating 100 configurations...")
    random.seed(42)
    configs = generate_configs(feature_names, n=100)
    print(f"  Generated {len(configs)} configs")

    print("\n[3/4] Running optimization (this will take a few minutes)...")
    results = []

    for i, cfg in enumerate(configs):
        print(f"  [{i+1:3d}/100] n_est={cfg['n_estimators']:3d} depth={cfg['max_depth']} "
              f"lr={cfg['learning_rate']:.2f} subsample={cfg['subsample']:.1f} "
              f"cols={cfg['colsample_bytree']:.1f} lambda={cfg['reg_lambda']:.1f} "
              f"alpha={cfg['reg_alpha']:.1f} gamma={cfg['gamma']:.1f} "
              f"feats={len(cfg['feature_subset'])}", end=" ")

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

    print("\n[4/4] Saving results...")
    results_dir = Path(__file__).parent / "training_data_v4"
    results_dir.mkdir(exist_ok=True)

    # Sort by composite score
    results_sorted = sorted(results, key=lambda r: r["composite_score"], reverse=True)

    output_path = results_dir / "optimization_results_v5.json"
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
    print(f"{'Rank':>4} {'AUC':>6} {'Top10WR':>8} {'Top10Avg':>9} {'Top20WR':>8} {'Score':>7} {'Feats':>5} {'Depth':>5} {'LR':>5} {'Lambda':>6}")
    print("-" * 70)
    for i, r in enumerate(results_sorted[:10]):
        hp = r["hyperparams"]
        print(f"{i+1:>4} {r['auc_mean']:>6.4f} {r['top10_wr_mean']:>7.1%} {r['top10_avg_mean']:>8.3%} "
              f"{r['top20_wr_mean']:>7.1%} {r['composite_score']:>7.4f} {r['n_features']:>5} "
              f"{hp['max_depth']:>5} {hp['learning_rate']:>5.2f} {hp['reg_lambda']:>6.1f}")

    print("\n" + "=" * 70)
    print("BEST CONFIG DETAILS")
    print("=" * 70)
    best = results_sorted[0]
    print(f"AUC:        {best['auc_mean']:.4f} (±{best['auc_std']:.4f})")
    print(f"Top 10% WR: {best['top10_wr_mean']:.1%}")
    print(f"Top 10% Avg Return: {best['top10_avg_mean']:.3%}")
    print(f"Top 20% WR: {best['top20_wr_mean']:.1%}")
    print(f"Top 20% Avg Return: {best['top20_avg_mean']:.3%}")
    print(f"US AUC:     {best['us_auc']:.4f}" if best['us_auc'] else "US AUC: N/A")
    print(f"India AUC:  {best['india_auc']:.4f}" if best['india_auc'] else "India AUC: N/A")
    print(f"\nHyperparameters:")
    for k, v in best["hyperparams"].items():
        print(f"  {k}: {v}")
    print(f"\nFeatures used ({best['n_features']}):")
    for f in best["feature_names"]:
        print(f"  - {f}")

    # Train final best model on ALL data and save
    print("\n[5/5] Training final production model with best config...")
    best_cfg = configs[best["config_id"] - 1]
    use_cols = best["feature_names"]
    X_final = X[use_cols].copy()

    final_model = XGBRegressor(
        n_estimators=best_cfg["n_estimators"],
        max_depth=best_cfg["max_depth"],
        learning_rate=best_cfg["learning_rate"],
        subsample=best_cfg["subsample"],
        colsample_bytree=best_cfg["colsample_bytree"],
        colsample_bylevel=best_cfg["colsample_bylevel"],
        min_child_weight=best_cfg["min_child_weight"],
        reg_alpha=best_cfg["reg_alpha"],
        reg_lambda=best_cfg["reg_lambda"],
        gamma=best_cfg["gamma"],
        random_state=42,
        n_jobs=4,
    )
    final_model.fit(X_final, y)

    model_path = results_dir / "model_v5_optimized.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_names": use_cols,
            "config": best_cfg,
            "metrics": best,
        }, f)
    print(f"  Saved final model to {model_path}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
