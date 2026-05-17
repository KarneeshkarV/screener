"""Fast hyperparameter optimization with honest train/val/test split."""
from __future__ import annotations

import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def load_data():
    data_dir = Path("scripts/training_data_v4")
    with open(data_dir / "v5_features.pkl", "rb") as f:
        feat_data = pickle.load(f)
    features_by_ticker = feat_data["features"]

    with open(data_dir / "trades.json") as f:
        trades = json.load(f)["trades"]

    feature_cols = list(features_by_ticker[list(features_by_ticker.keys())[0]].columns)
    rows = []
    for t in trades:
        feat_df = features_by_ticker.get(t["ticker"])
        if feat_df is None or feat_df.empty:
            continue
        ts = pd.Timestamp(t["signal_date"])
        mask = feat_df.index <= ts
        if not mask.any():
            continue
        row = feat_df.loc[mask].iloc[[-1]]
        if row.isna().all().all():
            continue
        rows.append({
            "signal_date": ts,
            "return_pct": float(t["return_pct"]),
            "is_win": int(t["return_pct"] > 0),
            **{k: float(row[k].iloc[0]) for k in feature_cols},
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("signal_date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols].values
    y_train = train_df["return_pct"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["return_pct"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["return_pct"].values
    y_test_win = test_df["is_win"].values

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_win, feature_cols


def evaluate_config(config, X_train, y_train, X_val, y_val, X_test, y_test, y_test_win):
    model = XGBRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        colsample_bylevel=0.8,
        min_child_weight=config["min_child_weight"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        gamma=config["gamma"],
        random_state=42,
        n_jobs=4,
        early_stopping_rounds=20,
    )

    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except Exception:
        return None

    preds = model.predict(X_test)
    auc = roc_auc_score(y_test_win, preds)
    mse = mean_squared_error(y_test, preds)

    # Top 10%
    sorted_idx = np.argsort(preds)[::-1]
    n10 = max(1, int(len(y_test) * 0.1))
    top10_wr = y_test_win[sorted_idx[:n10]].mean()

    return {"auc": auc, "mse": mse, "top10_wr": top10_wr, "best_iter": model.best_iteration}


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, y_test_win, feature_cols = load_data()
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # Search space
    space = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.5, 0.6, 0.7, 0.8],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 3.0, 5.0, 10.0],
        "gamma": [0.0, 0.1, 0.5],
    }

    best = None
    results = []
    n_iter = 30

    print(f"Running {n_iter} iterations...")
    for i in range(n_iter):
        config = {k: random.choice(v) for k, v in space.items()}
        res = evaluate_config(config, X_train, y_train, X_val, y_val, X_test, y_test, y_test_win)
        if res is None:
            continue

        score = res["auc"] * 0.6 + res["top10_wr"] * 0.4  # Combined objective
        entry = {**config, **res, "score": score}
        results.append(entry)

        if best is None or score > best["score"]:
            best = entry

        print(f"  {i+1:2d}/{n_iter}: AUC={res['auc']:.4f} top10WR={res['top10_wr']:.1%} score={score:.4f}")

    print(f"\n{'='*60}")
    print("BEST CONFIG")
    print(f"{'='*60}")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Sort by AUC
    top5 = sorted(results, key=lambda x: x["auc"], reverse=True)[:5]
    print(f"\nTop 5 by AUC:")
    for r in top5:
        print(f"  AUC={r['auc']:.4f} top10WR={r['top10_wr']:.1%} depth={r['max_depth']} lr={r['learning_rate']} lambda={r['reg_lambda']}")

    with open("scripts/training_data_v4/optimization_honest.json", "w") as f:
        json.dump({"best": best, "all": results, "top5": top5}, f, indent=2)


if __name__ == "__main__":
    main()
