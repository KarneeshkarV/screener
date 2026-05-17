"""Fast XGBRanker baseline using pre-computed v5 features."""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRanker

warnings.filterwarnings("ignore")


def main():
    data_dir = Path("scripts/training_data_v4")
    print("Loading pre-computed features...")
    with open(data_dir / "v5_features.pkl", "rb") as f:
        feat_data = pickle.load(f)
    features_by_ticker = feat_data["features"]

    with open(data_dir / "trades.json") as f:
        trades = json.load(f)["trades"]

    print(f"Loaded {len(features_by_ticker)} tickers, {len(trades)} trades")

    # Build rows by looking up pre-computed features
    rows = []
    feature_cols = list(features_by_ticker[list(features_by_ticker.keys())[0]].columns)

    for t in trades:
        ticker = t["ticker"]
        feat_df = features_by_ticker.get(ticker)
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
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df.sort_values("signal_date").reset_index(drop=True)
    n = len(df)

    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Ranking target: integer relevance labels (required by XGBRanker)
    # Within each date, rank returns and cap at max 5 labels
    def _to_relevance(x):
        n = len(x)
        if n <= 1:
            return pd.Series([1] * n, index=x.index)
        # Rank ascending (1=worst, n=best), invert, scale to 1-5
        ranks = x.rank(method="first", ascending=True)
        best_rank = n - ranks + 1  # best return = n
        scaled = (best_rank - 1) / max(n - 1, 1) * 4 + 1
        return pd.Series(scaled.round().astype(int).clip(1, 5).values, index=x.index)

    for d in [train_df, val_df, test_df]:
        d["y_rank"] = d.groupby("signal_date")["return_pct"].transform(_to_relevance)

    feature_names = feature_cols

    # XGBRanker
    model = XGBRanker(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0,
        objective="rank:pairwise", random_state=42, n_jobs=4,
    )

    train_groups = train_df.groupby("signal_date").size().to_numpy()
    val_groups = val_df.groupby("signal_date").size().to_numpy()

    print("Training XGBRanker...")
    model.fit(
        train_df[feature_names].values, train_df["y_rank"].values,
        group=train_groups,
        eval_set=[(val_df[feature_names].values, val_df["y_rank"].values)],
        eval_group=[val_groups],
        verbose=False,
    )

    # Test
    preds = model.predict(test_df[feature_names].values)
    test_df["pred_score"] = preds

    auc = roc_auc_score(test_df["is_win"].values, preds)

    sorted_idx = np.argsort(preds)[::-1]
    n10 = max(1, int(len(test_df) * 0.1))
    top10_wr = test_df["is_win"].iloc[sorted_idx[:n10]].mean()
    top10_avg = test_df["return_pct"].iloc[sorted_idx[:n10]].mean()

    # Post-cost Sharpe (15bps round-trip per trade)
    post_cost = test_df["return_pct"] - 0.0015
    monthly_sharpe = post_cost.mean() / (post_cost.std() + 1e-9) * np.sqrt(12)

    # Kelly on top 20%
    test_sorted = test_df.sort_values("pred_score", ascending=False)
    n_kelly = max(1, int(len(test_sorted) * 0.2))
    kelly_subset = test_sorted.iloc[:n_kelly]
    wins = kelly_subset[kelly_subset["return_pct"] > 0]
    losses = kelly_subset[kelly_subset["return_pct"] <= 0]
    if len(wins) > 0 and len(losses) > 0:
        wr = len(wins) / len(kelly_subset)
        avg_win = wins["return_pct"].mean()
        avg_loss = abs(losses["return_pct"].mean())
        kelly_f = wr / avg_loss - (1 - wr) / avg_win if avg_loss > 0 and avg_win > 0 else 0.0
        kelly_f = max(0, min(kelly_f, 0.25))
    else:
        kelly_f = 0.0
    kelly_pnl = kelly_subset["return_pct"].sum() * kelly_f

    print(f"\n{'='*60}")
    print("XGBRanker RESULTS (honest test set)")
    print(f"{'='*60}")
    print(f"AUC:          {auc:.4f}")
    print(f"Top 10% WR:   {top10_wr:.1%}")
    print(f"Top 10% Avg:  {top10_avg:.3%}")
    print(f"Baseline WR:  {test_df['is_win'].mean():.1%}")
    print(f"Monthly Sharpe: {monthly_sharpe:.3f}")
    print(f"Kelly f:      {kelly_f:.3f}")
    print(f"Kelly PnL:    {kelly_pnl:.3%}")
    print(f"{'='*60}")

    # Save
    with open(data_dir / "model_v5_ranker.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    print(f"Saved model to {data_dir / 'model_v5_ranker.pkl'}")


if __name__ == "__main__":
    main()
