"""V5 XGBRanker — cross-sectional ranking with walk-forward validation.

Reframes the problem from "predict return" to "rank signals within each date".
Robust to regime shifts because we're always comparing signals at the same date.

Walk-forward: retrain on rolling N months, predict next month.
Headline metric: post-cost Sharpe of top-K portfolio.
"""
from __future__ import annotations

import json
import pickle
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRanker

from screener.ml_signal_v5 import V5FeatureExtractor, FEATURE_NEUTRAL_VALUES

warnings.filterwarnings("ignore")


def load_data(data_dir: Path):
    """Load trades, bars, and pre-computed features."""
    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)
    with open(data_dir / "bars.json") as f:
        bars_json = json.load(f)

    trades = trades_data["trades"]

    bars_by_symbol = {}
    for key, records in bars_json.get("bars", {}).items():
        if not isinstance(records, list):
            continue
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        if ":" in key:
            _, sym = key.split(":", 1)
        else:
            sym = key
        bars_by_symbol[sym] = df

    return trades, bars_by_symbol


def build_features_for_trades(trades, bars_by_symbol):
    """Extract features for all trades, group by signal date."""
    extractor = V5FeatureExtractor()
    feature_names = V5FeatureExtractor.FEATURE_COLUMNS

    rows = []
    for t in trades:
        bars = bars_by_symbol.get(t["ticker"])
        if bars is None or bars.empty:
            continue
        feat = extractor.extract(bars)
        if feat.empty:
            continue
        ts = pd.Timestamp(t["signal_date"])
        mask = feat.index <= ts
        if not mask.any():
            continue
        row = feat.loc[mask].iloc[[-1]].copy()
        if row.isna().all().all():
            continue
        for col in feature_names:
            if col in FEATURE_NEUTRAL_VALUES:
                row[col] = row[col].fillna(FEATURE_NEUTRAL_VALUES[col])
            else:
                row[col] = row[col].fillna(0.0)

        rows.append({
            "ticker": t["ticker"],
            "market": t.get("market", "us"),
            "strategy": t["strategy"],
            "signal_date": ts,
            "return_pct": t["return_pct"],
            **{k: float(row[k].iloc[0]) for k in feature_names},
        })

    df = pd.DataFrame(rows)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["ym"] = df["signal_date"].dt.to_period("M")
    df["is_win"] = (df["return_pct"] > 0).astype(int)
    df["rank"] = df.groupby("signal_date")["return_pct"].rank(ascending=False, method="first")
    return df


def train_ranker_walk_forward(df: pd.DataFrame, feature_names: list[str], train_months: int = 12):
    """Walk-forward: train on rolling N months, predict next month.

    Returns list of (month, predictions_df, metrics) tuples.
    """
    months = sorted(df["ym"].unique())
    results = []

    for i in range(train_months, len(months)):
        train_end = months[i - 1]
        test_month = months[i]

        train_df = df[df["ym"] <= train_end]
        test_df = df[df["ym"] == test_month]

        if len(train_df) < 200 or len(test_df) < 20:
            continue

        # Build groups for ranker (group by signal_date)
        train_groups = train_df.groupby("signal_date").size().to_numpy()
        test_groups = test_df.groupby("signal_date").size().to_numpy()

        X_train = train_df[feature_names].values
        y_train = train_df["rank"].values  # lower rank = better (1st place = best return)
        # Invert rank so higher = better for ranking (rank:pairwise wants higher = better)
        y_train = train_df.groupby("signal_date")["return_pct"].transform("max") - train_df["return_pct"].values

        X_test = test_df[feature_names].values

        model = XGBRanker(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=3.0,
            objective="rank:pairwise",
            random_state=42,
            n_jobs=4,
        )

        try:
            model.fit(
                X_train, y_train,
                group=train_groups,
                eval_set=[(X_test, test_df.groupby("signal_date")["return_pct"].transform("max") - test_df["return_pct"].values)],
                eval_group=[test_groups],
                verbose=False,
            )
        except Exception as e:
            print(f"  Month {test_month}: fit failed: {e}")
            continue

        preds = model.predict(X_test)
        test_df = test_df.copy()
        test_df["pred_score"] = preds

        # Compute metrics
        labels = test_df["is_win"].values
        auc = roc_auc_score(labels, preds) if len(set(labels)) > 1 else float("nan")

        # Top 10% within each test date
        top10_by_date = []
        for date_key, group in test_df.groupby("signal_date"):
            if len(group) < 5:
                continue
            n_top = max(1, int(len(group) * 0.1))
            top = group.nlargest(n_top, "pred_score")
            top10_by_date.append(top)

        if top10_by_date:
            top10 = pd.concat(top10_by_date)
            top10_wr = top10["is_win"].mean()
            top10_avg = top10["return_pct"].mean()
        else:
            top10_wr = float("nan")
            top10_avg = float("nan")

        # Post-cost Sharpe: assume 10bps commission + 5bps slippage per trade
        post_cost = test_df["return_pct"] - 0.0015  # 15bps round-trip
        sharpe = post_cost.mean() / (post_cost.std() + 1e-9) * np.sqrt(12)  # monthly Sharpe

        # Kelly-optimal fraction for top 20% signals
        test_df_sorted = test_df.sort_values("pred_score", ascending=False)
        n_kelly = max(1, int(len(test_df_sorted) * 0.2))
        kelly_subset = test_df_sorted.iloc[:n_kelly]
        wins = kelly_subset[kelly_subset["return_pct"] > 0]
        losses = kelly_subset[kelly_subset["return_pct"] <= 0]
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(kelly_subset)
            avg_win = wins["return_pct"].mean()
            avg_loss = abs(losses["return_pct"].mean())
            if avg_loss > 0:
                kelly_f = win_rate / avg_loss - (1 - win_rate) / avg_win
                kelly_f = max(0, min(kelly_f, 0.25))  # cap at 25%
            else:
                kelly_f = 0.0
        else:
            kelly_f = 0.0

        kelly_pnl = kelly_subset["return_pct"].sum() * kelly_f if kelly_f > 0 else 0.0

        metrics = {
            "month": str(test_month),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "auc": float(auc),
            "top10_wr": float(top10_wr),
            "top10_avg": float(top10_avg),
            "baseline_wr": float(labels.mean()),
            "baseline_avg": float(test_df["return_pct"].mean()),
            "sharpe_monthly": float(sharpe),
            "kelly_fraction": float(kelly_f),
            "kelly_pnl": float(kelly_pnl),
        }

        results.append((test_month, test_df, metrics))
        print(f"  {test_month}: train={len(train_df)} test={len(test_df)} AUC={auc:.4f} top10WR={top10_wr:.1%} KellyPnL={kelly_pnl:.3%}")

    return results


def summarize(results: list):
    if not results:
        print("No walk-forward results.")
        return

    aucs = [r[2]["auc"] for r in results if not np.isnan(r[2]["auc"])]
    top10_wrs = [r[2]["top10_wr"] for r in results if not np.isnan(r[2]["top10_wr"])]
    kelly_pnls = [r[2]["kelly_pnl"] for r in results]
    sharpes = [r[2]["sharpe_monthly"] for r in results]

    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*60}")
    print(f"Months evaluated: {len(results)}")
    print(f"AUC:  mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}  min={np.min(aucs):.4f}  max={np.max(aucs):.4f}")
    print(f"Top10% WR: mean={np.mean(top10_wrs):.1%}  std={np.std(top10_wrs):.1%}")
    print(f"Kelly PnL (sum): {np.sum(kelly_pnls):.3%}")
    print(f"Monthly Sharpe: mean={np.mean(sharpes):.3f}  std={np.std(sharpes):.3f}")
    print(f"{'='*60}")


def main():
    data_dir = Path(__file__).parent / "training_data_v4"
    print("Loading data...")
    trades, bars_by_symbol = load_data(data_dir)
    print(f"Loaded {len(trades)} trades, {len(bars_by_symbol)} symbols")

    print("Building features...")
    df = build_features_for_trades(trades, bars_by_symbol)
    print(f"Feature matrix: {len(df)} rows, {len(df.columns)} cols")
    print(f"Date range: {df['signal_date'].min().date()} to {df['signal_date'].max().date()}")
    print(f"Unique months: {df['ym'].nunique()}")

    feature_names = list(V5FeatureExtractor.FEATURE_COLUMNS)
    print(f"\nTraining XGBRanker (walk-forward, 12-month window)...")
    results = train_ranker_walk_forward(df, feature_names, train_months=12)

    summarize(results)

    # Save results
    out_dir = data_dir
    out_dir.mkdir(exist_ok=True)

    # Save combined predictions
    all_preds = pd.concat([r[1] for r in results]) if results else pd.DataFrame()
    if not all_preds.empty:
        all_preds.to_parquet(out_dir / "ranker_walkforward_preds.parquet")
        print(f"\nSaved predictions to {out_dir / 'ranker_walkforward_preds.parquet'}")

    # Save metrics
    metrics_list = [r[2] for r in results]
    with open(out_dir / "ranker_walkforward_metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2)

    # Train final model on ALL data and save
    print("\nTraining final model on all data...")
    final_groups = df.groupby("signal_date").size().to_numpy()
    y_rank = df.groupby("signal_date")["return_pct"].transform("max") - df["return_pct"].values

    final_model = XGBRanker(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=3.0,
        objective="rank:pairwise",
        random_state=42,
        n_jobs=4,
    )
    final_model.fit(df[feature_names].values, y_rank, group=final_groups, verbose=False)

    with open(out_dir / "model_v5_ranker.pkl", "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_names": feature_names,
            "metrics_summary": {
                "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
                "auc_std": float(np.std(aucs)) if aucs else float("nan"),
                "top10_wr_mean": float(np.mean(top10_wrs)) if top10_wrs else float("nan"),
                "kelly_pnl_sum": float(np.sum(kelly_pnls)),
            },
        }, f)
    print(f"Saved final model to {out_dir / 'model_v5_ranker.pkl'}")


if __name__ == "__main__":
    main()
