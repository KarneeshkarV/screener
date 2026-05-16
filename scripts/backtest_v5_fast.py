"""Fast backtest v5 model with pre-computed features."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

from screener.ml_signal_v5 import V5FeatureExtractor


def main() -> None:
    data_dir = Path(__file__).parent / "training_data_v4"
    cache_path = data_dir / "v5_features.pkl"

    # Load trades
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

    @dataclass
    class SimpleTrade:
        ticker: str; market: str; strategy: str; rank: int
        signal_date: date; entry_date: date; entry_price: float
        exit_date: date; exit_price: float; exit_reason: ExitReason
        shares: float; entry_cost: float; exit_value: float
        pnl: float; return_pct: float; dividend_income: float

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

    # Pre-compute or load features
    if cache_path.exists():
        print("Loading cached features...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        features_cache = cache["features"]
    else:
        print("Pre-computing features...")
        extractor = V5FeatureExtractor()
        features_cache = {}
        for sym, bars in bars_lookup.items():
            if bars is None or bars.empty:
                continue
            features_cache[sym] = extractor.extract(bars)
        with open(cache_path, "wb") as f:
            pickle.dump({"features": features_cache}, f)
        print("Features cached.")

    # Build feature matrix
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
    feature_names = V5FeatureExtractor.FEATURE_COLUMNS
    X = X[feature_names].fillna(0.0)
    y_arr = np.array(y)
    dates_arr = pd.to_datetime(dates)
    markets_arr = np.array(markets)

    monthly_periods = pd.date_range(
        start=dates_arr.min() + pd.DateOffset(months=6),
        end=dates_arr.max(),
        freq="MS",
    )

    for window in [3, 6, 12]:
        print("=" * 70)
        print(f"V5 WALK-FORWARD — {window}-MONTH ROLLING WINDOW")
        print("=" * 70)

        all_preds = np.full(len(y_arr), np.nan)

        for i in range(len(monthly_periods)):
            test_start = monthly_periods[i]
            test_end = (
                monthly_periods[i + 1] - pd.Timedelta(days=1)
                if i + 1 < len(monthly_periods)
                else dates_arr.max()
            )
            train_end = test_start - pd.Timedelta(days=1)
            train_start = test_start - pd.DateOffset(months=window)

            train_mask = (dates_arr >= train_start) & (dates_arr <= train_end)
            test_mask = (dates_arr >= test_start) & (dates_arr <= test_end)

            if train_mask.sum() < 30 or test_mask.sum() < 3:
                continue

            reg = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                random_state=42,
            )
            reg.fit(X[train_mask], y_arr[train_mask])
            all_preds[np.where(test_mask)[0]] = reg.predict(X[test_mask])

        predicted_mask = ~np.isnan(all_preds)
        if predicted_mask.sum() < 100:
            print("Not enough predictions.")
            continue

        y_test = y_arr[predicted_mask]
        y_pred = all_preds[predicted_mask]
        m_test = markets_arr[predicted_mask]
        t_test = [valid_trades[i] for i in np.where(predicted_mask)[0]]

        auc = roc_auc_score((y_test > 0).astype(int), y_pred)
        baseline_wr = (y_test > 0).mean()
        baseline_avg = y_test.mean()
        baseline_pnl = sum(t.pnl for t in t_test)

        print(f"\nPredicted trades: {len(y_test)}")
        print(f"Baseline — WR: {baseline_wr:.1%} | Avg: {baseline_avg:.3%} | P&L: ${baseline_pnl:,.2f}")
        print(f"Direction AUC: {auc:.4f}")

        sorted_idx = np.argsort(y_pred)[::-1]
        for pct in [0.1, 0.2, 0.3]:
            n = max(1, int(len(y_test) * pct))
            sel_idx = sorted_idx[:n]
            sel_rets = y_test[sel_idx]
            sel_trades = [t_test[j] for j in sel_idx]
            sel_wr = (sel_rets > 0).mean()
            sel_avg = sel_rets.mean()
            sel_pnl = sum(t.pnl for t in sel_trades)
            print(f"\n  Top {pct:.0%} ({n} trades):")
            print(f"    WR: {sel_wr:.1%} | Avg: {sel_avg:.3%} | P&L: ${sel_pnl:,.2f}")
            print(f"    Δ WR: {sel_wr - baseline_wr:+.1%} | Δ Avg: {sel_avg - baseline_avg:+.3%} | Δ P&L: {sel_pnl - baseline_pnl:+.2f}")

        print("\n  By market:")
        for mkt in ["us", "india"]:
            mask = m_test == mkt
            if mask.sum() < 30:
                continue
            y_m = y_test[mask]
            y_p = y_pred[mask]
            auc_m = roc_auc_score((y_m > 0).astype(int), y_p) if len(set(y_m > 0)) > 1 else float("nan")
            print(f"    {mkt.upper()}: n={mask.sum()}, AUC={auc_m:.4f}, WR={(y_m>0).mean():.1%}")
        print()


if __name__ == "__main__":
    main()
