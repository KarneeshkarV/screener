"""Train enhanced model v2 on the same data as v1 for fair comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from screener.ml_signal_v2 import EnhancedSignalModel


def main() -> None:
    data_dir = Path(__file__).parent / "training_data"

    print("Loading existing training data...")
    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)
    with open(data_dir / "bars.json") as f:
        bars_json = json.load(f)

    trades_list = trades_data["trades"]
    print(f"Loaded {len(trades_list)} trades")

    # Convert bars_json back to DataFrames
    bars_json = bars_json.get("bars", bars_json)
    bars_by_tv = {}
    for sym, records in bars_json.items():
        if not isinstance(records, list):
            continue
        df = pd.DataFrame(records)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        bars_by_tv[sym] = df

    print(f"Loaded {len(bars_by_tv)} symbols")

    # Build Trade objects from dicts
    from dataclasses import dataclass
    from datetime import date
    from enum import Enum

    class ExitReason(Enum):
        HOLD = "hold"
        STOP_LOSS = "stop"
        TAKE_PROFIT = "target"
        TRAILING_STOP = "trail"
        EXIT_SIGNAL = "exit_expr"
        TIME = "time"
        EOD = "eod"

    @dataclass
    class SimpleTrade:
        ticker: str
        rank: int
        signal_date: date
        entry_date: date
        entry_price: float
        exit_date: date
        exit_price: float
        exit_reason: ExitReason
        shares: float
        entry_cost: float
        exit_value: float
        pnl: float
        return_pct: float
        dividend_income: float

    trades = []
    for t in trades_list:
        trades.append(SimpleTrade(
            ticker=t["ticker"],
            rank=t["rank"],
            signal_date=date.fromisoformat(t["signal_date"]),
            entry_date=date.fromisoformat(t["entry_date"]),
            entry_price=t["entry_price"],
            exit_date=date.fromisoformat(t["exit_date"]),
            exit_price=t["exit_price"],
            exit_reason=ExitReason(t["exit_reason"]),
            shares=t["shares"],
            entry_cost=t["entry_cost"],
            exit_value=t["exit_value"],
            pnl=t["pnl"],
            return_pct=t["return_pct"],
            dividend_income=t["dividend_income"],
        ))

    wins = sum(1 for t in trades if t.return_pct > 0)
    print(f"Baseline win rate: {wins}/{len(trades)} = {wins/len(trades):.1%}")

    # Get SPY as benchmark
    bench_df = bars_by_tv.get("SPY")
    all_bars = {sym: df for sym, df in bars_by_tv.items() if sym != "SPY"}

    print("\nTraining enhanced model v2...")
    model = EnhancedSignalModel()
    model.train(
        trades=trades,
        bars_by_symbol=bars_by_tv,
        benchmark_bars=bench_df,
        all_bars=all_bars,
    )

    print(f"\nModel metrics:")
    for k, v in (model.metrics or {}).items():
        print(f"  {k}: {v}")

    print("\nTop 15 features by importance:")
    fi = model.feature_importance().head(15)
    for _, row in fi.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    output_path = data_dir / "model_v2.pkl"
    model.save(output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
