"""Train ML Signal v5 model on the full dataset."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import pandas as pd

from screener.ml_signal_v5 import V5SignalModel


def main() -> None:
    data_dir = Path(__file__).parent / "training_data_v4"

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
        market: str
        strategy: str
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
            market=t.get("market", "us"),
            strategy=t.get("strategy", ""),
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

    # Fix bar lookup: bars keys are "market:symbol", trades have "ticker"
    bars_lookup = {}
    for key, df in bars_by_tv.items():
        if ":" in key:
            _, sym = key.split(":", 1)
        else:
            sym = key
        bars_lookup[sym] = df

    print("=" * 60)
    print("V5 MODEL — FULL DATASET")
    print("=" * 60)

    model = V5SignalModel(rolling_window_months=0)
    model.train(trades=trades, bars_by_symbol=bars_lookup, benchmark_bars=benchmark_by_market)

    print(f"\nModel metrics:")
    for k, v in (model.metrics or {}).items():
        print(f"  {k}: {v}")

    print("\nFeature importance (top 15):")
    fi = model.feature_importance()
    for _, row in fi.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    model.save(data_dir / "model_v5.pkl")
    print(f"\nSaved to {data_dir / 'model_v5.pkl'}")

    # Train rolling-window models (regime-adaptive)
    for window in [6, 12, 18]:
        print(f"\n{'=' * 60}")
        print(f"V5 MODEL — ROLLING {window} MONTHS")
        print("=" * 60)
        rw_model = V5SignalModel(rolling_window_months=window)
        try:
            rw_model.train(trades=trades, bars_by_symbol=bars_lookup, benchmark_bars=benchmark_by_market)
            print(f"Metrics: {rw_model.metrics}")
            rw_model.save(data_dir / f"model_v5_rw{window}.pkl")
            print(f"Saved to {data_dir / f'model_v5_rw{window}.pkl'}")
        except ValueError as exc:
            print(f"Skipped: {exc}")

    # Train market-specific models
    for market in ["us", "india"]:
        m_trades = [t for t in trades if t.market == market]
        if len(m_trades) < 200:
            continue
        print(f"\n{'=' * 60}")
        print(f"V5 MODEL — {market.upper()} SPECIFIC")
        print("=" * 60)
        m_model = V5SignalModel(rolling_window_months=0)
        m_model.train(trades=m_trades, bars_by_symbol=bars_lookup, benchmark_bars=benchmark_by_market.get(market))
        print(f"Metrics: {m_model.metrics}")
        m_model.save(data_dir / f"model_v5_{market}.pkl")
        print(f"Saved to {data_dir / f'model_v5_{market}.pkl'}")


if __name__ == "__main__":
    main()
