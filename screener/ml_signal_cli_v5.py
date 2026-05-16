"""CLI commands for ML Signal Confidence v5.

Usage:
    python -m screener.ml_signal_cli_v5 train --data-dir scripts/training_data_v4
    python -m screener.ml_signal_cli_v5 predict --model model_v5_us.pkl --ticker AAPL
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

import click
import pandas as pd

from screener.ml_signal_v5 import V5SignalModel, V5FeatureExtractor


@click.group()
def cli():
    """ML Signal Confidence v5 — regression-based expected return prediction."""


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--market", default=None, help="Train market-specific model (us/india)")
@click.option("--window", default=3, help="Rolling training window in months")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def train(data_dir: Path, market: str | None, window: int, output: Path | None) -> None:
    """Train v5 model on historical trades."""
    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)
    with open(data_dir / "bars.json") as f:
        bars_json = json.load(f)

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
    for t in trades_data["trades"]:
        trades.append(SimpleTrade(
            ticker=t["ticker"], market=t.get("market", "us"), strategy=t.get("strategy", ""),
            rank=t["rank"], signal_date=date.fromisoformat(t["signal_date"]),
            entry_date=date.fromisoformat(t["entry_date"]), entry_price=t["entry_price"],
            exit_date=date.fromisoformat(t["exit_date"]), exit_price=t["exit_price"],
            exit_reason=ExitReason(t["exit_reason"]), shares=t["shares"],
            entry_cost=t["entry_cost"], exit_value=t["exit_value"],
            pnl=t["pnl"], return_pct=t["return_pct"], dividend_income=t["dividend_income"],
        ))

    if market:
        trades = [t for t in trades if t.market == market]
        click.echo(f"Training {market.upper()}-specific model on {len(trades)} trades...")
    else:
        click.echo(f"Training global model on {len(trades)} trades...")

    bars_data = bars_json.get("bars", bars_json)
    bars_lookup = {}
    for key, df_raw in bars_data.items():
        if not isinstance(df_raw, list):
            continue
        df = pd.DataFrame(df_raw)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        sym = key.split(":", 1)[1] if ":" in key else key
        bars_lookup[sym] = df

    benchmark_by_market = {}
    for mkt, records in bars_json.get("benchmarks", {}).items():
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        benchmark_by_market[mkt] = df

    model = V5SignalModel(rolling_window_months=window)
    model.train(trades=trades, bars_by_symbol=bars_lookup, benchmark_bars=benchmark_by_market)

    click.echo("\nMetrics:")
    for k, v in (model.metrics or {}).items():
        click.echo(f"  {k}: {v}")

    click.echo("\nFeature importance (top 10):")
    fi = model.feature_importance()
    for _, row in fi.head(10).iterrows():
        click.echo(f"  {row['feature']}: {row['importance']:.4f}")

    out_path = output or data_dir / f"model_v5{'_' + market if market else ''}_rw{window}.pkl"
    model.save(out_path)
    click.echo(f"\nSaved to {out_path}")


@cli.command()
@click.option("--model-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--ticker", required=True)
@click.option("--bars-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--benchmark-file", type=click.Path(path_type=Path), default=None)
def predict(model_path: Path, ticker: str, bars_file: Path, benchmark_file: Path | None) -> None:
    """Predict expected return for a single ticker."""
    model = V5SignalModel.load(model_path)

    bars = pd.read_json(bars_file)
    if "date" in bars.columns:
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.set_index("date")

    bench = None
    if benchmark_file:
        bench = pd.read_json(benchmark_file)
        if "date" in bench.columns:
            bench["date"] = pd.to_datetime(bench["date"])
            bench = bench.set_index("date")

    extractor = V5FeatureExtractor()
    features = extractor.extract(bars, benchmark_bars=bench)
    pred = model.predict(features.iloc[[-1]])[0]
    conf = model.predict_confidence(features.iloc[[-1]])[0]

    click.echo(f"Expected return: {pred:.3%}")
    click.echo(f"Confidence:      {conf:.3f}")


if __name__ == "__main__":
    cli()
