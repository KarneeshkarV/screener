"""CLI commands for ML signal confidence."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.models import Trade
from screener.ml_signal import (
    BreakoutFeatureExtractor,
    EnsembleConfidence,
    MissingMLDependencyError,
    SignalConfidenceModel,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _trade_from_dict(d: dict) -> Trade:
    """Build a Trade from a partial dict (tolerates missing fields)."""
    defaults = {
        "rank": 0,
        "entry_date": d.get("signal_date", "1970-01-01"),
        "entry_price": 0.0,
        "exit_date": d.get("signal_date", "1970-01-01"),
        "exit_price": 0.0,
        "exit_reason": "exit_expr",
        "shares": 0.0,
        "entry_cost": 0.0,
        "exit_value": 0.0,
        "pnl": 0.0,
        "return_pct": 0.0,
        "dividend_income": 0.0,
    }
    merged = {**defaults, **d}
    for date_key in ("signal_date", "entry_date", "exit_date"):
        val = merged.get(date_key)
        if isinstance(val, str):
            merged[date_key] = date.fromisoformat(val)
    return Trade(**merged)


def _load_bars(bars_data: dict) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym, records in bars_data.get("bars", {}).items():
        if not records:
            out[sym] = pd.DataFrame()
            continue
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        out[sym] = df
    return out


@click.command(name="train-model")
@click.option(
    "--trades",
    "trades_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trades JSON.",
)
@click.option(
    "--bars",
    "bars_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to bars JSON (dict of symbol -> OHLCV list).",
)
@click.option(
    "--benchmark",
    "benchmark_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to benchmark bars JSON.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to save trained model.",
)
def train_model(
    trades_path: Path,
    bars_path: Path,
    benchmark_path: Path | None,
    output_path: Path,
) -> None:
    """Train an XGBoost confidence model on historical backtest trades."""
    console = Console()
    try:
        trades_data = _load_json(trades_path)
        bars_data = _load_json(bars_path)
    except Exception as exc:
        raise click.UsageError(f"Failed to load input JSON: {exc}") from exc

    trades = [_trade_from_dict(t) for t in trades_data.get("trades", [])]
    if not trades:
        raise click.UsageError("No trades found in JSON.")

    bars_by_symbol = _load_bars(bars_data)

    benchmark_bars = pd.DataFrame()
    if benchmark_path:
        bench_data = _load_json(benchmark_path)
        benchmark_bars = _load_bars(bench_data).get("benchmark", pd.DataFrame())

    try:
        model = SignalConfidenceModel()
        model.train(trades, bars_by_symbol)
    except MissingMLDependencyError as exc:
        raise click.UsageError(str(exc)) from exc
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    model.save(output_path)
    console.print(f"[green]Model saved to {output_path}[/green]")
    if model.metrics:
        table = Table(title="Training Metrics")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for k, v in model.metrics.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)


@click.command(name="predict")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input screen JSON.",
)
@click.option(
    "--bars",
    "bars_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to bars JSON.",
)
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output JSON path.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Minimum confidence to include.",
)
def predict_cmd(
    input_path: Path,
    bars_path: Path,
    model_path: Path,
    output_path: Path,
    threshold: float | None,
) -> None:
    """Append confidence scores to screen output JSON."""
    console = Console()
    try:
        data = _load_json(input_path)
        bars_data = _load_json(bars_path)
    except Exception as exc:
        raise click.UsageError(f"Failed to load input: {exc}") from exc

    try:
        model = SignalConfidenceModel.load(model_path)
    except MissingMLDependencyError as exc:
        raise click.UsageError(str(exc)) from exc

    extractor = BreakoutFeatureExtractor()
    bars_by_symbol = _load_bars(bars_data)

    # Normalize input: accept list of rows or rs-breakout-style dict
    if isinstance(data, list):
        rows = data
    else:
        rows = []
        for bucket in ("full", "relaxed"):
            rows.extend(data.get(bucket, []))

    results: list[dict] = []
    for row in rows:
        sym = row.get("symbol") or row.get("ticker") or row.get("name")
        as_of = row.get("date")
        if not sym or not as_of:
            row = {**row, "ml_confidence": None}
            results.append(row)
            continue
        bars = bars_by_symbol.get(sym)
        if bars is None or bars.empty:
            row = {**row, "ml_confidence": None}
            results.append(row)
            continue
        features = extractor.extract(bars)
        if features.empty:
            row = {**row, "ml_confidence": None}
            results.append(row)
            continue
        as_of_ts = pd.Timestamp(as_of)
        mask = features.index <= as_of_ts
        if not mask.any():
            row = {**row, "ml_confidence": None}
            results.append(row)
            continue
        feat_row = features.loc[mask].iloc[[-1]]
        prob = float(model.predict(feat_row)[0])
        row = {**row, "ml_confidence": round(prob, 4)}
        if threshold is None or prob >= threshold:
            results.append(row)

    output_data = {
        "ml_confidence": True,
        "model_path": str(model_path),
        "results": results,
    }
    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    console.print(f"[green]Wrote {len(results)} rows to {output_path}[/green]")


@click.command(name="feature-importance")
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model.",
)
def feature_importance(model_path: Path) -> None:
    """Show feature importance for a trained model."""
    console = Console()
    try:
        model = SignalConfidenceModel.load(model_path)
    except MissingMLDependencyError as exc:
        raise click.UsageError(str(exc)) from exc

    df = model.feature_importance()
    table = Table(title="Feature Importance")
    table.add_column("Feature")
    table.add_column("Importance", justify="right")
    for _, row in df.iterrows():
        table.add_row(row["feature"], f"{row['importance']:.4f}")
    console.print(table)
