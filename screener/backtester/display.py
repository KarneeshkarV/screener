"""Render backtest results: summary metrics table + per-trade ledger."""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from screener.backtester.models import BacktestResult


console = Console()


_METRIC_LABELS = {
    "total_return": "Total Return",
    "invested_return": "Invested Return",
    "cagr": "CAGR",
    "vol_annual": "Volatility (ann.)",
    "sharpe": "Sharpe",
    "max_drawdown": "Max Drawdown",
    "hit_rate": "Hit Rate",
    "alpha_annual": "Alpha (ann.)",
    "beta": "Beta",
    "exposure": "Avg Exposure",
    "benchmark_return": "Benchmark Return",
    "trade_count": "Trades",
}

_PCT_METRICS = {
    "total_return",
    "invested_return",
    "cagr",
    "vol_annual",
    "max_drawdown",
    "hit_rate",
    "alpha_annual",
    "exposure",
    "benchmark_return",
}


def _format_metric(key: str, value) -> str:
    if isinstance(value, float):
        if key in _PCT_METRICS:
            return f"{value * 100:+.2f}%"
        return f"{value:+.3f}"
    return str(value)


def print_backtest(result: BacktestResult) -> None:
    cfg = result.config
    console.print(
        Panel.fit(
            f"[bold]Backtest[/bold] [cyan]{cfg.market.upper()}[/cyan]  "
            f"as-of [yellow]{cfg.as_of}[/yellow]  hold=[green]{cfg.hold}[/green]  "
            f"top=[green]{cfg.top}[/green]  benchmark=[magenta]{cfg.benchmark}[/magenta]"
        )
    )

    for w in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")

    metrics_table = Table(title="Performance", show_header=True, header_style="bold")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")
    for key, label in _METRIC_LABELS.items():
        if key in result.metrics:
            metrics_table.add_row(label, _format_metric(key, result.metrics[key]))
    console.print(metrics_table)

    if not result.trades:
        console.print("[dim]No trades.[/dim]")
        return

    ledger = Table(title="Trade Ledger", show_header=True, header_style="bold")
    for col in [
        "Rank",
        "Ticker",
        "Signal",
        "Entry",
        "Entry $",
        "Exit",
        "Exit $",
        "Reason",
        "Return",
        "PnL",
    ]:
        justify = "right" if col not in {"Ticker", "Reason"} else "left"
        ledger.add_column(col, justify=justify)
    for t in sorted(result.trades, key=lambda tr: tr.rank):
        ledger.add_row(
            str(t.rank),
            t.ticker,
            str(t.signal_date),
            str(t.entry_date),
            f"{t.entry_price:.2f}",
            str(t.exit_date),
            f"{t.exit_price:.2f}",
            t.exit_reason,
            f"{t.return_pct * 100:+.2f}%",
            f"{t.pnl:+.2f}",
        )
    console.print(ledger)


def trades_dataframe(result: BacktestResult) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame(
            columns=[
                "ticker", "rank", "signal_date", "entry_date", "entry_price",
                "exit_date", "exit_price", "exit_reason", "shares",
                "entry_cost", "exit_value", "pnl", "return_pct",
            ]
        )
    rows = [asdict(t) for t in sorted(result.trades, key=lambda tr: tr.rank)]
    return pd.DataFrame(rows)


def print_ledger_csv(result: BacktestResult) -> None:
    df = trades_dataframe(result)
    print(df.to_csv(index=False), end="")
