"""Render backtest results: summary metrics table + per-trade ledger."""

from __future__ import annotations

import pandas as pd
from rich.console import Console, JustifyMethod
from rich.panel import Panel
from rich.table import Table

from screener.backtester.models import BacktestResult
from screener.format import fmt_pct


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
    "unique_tickers": "Unique Tickers",
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


_REGIME_LABELS = ("bull", "pullback", "bear", "unknown")


def _format_metric(key: str, value) -> str:
    if isinstance(value, float):
        if key in _PCT_METRICS:
            return fmt_pct(value * 100)
        return f"{value:+.3f}"
    return str(value)


def _print_regime_metrics(metrics: dict) -> None:
    """Render per-regime trade stats when regime_* keys are present."""
    rows = [label for label in _REGIME_LABELS if f"regime_{label}_trades" in metrics]
    if not rows:
        return
    table = Table(title="Trades by Entry Regime", show_header=True, header_style="bold")
    table.add_column("Regime")
    table.add_column("Trades", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Return", justify="right")
    for label in rows:
        table.add_row(
            label,
            str(metrics[f"regime_{label}_trades"]),
            f"{metrics[f'regime_{label}_win_rate'] * 100:.1f}%",
            f"{metrics[f'regime_{label}_avg_return'] * 100:+.2f}%",
        )
    console.print(table)


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
    _print_regime_metrics(result.metrics)

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
        justify: JustifyMethod = "right" if col not in {"Ticker", "Reason"} else "left"
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
                "ticker",
                "rank",
                "signal_date",
                "entry_date",
                "entry_price",
                "exit_date",
                "exit_price",
                "exit_reason",
                "shares",
                "entry_cost",
                "exit_value",
                "pnl",
                "return_pct",
            ]
        )
    rows = [t.model_dump() for t in sorted(result.trades, key=lambda tr: tr.rank)]
    return pd.DataFrame(rows)


def print_ledger_csv(result: BacktestResult) -> None:
    df = trades_dataframe(result)
    print(df.to_csv(index=False), end="")
