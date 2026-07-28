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
    "starting_equity": "Starting Capital",
    "final_equity": "Final Equity",
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
    "median_trade_return": "Median Trade Return",
    "avg_trade_return": "Avg Trade Return",
    "best_trade_return": "Best Trade",
    "worst_trade_return": "Worst Trade",
    "profit_factor": "Profit Factor",
    "expectancy": "Expectancy",
    "winning_trades": "Winning Trades",
    "losing_trades": "Losing Trades",
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
    "median_trade_return",
    "avg_trade_return",
    "best_trade_return",
    "worst_trade_return",
    "expectancy",
}


_MONEY_METRICS = {"starting_equity", "final_equity"}


_REGIME_LABELS = ("bull", "pullback", "bear", "unknown")


def _format_metric(key: str, value) -> str:
    if isinstance(value, float):
        if key in _MONEY_METRICS:
            return f"{value:,.2f}"
        if key in _PCT_METRICS:
            return fmt_pct(value * 100)
        return f"{value:+.3f}"
    return str(value)


_FEE_COMPONENT_LABELS = {
    "commission": "Commission",
    "brokerage": "Brokerage",
    "stt": "STT",
    "stamp_duty": "Stamp Duty",
    "exchange_txn": "Exchange Txn",
    "sebi": "SEBI",
    "gst": "GST",
    "ipft": "IPFT",
    "sec_fee": "SEC Section 31",
    "taf": "FINRA TAF",
}


def _print_cost_metrics(metrics: dict) -> None:
    """Render the fee-attribution breakdown when cost metrics are present."""
    if "total_fees" not in metrics:
        return
    total = float(metrics["total_fees"])
    components = sorted(
        (
            (key[len("fee_") :], float(value))
            for key, value in metrics.items()
            if key.startswith("fee_") and float(value) != 0.0
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if total == 0.0 and not components:
        return
    table = Table(title="Costs", show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Amount", justify="right")
    table.add_column("% of Total", justify="right")
    for name, amount in components:
        label = _FEE_COMPONENT_LABELS.get(name, name.replace("_", " ").title())
        share = amount / total * 100 if total else 0.0
        table.add_row(label, f"{amount:,.2f}", f"{share:.1f}%")
    table.add_row("[bold]Total Costs[/bold]", f"[bold]{total:,.2f}[/bold]", "100.0%")
    console.print(table)
    console.print(
        f"[dim]Total costs {total:,.2f} = "
        f"{metrics.get('fees_pct_capital', 0.0) * 100:.3f}% of initial capital, "
        f"{metrics.get('fees_pct_net_pnl', 0.0) * 100:.2f}% of net PnL[/dim]"
    )


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
    _print_cost_metrics(result.metrics)
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
