"""Render backtest results: summary metrics table + per-trade ledger."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rich.console import Console, JustifyMethod
from rich.panel import Panel
from rich.table import Table

from screener import agentio
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


#: Metrics worth spending digest tokens on. The full table has 23 rows;
#: these are the ones a decision actually turns on, and they cost one line
#: per four instead of one line plus box drawing per metric.
_AGENT_HEADLINE_METRICS = (
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "hit_rate",
    "profit_factor",
    "expectancy",
    "trade_count",
    "avg_trade_return",
    "exposure",
    "alpha_annual",
    "beta",
    "benchmark_return",
    "unique_tickers",
)


def _agent_cell(column: str, value: Any) -> str:
    """Round trade-ledger cells for the digest.

    ``trades_dataframe`` carries full float precision for the CSV; inlining
    ``173.85844036702034`` costs tokens and tells an agent nothing that
    ``173.86`` does not.
    """
    if column.endswith("_price"):
        return f"{float(value):.2f}"
    if column == "return_pct":
        return f"{float(value) * 100:+.2f}%"
    return str(value)


#: Tickers shown per side in the attribution block when the universe is
#: large. Keeps the digest bounded: a 500-name universe still costs 10 lines.
_ATTRIBUTION_SIDE = 5


def _print_ticker_attribution(trades: pd.DataFrame, out) -> None:
    """Per-ticker PnL, trade count, and win rate.

    Measured need: asked which ticker lost the most, agents reading only the
    metrics digest -- or even the full inline ledger, which carries no ``pnl``
    column -- answered by eyeballing the worst *single* trade and named the
    wrong ticker, quoting invented totals. Aggregating 69 rows in-head is not
    something to ask of a model when three lines of arithmetic here settle it.
    """
    if "pnl" not in trades.columns or trades.empty:
        return
    grouped = (
        trades.assign(_win=trades["return_pct"] > 0)
        .groupby("ticker")
        .agg(trades=("pnl", "size"), pnl=("pnl", "sum"), win=("_win", "mean"))
        .sort_values("pnl")
    )
    if len(grouped) > _ATTRIBUTION_SIDE * 2:
        shown = pd.concat(
            [grouped.head(_ATTRIBUTION_SIDE), grouped.tail(_ATTRIBUTION_SIDE)]
        )
        note = f" (worst/best {_ATTRIBUTION_SIDE} of {len(grouped)})"
    else:
        shown, note = grouped, ""
    out.print(f"pnl_by_ticker{note}: ticker trades win% pnl")
    for ticker, row in shown.iterrows():
        out.print(
            f"  {ticker} {int(row['trades'])} {row['win'] * 100:.1f} {row['pnl']:+,.2f}"
        )


def _print_backtest_agent(result: BacktestResult) -> None:
    """Bounded digest for agent mode: headline metrics, regimes, spill path.

    Size is independent of trade count -- the ledger goes to a CSV and only
    its path (plus a sample at higher detail levels) reaches stdout.
    """
    cfg = result.config
    out = agentio.get_console()
    detail = agentio.detail_level()

    out.print(
        f"backtest {cfg.market} as-of={cfg.as_of} hold={cfg.hold} "
        f"top={cfg.top} benchmark={cfg.benchmark}"
    )
    for warning in result.warnings:
        out.print(f"warning: {warning}")

    pairs = [
        (key, _format_metric(key, result.metrics[key]))
        for key in _AGENT_HEADLINE_METRICS
        if key in result.metrics
    ]
    for line in agentio.kv_line(pairs):
        out.print(line)

    if "total_fees" in result.metrics:
        out.print(
            f"costs total={float(result.metrics['total_fees']):,.2f} "
            f"pct_capital={result.metrics.get('fees_pct_capital', 0.0) * 100:.3f}% "
            f"pct_net_pnl={result.metrics.get('fees_pct_net_pnl', 0.0) * 100:.2f}%"
        )

    regimes = [
        label for label in _REGIME_LABELS if f"regime_{label}_trades" in result.metrics
    ]
    if regimes:
        out.print("regime trades win% avg%")
        for label in regimes:
            out.print(
                f"{label} {result.metrics[f'regime_{label}_trades']} "
                f"{result.metrics[f'regime_{label}_win_rate'] * 100:.1f} "
                f"{result.metrics[f'regime_{label}_avg_return'] * 100:+.2f}"
            )

    if not result.trades:
        out.print("trades: none")
        return

    trades = trades_dataframe(result)
    path = agentio.spill(trades, f"backtest-{cfg.market}")
    _print_ticker_attribution(trades, out)
    limit = (
        len(trades)
        if detail == "full"
        else (0 if detail == "summary" else agentio.HEAD_ROWS)
    )
    out.print(f"trades: {len(trades)} rows -> {path}")
    if limit:
        columns = [
            "ticker",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "return_pct",
            "exit_reason",
        ]
        columns = [col for col in columns if col in trades.columns]
        head = trades.head(limit)
        out.print("  " + " ".join(columns))
        for _, row in head.iterrows():
            out.print("  " + " ".join(_agent_cell(col, row[col]) for col in columns))
        if len(trades) > limit:
            out.print(f"  ... {len(trades) - limit} more rows in the CSV above")


def print_backtest(result: BacktestResult) -> None:
    if agentio.is_agent_mode():
        _print_backtest_agent(result)
        return

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
