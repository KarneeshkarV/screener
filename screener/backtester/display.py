"""Render backtest results: shared metrics plus a per-trade ledger."""

from __future__ import annotations

import pandas as pd
from rich.console import Console, JustifyMethod
from rich.panel import Panel
from rich.table import Table

from screener import agentio
from screener.backtester.metrics import ResultView, result_view
from screener.backtester.models import BacktestResult

console = Console()


_ATTRIBUTION_SIDE = 5


def _performance_table(view: ResultView) -> Table:
    """Draw the common result view as a Rich table."""
    table = Table(title="Performance", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for row in view:
        table.add_row(row.label, row.formatted)
    return table


def print_reinvestment_comparison(
    fixed_slots: BacktestResult,
    reinvested_slots: BacktestResult,
) -> None:
    """Print fixed-slot and reinvested-slot results from one signal panel."""
    table = Table(
        title="Fixed slots vs reinvested slots",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric")
    table.add_column("Fixed slots", justify="right")
    table.add_column("Reinvested slots", justify="right")
    rows = (
        ("Final Equity", "final_equity", "money"),
        ("Total Return", "total_return", "pct"),
        ("CAGR", "cagr", "pct"),
        ("Max Drawdown", "max_drawdown", "pct"),
        ("Sharpe", "sharpe", "float"),
    )
    for label, key, kind in rows:
        fixed = float(fixed_slots.metrics.get(key, 0.0))
        reinvested = float(reinvested_slots.metrics.get(key, 0.0))
        if kind == "money":
            fixed_text = f"{fixed:,.2f}"
            reinvested_text = f"{reinvested:,.2f}"
        elif kind == "pct":
            fixed_text = f"{fixed:+.2%}"
            reinvested_text = f"{reinvested:+.2%}"
        else:
            fixed_text = f"{fixed:+.3f}"
            reinvested_text = f"{reinvested:+.3f}"
        table.add_row(label, fixed_text, reinvested_text)
    agentio.render_table(table, agentio.get_console(), detail="full")


def _ledger_table(result: BacktestResult) -> Table:
    """Build the trade ledger independently of the result metric view."""
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
    for trade in sorted(result.trades, key=lambda item: item.rank):
        ledger.add_row(
            str(trade.rank),
            trade.ticker,
            str(trade.signal_date),
            str(trade.entry_date),
            f"{trade.entry_price:.2f}",
            str(trade.exit_date),
            f"{trade.exit_price:.2f}",
            trade.exit_reason,
            f"{trade.return_pct * 100:+.2f}%",
            f"{trade.pnl:+.2f}",
        )
    return ledger


def _print_ticker_attribution(trades: pd.DataFrame, out) -> None:
    """Print bounded per-ticker PnL, trade count, and win rate."""
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
    """Render the same shared metric rows in bounded, plain agent output."""
    cfg = result.config
    sizing_rule = getattr(cfg, "sizing_rule", "equal_slot")
    out = agentio.get_console()
    out.print(
        f"backtest {cfg.market} as-of={cfg.as_of} hold={cfg.hold} "
        f"top={cfg.top} sizing={sizing_rule} benchmark={cfg.benchmark}"
    )
    for warning in result.warnings:
        out.print(f"warning: {warning}")

    # Metrics are a small, fixed result view. Show all rows so no metric is
    # hidden from an agent digest, while the large trade ledger remains capped.
    agentio.render_table(
        _performance_table(result_view(result.metrics)), out, detail="full"
    )

    if not result.trades:
        out.print("trades: none")
        return

    trades = trades_dataframe(result)
    path = agentio.spill(trades, f"backtest-{cfg.market}")
    _print_ticker_attribution(trades, out)
    out.print(f"trades: {len(trades)} rows -> {path}")
    agentio.render_table(_ledger_table(result), out)


def print_backtest(result: BacktestResult) -> None:
    if agentio.is_agent_mode():
        _print_backtest_agent(result)
        return

    cfg = result.config
    sizing_rule = getattr(cfg, "sizing_rule", "equal_slot")
    console.print(
        Panel.fit(
            f"[bold]Backtest[/bold] [cyan]{cfg.market.upper()}[/cyan]  "
            f"as-of [yellow]{cfg.as_of}[/yellow]  hold=[green]{cfg.hold}[/green]  "
            f"top=[green]{cfg.top}[/green]  sizing=[green]{sizing_rule}[/green]  "
            f"benchmark=[magenta]{cfg.benchmark}[/magenta]"
        )
    )
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")

    console.print(_performance_table(result_view(result.metrics)))
    if not result.trades:
        console.print("[dim]No trades.[/dim]")
        return
    console.print(_ledger_table(result))


_SERIALIZED_EQUITY_TRADE_COLUMNS = [
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
    "dividend_income",
]


def trades_dataframe(result: BacktestResult) -> pd.DataFrame:
    if not result.trades:
        # Preserve the legacy no-trade ledger shape, which predates dividends.
        return pd.DataFrame(columns=_SERIALIZED_EQUITY_TRADE_COLUMNS[:-1])
    rows = [
        trade.model_dump()
        for trade in sorted(result.trades, key=lambda item: item.rank)
    ]
    # Trade lifecycle inheritance orders base fields first. Reindex explicitly
    # so established CSV and Lab JSON field ordering remains byte-for-byte stable.
    return pd.DataFrame(rows).reindex(columns=_SERIALIZED_EQUITY_TRADE_COLUMNS)


def print_ledger_csv(result: BacktestResult) -> None:
    df = trades_dataframe(result)
    print(df.to_csv(index=False), end="")
