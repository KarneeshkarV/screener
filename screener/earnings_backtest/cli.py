"""CLI entry-point for ``screener earnings-backtest``."""

from __future__ import annotations

import csv
import sys

import click
from rich.console import Console
from rich.table import Table

from screener.earnings_backtest.engine import (
    EarningsTrade,
    compute_backtest_summary,
    run_earnings_backtest,
)
from screener.earnings_backtest.pead import (
    PeadTrade,
    compute_pead_summary,
    run_pead_backtest,
)

STRATEGY_CHOICES = [
    "price_momentum",
    "volume_surge",
    "analyst_sentiment",
    "iv_sentiment",
    "combined_score",
]

console = Console()


@click.command(name="earnings-backtest")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    help="Market: US (S&P 500) or India (Nifty 500).",
)
@click.option(
    "--years",
    type=int,
    default=3,
    show_default=True,
    help="Look-back period for earnings history.",
)
@click.option(
    "--strategy",
    type=click.Choice(STRATEGY_CHOICES),
    default="combined_score",
    show_default=True,
    help="Sentiment strategy to use.",
)
@click.option(
    "--days-before",
    type=int,
    default=1,
    show_default=True,
    help="Days before earnings to enter (1=E-1, 2=E-2).",
)
@click.option(
    "--min-score",
    type=float,
    default=0.55,
    show_default=True,
    help="Minimum strategy score to take a trade (0-1).",
)
@click.option(
    "--commission-bps",
    type=float,
    default=10.0,
    show_default=True,
    help="Round-trip commission in basis points.",
)
@click.option(
    "--slippage-bps",
    type=float,
    default=5.0,
    show_default=True,
    help="Slippage per fill in basis points.",
)
@click.option(
    "--batch-size",
    type=int,
    default=50,
    show_default=True,
    help="Symbols per API batch (controls RAM).",
)
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated ticker list (overrides universe).",
)
@click.option(
    "--csv",
    "output_csv",
    is_flag=True,
    help="Output trade ledger as CSV.",
)
def earnings_backtest(
    market: str,
    years: int,
    strategy: str,
    days_before: int,
    min_score: float,
    commission_bps: float,
    slippage_bps: float,
    batch_size: int,
    tickers: str | None,
    output_csv: bool,
) -> None:
    """Backtest earnings-drift entry (E-1/E-2 → E) with sentiment filters."""
    ticker_list = None
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    with console.status(
        f"[bold green]Running earnings-backtest ({market}, {strategy}, {years}y)…"
    ):
        trades = run_earnings_backtest(
            market=market,
            years=years,
            strategy=strategy,
            days_before=days_before,
            min_score=min_score,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            batch_size=batch_size,
            tickers=ticker_list,
        )

    if not trades:
        console.print(
            "[yellow]No earnings events found for the given parameters.[/yellow]"
        )
        return

    # Summary
    summary = compute_backtest_summary(trades, strategy=strategy)
    taken = [t for t in trades if t.passed_filter]

    if output_csv:
        _print_csv(taken)
        return

    # Rich output
    _print_summary(summary)
    if taken:
        _print_trade_table(taken)


def _print_summary(summary: dict) -> None:
    table = Table(
        title="Earnings-Drift Backtest Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    labels = {
        "total_events": "Total Events Scanned",
        "trades_taken": "Trades Taken",
        "strategy": "Strategy",
        "win_rate": "Win Rate (%)",
        "avg_return_pct": "Avg Return (%)",
        "median_return_pct": "Median Return (%)",
        "total_return_pct": "Cumulative Return (%)",
        "max_winner_pct": "Best Trade (%)",
        "max_loser_pct": "Worst Trade (%)",
        "profit_factor": "Profit Factor",
        "avg_holding_days": "Avg Hold (days)",
        "sharpe_approx": "Sharpe (approx)",
    }
    for key, label in labels.items():
        val = summary.get(key, "")
        if isinstance(val, float):
            val = f"{val:,.4f}"
        table.add_row(label, str(val))

    console.print(table)


def _print_trade_table(trades: list[EarningsTrade]) -> None:
    # Limit display to top 30 by absolute return
    shown = sorted(trades, key=lambda t: t.return_pct, reverse=True)[:30]
    table = Table(
        title=f"Top Trades (showing {len(shown)} of {len(trades)})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="bold")
    table.add_column("Earnings", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("Entry $", justify="right")
    table.add_column("Exit $", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Score", justify="right")

    for t in shown:
        ret_color = "green" if t.return_pct > 0 else "red"
        table.add_row(
            t.ticker,
            str(t.earnings_date),
            str(t.entry_date),
            str(t.exit_date),
            f"{t.entry_price:,.2f}",
            f"{t.exit_price:,.2f}",
            f"[{ret_color}]{t.return_pct:+.2f}%[/{ret_color}]",
            f"{t.score:.3f}",
        )

    console.print(table)


def _print_csv(trades: list[EarningsTrade]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "ticker",
            "earnings_date",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "return_pct",
            "strategy",
            "score",
            "passed_filter",
            "price_momentum_score",
            "volume_surge_score",
            "analyst_sentiment_score",
            "iv_sentiment_score",
        ]
    )
    for t in trades:
        scores = t.details.get("scores", {})
        writer.writerow(
            [
                t.ticker,
                t.earnings_date,
                t.entry_date,
                t.exit_date,
                t.entry_price,
                t.exit_price,
                t.return_pct,
                t.strategy,
                t.score,
                t.passed_filter,
                scores.get("price_momentum", ""),
                scores.get("volume_surge", ""),
                scores.get("analyst_sentiment", ""),
                scores.get("iv_sentiment", ""),
            ]
        )


# ── PEAD (post-earnings-announcement drift) ─────────────────────────────


@click.command(name="earnings-pead")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    help="Market: US (S&P 500) or India (Nifty 500).",
)
@click.option(
    "--years",
    type=int,
    default=3,
    show_default=True,
    help="Look-back period for earnings history.",
)
@click.option(
    "--min-surprise",
    type=float,
    default=5.0,
    show_default=True,
    help="Minimum EPS surprise (%) to take a trade.",
)
@click.option(
    "--hold-days",
    type=int,
    default=40,
    show_default=True,
    help="Trading days to hold after the next-open entry.",
)
@click.option(
    "--commission-bps",
    type=float,
    default=10.0,
    show_default=True,
    help="Round-trip commission in basis points.",
)
@click.option(
    "--slippage-bps",
    type=float,
    default=5.0,
    show_default=True,
    help="Slippage per fill in basis points.",
)
@click.option(
    "--batch-size",
    type=int,
    default=50,
    show_default=True,
    help="Symbols per API batch (controls RAM).",
)
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated ticker list (overrides universe).",
)
@click.option(
    "--csv",
    "output_csv",
    is_flag=True,
    help="Output trade ledger as CSV.",
)
def earnings_pead(
    market: str,
    years: int,
    min_surprise: float,
    hold_days: int,
    commission_bps: float,
    slippage_bps: float,
    batch_size: int,
    tickers: str | None,
    output_csv: bool,
) -> None:
    """Backtest post-earnings-announcement drift (next open → hold N days)."""
    ticker_list = None
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    with console.status(
        f"[bold green]Running PEAD backtest ({market}, surprise≥{min_surprise}%, "
        f"{hold_days}d hold, {years}y)…"
    ):
        trades = run_pead_backtest(
            market=market,
            years=years,
            min_surprise=min_surprise,
            hold_days=hold_days,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            batch_size=batch_size,
            tickers=ticker_list,
        )

    if not trades:
        console.print(
            "[yellow]No qualifying PEAD events found for the given parameters.[/yellow]"
        )
        return

    if output_csv:
        _print_pead_csv(trades)
        return

    summary = compute_pead_summary(trades, min_surprise, hold_days)
    _print_pead_summary(summary)
    _print_pead_quintiles(summary.get("surprise_quintiles", {}))
    _print_pead_trade_table(trades)


def _print_pead_summary(summary: dict) -> None:
    table = Table(
        title="PEAD Backtest Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    labels = {
        "total_events": "Qualifying Events",
        "trades_taken": "Trades Taken",
        "min_surprise_pct": "Min Surprise (%)",
        "hold_days": "Hold (trading days)",
        "win_rate": "Win Rate (%)",
        "avg_return_pct": "Avg Return (%)",
        "median_return_pct": "Median Return (%)",
        "total_return_pct": "Cumulative Return (%)",
        "max_winner_pct": "Best Trade (%)",
        "max_loser_pct": "Worst Trade (%)",
        "profit_factor": "Profit Factor",
        "sharpe_approx": "Sharpe (approx)",
    }
    for key, label in labels.items():
        val = summary.get(key, "")
        if isinstance(val, float):
            val = f"{val:,.4f}"
        table.add_row(label, str(val))

    console.print(table)


def _print_pead_quintiles(quintiles: dict) -> None:
    if not quintiles:
        return
    table = Table(
        title="Drift by EPS-Surprise Quintile (Q1 lowest → Q5 highest)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Quintile", style="bold")
    table.add_column("Trades", justify="right")
    table.add_column("Avg Surprise %", justify="right")
    table.add_column("Avg Return %", justify="right")
    table.add_column("Median Return %", justify="right")
    table.add_column("Win Rate %", justify="right")

    for name in sorted(quintiles):
        row = quintiles[name]
        table.add_row(
            name,
            str(row["trades"]),
            f"{row['avg_surprise_pct']:,.2f}",
            f"{row['avg_return_pct']:+,.4f}",
            f"{row['median_return_pct']:+,.4f}",
            f"{row['win_rate']:,.2f}",
        )

    console.print(table)


def _print_pead_trade_table(trades: list[PeadTrade]) -> None:
    shown = sorted(trades, key=lambda t: t.return_pct, reverse=True)[:30]
    table = Table(
        title=f"Top Trades (showing {len(shown)} of {len(trades)})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="bold")
    table.add_column("Earnings", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("Entry $", justify="right")
    table.add_column("Exit $", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Surprise %", justify="right")

    for t in shown:
        ret_color = "green" if t.return_pct > 0 else "red"
        table.add_row(
            t.ticker,
            str(t.earnings_date),
            str(t.entry_date),
            str(t.exit_date),
            f"{t.entry_price:,.2f}",
            f"{t.exit_price:,.2f}",
            f"[{ret_color}]{t.return_pct:+.2f}%[/{ret_color}]",
            f"{t.surprise_pct:+.2f}",
        )

    console.print(table)


def _print_pead_csv(trades: list[PeadTrade]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "ticker",
            "earnings_date",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "return_pct",
            "surprise_pct",
            "holding_days",
        ]
    )
    for t in trades:
        writer.writerow(
            [
                t.ticker,
                t.earnings_date,
                t.entry_date,
                t.exit_date,
                t.entry_price,
                t.exit_price,
                t.return_pct,
                t.surprise_pct,
                t.holding_days,
            ]
        )
