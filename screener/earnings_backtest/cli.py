"""CLI entry-points for ``screener earnings-backtest`` and ``earnings-pead``.

The two commands run genuinely different entry/exit policies but share all of
their plumbing: option surface, ticker parsing, spinner/empty-result handling,
and the summary/trade-table renderers. That shared scaffolding lives here as
:func:`_common_options`, :func:`_parse_tickers`, :func:`_render_summary`, and
:func:`_render_trade_table`; only the policy-specific options, CSV schemas, and
summary labels stay per-command.
"""

from __future__ import annotations

from contextlib import nullcontext
import csv
import sys
from typing import Any, Callable

import click
from rich.console import Console
from rich.table import Table

from screener.markets import market_option
from screener.earnings_backtest.engine import (
    EarningsTrade,
    compute_backtest_summary,
    run_earnings_backtest,
)
from screener.earnings_backtest.pead import (
    EXIT_MODES,
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


# ── Shared option surface ───────────────────────────────────────────────


def _common_options(func: Callable) -> Callable:
    """Apply the options both earnings commands share.

    Kept as a single decorator so the market/years/slippage/batch/tickers/csv
    surface stays identical between the two commands.
    """
    decorators = [
        market_option(default="us", help="Market: US (S&P 500) or India (Nifty 500)."),
        click.option(
            "--years",
            type=int,
            default=3,
            show_default=True,
            help="Look-back period for earnings history.",
        ),
        click.option(
            "--slippage-bps",
            type=float,
            default=5.0,
            show_default=True,
            help="Slippage per fill in basis points.",
        ),
        click.option(
            "--batch-size",
            type=int,
            default=50,
            show_default=True,
            help="Symbols per API batch (controls RAM).",
        ),
        click.option(
            "--tickers",
            default=None,
            help="Comma-separated ticker list (overrides universe).",
        ),
        click.option(
            "--csv",
            "output_csv",
            is_flag=True,
            help="Output trade ledger as CSV.",
        ),
    ]
    for decorator in reversed(decorators):
        func = decorator(func)
    return func


def _parse_tickers(tickers: str | None) -> list[str] | None:
    """Normalize a comma-separated ``--tickers`` value to an upper-cased list."""
    if not tickers:
        return None
    return [t.strip().upper() for t in tickers.split(",") if t.strip()]


# ── Shared renderers ────────────────────────────────────────────────────


def _render_summary(
    summary: dict, *, title: str, labels: dict[str, str], footer: str | None = None
) -> None:
    """Render a two-column Metric/Value summary table with an optional footer."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for key, label in labels.items():
        val = summary.get(key, "")
        if isinstance(val, float):
            val = f"{val:,.4f}"
        table.add_row(label, str(val))
    console.print(table)
    if footer is not None:
        console.print(footer)


def _render_trade_table(
    trades: list,
    *,
    last_header: str,
    last_value: Callable[[Any], str],
) -> None:
    """Render the top-30 trade table shared by both commands.

    The first seven columns are identical across policies; the eighth column
    (score vs surprise) is supplied via *last_header* / *last_value*.
    """
    shown = sorted(trades, key=lambda t: t.return_pct, reverse=True)[:30]
    table = Table(
        title=f"Top Trades (showing {len(shown)} of {len(trades)})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Ticker", style="bold")
    for col in ("Earnings", "Entry", "Exit", "Entry $", "Exit $", "Return %"):
        table.add_column(col, justify="right")
    table.add_column(last_header, justify="right")

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
            last_value(t),
        )
    console.print(table)


# ── earnings-drift (E-1/E-2 -> E) ────────────────────────────────────────


@click.command(name="earnings-backtest")
@_common_options
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
    help=(
        "Round-trip commission in basis points when --cost-model=flat "
        "(split evenly across buy and sell)."
    ),
)
@click.option(
    "--cost-model",
    type=click.Choice(["flat", "india", "us_vested"]),
    default="flat",
    show_default=True,
    help=(
        "Statutory fee model. 'flat' applies --commission-bps as a round-trip "
        "total (legacy). 'india' applies NSE equity delivery fees. "
        "'us_vested' applies the Vested/DriveWealth US equity fee stack."
    ),
)
def earnings_backtest(
    market: str,
    years: int,
    strategy: str,
    days_before: int,
    min_score: float,
    commission_bps: float,
    cost_model: str,
    slippage_bps: float,
    batch_size: int,
    tickers: str | None,
    output_csv: bool,
) -> None:
    """Backtest earnings-drift entry (E-1/E-2 → E) with sentiment filters."""
    ticker_list = _parse_tickers(tickers)

    status = (
        nullcontext()
        if output_csv
        else console.status(
            f"[bold green]Running earnings-backtest ({market}, {strategy}, {years}y)…"
        )
    )
    with status:
        trades = run_earnings_backtest(
            market=market,
            years=years,
            strategy=strategy,
            days_before=days_before,
            min_score=min_score,
            commission_bps=commission_bps,
            cost_model=cost_model,
            slippage_bps=slippage_bps,
            batch_size=batch_size,
            tickers=ticker_list,
        )
    if not trades:
        console.print(
            "[yellow]No earnings events found for the given parameters.[/yellow]"
        )
        return

    summary = compute_backtest_summary(trades, strategy=strategy)
    taken = [t for t in trades if t.passed_filter]

    if output_csv:
        _print_csv(taken)
        return

    _print_summary(summary)
    if taken:
        _render_trade_table(
            taken, last_header="Score", last_value=lambda t: f"{t.score:.3f}"
        )


_FEE_COMPONENT_LABELS = {
    "commission": "brokerage",
    "brokerage": "brokerage",
    "stt": "stt",
    "stamp_duty": "stamp_duty",
    "exchange_txn": "exchange_txn",
    "sebi": "sebi",
    "gst": "gst",
    "ipft": "ipft",
    "sec_fee": "sec_fee",
    "taf": "taf",
}


def _format_costs_line(summary: dict) -> str | None:
    total = summary.get("total_fees")
    if total is None:
        return None
    total_f = float(total)
    parts: list[str] = []
    for key, value in summary.items():
        if not key.startswith("fee_") or float(value) == 0.0:
            continue
        name = key[len("fee_") :]
        label = _FEE_COMPONENT_LABELS.get(name, name)
        parts.append(f"{label}: {float(value):,.4f}")
    if parts:
        return f"Total costs: {total_f:,.4f} ({', '.join(parts)})"
    return f"Total costs: {total_f:,.4f}"


_EARNINGS_SUMMARY_LABELS = {
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


def _print_summary(summary: dict) -> None:
    costs_line = _format_costs_line(summary)
    _render_summary(
        summary,
        title="Earnings-Drift Backtest Summary",
        labels=_EARNINGS_SUMMARY_LABELS,
        footer=f"[dim]{costs_line}[/dim]" if costs_line is not None else None,
    )


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
@_common_options
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
    help="Trading days to hold after the next-open entry (fixed exit mode).",
)
@click.option(
    "--exit-mode",
    type=click.Choice(list(EXIT_MODES)),
    default="fixed",
    show_default=True,
    help=(
        "fixed: hold --hold-days sessions. dynamic: hold until a later report "
        "fails the surprise criterion (or the price history ends)."
    ),
)
@click.option(
    "--commission-bps",
    type=float,
    default=10.0,
    show_default=True,
    help="Round-trip commission in basis points.",
)
def earnings_pead(
    market: str,
    years: int,
    min_surprise: float,
    hold_days: int,
    exit_mode: str,
    commission_bps: float,
    slippage_bps: float,
    batch_size: int,
    tickers: str | None,
    output_csv: bool,
) -> None:
    """Backtest post-earnings-announcement drift (next open → hold N days)."""
    ticker_list = _parse_tickers(tickers)

    hold_label = "dynamic hold" if exit_mode == "dynamic" else f"{hold_days}d hold"
    with console.status(
        f"[bold green]Running PEAD backtest ({market}, surprise≥{min_surprise}%, "
        f"{hold_label}, {years}y)…"
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
            exit_mode=exit_mode,
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
    _render_trade_table(
        trades,
        last_header="Surprise %",
        last_value=lambda t: f"{t.surprise_pct:+.2f}",
    )


_PEAD_SUMMARY_LABELS = {
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


def _print_pead_summary(summary: dict) -> None:
    _render_summary(summary, title="PEAD Backtest Summary", labels=_PEAD_SUMMARY_LABELS)


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


def _print_pead_csv(trades: list[PeadTrade]) -> None:
    writer = csv.writer(sys.stdout)
    # The dynamic exit mode records an exit_reason per trade; surface it as a
    # trailing column only when present so the fixed-mode ledger is unchanged.
    has_exit_reason = any("exit_reason" in t.details for t in trades)
    header = [
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
    if has_exit_reason:
        header.append("exit_reason")
    writer.writerow(header)
    for t in trades:
        row = [
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
        if has_exit_reason:
            row.append(t.details.get("exit_reason", ""))
        writer.writerow(row)
