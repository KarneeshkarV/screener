"""CLI command listing persisted screen runs from the history database."""

from __future__ import annotations

import click

from screener import history


@click.command(name="history")
@click.option("--market", "-m", default=None, help="Filter by market (e.g. us, india).")
@click.option("--criteria", "-c", default=None, help="Filter by criteria label.")
@click.option(
    "--limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of runs to show (newest first).",
)
@click.option("--csv", "output_csv", is_flag=True, help="Emit runs as CSV.")
def history_list(market, criteria, limit, output_csv):
    """List persisted screen runs (replay them with `backtest-historical --from-run`)."""
    df = history.list_runs(market=market, criteria=criteria, limit=limit)
    if output_csv:
        click.echo(df.to_csv(index=False), nl=False)
        return
    if df.empty:
        click.echo(f"No screen runs recorded yet in {history.DB_PATH}.")
        return

    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"Screen runs ({history.DB_PATH})")
    table.add_column("ID", justify="right")
    table.add_column("Run (UTC)")
    table.add_column("Market")
    table.add_column("Criteria")
    table.add_column("Matches", justify="right")
    table.add_column("Saved rows", justify="right")
    for _, row in df.iterrows():
        table.add_row(
            str(row["id"]),
            str(row["run_ts"]),
            str(row["market"]),
            str(row["criteria"]),
            str(row["total_matches"]),
            str(row["saved_rows"]),
        )
    Console().print(table)
    click.echo(
        "Replay a run: screener backtest-historical --from-run <ID> "
        "(or --from-run MARKET:CRITERIA --run-age-days N)"
    )


__all__ = ["history_list"]
