"""CLI for video-inspired sector baskets, without changing trading rules."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import click
from rich.table import Table

from screener import agentio
from screener.markets import as_of_option, get_price_fetcher, resolve_as_of
from screener.sector_rotation.analysis import fetch_sector_rotation
from screener.sector_rotation.catalogue import (
    load_sector_catalogue,
    select_sector_groups,
)

_PERIOD_DAYS = {
    "1d": 1,
    "1w": 7,
    "1m": 30,
    "3m": 91,
    "6m": 183,
    "1y": 365,
    "3y": 1096,
    "5y": 1826,
}


@click.command(name="sector-rotation")
@click.option(
    "--market", type=click.Choice(["india"]), default="india", show_default=True
)
@click.option(
    "--catalogue",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Replacement group YAML catalogue.",
)
@click.option(
    "--group",
    "group_ids",
    multiple=True,
    help="Exact group ID; repeat to select several.",
)
@click.option(
    "--list-groups",
    is_flag=True,
    help="List group IDs and selected members without fetching prices.",
)
@click.option(
    "--period", type=click.Choice(list(_PERIOD_DAYS)), default="3m", show_default=True
)
@as_of_option(help="Last completed session on or before this date; default yesterday.")
@click.option(
    "--benchmark", help="Price-provider symbol override; default Nifty 500 (^CRSLDX)."
)
@click.option(
    "--minimum-coverage",
    type=click.FloatRange(min=0, max=1, min_open=True),
    default=0.8,
    show_default=True,
)
@click.option(
    "--csv", "csv_output", is_flag=True, help="Emit all group metrics as CSV."
)
@click.option(
    "--members-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the full group-member metrics CSV.",
)
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write a self-contained HTML report.",
)
def sector_rotation(
    market: str,
    catalogue: Path | None,
    group_ids: tuple[str, ...],
    list_groups: bool,
    period: str,
    as_of_arg,
    benchmark: str | None,
    minimum_coverage: float,
    csv_output: bool,
    members_csv: Path | None,
    report: Path | None,
) -> None:
    """Compare curated sector and industry baskets with Nifty 500.

    Current selected members are not historical index constituents. This
    research command does not generate entry signals or change screen gates.
    """
    try:
        selection = select_sector_groups(load_sector_catalogue(catalogue), group_ids)
        if market != selection.catalogue.market:
            raise ValueError(
                "Sector rotation: catalogue market does not match --market"
            )
        if list_groups:
            for group in selection.groups:
                click.echo(
                    f"{group.id}\t{group.name}\t{group.parent}\t{','.join(group.members)}"
                )
            return
        end = (
            resolve_as_of(as_of_arg)
            if as_of_arg
            else resolve_as_of(None) - timedelta(days=1)
        )
        if end >= resolve_as_of(None):
            raise ValueError(
                "Sector rotation: --as-of must be before today to exclude incomplete sessions"
            )
        start = end - timedelta(days=_PERIOD_DAYS[period])
        result = fetch_sector_rotation(
            selection,
            get_price_fetcher(click.get_current_context().obj),
            start=start,
            end=end,
            benchmark=benchmark,
            minimum_coverage=minimum_coverage,
        )
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)
        if members_csv:
            members_csv.parent.mkdir(parents=True, exist_ok=True)
            result.members.to_csv(members_csv, index=False)
            click.echo(f"members: {members_csv}", err=True)
        if report:
            from screener.sector_rotation.report import render_sector_report

            render_sector_report(result, report)
            click.echo(f"report: {report}", err=True)
        if csv_output:
            click.echo(result.summary.to_csv(index=False), nl=False)
            return
        table = Table(
            title=f"Sector rotation | {result.start} to {result.end} | {result.benchmark}"
        )
        for column in ("Group", "Return %", "Excess pp", "Coverage %", "Status"):
            table.add_column(column)
        for row in result.summary.to_dict("records"):
            table.add_row(
                row["group"],
                f"{row['return_pct']:.2f}",
                f"{row['excess_return_pp']:.2f}",
                f"{row['minimum_coverage_pct']:.0f}",
                row["status"],
            )
        console = agentio.get_console()
        if agentio.is_agent_mode():
            agentio.render_table(
                table,
                console,
                spill_path=agentio.spill(result.summary, "sector-rotation"),
            )
        else:
            console.print(table)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
