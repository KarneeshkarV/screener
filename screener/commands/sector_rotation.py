"""CLI for India and US sector research, without changing trading rules."""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from pathlib import Path

import click
import pandas as pd
from rich.table import Table

from screener import agentio
from screener.backtester.data import (
    ExchangeFallbackPriceFetcher,
    FMPPriceFetcher,
    PriceFetcher,
    build_price_fetcher,
)
from screener.fmp import resolve_api_key
from screener.markets import as_of_option, resolve_as_of
from screener.sector_rotation.analysis import fetch_sector_rotation
from screener.sector_rotation.catalogue import (
    SectorSelection,
    load_sector_catalogue,
    select_sector_groups,
)

_PERIOD_PATTERN = re.compile(r"(?P<count>[1-9]\d*)(?P<unit>[dwmy])")


def _period_start(end: date, period: str) -> date:
    """Calendar start for ``N`` days, weeks, months, or years before ``end``."""
    count, unit = int(period[:-1]), period[-1]
    if unit == "d":
        return end - timedelta(days=count)
    if unit == "w":
        return end - timedelta(weeks=count)
    months = count * (12 if unit == "y" else 1)
    return (pd.Timestamp(end) - pd.DateOffset(months=months)).date()


def _parse_period(_ctx: click.Context, _param: click.Parameter, value: str) -> str:
    period = value.strip().lower()
    match = _PERIOD_PATTERN.fullmatch(period)
    if match is None:
        raise click.BadParameter(
            "use a positive whole number and a unit d, w, m, or y, such as 9m or 18m"
        )
    if _period_start(date.today(), period).year < 1900:
        raise click.BadParameter("period is too long")
    return period


def _price_fetchers(
    injected: PriceFetcher | None,
) -> tuple[PriceFetcher, PriceFetcher | None]:
    """yfinance first, with FMP as a whole-ticker fallback when a key is set.

    The fallback is kept separate so the command can say which tickers came
    from FMP. An injected fetcher or an explicit provider choice is used as is.
    """
    if injected is not None:
        return injected, None
    provider = os.environ.get("SCREENER_PRICE_PROVIDER", "auto").lower()
    if provider not in {"auto", "default"}:
        return build_price_fetcher(provider), None
    fallback = (
        ExchangeFallbackPriceFetcher(FMPPriceFetcher()) if resolve_api_key() else None
    )
    return build_price_fetcher("yfinance"), fallback


def _describe_sector_groups(selection: SectorSelection) -> None:
    """Show offline group details, with a bounded agent preview and complete CSV."""
    catalogue = selection.catalogue
    spill_path = None
    if agentio.is_agent_mode():
        rows = [
            {
                "market": catalogue.market,
                "benchmark": catalogue.benchmark,
                "benchmark_name": catalogue.benchmark_name,
                "membership_note": catalogue.membership_note,
                "catalogue_source": catalogue.source,
                "group_id": group.id,
                "group": group.name,
                "parent": group.parent,
                "kind": group.kind,
                "description": group.description,
                "drivers": "; ".join(group.drivers),
                "risks": "; ".join(group.risks),
                "members": ", ".join(group.members),
                "notes": group.notes,
                "sources": "; ".join(group.sources),
                "video_at": group.video_at or "",
            }
            for group in selection.groups
        ]
        spill_path = agentio.spill(pd.DataFrame(rows), "sector-group-details")
        if agentio.detail_level() != "full":
            table = Table(
                title=f"Sector group details | {catalogue.market} | {catalogue.benchmark}"
            )
            for column in ("Group ID", "Name", "Parent", "Kind"):
                table.add_column(column)
            for group in selection.groups:
                table.add_row(group.id, group.name, group.parent, group.kind)
            agentio.render_table(table, agentio.get_console(), spill_path=spill_path)
            return
    click.echo(
        f"Market: {catalogue.market} | Benchmark: {catalogue.benchmark_name} "
        f"({catalogue.benchmark})"
    )
    click.echo(f"Membership: {catalogue.membership_note}")
    for group in selection.groups:
        click.echo(f"\n{group.id} | {group.name} | {group.parent} | {group.kind}")
        if group.description:
            click.echo(group.description)
        for label, items in (("Drivers", group.drivers), ("Risks", group.risks)):
            if items:
                click.echo(f"{label}: {'; '.join(items)}")
        click.echo(f"Members: {', '.join(group.members)}")
        if group.notes:
            click.echo(f"Exposure notes: {group.notes}")
        click.echo(f"Sources: {'; '.join(group.sources) or catalogue.source}")
    if spill_path is not None:
        click.echo(f"full: {spill_path}")


@click.command(name="sector-rotation")
@click.option(
    "--market", type=click.Choice(["india", "us"]), default="india", show_default=True
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
    "--describe-groups",
    is_flag=True,
    help="Show descriptions, drivers, risks, sources, and members without fetching prices.",
)
@click.option(
    "--period",
    default="3m",
    show_default=True,
    callback=_parse_period,
    help="Calendar window: a whole number and d, w, m, or y, such as 9m or 18m.",
)
@as_of_option(help="Last completed session on or before this date; default yesterday.")
@click.option(
    "--benchmark",
    help="Price-provider symbol override; defaults to Nifty 500 for India or S&P 500 for US.",
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
    describe_groups: bool,
    period: str,
    as_of_arg,
    benchmark: str | None,
    minimum_coverage: float,
    csv_output: bool,
    members_csv: Path | None,
    report: Path | None,
) -> None:
    """Compare curated India or US baskets with a market benchmark.

    Current selected members are not historical index constituents. This
    research command does not generate entry signals or change screen gates.
    """
    try:
        if list_groups and describe_groups:
            raise ValueError(
                "Sector rotation: use either --list-groups or --describe-groups"
            )
        selection = select_sector_groups(
            load_sector_catalogue(catalogue, market=market), group_ids
        )
        if market != selection.catalogue.market:
            raise ValueError(
                "Sector rotation: catalogue market does not match --market"
            )
        if describe_groups:
            _describe_sector_groups(selection)
            return
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
        fetcher, fallback = _price_fetchers(click.get_current_context().obj)
        result = fetch_sector_rotation(
            selection,
            fetcher,
            start=_period_start(end, period),
            end=end,
            benchmark=benchmark,
            minimum_coverage=minimum_coverage,
            fallback=fallback,
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
