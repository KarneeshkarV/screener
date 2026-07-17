"""Click command group for the FMP-backed US SEC filings reader (US only)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.table import Table

from screener.filings import VALID_PERIODS
from screener.fmp import resolve_api_key
from screener.markets import market_option


if TYPE_CHECKING:
    from screener.filings import Filing, FinancialReport, ReportSection


console = Console()


def _require_api_key() -> str:
    api_key = resolve_api_key()
    if not api_key:
        raise click.ClickException(
            "FMP_API_KEY is not set. Export it or add it to the project .env "
            "to use the filings command."
        )
    return api_key


@click.group(name="filings")
def filings() -> None:
    """Read US SEC filings (10-K/10-Q/8-K) via Financial Modeling Prep."""


@filings.command(name="list")
@click.argument("ticker")
@market_option(
    choices=("us",),
    default="us",
    help="Market to query. Only 'us' is supported (SEC filings).",
)
@click.option(
    "--type",
    "filing_type",
    default=None,
    help="Filter by filing type, e.g. 10-K, 10-Q, 8-K.",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of filings to show (newest first).",
)
@click.option("--csv", "output_csv", is_flag=True, help="Output as CSV.")
@click.option("--refresh", is_flag=True, help="Bypass cached FMP data.")
def filings_list(
    ticker: str,
    market: str,
    filing_type: str | None,
    limit: int,
    output_csv: bool,
    refresh: bool,
) -> None:
    """List recent SEC filings for TICKER (e.g. `filings list AAPL --type 10-K`)."""
    from screener.filings import load_filings

    symbol = ticker.strip().upper()
    if not symbol:
        raise click.ClickException("A ticker symbol is required.")
    if limit < 1:
        raise click.ClickException("--limit must be at least 1.")

    api_key = _require_api_key()
    records = load_filings(
        symbol,
        api_key=api_key,
        filing_type=filing_type.strip() if filing_type else None,
        limit=limit,
        refresh=refresh,
    )

    if not records:
        click.echo(f"No filings found for {symbol}.")
        return

    if output_csv:
        _print_filings_csv(records)
        return

    _print_filings_table(symbol, records)


def _fmt_date(value: date | None) -> str:
    return value.isoformat() if value is not None else "-"


def _print_filings_csv(records: list[Filing]) -> None:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["symbol", "type", "filing_date", "accepted_date", "final_link", "index_link"]
    )
    for f in records:
        writer.writerow(
            [
                f.symbol,
                f.type,
                _fmt_date(f.filing_date),
                _fmt_date(f.accepted_date),
                f.final_link,
                f.link,
            ]
        )
    click.echo(buf.getvalue(), nl=False)


def _print_filings_table(symbol: str, records: list[Filing]) -> None:
    console.print(f"\n[bold]SEC filings (US)[/bold] — {symbol}, {len(records)} shown\n")
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Filing Date")
    table.add_column("Accepted")
    table.add_column("Type")
    table.add_column("Document")
    for f in records:
        table.add_row(
            _fmt_date(f.filing_date),
            _fmt_date(f.accepted_date),
            f.type or "-",
            f.final_link or f.link or "-",
        )
    console.print(table)


@filings.command(name="report")
@click.argument("ticker")
@market_option(
    choices=("us",),
    default="us",
    help="Market to query. Only 'us' is supported (SEC filings).",
)
@click.option("--year", type=int, required=True, help="Fiscal year, e.g. 2024.")
@click.option(
    "--period",
    type=click.Choice(list(VALID_PERIODS)),
    default="FY",
    show_default=True,
    help="Report period: FY (10-K) or Q1..Q3 (10-Q).",
)
@click.option(
    "--section",
    "sections",
    multiple=True,
    help="Case-insensitive section-name substring; repeatable.",
)
@click.option(
    "--list-sections",
    is_flag=True,
    help="List available section names and exit.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the matched (or full) section JSON to PATH.",
)
@click.option("--refresh", is_flag=True, help="Bypass cached FMP data.")
def filings_report(
    ticker: str,
    market: str,
    year: int,
    period: str,
    sections: tuple[str, ...],
    list_sections: bool,
    json_path: Path | None,
    refresh: bool,
) -> None:
    """Read a 10-K/10-Q for TICKER by section (`filings report AAPL --year 2024`)."""
    from screener.filings import load_report, match_sections

    symbol = ticker.strip().upper()
    if not symbol:
        raise click.ClickException("A ticker symbol is required.")

    api_key = _require_api_key()
    report = load_report(
        symbol,
        api_key=api_key,
        year=year,
        period=period,
        refresh=refresh,
    )
    if report is None:
        click.echo(f"No {period} report found for {symbol} {year}.")
        return

    patterns = [p for p in sections if p.strip()]

    # Default (no --section) and --list-sections both just list the sections.
    if list_sections or not patterns:
        _print_section_names(symbol, report)
        return

    matched = []
    seen: set[str] = set()
    unmatched: list[str] = []
    for pattern in patterns:
        hits = match_sections(report, pattern)
        if not hits:
            unmatched.append(pattern)
            continue
        for section in hits:
            if section.name not in seen:
                seen.add(section.name)
                matched.append(section)

    for pattern in unmatched:
        click.echo(f"No section matched '{pattern}'.", err=True)

    if not matched:
        if json_path is not None:
            raise click.ClickException(
                f"No sections matched; nothing written to {json_path}."
            )
        _print_section_names(symbol, report)
        return

    if json_path is not None:
        payload = {section.name: section.raw() for section in matched}
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        click.echo(f"Wrote {len(matched)} section(s) to {json_path}.")
        return

    for section in matched:
        _print_section(symbol, report, section)


def _print_section_names(symbol: str, report: FinancialReport) -> None:
    names = report.section_names()
    console.print(
        f"\n[bold]{symbol} {report.year} {report.period}[/bold] — "
        f"{len(names)} sections\n"
    )
    for name in names:
        click.echo(name)


def _print_section(
    symbol: str, report: FinancialReport, section: ReportSection
) -> None:
    rows = section.rows
    name = section.name
    period = report.period
    year = report.year
    n_cols = max((len(row.values) for row in rows), default=0)

    console.print(f"\n[bold]{symbol} {year} {period}[/bold] — [cyan]{name}[/cyan]\n")
    if not rows:
        console.print("[dim]No rows in this section.[/dim]")
        return

    table = Table(show_header=False, show_lines=False)
    table.add_column("", justify="left", overflow="fold")
    for _ in range(n_cols):
        table.add_column("", justify="right", overflow="fold")
    for row in rows:
        cells = [str(v) for v in row.values]
        cells += [""] * (n_cols - len(cells))
        table.add_row(row.label, *cells)
    console.print(table)


__all__ = ["filings"]
