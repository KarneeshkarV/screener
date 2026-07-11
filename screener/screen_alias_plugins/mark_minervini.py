"""Mark Minervini ``screen -c`` compatibility alias."""

from __future__ import annotations

from datetime import date

from rich.console import Console

from screener.minervini import render_rows, scan_minervini


def mark_minervini_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool = False,
    refresh: bool = False,
    cache_ttl: str = "15m",
) -> None:
    console = Console()
    rows = scan_minervini(
        market,
        as_of=date.today(),
        limit=int(limit),
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    render_rows(rows, console, market)
    console.print(
        "\n[dim]RS Rank is a universe-relative 12-month percentile proxy, not "
        "Investor's Business Daily's proprietary RS Rating.[/dim]"
    )
