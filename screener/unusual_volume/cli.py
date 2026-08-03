"""Click sub-command for unusual-volume detection.

The command is registered on the main ``cli`` group in ``main.py`` via:

    from screener.unusual_volume.cli import unusual_volume
    cli.add_command(unusual_volume)
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from screener.markets import as_of_option, market_option, resolve_as_of

from .buildup import (
    DEFAULT_MIN_SCORE as DEFAULT_BUILDUP_MIN,
)
from .buildup import (
    DEFAULT_WINDOW as DEFAULT_BUILDUP_WINDOW,
)
from .detector import (
    DEFAULT_MIN_RVOL,
    DEFAULT_MIN_Z,
)
from .enrichment import Enrichment
from .output import render_rich, sort_events, write_json, write_markdown
from .service import (
    DEFAULT_MIN_AVG_VOLUME,
    UnusualVolumeRequest,
    run_unusual_volume_scan,
)


def _resolve_universe(
    market: str,
    tickers: str | None,
    universe_file: str | None,
) -> list[str]:
    from screener.universes import (
        UniverseRequest,
        UniverseSource,
        parse_ticker_csv,
        resolve_universe,
    )

    def _tv_loader() -> list[str]:
        # Lazy import; pulls TradingView. Kept module-local so it stays patchable.
        from screener.research.pine_runner import load_universe

        return load_universe(market)

    request = UniverseRequest(
        source=UniverseSource.TV_LIQUIDITY,
        market=market,
        tickers=tuple(parse_ticker_csv(tickers)) if tickers else None,
        file=universe_file,
    )
    try:
        return resolve_universe(request, tv_loader=_tv_loader)
    except FileNotFoundError as exc:
        raise click.UsageError(f"--universe-file not found: {universe_file}") from exc


@click.command(name="unusual-volume")
@market_option(default="us", help="Market to scan.")
@as_of_option()
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated ticker list. Falls back to load_universe() when omitted.",
)
@click.option(
    "--universe-file",
    default=None,
    help="Newline-separated ticker file (alternative to --tickers).",
)
@click.option(
    "--min-rvol",
    type=float,
    default=DEFAULT_MIN_RVOL,
    help=f"RVOL floor for the moderate tier (default {DEFAULT_MIN_RVOL}).",
)
@click.option(
    "--min-z",
    type=float,
    default=DEFAULT_MIN_Z,
    help=f"Volume Z-score floor for the moderate tier (default {DEFAULT_MIN_Z}).",
)
@click.option(
    "--strength",
    "strength_floor",
    type=click.Choice(["moderate", "high", "extreme"]),
    default="moderate",
    help="Drop events below this strength tier.",
)
@click.option(
    "--min-avg-volume",
    type=float,
    default=DEFAULT_MIN_AVG_VOLUME,
    help="Minimum 20-day average daily volume (shares).",
)
@click.option(
    "--min-market-cap",
    type=float,
    default=None,
    help="Minimum market cap. Defaults to $300M (US) / ₹500 cr (India).",
)
@click.option(
    "--include-fno-ban",
    is_flag=True,
    default=False,
    help="(India) include tickers in the F&O ban list. Default: drop them.",
)
@click.option(
    "--deep-india",
    is_flag=True,
    default=False,
    help="(India) enrich flagged events with promoter holding via openscreener.",
)
@click.option(
    "--json",
    "json_path",
    default=None,
    help="JSON output path. Default: unusual_volume_<market>_<as_of>.json",
)
@click.option(
    "--md",
    "md_path",
    default=None,
    help="Markdown output path. Default: unusual_volume_<market>_<as_of>.md",
)
@click.option(
    "--no-output-files",
    is_flag=True,
    default=False,
    help="Skip JSON/MD writes (rich-table only).",
)
@click.option(
    "--option-chain",
    is_flag=True,
    default=False,
    help="(India) attach live NSE option-chain PCR / call-put OI ratio and "
    "accumulate a daily snapshot panel.",
)
@click.option(
    "--fii-dii",
    is_flag=True,
    default=False,
    help="(India) attach market-wide FII/DII 5d net + trend and accumulate a "
    "daily snapshot panel.",
)
@click.option(
    "--pledge",
    is_flag=True,
    default=False,
    help="(India) attach promoter pledge %% (NSE filings, openscreener fallback).",
)
@click.option(
    "--refresh", is_flag=True, help="Bypass cached yfinance and enrichment data."
)
@click.option(
    "-n",
    "--limit",
    type=int,
    default=50,
    help="Cap rich-table rows (sorted by strength then RVOL).",
)
@click.option(
    "--buildup/--no-buildup",
    "buildup_enabled",
    default=False,
    help="Score every ticker for multi-week build-up patterns and emit a "
    "BUILDUP bucket. Adds buildup_score+flags onto detected events too.",
)
@click.option(
    "--buildup-window",
    type=int,
    default=DEFAULT_BUILDUP_WINDOW,
    show_default=True,
    help="Bars of lookback for build-up scoring.",
)
@click.option(
    "--buildup-min-score",
    type=float,
    default=DEFAULT_BUILDUP_MIN,
    show_default=True,
    help="Composite score floor for the BUILDUP bucket.",
)
def unusual_volume(
    market: str,
    as_of_arg,
    tickers: str | None,
    universe_file: str | None,
    min_rvol: float,
    min_z: float,
    strength_floor: str,
    min_avg_volume: float,
    min_market_cap: float | None,
    include_fno_ban: bool,
    deep_india: bool,
    option_chain: bool,
    fii_dii: bool,
    pledge: bool,
    json_path: str | None,
    md_path: str | None,
    no_output_files: bool,
    refresh: bool,
    limit: int,
    buildup_enabled: bool,
    buildup_window: int,
    buildup_min_score: float,
) -> None:
    """Detect abnormal trading volume across a market on a given day."""
    universe = _resolve_universe(market, tickers, universe_file)
    if not universe:
        raise click.UsageError("Empty universe — pass --tickers or --universe-file.")
    enrichments = frozenset(
        enrichment
        for enabled, enrichment in (
            (deep_india, Enrichment.DEEP_INDIA),
            (buildup_enabled, Enrichment.BUILDUP),
            (option_chain, Enrichment.OPTION_CHAIN),
            (fii_dii, Enrichment.FII_DII),
            (pledge, Enrichment.PLEDGE),
        )
        if enabled
    )
    request = UnusualVolumeRequest(
        market=market,
        as_of=resolve_as_of(as_of_arg),
        universe=universe,
        min_rvol=min_rvol,
        min_z=min_z,
        strength_floor=strength_floor,
        min_avg_volume=min_avg_volume,
        min_market_cap=min_market_cap,
        include_fno_ban=include_fno_ban,
        enrichments=enrichments,
        buildup_window=buildup_window,
        buildup_min_score=buildup_min_score,
        refresh=refresh,
    )
    ok = run_unusual_volume(
        request,
        limit=limit,
        json_path=json_path,
        md_path=md_path,
        no_output_files=no_output_files,
    )
    if not ok:
        import sys

        sys.exit(1)


def run_unusual_volume(
    request: UnusualVolumeRequest,
    *,
    limit: int = 50,
    json_path: str | None = None,
    md_path: str | None = None,
    no_output_files: bool = False,
    console: Console | None = None,
) -> bool:
    """Run a prepared scan request and render it (no Click context required)."""
    console = console or Console()
    market = request.market
    as_of = request.as_of
    result = run_unusual_volume_scan(request, console)
    if not result.events and result.fetched_count == 0:
        console.print("[red]No OHLCV data fetched. Aborting.[/red]")
        return False
    if not result.events and result.liquid_count == 0:
        console.print("[yellow]No tickers passed the volume floor.[/yellow]")
        return True
    if not result.events:
        console.print(
            f"[yellow]No unusual-volume events on {as_of} for {market.upper()}.[/yellow]"
        )
        return True

    events = result.events
    sorted_events = sort_events(events)
    render_rich(sorted_events[:limit], market, as_of, console)

    if not no_output_files:
        json_default = f"unusual_volume_{market}_{as_of.isoformat()}.json"
        md_default = f"unusual_volume_{market}_{as_of.isoformat()}.md"
        write_json(events, Path(json_path or json_default))
        write_markdown(events, Path(md_path or md_default), market, as_of)
        console.print(
            f"\n[dim]Wrote {json_path or json_default} + {md_path or md_default}[/dim]"
        )
    return True
