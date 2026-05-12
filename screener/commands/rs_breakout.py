"""Click command and orchestration helpers for RS breakout scans."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import click
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
import requests
from rich.console import Console

from screener.backtester.data import PriceFetcher, build_price_fetcher
from screener.cache import parse_ttl
from screener.rs_breakout import (
    DEFAULT_BENCHMARKS,
    RsBreakoutResult,
    fetch_price_data,
    load_india_delivery_for_scan,
    render_result,
    scan_rs_breakouts,
    write_json,
    write_markdown,
)
from screener.scanner import scan


class RsBreakoutRequest(BaseModel):
    market: str
    as_of: date
    universe: list[str]
    benchmark: str
    history_days: int = Field(ge=1)
    require_delivery: bool

    model_config = ConfigDict(frozen=True)

    @field_validator("market", "benchmark")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("universe")
    @classmethod
    def _normalize_universe(cls, value: list[str]) -> list[str]:
        normalized = [ticker.strip() for ticker in value if ticker.strip()]
        if not normalized:
            raise ValueError("universe must include at least one ticker")
        return normalized


def resolve_universe(
    market: str,
    tickers: str | None,
    universe_file: str | None,
    universe_limit: int,
    *,
    cache_ttl: float | None = 900,
    refresh: bool = False,
) -> list[str]:
    if tickers:
        return [t.strip() for t in tickers.split(",") if t.strip()]
    if universe_file:
        path = Path(universe_file)
        if not path.exists():
            raise click.UsageError(f"--universe-file not found: {universe_file}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return load_universe(
        market, int(universe_limit), cache_ttl=cache_ttl, refresh=refresh
    )


def load_universe(
    market: str,
    universe_limit: int,
    *,
    cache_ttl: float | None = 900,
    refresh: bool = False,
) -> list[str]:
    from tradingview_screener import col

    price_floor = {"india": 50.0, "us": 5.0}[market]
    requested_limit = 5000 if universe_limit == 0 else universe_limit
    filters = [col("type") == "stock", col("close") >= price_floor]
    _total, df = scan(
        market=market,
        filters=filters,
        limit=requested_limit,
        order_by="volume",
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return [str(t) for t in df["name"].dropna().tolist()]


def run_rs_breakout_scan(
    request: RsBreakoutRequest,
    fetcher: PriceFetcher,
    console: Console,
) -> RsBreakoutResult:
    console.print(
        f"[dim]Scanning {len(request.universe)} {request.market.upper()} "
        f"tickers as of {request.as_of}...[/dim]"
    )
    bars_by_symbol, benchmark_bars = fetch_price_data(
        request.universe,
        request.market,
        request.as_of,
        fetcher,
        benchmark=request.benchmark,
        history_days=request.history_days,
    )
    delivery_panel = pd.DataFrame()
    if request.market == "india":
        try:
            delivery_panel = load_india_delivery_for_scan(
                request.universe, request.as_of
            )
        except (
            requests.RequestException,
            OSError,
            RuntimeError,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            console.print(
                f"[yellow]Delivery data load failed: {exc}. Full bucket may be empty.[/yellow]"
            )

    return scan_rs_breakouts(
        bars_by_symbol,
        benchmark_bars,
        request.as_of,
        delivery_panel=delivery_panel,
        benchmark_symbol=request.benchmark,
        require_delivery=request.require_delivery,
    )


def write_default_outputs(
    result: RsBreakoutResult,
    market: str,
    json_path: str | None,
    md_path: str | None,
) -> tuple[str, str]:
    json_default = f"rs_breakout_{market}_{result.as_of.isoformat()}.json"
    md_default = f"rs_breakout_{market}_{result.as_of.isoformat()}.md"
    write_json(result, Path(json_path or json_default))
    write_markdown(result, Path(md_path or md_default), market=market)
    return json_path or json_default, md_path or md_default


@click.command(name="rs-breakout")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["india", "us"]),
    default="india",
    show_default=True,
    help="Market to scan. India includes delivery-percent filter; US skips delivery.",
)
@click.option(
    "--as-of",
    "as_of_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Trading date to evaluate (default: today).",
)
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated ticker list. Falls back to the India universe when omitted.",
)
@click.option(
    "--universe-file", default=None, help="Path to newline-separated tickers."
)
@click.option(
    "--universe-limit",
    type=int,
    default=500,
    show_default=True,
    help="TradingView universe size. Use 0 for broad market scan.",
)
@click.option(
    "--benchmark",
    default=None,
    help="Benchmark ticker for 55-day relative strength.",
)
@click.option(
    "--history-days",
    type=int,
    default=220,
    show_default=True,
    help="Calendar days of OHLCV history to fetch.",
)
@click.option("-n", "--limit", type=int, default=50, show_default=True)
@click.option("--json", "json_path", default=None, help="JSON output path.")
@click.option("--md", "md_path", default=None, help="Markdown output path.")
@click.option(
    "--refresh", is_flag=True, help="Bypass cached TradingView/yfinance data."
)
@click.option(
    "--cache-ttl",
    default="15m",
    show_default=True,
    help="TradingView universe cache TTL, e.g. 30s, 15m, 1h, off.",
)
@click.option(
    "--no-output-files",
    is_flag=True,
    default=False,
    help="Skip JSON/Markdown writes.",
)
def rs_breakout(
    market: str,
    as_of_arg: datetime | None,
    tickers: str | None,
    universe_file: str | None,
    universe_limit: int,
    benchmark: str | None,
    history_days: int,
    limit: int,
    json_path: str | None,
    md_path: str | None,
    refresh: bool,
    cache_ttl: str,
    no_output_files: bool,
) -> None:
    """Screen stocks for RS + SuperTrend + breakout/volume setups."""
    console = Console()
    as_of_date = as_of_arg.date() if isinstance(as_of_arg, datetime) else date.today()
    resolved_benchmark = benchmark or DEFAULT_BENCHMARKS[market]
    parsed_ttl = parse_ttl(cache_ttl, default=900)
    universe = resolve_universe(
        market,
        tickers,
        universe_file,
        int(universe_limit),
        cache_ttl=parsed_ttl,
        refresh=refresh,
    )
    if not universe:
        raise click.UsageError("Empty universe: pass --tickers or --universe-file.")

    fetcher = click.get_current_context().obj or build_price_fetcher(refresh=refresh)
    request = RsBreakoutRequest(
        market=market,
        as_of=as_of_date,
        universe=universe,
        benchmark=resolved_benchmark,
        history_days=int(history_days),
        require_delivery=market == "india",
    )
    result = run_rs_breakout_scan(request, fetcher, console)
    render_result(result, console, limit=int(limit), market=market)

    if not no_output_files:
        json_written, md_written = write_default_outputs(
            result, market, json_path, md_path
        )
        console.print(f"\n[dim]Wrote {json_written} + {md_written}[/dim]")
