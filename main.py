from datetime import date, datetime

import click
import pandas as pd

from screener import history
from screener.backtester.historical import backtest_historical
from screener.backtester.rolling import backtest_rolling
from screener.criteria import CRITERIA, combine
from screener.operator.cli import register as _register_operator_cli
from screener.scanner import scan, MARKETS
from screener.display import print_results, print_csv, print_insider_results
from screener.rs_breakout import (
    DEFAULT_BENCHMARKS as RS_BREAKOUT_DEFAULT_BENCHMARKS,
    fetch_price_data as fetch_rs_breakout_price_data,
    load_india_delivery_for_scan,
    render_result as render_rs_breakout_result,
    scan_rs_breakouts,
    write_json as write_rs_breakout_json,
    write_markdown as write_rs_breakout_markdown,
)
from screener.unusual_volume.cli import unusual_volume
from screener.resilience import call_with_resilience
from screener.backtester.optimization.cli import optimize


@click.group()
def cli():
    """Stock screener for US and Indian markets."""


cli.add_command(unusual_volume)
cli.add_command(backtest_historical)
cli.add_command(backtest_rolling)
_register_operator_cli(cli)
cli.add_command(optimize)


@cli.command()
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
    default="us",
    help="Market to screen.",
)
@click.option(
    "-c",
    "--criteria",
    "criteria_names",
    type=click.Choice(list(CRITERIA.keys())),
    multiple=True,
    default=("ema",),
    help="Screening criteria (repeat to combine, e.g. -c ema -c breakout).",
)
@click.option("-n", "--limit", default=50, help="Number of results.")
@click.option(
    "--sort",
    "order_by",
    default="setup_score",
    help="Sort by column. Use setup_score for local composite ranking.",
)
@click.option("--csv", "output_csv", is_flag=True, help="Output as CSV.")
@click.option("--detail", is_flag=True, help="Show fundamental details (P/E, ROE, etc.).")
def screen(market, criteria_names, limit, order_by, output_csv, detail):
    """Screen stocks based on technical criteria."""
    criteria_fns = [CRITERIA[name] for name in criteria_names]
    filters = combine(*criteria_fns)()

    label = "+".join(criteria_names)

    total, df = scan(
        market=market,
        filters=filters,
        limit=limit,
        order_by=order_by,
        detail=detail,
    )

    if output_csv:
        print_csv(df)
        return

    run_id = history.save_run(market, label, total, df)
    prev = history.previous_run(market, label, before_id=run_id)
    if prev is None:
        added, removed, first_run = [], [], True
    else:
        added, removed = history.diff(df, prev)
        first_run = False

    print_results(
        df,
        total,
        market,
        label,
        added=added,
        removed=removed,
        first_run=first_run,
    )


@cli.command(name="rs-breakout")
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
@click.option("--universe-file", default=None, help="Path to newline-separated tickers.")
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
    "--no-output-files",
    is_flag=True,
    default=False,
    help="Skip JSON/Markdown writes.",
)
def rs_breakout(
    market,
    as_of_arg,
    tickers,
    universe_file,
    universe_limit,
    benchmark,
    history_days,
    limit,
    json_path,
    md_path,
    no_output_files,
):
    """Screen Indian stocks for RS + SuperTrend + breakout/volume setups."""
    from pathlib import Path

    from rich.console import Console

    from screener.backtester.data import YFinancePriceFetcher

    console = Console()
    as_of_date: date = (
        as_of_arg.date() if isinstance(as_of_arg, datetime) else (as_of_arg or date.today())
    )

    resolved_benchmark = benchmark or RS_BREAKOUT_DEFAULT_BENCHMARKS[market]

    if tickers:
        universe = [t.strip() for t in tickers.split(",") if t.strip()]
    elif universe_file:
        path = Path(universe_file)
        if not path.exists():
            raise click.UsageError(f"--universe-file not found: {universe_file}")
        universe = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        universe = _load_rs_universe(market, int(universe_limit))

    if not universe:
        raise click.UsageError("Empty universe: pass --tickers or --universe-file.")

    fetcher = click.get_current_context().obj or YFinancePriceFetcher()
    console.print(
        f"[dim]Scanning {len(universe)} {market.upper()} tickers as of {as_of_date}...[/dim]"
    )
    bars_by_symbol, benchmark_bars = fetch_rs_breakout_price_data(
        universe,
        market,
        as_of_date,
        fetcher,
        benchmark=resolved_benchmark,
        history_days=int(history_days),
    )
    if market == "india":
        try:
            delivery_panel = load_india_delivery_for_scan(universe, as_of_date)
        except Exception as exc:
            console.print(
                f"[yellow]Delivery data load failed: {exc}. Full bucket may be empty.[/yellow]"
            )
            import pandas as pd

            delivery_panel = pd.DataFrame()
    else:
        import pandas as pd

        delivery_panel = pd.DataFrame()

    result = scan_rs_breakouts(
        bars_by_symbol,
        benchmark_bars,
        as_of_date,
        delivery_panel=delivery_panel,
        benchmark_symbol=resolved_benchmark,
        require_delivery=market == "india",
    )
    render_rs_breakout_result(result, console, limit=int(limit), market=market)

    if not no_output_files:
        json_default = f"rs_breakout_{market}_{as_of_date.isoformat()}.json"
        md_default = f"rs_breakout_{market}_{as_of_date.isoformat()}.md"
        write_rs_breakout_json(result, Path(json_path or json_default))
        write_rs_breakout_markdown(result, Path(md_path or md_default), market=market)
        console.print(
            f"\n[dim]Wrote {json_path or json_default} + {md_path or md_default}[/dim]"
        )


def _load_rs_universe(market: str, universe_limit: int) -> list[str]:
    from tradingview_screener import col

    price_floor = {"india": 50.0, "us": 5.0}[market]
    requested_limit = 5000 if universe_limit == 0 else universe_limit
    filters = [col("type") == "stock", col("close") >= price_floor]
    _total, df = scan(
        market=market,
        filters=filters,
        limit=requested_limit,
        order_by="volume",
    )
    return [str(t) for t in df["name"].dropna().tolist()]


@cli.command(name="promoter-buys")
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
    default="india",
    help="Market to screen. india => promoter % from screener.in (+ yfinance "
    "cross-check). us => yfinance Form 4 insider buys.",
)
@click.option(
    "--universe-size",
    type=int,
    default=200,
    help="Number of liquid tickers to fetch from TradingView before "
    "enrichment. Each ticker costs one HTTP/yfinance call.",
)
@click.option(
    "-n",
    "--limit",
    type=int,
    default=30,
    help="Maximum rows to display.",
)
@click.option(
    "--min-change",
    "min_change_pct",
    type=float,
    default=0.0,
    help="Minimum increase. India: percentage points of promoter holding "
    "vs. previous quarter (e.g. 0.5 = +0.5pp). US: ignored.",
)
@click.option(
    "--min-yf-net-pct",
    type=float,
    default=None,
    help="US only: minimum 6m net buy as fraction of total insider holding.",
)
@click.option(
    "--require-both",
    is_flag=True,
    help="India only: require BOTH screener.in promoter increase AND positive "
    "yfinance net insider buys. Default uses screener.in alone.",
)
@click.option(
    "--min-market-cap",
    type=float,
    default=None,
    help="Optional TradingView market_cap_basic floor before enrichment.",
)
@click.option(
    "--workers",
    type=int,
    default=10,
    help="Parallel enrichment workers.",
)
@click.option("--csv", "output_csv", is_flag=True, help="Output as CSV.")
def promoter_buys(
    market,
    universe_size,
    limit,
    min_change_pct,
    min_yf_net_pct,
    require_both,
    min_market_cap,
    workers,
    output_csv,
):
    """Find stocks where promoter/insider holding has increased.

    India: pulls quarterly shareholding from screener.in (via openscreener)
    and computes ΔPromoter % vs. the previous quarter. Cross-checked against
    yfinance 6-month insider buys.

    US: uses yfinance Form 4 aggregate (Purchases - Sales over 6m).
    """
    from tradingview_screener import Query, col
    from screener.scanner import _dedupe_listings
    from screener.insiders import (
        fetch_yfinance_insiders,
        fetch_openscreener_promoters,
        filter_promoter_increased,
    )

    exchanges = ("NSE", "BSE") if market == "india" else ("NASDAQ", "NYSE", "AMEX")
    min_close = 10.0 if market == "india" else 1.0
    base = [
        col("type") == "stock",
        col("close") >= min_close,
        col("volume") >= 1_000,
        col("exchange").isin(exchanges),
    ]
    if min_market_cap is not None:
        base.append(col("market_cap_basic") >= float(min_market_cap))

    query = (
        Query()
        .set_markets(MARKETS[market])
        .select("name", "description", "close", "change", "volume", "market_cap_basic")
        .where(*base)
        .order_by("volume", ascending=False)
        .limit(int(universe_size))
    )

    total, universe = call_with_resilience(
        "tradingview",
        "promoter universe",
        query.get_scanner_data,
        fallback=(0, pd.DataFrame()),
    )
    if not universe.empty:
        universe = _dedupe_listings(universe)

    if universe.empty:
        click.echo("No tickers returned from the base universe scan.")
        return

    click.echo(
        f"Universe: {len(universe)} liquid tickers (out of {total} in "
        f"{market}). Enriching..."
    )

    yf_df = fetch_yfinance_insiders(universe, market, max_workers=int(workers))

    if market == "india":
        os_df = fetch_openscreener_promoters(universe, max_workers=int(workers))
        if os_df.empty:
            click.echo("No openscreener data returned. Falling back to yfinance only.")
            insiders = yf_df
        else:
            insiders = os_df.merge(yf_df, on="name", how="left") if not yf_df.empty else os_df
    else:
        insiders = yf_df

    if insiders.empty:
        click.echo("No insider data returned for this universe.")
        return

    matches = filter_promoter_increased(
        insiders,
        market=market,
        min_promoter_change_pct=float(min_change_pct),
        min_yf_net_pct=min_yf_net_pct,
        require_both=bool(require_both),
    )
    if matches.empty:
        click.echo("No tickers passed the holding-increase filter.")
        return

    enriched = matches.merge(
        universe[["name", "description", "close", "change", "volume", "market_cap_basic"]],
        on="name",
        how="left",
    )
    if market == "india":
        enriched = enriched.sort_values(
            ["promoter_change", "yf_net_pct_6m"], ascending=False, na_position="last"
        )
    else:
        enriched = enriched.sort_values(
            ["yf_net_pct_6m", "yf_net_shares_6m"], ascending=False, na_position="last"
        )
    enriched = enriched.head(limit)

    if output_csv:
        print_csv(enriched)
        return

    print_insider_results(enriched, market, len(universe), len(matches))


if __name__ == "__main__":
    cli()
