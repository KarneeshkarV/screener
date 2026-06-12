"""Click command for promoter and insider buying screens."""

from __future__ import annotations

import click

from screener.cache import parse_ttl
from screener.display import print_csv, print_insider_results
from screener.scanner import MARKETS, _dedupe_listings, get_scanner_data_cached


@click.command(name="promoter-buys")
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
    default="india",
    help="Market to screen. india => promoter % from screener.in (+ yfinance "
    "cross-check). us => FMP Form 4 insider buys when FMP_API_KEY is set "
    "(+ yfinance cross-check), else yfinance only.",
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
@click.option(
    "--refresh",
    is_flag=True,
    help="Bypass cached TradingView/yfinance/screener.in data.",
)
@click.option(
    "--cache-ttl",
    default="15m",
    show_default=True,
    help="TradingView universe cache TTL, e.g. 30s, 15m, 1h, off.",
)
def promoter_buys(
    market: str,
    universe_size: int,
    limit: int,
    min_change_pct: float,
    min_yf_net_pct: float | None,
    require_both: bool,
    min_market_cap: float | None,
    workers: int,
    output_csv: bool,
    refresh: bool,
    cache_ttl: str,
) -> None:
    """Find stocks where promoter/insider holding has increased."""
    run_promoter_buys(
        market=market,
        universe_size=universe_size,
        limit=limit,
        min_change_pct=min_change_pct,
        min_yf_net_pct=min_yf_net_pct,
        require_both=require_both,
        min_market_cap=min_market_cap,
        workers=workers,
        output_csv=output_csv,
        refresh=refresh,
        cache_ttl=cache_ttl,
    )


def run_promoter_buys(
    *,
    market: str,
    universe_size: int,
    limit: int,
    min_change_pct: float,
    min_yf_net_pct: float | None,
    require_both: bool,
    min_market_cap: float | None,
    workers: int,
    output_csv: bool,
    refresh: bool,
    cache_ttl: str,
) -> None:
    """Run the promoter/insider-buying screen (no Click context required)."""
    from tradingview_screener import Query, col

    from screener.insiders import (
        fetch_fmp_insiders,
        fetch_openscreener_promoters,
        fetch_yfinance_insiders,
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

    columns = ["name", "description", "close", "change", "volume", "market_cap_basic"]
    parsed_ttl = parse_ttl(cache_ttl, default=900)
    total, universe = get_scanner_data_cached(
        query,
        key_parts=(
            "promoter_universe",
            market,
            [repr(f) for f in base],
            columns,
            int(universe_size),
        ),
        columns=columns,
        operation="promoter universe",
        cache_ttl=parsed_ttl,
        refresh=refresh,
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

    yf_df = fetch_yfinance_insiders(
        universe,
        market,
        max_workers=int(workers),
        refresh=refresh,
    )

    if market == "india":
        os_df = fetch_openscreener_promoters(
            universe,
            max_workers=int(workers),
            refresh=refresh,
        )
        if os_df.empty:
            click.echo("No openscreener data returned. Falling back to yfinance only.")
            insiders = yf_df
        else:
            insiders = (
                os_df.merge(yf_df, on="name", how="left") if not yf_df.empty else os_df
            )
    else:
        fmp_df = fetch_fmp_insiders(
            universe,
            market,
            max_workers=int(workers),
            refresh=refresh,
        )
        if not fmp_df.empty and "fmp_truncated" in fmp_df.columns:
            truncated = fmp_df.loc[
                fmp_df["fmp_truncated"].fillna(False).astype(bool), "fmp_symbol"
            ].astype(str)
            if not truncated.empty:
                click.echo(
                    "Warning: FMP insider history hit the page cap for: "
                    f"{', '.join(sorted(truncated))} — 6m net-buy totals may "
                    "be incomplete.",
                    err=True,
                )
        if fmp_df.empty:
            insiders = yf_df
        elif yf_df.empty:
            insiders = fmp_df
        else:
            insiders = fmp_df.merge(yf_df, on="name", how="outer")

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
        universe[
            ["name", "description", "close", "change", "volume", "market_cap_basic"]
        ],
        on="name",
        how="left",
    )
    if market == "india":
        enriched = enriched.sort_values(
            ["promoter_change", "yf_net_pct_6m"], ascending=False, na_position="last"
        )
    else:
        sort_cols = [
            c
            for c in ("fmp_net_shares_6m", "yf_net_pct_6m", "yf_net_shares_6m")
            if c in enriched.columns
        ]
        enriched = enriched.sort_values(sort_cols, ascending=False, na_position="last")
    enriched = enriched.head(limit)

    if output_csv:
        print_csv(enriched)
        return

    print_insider_results(enriched, market, len(universe), len(matches))
