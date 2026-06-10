"""Click command for FMP institutional ownership lookups (US only)."""

from __future__ import annotations

import click
import pandas as pd

from screener.display import print_csv, print_institutional_results


@click.command(name="institutional")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us"]),
    default="us",
    help="Market to query. Only 'us' is supported (FMP 13F filings).",
)
@click.option(
    "--tickers",
    required=True,
    help="Comma-separated US symbols, e.g. AAPL,MSFT.",
)
@click.option("--csv", "output_csv", is_flag=True, help="Output as CSV.")
@click.option("--refresh", is_flag=True, help="Bypass cached FMP data.")
@click.option(
    "--workers",
    type=int,
    default=8,
    help="Parallel FMP fetch workers.",
)
def institutional(
    market: str,
    tickers: str,
    output_csv: bool,
    refresh: bool,
    workers: int,
) -> None:
    """Show FMP institutional ownership per ticker, ranked by QoQ change."""
    from screener.insiders import _fmp_api_key
    from screener.institutional import fetch_fmp_institutional

    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not symbols:
        raise click.ClickException("--tickers must list at least one symbol.")

    api_key = _fmp_api_key()
    if not api_key:
        raise click.ClickException(
            "FMP_API_KEY is not set. Export it or add it to the project .env "
            "to use the institutional command."
        )

    df = fetch_fmp_institutional(
        symbols,
        api_key=api_key,
        max_workers=int(workers),
        refresh=refresh,
    )

    found = set(df["symbol"].astype(str)) if not df.empty else set()
    missing = sorted(set(symbols) - found)
    if missing:
        click.echo(
            f"No institutional data for: {', '.join(missing)}",
            err=True,
        )
    if df.empty:
        click.echo("No institutional ownership data returned.")
        return

    df = df.copy()
    df["qoq_change_shares"] = pd.to_numeric(
        df.get("qoq_change_shares"), errors="coerce"
    )
    df = df.sort_values(
        ["qoq_change_shares", "qoq_change_pct"],
        ascending=False,
        na_position="last",
    )

    if output_csv:
        print_csv(df)
        return

    print_institutional_results(df)
