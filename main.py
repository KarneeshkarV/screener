import click

from screener.criteria import CRITERIA
from screener.scanner import scan, MARKETS
from screener.display import print_results, print_csv


@click.group()
def cli():
    """Stock screener for US and Indian markets."""


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
    "criteria_name",
    type=click.Choice(list(CRITERIA.keys())),
    default="ema",
    help="Screening criteria.",
)
@click.option("-n", "--limit", default=50, help="Number of results.")
@click.option("--sort", "order_by", default="volume", help="Sort by column.")
@click.option("--csv", "output_csv", is_flag=True, help="Output as CSV.")
@click.option("--detail", is_flag=True, help="Show fundamental details (P/E, ROE, etc.).")
def screen(market, criteria_name, limit, order_by, output_csv, detail):
    """Screen stocks based on technical criteria."""
    criteria_fn = CRITERIA[criteria_name]
    filters = criteria_fn()

    total, df = scan(
        market=market,
        filters=filters,
        limit=limit,
        order_by=order_by,
        detail=detail,
    )

    if output_csv:
        print_csv(df)
    else:
        print_results(df, total, market, criteria_name)


if __name__ == "__main__":
    cli()
