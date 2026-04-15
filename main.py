import click

from screener import history
from screener.criteria import CRITERIA, combine
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


if __name__ == "__main__":
    cli()
