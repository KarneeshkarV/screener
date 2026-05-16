"""Click command for the TradingView-based technical screener."""

from __future__ import annotations

import click
from pydantic import ValidationError

from screener.cache import parse_ttl
from screener import history
from screener.commands.requests import ScreenRequest
from screener.criteria import (
    CRITERIA,
    combine,
    is_pipeline,
    registry as criteria_registry,
)
from screener.display import print_csv, print_results
from screener.scanner import MARKETS, scan


@click.command()
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
@click.option(
    "--detail", is_flag=True, help="Show fundamental details (P/E, ROE, etc.)."
)
@click.option("--refresh", is_flag=True, help="Bypass cached TradingView data.")
@click.option(
    "--cache-ttl",
    default="15m",
    show_default=True,
    help="TradingView cache TTL, e.g. 30s, 15m, 1h, off.",
)
def screen(
    market: str,
    criteria_names: tuple[str, ...],
    limit: int,
    order_by: str,
    output_csv: bool,
    detail: bool,
    refresh: bool,
    cache_ttl: str,
) -> None:
    """Screen stocks based on technical criteria."""
    try:
        req = ScreenRequest(
            market=market,
            criteria_names=criteria_names,
            limit=limit,
            order_by=order_by,
            output_csv=output_csv,
            detail=detail,
            refresh=refresh,
            cache_ttl=cache_ttl,
        )
    except ValidationError as exc:
        msg = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise click.UsageError(msg) from exc

    pipeline_names = [n for n in req.criteria_names if is_pipeline(n)]
    if pipeline_names:
        if len(req.criteria_names) > 1:
            raise click.UsageError(
                f"Pipeline criterion {pipeline_names[0]!r} cannot be combined "
                f"with other -c values; got {list(req.criteria_names)!r}."
            )
        runner = criteria_registry.get(pipeline_names[0])
        runner(
            market=req.market,
            limit=req.limit,
            output_csv=req.output_csv,
            refresh=req.refresh,
            cache_ttl=req.cache_ttl,
        )
        return

    criteria_fns = [CRITERIA[name] for name in req.criteria_names]
    filters = combine(*criteria_fns)()
    run_label = "+".join(req.criteria_names)

    total, df = scan(
        market=req.market,
        filters=filters,
        limit=req.limit,
        order_by=req.order_by,
        detail=req.detail,
        cache_ttl=parse_ttl(req.cache_ttl, default=900),
        refresh=req.refresh,
    )

    if req.output_csv:
        print_csv(df)
        return

    run_id = history.save_run(req.market, run_label, total, df)
    prev = history.previous_run(req.market, run_label, before_id=run_id)
    if prev is None:
        added, removed, first_run = [], [], True
    else:
        added, removed = history.diff(df, prev)
        first_run = False

    print_results(
        df,
        total,
        req.market,
        run_label,
        added=added,
        removed=removed,
        first_run=first_run,
    )
