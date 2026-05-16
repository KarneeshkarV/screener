from datetime import date, datetime
from pathlib import Path

import click

from screener import history
from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.historical import backtest_historical
from screener.backtester.rolling import backtest_rolling
from screener.criteria import CRITERIA, combine
from screener.display import print_csv, print_results
from screener.ml_signal import BreakoutFeatureExtractor, MissingMLDependencyError, SignalConfidenceModel
from screener.ml_signal_cli import predict_cmd, feature_importance, train_model
from screener.scanner import MARKETS, scan
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


@click.group()
def cli():
    """Stock screener for US and Indian markets."""


cli.add_command(unusual_volume)
cli.add_command(backtest_historical)
cli.add_command(backtest_rolling)
cli.add_command(train_model)
cli.add_command(predict_cmd)
cli.add_command(feature_importance)


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
@click.option(
    "--confidence-model",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to trained SignalConfidenceModel for filtering.",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=None,
    help="Minimum ML confidence (0-1) to include a breakout.",
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
    confidence_model,
    confidence_threshold,
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

    model = None
    if confidence_model:
        try:
            model = SignalConfidenceModel.load(confidence_model)
            console.print(f"[dim]Loaded confidence model from {confidence_model}[/dim]")
        except MissingMLDependencyError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Failed to load model: {exc}[/yellow]")

    result = scan_rs_breakouts(
        bars_by_symbol,
        benchmark_bars,
        as_of_date,
        delivery_panel=delivery_panel,
        benchmark_symbol=resolved_benchmark,
        require_delivery=market == "india",
        confidence_model=model,
        confidence_threshold=confidence_threshold,
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


if __name__ == "__main__":
    cli()
