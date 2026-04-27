from datetime import date, datetime

import click

from screener import history
from screener.criteria import CRITERIA, combine
from screener.scanner import scan, MARKETS
from screener.display import print_results, print_csv, print_insider_results


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


_DEFAULT_BENCHMARK = {"us": "SPY", "india": "^NSEI"}
_DEFAULT_MIN_PRICE = {"us": 1.0, "india": 10.0}
_DEFAULT_MIN_ADV = {"us": 1_000.0, "india": 100_000.0}  # avg daily dollar volume


@cli.command(name="backtest-historical")
@click.option(
    "-m",
    "--market",
    type=click.Choice(list(MARKETS.keys())),
    default="us",
    help="Market to backtest.",
)
@click.option(
    "--as-of",
    "as_of",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Signal evaluation date (YYYY-MM-DD).",
)
@click.option("--hold", type=int, default=20, help="Holding period (trading days).")
@click.option("--top", type=int, default=10, help="Top N tickers to select.")
@click.option("--entry", "entry_expr", default=None, help="Pine-like entry expression.")
@click.option("--exit", "exit_expr", default=None, help="Pine-like exit expression.")
@click.option(
    "--strategy",
    "strategy_name",
    default=None,
    help="Named strategy shortcut (overrides --entry/--exit if given).",
)
@click.option("--stop-loss", type=float, default=None, help="Stop loss (fraction, e.g. 0.08).")
@click.option("--take-profit", type=float, default=None, help="Take profit (fraction).")
@click.option("--trailing-stop", type=float, default=None, help="Trailing stop (fraction).")
@click.option("--slippage-bps", type=float, default=0.0, help="Slippage per fill (bps).")
@click.option("--commission-bps", type=float, default=0.0, help="Commission per fill (bps).")
@click.option("--initial-capital", type=float, default=100_000.0)
@click.option("--benchmark", default=None, help="Benchmark symbol (default: SPY for US, ^NSEI for India).")
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated ticker list.",
)
@click.option("--universe-file", default=None, help="Path to newline-separated ticker file.")
@click.option(
    "--max-universe",
    type=int,
    default=200,
    help="Cap supplied universe size before fetching prices. Pass 0 to disable.",
)
@click.option(
    "--min-price",
    type=float,
    default=None,
    help="Minimum as-of close to admit a ticker. Default: $1 (US) / ₹10 (India). "
    "Pass 0 to disable.",
)
@click.option(
    "--min-avg-dollar-volume",
    type=float,
    default=None,
    help="Minimum rolling-mean dollar volume (close*volume) over --adv-window. "
    "Default: $1,000 (US) / ₹100,000 (India). Pass 0 to disable.",
)
@click.option(
    "--adv-window",
    type=int,
    default=20,
    help="Lookback (bars) for average dollar-volume filter.",
)
@click.option(
    "--reserve-multiple",
    type=int,
    default=3,
    help="Deepen the selection pool to top*N for reserve rotation on exits.",
)
@click.option(
    "--no-reinvest",
    is_flag=True,
    default=False,
    help="Disable reserve rotation (freed cash stays idle, matches legacy behavior).",
)
@click.option(
    "--slippage-model",
    type=click.Choice(["fixed", "half-spread", "vol-impact", "composite"]),
    default="fixed",
    help="Slippage model. 'fixed' = constant bps (legacy); 'half-spread' adds "
    "quoted-spread cost; 'vol-impact' adds Almgren-Chriss sqrt-law impact; "
    "'composite' sums all three.",
)
@click.option(
    "--half-spread-bps",
    type=float,
    default=0.0,
    help="Half-spread charged on every fill (bps). Used by half-spread/composite.",
)
@click.option(
    "--vol-impact-k",
    type=float,
    default=0.1,
    help="Coefficient for sqrt-law market impact (vol-impact/composite).",
)
@click.option(
    "--no-gap-fills",
    is_flag=True,
    default=False,
    help="Disable gap-aware stop/target fills (fills always at reference price).",
)
@click.option(
    "--entry-order",
    type=click.Choice(["moo", "moc", "limit"]),
    default="moo",
    help="Entry order type. moo=next-bar open (default); moc=next-bar close; "
    "limit=limit order at close*(1 - entry_limit_bps/1e4).",
)
@click.option(
    "--entry-limit-bps",
    type=float,
    default=None,
    help="Discount below signal-bar close for limit entries (bps).",
)
@click.option(
    "--allow-reentry",
    is_flag=True,
    default=False,
    help="After a position closes, re-enter the same ticker if the entry "
    "signal fires again (up to --max-reentries times).",
)
@click.option(
    "--max-reentries",
    type=int,
    default=0,
    help="Maximum number of re-entries per slot when --allow-reentry is set.",
)
@click.option(
    "--partial-exit",
    "partial_exit_args",
    multiple=True,
    help="Scale-out tier as 'PROFIT_FRAC:SHARES_FRAC' (e.g. 0.05:0.5 = close "
    "half at +5%). Repeat to configure multiple tiers.",
)
@click.option(
    "--price-adjustment",
    type=click.Choice(["full", "splits_only", "none"]),
    default="full",
    help="Price-adjustment regime. full=legacy (yfinance auto_adjust=True); "
    "splits_only=split-adjust OHLC and credit dividends as cash; none=raw OHLC.",
)
@click.option("--csv", "output_csv", is_flag=True, help="Emit trade ledger as CSV.")
def backtest_historical(
    market,
    as_of,
    hold,
    top,
    entry_expr,
    exit_expr,
    strategy_name,
    stop_loss,
    take_profit,
    trailing_stop,
    slippage_bps,
    commission_bps,
    initial_capital,
    benchmark,
    tickers,
    universe_file,
    max_universe,
    min_price,
    min_avg_dollar_volume,
    adv_window,
    reserve_multiple,
    no_reinvest,
    slippage_model,
    half_spread_bps,
    vol_impact_k,
    no_gap_fills,
    entry_order,
    entry_limit_bps,
    allow_reentry,
    max_reentries,
    partial_exit_args,
    price_adjustment,
    output_csv,
):
    """Run an accurate historical backtest with Pine-like entry/exit expressions."""
    from screener.backtester import BacktestConfig, run_backtest
    from screener.backtester.data import YFinancePriceFetcher
    from screener.backtester.display import print_backtest, print_ledger_csv
    from screener.backtester.slippage import (
        CompositeSlippage,
        FixedBpsSlippage,
        HalfSpreadSlippage,
        VolumeImpactSlippage,
    )
    from screener.backtester.strategies import resolve_strategy

    if strategy_name:
        s = resolve_strategy(strategy_name)
        entry_expr = entry_expr or s.entry
        exit_expr = exit_expr or s.exit

    if not entry_expr:
        raise click.UsageError("--entry (or --strategy) is required.")

    if slippage_model == "fixed":
        slip_model = FixedBpsSlippage(float(slippage_bps))
    elif slippage_model == "half-spread":
        slip_model = HalfSpreadSlippage(float(half_spread_bps))
    elif slippage_model == "vol-impact":
        slip_model = VolumeImpactSlippage(float(vol_impact_k))
    else:  # composite
        slip_model = CompositeSlippage(
            models=(
                FixedBpsSlippage(float(slippage_bps)),
                HalfSpreadSlippage(float(half_spread_bps)),
                VolumeImpactSlippage(float(vol_impact_k)),
            )
        )

    partial_exits: tuple[tuple[float, float], ...] = ()
    if partial_exit_args:
        parsed: list[tuple[float, float]] = []
        for raw in partial_exit_args:
            try:
                profit_s, shares_s = raw.split(":", 1)
                parsed.append((float(profit_s), float(shares_s)))
            except ValueError as exc:
                raise click.UsageError(
                    f"--partial-exit expects PROFIT_FRAC:SHARES_FRAC, got {raw!r}"
                ) from exc
        partial_exits = tuple(parsed)

    bench = benchmark or _DEFAULT_BENCHMARK.get(market, "SPY")
    as_of_date: date = as_of.date() if isinstance(as_of, datetime) else as_of

    ticker_tuple = None
    if tickers:
        ticker_tuple = tuple(t.strip() for t in tickers.split(",") if t.strip())

    if not ticker_tuple and not universe_file:
        raise click.UsageError(
            "No universe provided: pass --tickers or --universe-file. "
            "The TradingView current-screener fallback was removed because it "
            "injects survivorship bias."
        )

    resolved_min_price = (
        _DEFAULT_MIN_PRICE.get(market) if min_price is None else min_price
    )
    if resolved_min_price == 0:
        resolved_min_price = None
    resolved_min_adv = (
        _DEFAULT_MIN_ADV.get(market)
        if min_avg_dollar_volume is None
        else min_avg_dollar_volume
    )
    if resolved_min_adv == 0:
        resolved_min_adv = None

    cfg = BacktestConfig(
        market=market,
        as_of=as_of_date,
        hold=int(hold),
        top=int(top),
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        slippage_bps=float(slippage_bps),
        commission_bps=float(commission_bps),
        initial_capital=float(initial_capital),
        benchmark=bench,
        tickers=ticker_tuple,
        universe_file=universe_file,
        max_universe=int(max_universe),
        min_price=resolved_min_price,
        min_avg_dollar_volume=resolved_min_adv,
        avg_dollar_volume_window=int(adv_window),
        reserve_multiple=int(reserve_multiple),
        reinvest=not no_reinvest,
        slippage_model=slip_model,
        gap_fills=not no_gap_fills,
        entry_order_type=entry_order,
        entry_limit_bps=entry_limit_bps,
        allow_reentry=bool(allow_reentry),
        max_reentries=int(max_reentries),
        partial_exits=partial_exits,
        price_adjustment=price_adjustment,
    )

    auto_adjust = price_adjustment == "full"
    fetcher = click.get_current_context().obj or YFinancePriceFetcher(
        auto_adjust=auto_adjust
    )
    result = run_backtest(cfg, fetcher)

    if output_csv:
        print_ledger_csv(result)
        return

    print_backtest(result)


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

    total, universe = query.get_scanner_data()
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
