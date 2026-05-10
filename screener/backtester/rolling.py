"""Rolling backtest orchestration and CLI."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

from screener.backtester.cli_common import (
    DEFAULT_BENCHMARK,
    build_slippage_model,
    parse_partial_exits,
    resolve_min_filters,
    resolve_strategy_exprs,
)
from screener.backtester.core import (
    _active_or_pending_tickers,
    _bar_index_on_or_before,
    _close_slot_at_day,
    _SlotState,
    _force_close_open_slots,
    _make_slot_state,
    _passes_entry_filters,
    _precompute_entry_signals,
    _prepare_strategy_bars,
    _resolve_universe,
)
from screener.backtester.data import PriceFetcher, build_price_fetcher, fetch_benchmark
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.metrics import compute_metrics
from screener.backtester.models import BacktestConfig, BacktestResult
from screener.backtester.pine import parse, required_lookback
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.universes import load_current_universe


def _candidate_rows_for_day(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_signals_by_ticker: dict[str, pd.Series],
    day: pd.Timestamp,
    lookback_required: int,
    cfg: BacktestConfig,
    *,
    exclude: set[str],
) -> tuple[list[dict], list[str]]:
    """Evaluate entry signals for the full universe on one trading day."""
    rows: list[dict] = []
    warnings: list[str] = []
    for ticker, bars in bars_by_ticker.items():
        if ticker in exclude or bars is None or bars.empty:
            continue
        signal_idx = _bar_index_on_or_before(bars, day)
        if signal_idx is None or signal_idx + 1 >= len(bars):
            continue
        history = bars.iloc[: signal_idx + 1]
        if len(history) < lookback_required + 1:
            continue
        passes, _ = _passes_entry_filters(bars, day, cfg)
        if not passes:
            continue
        signal = entry_signals_by_ticker.get(ticker)
        if signal is None or signal.empty or day not in signal.index:
            continue
        last = signal.loc[day]
        if pd.isna(last) or not bool(last):
            continue
        last_bar = history.iloc[-1]
        close = float(last_bar["close"])
        volume = float(last_bar["volume"])
        rows.append(
            {
                "ticker": ticker,
                "signal_idx": signal_idx,
                "as_of_close": close,
                "as_of_volume": volume,
                "as_of_dollar_vol": close * volume,
            }
        )
    rows.sort(key=lambda row: row["as_of_dollar_vol"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
        row["role"] = "active"
    return rows, warnings


def run_rolling_backtest(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    *,
    start_date: date,
    end_date: date,
) -> BacktestResult:
    """Run a daily rolling simulation over ``[start_date, end_date]``."""
    warnings: list[str] = []
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date must be >= start_date")

    entry_ast = parse(cfg.entry_expr)
    exit_ast = parse(cfg.exit_expr) if cfg.exit_expr else None
    lookback = required_lookback(entry_ast)
    if exit_ast is not None:
        lookback = max(lookback, required_lookback(exit_ast))

    from screener.backtester.data import tv_to_yf

    tv_symbols, univ_warnings = _resolve_universe(cfg)
    warnings.extend(univ_warnings)
    yf_by_tv = {tv: tv_to_yf(tv, cfg.market) for tv in tv_symbols}
    yf_symbols = list(dict.fromkeys(list(yf_by_tv.values()) + [cfg.benchmark]))

    warmup_days = max(lookback * 3 + 30, 365)
    fetch_start = (start_ts - pd.Timedelta(days=warmup_days)).date()
    fetch_end = end_ts.date()
    price_panel = fetcher.fetch(yf_symbols, fetch_start, fetch_end)

    bars_by_tv = {
        tv: price_panel.get(yf_by_tv[tv], pd.DataFrame()) for tv in tv_symbols
    }
    bars_by_tv, strategy_lookback = _prepare_strategy_bars(
        cfg,
        bars_by_tv,
        price_panel,
        tv_symbols,
        fetch_start,
        fetch_end,
        fetcher,
        warnings,
    )
    lookback = max(lookback, strategy_lookback)
    entry_signals_by_tv = _precompute_entry_signals(bars_by_tv, entry_ast, warnings)

    day_set: set[pd.Timestamp] = set()
    for bars in bars_by_tv.values():
        if bars is None or bars.empty:
            continue
        day_set.update(day for day in bars.index if start_ts <= day <= end_ts)
    if not day_set:
        calendar = pd.bdate_range(start_ts, end_ts)
        equity = pd.Series(cfg.initial_capital, index=calendar, dtype=float)
        benchmark = fetch_benchmark(cfg.benchmark, fetch_start, fetch_end, fetcher)
        benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()
        metrics = compute_metrics(equity, benchmark_aligned, [], max(cfg.top, 1))
        metrics["unique_tickers"] = 0
        return BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=equity,
            benchmark_curve=benchmark_aligned,
            metrics=metrics,
            warnings=warnings + ["no trading days with price data in rolling window"],
            selection=pd.DataFrame(),
        )

    master_dates = sorted(day_set)
    portfolio = Portfolio(cfg.initial_capital, max(cfg.top, 1))
    slot_states: dict[int, _SlotState | None] = {
        slot_id: None for slot_id in range(max(cfg.top, 1))
    }
    slot_bars: dict[int, pd.DataFrame] = {}
    selection_rows: list[dict] = []

    for day in master_dates:
        free_slots: list[int] = []
        for slot_id, state in list(slot_states.items()):
            if state is None:
                free_slots.append(slot_id)
                continue
            bars = slot_bars[slot_id]
            if _close_slot_at_day(
                slot_id=slot_id,
                state=state,
                bars=bars,
                day=day,
                cfg=cfg,
                portfolio=portfolio,
                slot_states=slot_states,
            ):
                free_slots.append(slot_id)

        if not free_slots:
            continue

        candidates, day_warnings = _candidate_rows_for_day(
            bars_by_tv,
            entry_signals_by_tv,
            day,
            lookback,
            cfg,
            exclude=_active_or_pending_tickers(slot_states),
        )
        warnings.extend(day_warnings)
        if not candidates:
            continue

        for slot_id in free_slots:
            opened = False
            while candidates and not opened:
                row = candidates.pop(0)
                ticker = str(row["ticker"])
                if ticker in _active_or_pending_tickers(slot_states):
                    continue
                bars = bars_by_tv.get(ticker, pd.DataFrame())
                if bars is None or bars.empty:
                    continue
                state, warn = _make_slot_state(
                    ticker,
                    bars,
                    int(row["signal_idx"]),
                    cfg,
                    exit_ast,
                    int(row["rank"]),
                )
                if state is None:
                    if warn:
                        warnings.append(f"{ticker}: {warn}")
                    continue
                if pd.Timestamp(state.entry_date) > end_ts:
                    continue
                portfolio.assign(ticker, int(row["rank"]), day.date())
                portfolio.open(
                    ticker=ticker,
                    entry_date=state.entry_date,
                    entry_price=state.entry_fill,
                    commission_bps=cfg.commission_bps,
                )
                slot_states[slot_id] = state
                slot_bars[slot_id] = bars
                selection_rows.append(
                    {
                        "ticker": ticker,
                        "signal_date": day.date(),
                        "as_of_close": row["as_of_close"],
                        "as_of_volume": row["as_of_volume"],
                        "as_of_dollar_vol": row["as_of_dollar_vol"],
                        "rank": row["rank"],
                        "role": "active",
                    }
                )
                opened = True

    _force_close_open_slots(
        slot_states=slot_states,
        slot_bars=slot_bars,
        cfg=cfg,
        portfolio=portfolio,
        end_ts=end_ts,
    )
    trades = portfolio.closed_trades()

    date_set: set[pd.Timestamp] = set(master_dates)
    for trade in trades:
        frame = bars_by_tv.get(trade.ticker)
        if frame is None or frame.empty:
            continue
        dates = frame.loc[
            (frame.index >= pd.Timestamp(trade.entry_date))
            & (frame.index <= pd.Timestamp(trade.exit_date))
        ].index
        date_set.update(dates.tolist())
    calendar = pd.DatetimeIndex(sorted(date_set))
    equity = build_equity_curve(calendar, trades, bars_by_tv, cfg.initial_capital)
    benchmark = fetch_benchmark(cfg.benchmark, fetch_start, fetch_end, fetcher)
    benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()
    metrics = compute_metrics(equity, benchmark_aligned, trades, max(cfg.top, 1))
    metrics["unique_tickers"] = len({trade.ticker for trade in trades})

    selection = pd.DataFrame(
        selection_rows,
        columns=[
            "ticker",
            "signal_date",
            "as_of_close",
            "as_of_volume",
            "as_of_dollar_vol",
            "rank",
            "role",
        ],
    )
    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=equity,
        benchmark_curve=benchmark_aligned,
        metrics=metrics,
        warnings=warnings,
        selection=selection,
    )


@click.command(name="backtest-rolling")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    help="Market to backtest.",
)
@click.option(
    "--start", "start_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--end", "end_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--years",
    type=int,
    default=1,
    show_default=True,
    help="Trailing calendar years when --start is omitted.",
)
@click.option("--hold", type=int, default=20, help="Holding period (trading days).")
@click.option("--top", type=int, default=10, help="Concurrent portfolio slots.")
@click.option("--entry", "entry_expr", default=None, help="Pine-like entry expression.")
@click.option("--exit", "exit_expr", default=None, help="Pine-like exit expression.")
@click.option(
    "--strategy", "strategy_name", default=None, help="Named strategy shortcut."
)
@click.option(
    "--universe",
    type=click.Choice(["sp500", "nifty50"]),
    default=None,
    help="Current index universe. Defaults to sp500 for US and nifty50 for India.",
)
@click.option(
    "--no-universe-cache",
    is_flag=True,
    default=False,
    help="Force live constituent refresh instead of today's cache.",
)
@click.option("--tickers", default=None, help="Comma-separated ticker list.")
@click.option(
    "--universe-file", default=None, help="Path to newline-separated ticker file."
)
@click.option(
    "--max-universe",
    type=int,
    default=0,
    help="Cap universe size before fetching prices. Pass 0 to disable.",
)
@click.option(
    "--stop-loss", type=float, default=None, help="Stop loss (fraction, e.g. 0.08)."
)
@click.option("--take-profit", type=float, default=None, help="Take profit (fraction).")
@click.option(
    "--trailing-stop", type=float, default=None, help="Trailing stop (fraction)."
)
@click.option(
    "--slippage-bps", type=float, default=0.0, help="Slippage per fill (bps)."
)
@click.option(
    "--commission-bps", type=float, default=0.0, help="Commission per fill (bps)."
)
@click.option("--initial-capital", type=float, default=100_000.0)
@click.option(
    "--benchmark",
    default=None,
    help="Benchmark symbol (default: SPY for US, ^NSEI for India).",
)
@click.option(
    "--min-price",
    type=float,
    default=None,
    help="Minimum signal-day close. Pass 0 to disable.",
)
@click.option(
    "--min-avg-dollar-volume",
    type=float,
    default=None,
    help="Minimum rolling mean dollar volume. Pass 0 to disable.",
)
@click.option(
    "--adv-window",
    type=int,
    default=20,
    help="Lookback bars for average dollar-volume filter.",
)
@click.option(
    "--slippage-model",
    type=click.Choice(["fixed", "half-spread", "vol-impact", "composite"]),
    default="fixed",
)
@click.option("--half-spread-bps", type=float, default=0.0)
@click.option("--vol-impact-k", type=float, default=0.1)
@click.option("--no-gap-fills", is_flag=True, default=False)
@click.option(
    "--entry-order", type=click.Choice(["moo", "moc", "limit"]), default="moo"
)
@click.option("--entry-limit-bps", type=float, default=None)
@click.option(
    "--partial-exit",
    "partial_exit_args",
    multiple=True,
    help="Scale-out tier as PROFIT_FRAC:SHARES_FRAC.",
)
@click.option(
    "--price-adjustment",
    type=click.Choice(["full", "splits_only", "none"]),
    default="full",
)
@click.option("--csv", "output_csv", is_flag=True, help="Emit trade ledger as CSV.")
@click.option(
    "--dashboard",
    is_flag=True,
    default=False,
    help="Render and serve a local interactive dashboard for this run.",
)
@click.option(
    "--dashboard-port",
    type=int,
    default=8765,
    show_default=True,
    help="Local port used when --dashboard is enabled.",
)
@click.option(
    "--dashboard-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(".screener/dashboards"),
    show_default=True,
    help="Directory for generated dashboard HTML files.",
)
def backtest_rolling(
    market,
    start_arg,
    end_arg,
    years,
    hold,
    top,
    entry_expr,
    exit_expr,
    strategy_name,
    universe,
    no_universe_cache,
    tickers,
    universe_file,
    max_universe,
    stop_loss,
    take_profit,
    trailing_stop,
    slippage_bps,
    commission_bps,
    initial_capital,
    benchmark,
    min_price,
    min_avg_dollar_volume,
    adv_window,
    slippage_model,
    half_spread_bps,
    vol_impact_k,
    no_gap_fills,
    entry_order,
    entry_limit_bps,
    partial_exit_args,
    price_adjustment,
    output_csv,
    dashboard,
    dashboard_port,
    dashboard_dir,
):
    """Run a true daily rolling backtest over a date window."""
    if output_csv and dashboard:
        raise click.UsageError("--csv and --dashboard cannot be used together.")

    entry_expr, exit_expr = resolve_strategy_exprs(strategy_name, entry_expr, exit_expr)
    slip_model = build_slippage_model(
        slippage_model, slippage_bps, half_spread_bps, vol_impact_k
    )
    partial_exits = parse_partial_exits(partial_exit_args)
    resolved_min_price, resolved_min_adv = resolve_min_filters(
        market, min_price, min_avg_dollar_volume
    )

    end_date = (
        end_arg.date() if isinstance(end_arg, datetime) else (end_arg or date.today())
    )
    start_date = (
        start_arg.date()
        if isinstance(start_arg, datetime)
        else (start_arg or (end_date - timedelta(days=365 * int(years))))
    )
    bench = benchmark or DEFAULT_BENCHMARK.get(market, "SPY")

    ticker_tuple = None
    universe_note = None
    if tickers:
        ticker_tuple = tuple(t.strip() for t in tickers.split(",") if t.strip())
    elif not universe_file:
        resolved_universe = universe or ("nifty50" if market == "india" else "sp500")
        loaded = load_current_universe(
            resolved_universe,
            as_of=end_date,
            use_cache=not no_universe_cache,
        )
        ticker_tuple = loaded.symbols
        universe_note = f"{loaded.name}: {len(loaded.symbols)} symbols from {loaded.source}; cache={loaded.cached_path}"

    cfg = BacktestConfig(
        market=market,
        as_of=end_date,
        hold=int(hold),
        top=int(top),
        strategy_name=strategy_name,
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
        reinvest=True,
        slippage_model=slip_model,
        gap_fills=not no_gap_fills,
        entry_order_type=entry_order,
        entry_limit_bps=entry_limit_bps,
        partial_exits=partial_exits,
        price_adjustment=price_adjustment,
    )

    fetcher = click.get_current_context().obj or build_price_fetcher(
        auto_adjust=price_adjustment == "full"
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=start_date, end_date=end_date
    )
    if output_csv:
        print_ledger_csv(result)
        return

    console = Console()
    console.print(
        f"[dim]Rolling window: {start_date.isoformat()} to {end_date.isoformat()}[/dim]"
    )
    if universe_note:
        console.print(f"[dim]Universe: {universe_note}[/dim]")
    print_backtest(result)
    if dashboard:
        from screener.backtester.dashboard import render_dashboard, serve_dashboard

        dashboard_path = render_dashboard(result, dashboard_dir)
        console.print(f"[green]Dashboard:[/green] {dashboard_path}")
        console.print(
            f"[green]Serving:[/green] http://127.0.0.1:{dashboard_port}/{dashboard_path.name}"
        )
        console.print("[dim]Press Ctrl+C to stop the dashboard server.[/dim]")
        serve_dashboard(dashboard_path.parent, int(dashboard_port))
