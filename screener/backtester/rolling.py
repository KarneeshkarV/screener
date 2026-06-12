"""Rolling backtest orchestration and CLI."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import numpy as np
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
    _SlotState,
    _force_close_open_slots,
    _make_slot_state,
    _precompute_entry_signals,
    _precompute_filter_signals,
    _prepare_strategy_bars,
    _resolve_universe,
)
from screener.backtester.day_loop import DayLoop
from screener.backtester.fills import FillModel
from screener.backtester.data import PriceFetcher, build_price_fetcher, fetch_benchmark
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.metrics import compute_metrics, compute_regime_metrics
from screener.backtester.models import BacktestConfig, BacktestResult
from screener.backtester.pine import parse, required_lookback
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.regime import TREND_LABELS, classify_regimes
from screener.universes import load_current_universe, load_sp500_membership


@dataclass(frozen=True)
class _RollingCandidateMatrices:
    """Precomputed per-day matrices for vectorized candidate selection."""

    signal_mat: pd.DataFrame
    lookback_ok_mat: pd.DataFrame
    filter_mat: pd.DataFrame | None
    dollar_vol_mat: pd.DataFrame
    close_mat: pd.DataFrame
    volume_mat: pd.DataFrame
    bar_idx_mat: pd.DataFrame


def _build_rolling_candidate_matrices(
    bars_by_tv: dict[str, pd.DataFrame],
    entry_signals_by_tv: dict[str, pd.Series],
    filter_signals_by_tv: dict[str, pd.Series],
    master_dates: list[pd.Timestamp],
    lookback_required: int,
    membership_added: dict[str, date] | None = None,
    regime_allowed: pd.Series | None = None,
) -> _RollingCandidateMatrices:
    """Build once-per-run matrices for daily candidate scans."""
    master_ix = pd.DatetimeIndex(master_dates)
    valid_tickers = [
        tv for tv, bars in bars_by_tv.items() if bars is not None and not bars.empty
    ]
    signal_mat = (
        pd.DataFrame(entry_signals_by_tv)
        .reindex(master_ix)
        .reindex(columns=valid_tickers)
        .fillna(False)
        .astype(bool)
    )
    # Point-in-time eligibility: suppress entry signals before a symbol's
    # index "date added" so today's constituents are not backtested through
    # history they were never selectable in.
    if membership_added:
        for tv, added in membership_added.items():
            if tv in signal_mat.columns:
                signal_mat.loc[master_ix < pd.Timestamp(added), tv] = False
    # Benchmark-regime gate: suppress every entry signal on days whose
    # benchmark regime is not allowed (days missing from the benchmark
    # calendar inherit the most recent prior regime; warmup days are blocked).
    if regime_allowed is not None:
        allowed = (
            regime_allowed.reindex(master_ix, method="ffill").fillna(False).astype(bool)
        )
        signal_mat.loc[~allowed.to_numpy(), :] = False
    # Empty dict sentinel: no min-price / ADV filters configured.
    filter_mat: pd.DataFrame | None
    if filter_signals_by_tv:
        filter_mat = (
            pd.DataFrame(filter_signals_by_tv)
            .reindex(master_ix)
            .reindex(columns=valid_tickers)
            .fillna(False)
            .astype(bool)
        )
    else:
        filter_mat = None

    bar_cols: dict[str, np.ndarray] = {}
    lookback_cols: dict[str, np.ndarray] = {}
    close_cols: dict[str, np.ndarray] = {}
    volume_cols: dict[str, np.ndarray] = {}
    for tv in valid_tickers:
        bars = bars_by_tv[tv]
        close = bars["close"].astype(float).to_numpy()
        volume = bars["volume"].astype(float).to_numpy()
        pos = bars.index.searchsorted(master_ix, side="right") - 1
        pos = np.where(pos < 0, -1, pos)
        n = len(bars)
        has_bar = pos >= 0
        bar_cols[tv] = pos
        lookback_cols[tv] = (pos + 1 >= lookback_required + 1) & (pos + 1 < n) & has_bar
        close_cols[tv] = np.where(has_bar, close[pos], np.nan)
        volume_cols[tv] = np.where(has_bar, volume[pos], np.nan)

    bar_idx_mat = pd.DataFrame(bar_cols, index=master_ix)
    lookback_ok_mat = pd.DataFrame(lookback_cols, index=master_ix).astype(bool)
    close_mat = pd.DataFrame(close_cols, index=master_ix)
    volume_mat = pd.DataFrame(volume_cols, index=master_ix)
    dollar_vol_mat = close_mat * volume_mat
    return _RollingCandidateMatrices(
        signal_mat=signal_mat,
        lookback_ok_mat=lookback_ok_mat,
        filter_mat=filter_mat,
        dollar_vol_mat=dollar_vol_mat,
        close_mat=close_mat,
        volume_mat=volume_mat,
        bar_idx_mat=bar_idx_mat,
    )


def _candidate_rows_for_day(
    day: pd.Timestamp,
    matrices: _RollingCandidateMatrices,
    *,
    exclude: set[str],
) -> tuple[list[dict], list[str]]:
    """Evaluate entry signals for the full universe on one trading day."""
    warnings: list[str] = []
    eligible = matrices.signal_mat.loc[day] & matrices.lookback_ok_mat.loc[day]
    if matrices.filter_mat is not None:
        eligible = eligible & matrices.filter_mat.loc[day]
    if exclude:
        eligible = eligible & ~eligible.index.isin(exclude)
    dollar_vol = matrices.dollar_vol_mat.loc[day]
    eligible = eligible & dollar_vol.notna()
    ranked = dollar_vol[eligible].sort_values(ascending=False, kind="mergesort")
    rows: list[dict] = []
    for rank, (ticker, as_of_dollar_vol) in enumerate(ranked.items(), start=1):
        rows.append(
            {
                "ticker": ticker,
                "signal_idx": int(matrices.bar_idx_mat.at[day, ticker]),
                "as_of_close": float(matrices.close_mat.at[day, ticker]),
                "as_of_volume": float(matrices.volume_mat.at[day, ticker]),
                "as_of_dollar_vol": float(as_of_dollar_vol),
                "rank": rank,
                "role": "active",
            }
        )
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
    filter_signals_by_tv = _precompute_filter_signals(bars_by_tv, cfg)

    # Fetched once (with warmup history so SMA200-based regimes are defined)
    # and reused for the regime gate, the aligned curve, and regime metrics.
    benchmark = fetch_benchmark(cfg.benchmark, fetch_start, fetch_end, fetcher)
    regime_allowed: pd.Series | None = None
    if cfg.regime_filter:
        regime_allowed = classify_regimes(benchmark).isin(set(cfg.regime_filter))

    day_arrays: list[np.ndarray] = []
    for bars in bars_by_tv.values():
        if bars is None or bars.empty:
            continue
        idx = bars.index
        mask = (idx >= start_ts) & (idx <= end_ts)
        if mask.any():
            day_arrays.append(idx[mask].to_numpy())
    if not day_arrays:
        calendar = pd.bdate_range(start_ts, end_ts)
        equity = pd.Series(cfg.initial_capital, index=calendar, dtype=float)
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

    master_dates = list(pd.DatetimeIndex(np.unique(np.concatenate(day_arrays))))
    candidate_matrices = _build_rolling_candidate_matrices(
        bars_by_tv,
        entry_signals_by_tv,
        filter_signals_by_tv,
        master_dates,
        lookback,
        membership_added=dict(cfg.membership_added) or None,
        regime_allowed=regime_allowed,
    )
    portfolio = Portfolio(cfg.initial_capital, max(cfg.top, 1))
    slot_states: dict[int, _SlotState | None] = {
        slot_id: None for slot_id in range(max(cfg.top, 1))
    }
    slot_bars: dict[int, pd.DataFrame] = {}
    selection_rows: list[dict] = []

    fill_model = FillModel(cfg)
    day_loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=slot_states,
        slot_bars=slot_bars,
        fill_model=fill_model,
    )

    for day in master_dates:
        # Run the shared exit sequence, then treat every slot that is now empty
        # (whether already idle or freed today) as available for refill. Order
        # is slot-id ascending, matching the original interleaved loop.
        day_loop.process_exits_for_day(day)
        free_slots: list[int] = [
            slot_id for slot_id, state in slot_states.items() if state is None
        ]

        if not free_slots:
            continue

        candidates, day_warnings = _candidate_rows_for_day(
            day,
            candidate_matrices,
            exclude=_active_or_pending_tickers(slot_states),
        )
        warnings.extend(day_warnings)
        if not candidates:
            continue
        candidate_queue: deque[dict] = deque(candidates)

        for slot_id in free_slots:
            opened = False
            while candidate_queue and not opened:
                row = candidate_queue.popleft()
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
                    fill_model,
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
        fill_model=fill_model,
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
    benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()
    metrics = compute_metrics(equity, benchmark_aligned, trades, max(cfg.top, 1))
    metrics["unique_tickers"] = len({trade.ticker for trade in trades})
    metrics.update(compute_regime_metrics(benchmark, trades))

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
@click.option(
    "--point-in-time",
    is_flag=True,
    default=False,
    help=(
        "Reduce survivorship bias: only allow entries after a symbol's index "
        "'date added' (sp500 universe only). Removed ex-members are still "
        "absent because their delisted history is unavailable."
    ),
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
@click.option(
    "--regime-filter",
    "regime_filter_args",
    multiple=True,
    type=click.Choice(list(TREND_LABELS)),
    help=(
        "Only allow entries on days whose benchmark trend regime matches "
        "(repeatable). Warmup days with an unknown regime are suppressed."
    ),
)
@click.option("--csv", "output_csv", is_flag=True, help="Emit trade ledger as CSV.")
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a static, self-contained HTML tear-sheet to this file.",
)
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
    point_in_time,
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
    regime_filter_args,
    output_csv,
    report_path,
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
    membership_added: tuple[tuple[str, date], ...] = ()
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
        if point_in_time:
            if resolved_universe != "sp500":
                raise click.UsageError(
                    "--point-in-time currently supports only the sp500 universe."
                )
            added_by_symbol = load_sp500_membership(
                as_of=end_date, use_cache=not no_universe_cache
            )
            membership_added = tuple(
                (symbol, added)
                for symbol, added in added_by_symbol.items()
                if added is not None
            )
            universe_note += (
                f"; point-in-time entries via 'date added' "
                f"({len(membership_added)} dated symbols; removed ex-members not reconstructed)"
            )
        else:
            universe_note += (
                "; survivorship bias: today's members applied to history "
                "(pass --point-in-time to filter by 'date added')"
            )
    if point_in_time and not membership_added:
        raise click.UsageError(
            "--point-in-time requires an index universe; it cannot be used with "
            "--tickers or --universe-file."
        )

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
        membership_added=membership_added,
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
        regime_filter=tuple(dict.fromkeys(regime_filter_args)),
    )

    fetcher = click.get_current_context().obj or build_price_fetcher(
        auto_adjust=price_adjustment == "full"
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=start_date, end_date=end_date
    )
    if report_path:
        from screener.backtester.tearsheet import render_tearsheet

        render_tearsheet(
            result,
            report_path,
            title="Rolling Backtest Tear Sheet",
            extra_notes=[universe_note] if universe_note else [],
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
    if report_path:
        console.print(f"[green]Report:[/green] {report_path}")
    if dashboard:
        from screener.backtester.dashboard import render_dashboard, serve_dashboard

        dashboard_path = render_dashboard(result, dashboard_dir)
        console.print(f"[green]Dashboard:[/green] {dashboard_path}")
        console.print(
            f"[green]Serving:[/green] http://127.0.0.1:{dashboard_port}/{dashboard_path.name}"
        )
        console.print("[dim]Press Ctrl+C to stop the dashboard server.[/dim]")
        serve_dashboard(dashboard_path.parent, int(dashboard_port))
