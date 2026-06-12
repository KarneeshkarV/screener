"""Historical backtest orchestration and CLI."""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from screener.backtester.cli_common import (
    DEFAULT_BENCHMARK,
    build_slippage_model,
    parse_partial_exits,
    resolve_min_filters,
    resolve_strategy_exprs,
)
from screener.backtester.core import (
    _SlotState,
    _eligible_reserve_signal_idx,
    _make_slot_state,
    _passes_entry_filters,
    _prepare_strategy_bars,
    _resolve_universe,
)
from screener.backtester.data import PriceFetcher, build_price_fetcher, fetch_benchmark
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.metrics import compute_metrics, compute_regime_metrics
from screener.backtester.models import BacktestConfig, BacktestResult
from screener.backtester.pine import PineError, evaluate, parse, required_lookback
from screener.backtester.portfolio import Portfolio, build_equity_curve


def select_candidates(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast,
    as_of: pd.Timestamp,
    top_n: int,
    lookback_required: int,
    cfg: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate entry AST at ``as_of`` for each ticker and rank survivors."""
    rows = []
    warnings: list[str] = []
    filtered_count = 0
    reserve_multiple = cfg.reserve_multiple if cfg is not None else 1
    pool_limit = max(top_n * max(reserve_multiple, 1), top_n)
    for ticker, bars in bars_by_ticker.items():
        if bars is None or bars.empty:
            warnings.append(f"no data: {ticker}")
            continue
        history = bars.loc[bars.index <= as_of]
        if len(history) < lookback_required + 1:
            warnings.append(f"insufficient lookback ({len(history)} bars): {ticker}")
            continue
        if cfg is not None:
            passes, _reason = _passes_entry_filters(bars, as_of, cfg)
            if not passes:
                filtered_count += 1
                continue
        try:
            signal = evaluate(entry_ast, history)
        except PineError as exc:
            warnings.append(f"entry eval failed: {ticker}: {exc}")
            continue
        if signal.empty:
            continue
        last = signal.iloc[-1]
        if pd.isna(last) or not bool(last):
            continue
        last_bar = history.iloc[-1]
        close = float(last_bar["close"])
        volume = float(last_bar["volume"])
        rows.append(
            {
                "ticker": ticker,
                "as_of_close": close,
                "as_of_volume": volume,
                "as_of_dollar_vol": close * volume,
            }
        )
    if filtered_count:
        warnings.append(f"filtered {filtered_count} tickers on price/liquidity filters")
    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "as_of_close",
                "as_of_volume",
                "as_of_dollar_vol",
                "rank",
                "role",
            ]
        ), warnings
    df = (
        pd.DataFrame(rows)
        .sort_values("as_of_dollar_vol", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    df = df.head(pool_limit).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["role"] = ["active" if i < top_n else "reserve" for i in range(len(df))]
    return df, warnings


def _run_event_driven_sim(
    *,
    portfolio: Portfolio,
    actives_df: pd.DataFrame,
    reserves_df: pd.DataFrame,
    bars_by_tv: dict[str, pd.DataFrame],
    as_of_ts: pd.Timestamp,
    cfg: BacktestConfig,
    entry_ast,
    exit_ast,
    lookback: int,
    warnings: list[str],
) -> None:
    """Chronological event-driven simulator with optional reserve rotation."""
    from screener.backtester.day_loop import DayLoop
    from screener.backtester.fills import FillModel

    fill_model = FillModel(cfg)
    slot_states: dict[int, _SlotState | None] = {}
    slot_bars: dict[int, pd.DataFrame] = {}
    reentries_left: dict[int, int] = {}
    pending_reentry: dict[int, str] = {}

    for raw_slot_id, row in actives_df.iterrows():
        slot_id = int(raw_slot_id)
        ticker = row["ticker"]
        bars = bars_by_tv.get(ticker, pd.DataFrame())
        if bars is None or bars.empty:
            warnings.append(f"no data during sim: {ticker}")
            slot_states[slot_id] = None
            continue
        mask = bars.index <= as_of_ts
        if not mask.any():
            warnings.append(f"no history at as_of: {ticker}")
            slot_states[slot_id] = None
            continue
        signal_idx = int(np.where(mask)[0][-1])
        state, warn = _make_slot_state(
            ticker, bars, signal_idx, cfg, exit_ast, int(row["rank"]), fill_model
        )
        if state is None:
            if warn:
                warnings.append(f"{ticker}: {warn}")
            slot_states[slot_id] = None
            continue
        portfolio.assign(ticker, int(row["rank"]), cfg.as_of)
        portfolio.open(
            ticker=ticker,
            entry_date=state.entry_date,
            entry_price=state.entry_fill,
            commission_bps=cfg.commission_bps,
        )
        slot_states[slot_id] = state
        slot_bars[slot_id] = bars
        reentries_left[slot_id] = cfg.max_reentries if cfg.allow_reentry else 0

    taken = {state.ticker for state in slot_states.values() if state is not None}
    reserve_queue: deque[dict] = deque(reserves_df.to_dict("records"))

    horizon_end = as_of_ts + pd.Timedelta(days=max(cfg.hold * 3 + 60, 90))
    day_set: set[pd.Timestamp] = set()
    for bars in bars_by_tv.values():
        if bars is None or bars.empty:
            continue
        for current_day in bars.index:
            if as_of_ts < current_day <= horizon_end:
                day_set.add(current_day)
    master_dates = sorted(day_set)

    day_loop = DayLoop(
        portfolio=portfolio,
        cfg=cfg,
        slot_states=slot_states,
        slot_bars=slot_bars,
        fill_model=fill_model,
    )

    for day in master_dates:
        if pending_reentry:
            for slot_id, ticker in list(pending_reentry.items()):
                slot_frame = slot_bars.get(slot_id)
                if slot_frame is None or slot_frame.empty:
                    del pending_reentry[slot_id]
                    continue
                reentry_signal_idx = _eligible_reserve_signal_idx(
                    slot_frame, day, cfg, entry_ast, lookback
                )
                if reentry_signal_idx is None:
                    continue
                new_rank = portfolio._ranks.get(ticker, 0)
                state, warn = _make_slot_state(
                    ticker,
                    slot_frame,
                    reentry_signal_idx,
                    cfg,
                    exit_ast,
                    new_rank,
                    fill_model,
                )
                if state is None:
                    if warn:
                        warnings.append(f"{ticker} re-entry: {warn}")
                    del pending_reentry[slot_id]
                    continue
                portfolio.assign(ticker, new_rank, day.date())
                portfolio.open(
                    ticker=ticker,
                    entry_date=state.entry_date,
                    entry_price=state.entry_fill,
                    commission_bps=cfg.commission_bps,
                )
                slot_states[slot_id] = state
                del pending_reentry[slot_id]

        freed_slots = day_loop.process_exits_for_day(day)
        freed: list[int] = []
        for freed_slot in freed_slots:
            slot_id = freed_slot.slot_id
            freed.append(slot_id)
            if cfg.allow_reentry and reentries_left.get(slot_id, 0) > 0:
                reentries_left[slot_id] -= 1
                pending_reentry[slot_id] = freed_slot.state.ticker

        if not cfg.reinvest or not freed:
            continue

        for slot_id in freed:
            if slot_id in pending_reentry:
                continue
            while reserve_queue:
                reserve = reserve_queue.popleft()
                ticker = str(reserve["ticker"])
                if ticker in taken:
                    continue
                reserve_bars = bars_by_tv.get(ticker, pd.DataFrame())
                if reserve_bars is None or reserve_bars.empty:
                    continue
                reserve_signal_idx = _eligible_reserve_signal_idx(
                    reserve_bars, day, cfg, entry_ast, lookback
                )
                if reserve_signal_idx is None:
                    continue
                state, warn = _make_slot_state(
                    ticker,
                    reserve_bars,
                    reserve_signal_idx,
                    cfg,
                    exit_ast,
                    int(reserve["rank"]),
                    fill_model,
                )
                if state is None:
                    if warn:
                        warnings.append(f"{ticker}: {warn}")
                    continue
                portfolio.assign(ticker, int(reserve["rank"]), day.date())
                portfolio.open(
                    ticker=ticker,
                    entry_date=state.entry_date,
                    entry_price=state.entry_fill,
                    commission_bps=cfg.commission_bps,
                )
                slot_states[slot_id] = state
                slot_bars[slot_id] = reserve_bars
                taken.add(ticker)
                break

    for slot_id, state in list(slot_states.items()):
        if state is None:
            continue
        bars = slot_bars[slot_id]
        tail = bars.loc[bars.index > pd.Timestamp(state.entry_date)]
        if tail.empty:
            continue
        last_bar = tail.iloc[-1]
        fill = fill_model.exit_price(
            reason="eod",
            close=float(last_bar["close"]),
            adv_shares=state.adv_shares,
            sigma_daily=state.sigma_daily,
        )
        portfolio.close(
            ticker=state.ticker,
            exit_date=tail.index[-1].date(),
            exit_price=fill,
            reason="eod",
            commission_bps=cfg.commission_bps,
        )
        slot_states[slot_id] = None


def run_backtest(cfg: BacktestConfig, fetcher: PriceFetcher) -> BacktestResult:
    warnings: list[str] = []
    as_of_ts = pd.Timestamp(cfg.as_of)

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

    start = (as_of_ts - pd.Timedelta(days=max(lookback * 2 + 30, 365))).date()
    end = (as_of_ts + pd.Timedelta(days=cfg.hold * 2 + 30)).date()
    price_panel = fetcher.fetch(yf_symbols, start, end)

    bars_by_tv = {
        tv: price_panel.get(yf_by_tv[tv], pd.DataFrame()) for tv in tv_symbols
    }
    bars_by_tv, strategy_lookback = _prepare_strategy_bars(
        cfg, bars_by_tv, price_panel, tv_symbols, start, end, fetcher, warnings
    )
    lookback = max(lookback, strategy_lookback)

    selection, sel_warnings = select_candidates(
        bars_by_tv, entry_ast, as_of_ts, cfg.top, lookback, cfg
    )
    warnings.extend(sel_warnings)

    if selection.empty:
        calendar = pd.date_range(
            as_of_ts, as_of_ts + pd.Timedelta(days=cfg.hold * 2), freq="B"
        )
        equity = pd.Series(cfg.initial_capital, index=calendar, dtype=float)
        benchmark = fetch_benchmark(cfg.benchmark, start, end, fetcher)
        benchmark = benchmark.reindex(calendar, method="ffill").dropna()
        metrics = compute_metrics(equity, benchmark, [], max(cfg.top, 1))
        return BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=equity,
            benchmark_curve=benchmark,
            metrics=metrics,
            warnings=warnings,
            selection=selection,
        )

    actives_df = selection[selection["role"] == "active"].reset_index(drop=True)
    reserves_df = selection[selection["role"] == "reserve"].reset_index(drop=True)
    slot_count = max(cfg.top, len(actives_df))
    portfolio = Portfolio(cfg.initial_capital, slot_count)

    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives_df,
        reserves_df=reserves_df,
        bars_by_tv=bars_by_tv,
        as_of_ts=as_of_ts,
        cfg=cfg,
        entry_ast=entry_ast,
        exit_ast=exit_ast,
        lookback=lookback,
        warnings=warnings,
    )

    trades = portfolio.closed_trades()
    date_set: set[pd.Timestamp] = {as_of_ts.normalize()}
    for trade in trades:
        frame = bars_by_tv.get(trade.ticker)
        if frame is None or frame.empty:
            continue
        dates = frame.loc[
            (frame.index >= pd.Timestamp(trade.entry_date))
            & (frame.index <= pd.Timestamp(trade.exit_date))
        ].index
        date_set.update(dates.tolist())
    if not date_set:
        date_set.update(
            pd.date_range(
                as_of_ts, as_of_ts + pd.Timedelta(days=cfg.hold * 2), freq="B"
            ).tolist()
        )
    calendar = pd.DatetimeIndex(sorted(date_set))
    equity = build_equity_curve(calendar, trades, bars_by_tv, cfg.initial_capital)

    benchmark = fetch_benchmark(cfg.benchmark, start, end, fetcher)
    benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()
    metrics = compute_metrics(equity, benchmark_aligned, trades, slot_count)
    metrics.update(compute_regime_metrics(benchmark, trades))

    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=equity,
        benchmark_curve=benchmark_aligned,
        metrics=metrics,
        warnings=warnings,
        selection=selection,
    )


@click.command(name="backtest-historical")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
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
@click.option("--tickers", default=None, help="Comma-separated ticker list.")
@click.option(
    "--universe-file", default=None, help="Path to newline-separated ticker file."
)
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
    help="Minimum as-of close to admit a ticker. Default: $1 (US) / ₹10 (India). Pass 0 to disable.",
)
@click.option(
    "--min-avg-dollar-volume",
    type=float,
    default=None,
    help="Minimum rolling-mean dollar volume (close*volume) over --adv-window. Default: $1,000 (US) / ₹100,000 (India). Pass 0 to disable.",
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
    help="Slippage model. 'fixed' = constant bps (legacy); 'half-spread' adds quoted-spread cost; 'vol-impact' adds Almgren-Chriss sqrt-law impact; 'composite' sums all three.",
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
    help="Entry order type. moo=next-bar open (default); moc=next-bar close; limit=limit order at close*(1 - entry_limit_bps/1e4).",
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
    help="After a position closes, re-enter the same ticker if the entry signal fires again (up to --max-reentries times).",
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
    help="Scale-out tier as 'PROFIT_FRAC:SHARES_FRAC' (e.g. 0.05:0.5 = close half at +5%). Repeat to configure multiple tiers.",
)
@click.option(
    "--price-adjustment",
    type=click.Choice(["full", "splits_only", "none"]),
    default="full",
    help="Price-adjustment regime. full=legacy (yfinance auto_adjust=True); splits_only=split-adjust OHLC and credit dividends as cash; none=raw OHLC.",
)
@click.option("--csv", "output_csv", is_flag=True, help="Emit trade ledger as CSV.")
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a static, self-contained HTML tear-sheet to this file.",
)
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
    report_path,
):
    """Run an accurate historical backtest with Pine-like entry/exit expressions."""
    entry_expr, exit_expr = resolve_strategy_exprs(strategy_name, entry_expr, exit_expr)
    slip_model = build_slippage_model(
        slippage_model, slippage_bps, half_spread_bps, vol_impact_k
    )
    partial_exits = parse_partial_exits(partial_exit_args)
    bench = benchmark or DEFAULT_BENCHMARK.get(market, "SPY")
    as_of_date: date = as_of.date() if isinstance(as_of, datetime) else as_of

    ticker_tuple = None
    if tickers:
        ticker_tuple = tuple(t.strip() for t in tickers.split(",") if t.strip())
    if not ticker_tuple and not universe_file:
        raise click.UsageError(
            "No universe provided: pass --tickers or --universe-file. "
            "The TradingView current-screener fallback was removed because it injects survivorship bias."
        )

    resolved_min_price, resolved_min_adv = resolve_min_filters(
        market, min_price, min_avg_dollar_volume
    )
    cfg = BacktestConfig(
        market=market,
        as_of=as_of_date,
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

    fetcher = click.get_current_context().obj or build_price_fetcher(
        auto_adjust=price_adjustment == "full"
    )
    result = run_backtest(cfg, fetcher)
    if report_path:
        from screener.backtester.tearsheet import render_tearsheet

        universe_note = (
            f"explicit universe: {len(ticker_tuple)} tickers via --tickers"
            if ticker_tuple
            else f"universe file: {universe_file}"
        ) + "; survivorship bias: supplied list is not point-in-time"
        render_tearsheet(
            result,
            report_path,
            title="Historical Backtest Tear Sheet",
            extra_notes=[universe_note],
        )
    if output_csv:
        print_ledger_csv(result)
        return
    print_backtest(result)
    if report_path:
        click.echo(f"Report: {report_path}")
