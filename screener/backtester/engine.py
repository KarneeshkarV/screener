"""Historical backtest engine.

Orchestrates: universe → price fetch → Pine entry-signal selection as-of a
past date → per-ticker trade simulation → portfolio equity curve → metrics.

Accuracy guarantees:
  * signal bar and entry-fill bar are distinct (fill = next trading day open)
  * stop-loss checks bar.low, take-profit checks bar.high
  * on same-bar stop+target conflict (long), stop wins (conservative)
  * trailing stop tracks peak-high, updated AFTER each bar's stop/target check
  * exit expression fires at close of its signaling bar
  * time exit fires at close of ``entry_bar + hold``
  * slippage bps widens fills (adverse); commission bps applied to both sides
  * selection rank is preserved in output; never re-sorted by realized return
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from screener.backtester.data import PriceFetcher, ensure_date, fetch_benchmark
from screener.backtester.models import BacktestConfig, BacktestResult, ExitReason, Trade
from screener.backtester.pine import (
    PineError,
    evaluate,
    parse,
    required_lookback,
)
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.backtester.metrics import compute_metrics


@dataclass
class _SimOutcome:
    trade: Optional[Trade]
    warning: Optional[str]


def _slippage_factor(bps: float, buy: bool) -> float:
    delta = bps / 10_000.0
    return 1.0 + delta if buy else 1.0 - delta


def _passes_entry_filters(
    bars: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    cfg: BacktestConfig,
) -> tuple[bool, Optional[str]]:
    """Check min-price and liquidity filters against history up to ``as_of_ts``.

    Applied both at initial selection (signal date) AND at reserve promotion
    (exit date of a freed slot), so a ticker that was liquid at the run's
    ``as_of`` but has since crashed below the floor is correctly rejected.
    """
    if cfg.min_price is None and cfg.min_avg_dollar_volume is None:
        return True, None
    history = bars.loc[bars.index <= as_of_ts]
    if history.empty:
        return False, "no history"
    last = history.iloc[-1]
    close = float(last["close"])
    if cfg.min_price is not None and close < cfg.min_price:
        return False, f"price {close:.4f} < {cfg.min_price}"
    if cfg.min_avg_dollar_volume is not None:
        window = max(int(cfg.avg_dollar_volume_window), 1)
        tail = history.tail(window)
        if tail.empty:
            return False, "no volume history"
        adv = float((tail["close"] * tail["volume"]).mean())
        if not np.isfinite(adv) or adv < cfg.min_avg_dollar_volume:
            return False, f"adv {adv:.0f} < {cfg.min_avg_dollar_volume}"
    return True, None


def select_candidates(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast,
    as_of: pd.Timestamp,
    top_n: int,
    lookback_required: int,
    cfg: Optional[BacktestConfig] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate entry AST at ``as_of`` for each ticker and rank survivors.

    Returns (selection_df, warnings). selection_df columns:
    ``ticker, as_of_close, as_of_volume, as_of_dollar_vol, rank, role``.
    ``role`` is ``"active"`` for the top_n slots and ``"reserve"`` for deeper
    candidates (used by the event-driven reallocation loop in run_backtest).
    Rank is 1-based by descending dollar volume across active+reserves.
    """
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
            warnings.append(
                f"insufficient lookback ({len(history)} bars): {ticker}"
            )
            continue
        if cfg is not None:
            passes, _reason = _passes_entry_filters(bars, as_of, cfg)
            if not passes:
                filtered_count += 1
                continue
        try:
            signal = evaluate(entry_ast, history)
        except PineError as e:
            warnings.append(f"entry eval failed: {ticker}: {e}")
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
        warnings.append(
            f"filtered {filtered_count} tickers on price/liquidity filters"
        )
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
    df = pd.DataFrame(rows).sort_values(
        "as_of_dollar_vol", ascending=False, kind="stable"
    ).reset_index(drop=True)
    df = df.head(pool_limit).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["role"] = ["active" if i < top_n else "reserve" for i in range(len(df))]
    return df, warnings


@dataclass
class _SlotState:
    """Mutable state for a single slot during the event-driven simulation."""
    ticker: str
    entry_idx: int
    entry_date: date
    entry_fill: float
    signal_date: date
    rank: int
    stop_ref: Optional[float]
    target_ref: Optional[float]
    hold_limit_idx: int
    peak: float
    exit_signal: Optional[pd.Series]


def _make_slot_state(
    ticker: str,
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    exit_ast,
    rank: int,
) -> tuple[Optional[_SlotState], Optional[str]]:
    """Build the mutable per-slot state used by both simulate_ticker and the
    event-driven loop. Returns (state, warning)."""
    if signal_idx + 1 >= len(bars):
        return None, "no post-signal entry bar"
    entry_idx = signal_idx + 1
    entry_bar = bars.iloc[entry_idx]
    entry_open = float(entry_bar["open"])
    entry_fill = entry_open * _slippage_factor(cfg.slippage_bps, buy=True)
    exit_signal = None
    if exit_ast is not None:
        try:
            exit_signal = evaluate(exit_ast, bars).fillna(False).astype(bool)
        except PineError as e:
            return None, f"exit eval failed: {e}"
    stop_ref = entry_fill * (1.0 - cfg.stop_loss) if cfg.stop_loss else None
    target_ref = entry_fill * (1.0 + cfg.take_profit) if cfg.take_profit else None
    return (
        _SlotState(
            ticker=ticker,
            entry_idx=entry_idx,
            entry_date=bars.index[entry_idx].date(),
            entry_fill=entry_fill,
            signal_date=bars.index[signal_idx].date(),
            rank=rank,
            stop_ref=stop_ref,
            target_ref=target_ref,
            hold_limit_idx=entry_idx + cfg.hold,
            peak=entry_fill,
            exit_signal=exit_signal,
        ),
        None,
    )


def _check_exit_at_bar(
    state: _SlotState,
    bars: pd.DataFrame,
    i: int,
    cfg: BacktestConfig,
) -> Optional[tuple[float, ExitReason]]:
    """Evaluate exit rules for ``state`` at bars[i]. Returns (fill, reason) if
    the position exits on this bar, else None. Mutates ``state.peak`` in place
    after the stop/target/trail checks (matching the original ordering).

    Priority on same bar: stop+target→stop, then stop, then trail, then target
    (all use bar.low / bar.high); after that, peak is updated; then exit_expr
    and time exits fire on close.
    """
    bar = bars.iloc[i]
    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])

    trail_ref = (
        state.peak * (1.0 - cfg.trailing_stop)
        if cfg.trailing_stop
        else None
    )
    stop_hit = state.stop_ref is not None and low <= state.stop_ref
    target_hit = state.target_ref is not None and high >= state.target_ref
    trail_hit = trail_ref is not None and low <= trail_ref

    if stop_hit and target_hit:
        return (
            state.stop_ref * _slippage_factor(cfg.slippage_bps, buy=False),
            "stop",
        )
    if stop_hit:
        return (
            state.stop_ref * _slippage_factor(cfg.slippage_bps, buy=False),
            "stop",
        )
    if trail_hit:
        return (
            trail_ref * _slippage_factor(cfg.slippage_bps, buy=False),
            "trail",
        )
    if target_hit:
        return (
            state.target_ref * _slippage_factor(cfg.slippage_bps, buy=False),
            "target",
        )

    # update peak AFTER stop checks (first post-entry bar's trail_ref uses
    # entry_fill, not max(entry_fill, this_bar.high))
    if high > state.peak:
        state.peak = high

    if state.exit_signal is not None and bool(state.exit_signal.iloc[i]):
        return (
            close * _slippage_factor(cfg.slippage_bps, buy=False),
            "exit_expr",
        )
    if i >= state.hold_limit_idx:
        return (
            close * _slippage_factor(cfg.slippage_bps, buy=False),
            "time",
        )
    return None


def simulate_ticker(
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    exit_ast=None,
) -> _SimOutcome:
    """Simulate a single long-only trade starting from the bar AFTER signal_idx.

    ``signal_idx`` is the positional index of the as-of/signal bar in ``bars``.
    Entry fills at bars[signal_idx + 1]['open'] (adverse slippage applied).
    Returns ``_SimOutcome`` with ``trade=None`` if no post-signal bar exists.
    """
    state, warning = _make_slot_state(
        ticker="", bars=bars, signal_idx=signal_idx, cfg=cfg, exit_ast=exit_ast, rank=0
    )
    if state is None:
        return _SimOutcome(trade=None, warning=warning)

    for i in range(state.entry_idx + 1, len(bars)):
        exit_ = _check_exit_at_bar(state, bars, i, cfg)
        if exit_ is not None:
            fill, reason = exit_
            return _SimOutcome(
                _make_exit(
                    state.entry_date,
                    state.entry_fill,
                    bars.index[i].date(),
                    fill,
                    reason,
                    signal_idx_bar=state.signal_date,
                ),
                None,
            )

    last_bar = bars.iloc[-1]
    last_date = bars.index[-1].date()
    fill = float(last_bar["close"]) * _slippage_factor(cfg.slippage_bps, buy=False)
    return _SimOutcome(
        _make_exit(
            state.entry_date,
            state.entry_fill,
            last_date,
            fill,
            "eod",
            signal_idx_bar=state.signal_date,
        ),
        None,
    )


def _make_exit(
    entry_date: date,
    entry_fill: float,
    exit_date: date,
    exit_fill: float,
    reason: ExitReason,
    signal_idx_bar: date,
) -> Trade:
    """Return a partial Trade with only the price/date/reason fields set.

    The caller (run_backtest) fills shares/cost/pnl via the shared Portfolio.
    rank and ticker are also set by the caller.
    """
    return Trade(
        ticker="",
        rank=0,
        signal_date=signal_idx_bar,
        entry_date=entry_date,
        entry_price=entry_fill,
        exit_date=exit_date,
        exit_price=exit_fill,
        exit_reason=reason,
        shares=0.0,
        entry_cost=0.0,
        exit_value=0.0,
        pnl=0.0,
        return_pct=0.0,
    )


_NO_UNIVERSE_MSG = (
    "No universe provided: pass --tickers or --universe-file. The TradingView "
    "current-screener fallback was removed because it injects survivorship bias "
    "(delisted or deleveraged tickers as of the signal date would be silently "
    "excluded)."
)


def _resolve_universe(cfg: BacktestConfig) -> tuple[list[str], list[str]]:
    """Return (tv_symbols, warnings) for the configured universe.

    Raises ValueError if neither ``cfg.tickers`` nor ``cfg.universe_file`` is
    set. The historical TradingView fallback was removed to avoid survivorship
    bias; callers must supply an as-of universe explicitly.
    """
    warnings: list[str] = []
    if cfg.tickers:
        return list(cfg.tickers), warnings
    if cfg.universe_file:
        from pathlib import Path
        content = Path(cfg.universe_file).read_text()
        tickers = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return tickers, warnings
    raise ValueError(_NO_UNIVERSE_MSG)


def _eligible_reserve_signal_idx(
    bars: pd.DataFrame,
    exit_day: pd.Timestamp,
    cfg: BacktestConfig,
    entry_ast,
    lookback: int,
) -> Optional[int]:
    """Return the bar index at which a reserve can enter (signal_idx) if it
    passes filters and the entry AST as-of ``exit_day``; else None.

    Implements the user-requested semantic: *"when it returns the market, find
    the filter on that day and enter"* — the min-price / liquidity filter is
    re-evaluated on the exit day, not on the original ``as_of``.
    """
    history_mask = bars.index <= exit_day
    if not history_mask.any():
        return None
    history = bars.loc[history_mask]
    if len(history) < lookback + 1:
        return None
    passes, _ = _passes_entry_filters(bars, exit_day, cfg)
    if not passes:
        return None
    try:
        signal = evaluate(entry_ast, history)
    except PineError:
        return None
    if signal.empty or pd.isna(signal.iloc[-1]) or not bool(signal.iloc[-1]):
        return None
    return int(np.where(history_mask)[0][-1])


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
    """Chronological event-driven simulator with optional reserve rotation.

    Each active slot is opened at its as-of entry bar. On each subsequent bar
    we check exits across all open slots (processed in slot_id order); on an
    exit, if ``cfg.reinvest`` is True, we walk the reserve list in rank order
    and fill the freed slot with the next reserve whose min-price/liquidity
    filter AND entry AST pass as-of the exit date. Slots that can't be
    refilled stay idle (proceeds live in ``portfolio.cash()``).

    Force-closes any still-open slot at its last available bar with
    reason=``eod`` (matching the single-ticker simulator's end-of-data rule).
    """
    slot_states: dict[int, Optional[_SlotState]] = {}
    slot_bars: dict[int, pd.DataFrame] = {}

    # Initial active opens
    for slot_id, row in actives_df.iterrows():
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
            ticker, bars, signal_idx, cfg, exit_ast, int(row["rank"])
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

    taken: set[str] = {
        s.ticker for s in slot_states.values() if s is not None
    }
    reserve_queue: list[dict] = reserves_df.to_dict("records")

    # Master calendar: union of all bars' trading days in the simulation window
    horizon_end = as_of_ts + pd.Timedelta(days=max(cfg.hold * 3 + 60, 90))
    day_set: set[pd.Timestamp] = set()
    for bars in bars_by_tv.values():
        if bars is None or bars.empty:
            continue
        for d in bars.index:
            if as_of_ts < d <= horizon_end:
                day_set.add(d)
    master_dates = sorted(day_set)

    for day in master_dates:
        # Step 1: check exits for every open slot
        freed: list[int] = []
        for slot_id, state in list(slot_states.items()):
            if state is None:
                continue
            bars = slot_bars[slot_id]
            if day not in bars.index:
                continue
            i = bars.index.get_loc(day)
            if isinstance(i, slice) or not isinstance(i, int):
                # duplicate index — defensive
                continue
            if i < state.entry_idx + 1:
                continue  # not yet past entry bar
            exit_ = _check_exit_at_bar(state, bars, i, cfg)
            if exit_ is None:
                continue
            fill, reason = exit_
            portfolio.close(
                ticker=state.ticker,
                exit_date=day.date(),
                exit_price=fill,
                reason=reason,
                commission_bps=cfg.commission_bps,
            )
            slot_states[slot_id] = None
            freed.append(slot_id)

        if not cfg.reinvest or not freed:
            continue

        # Step 2: fill freed slots from the reserve queue (rank order)
        for slot_id in freed:
            while reserve_queue:
                r = reserve_queue.pop(0)
                ticker = r["ticker"]
                if ticker in taken:
                    continue
                bars = bars_by_tv.get(ticker, pd.DataFrame())
                if bars is None or bars.empty:
                    continue
                signal_idx = _eligible_reserve_signal_idx(
                    bars, day, cfg, entry_ast, lookback
                )
                if signal_idx is None:
                    continue
                state, warn = _make_slot_state(
                    ticker, bars, signal_idx, cfg, exit_ast, int(r["rank"])
                )
                if state is None:
                    if warn:
                        warnings.append(f"{ticker}: {warn}")
                    continue
                portfolio.assign(ticker, int(r["rank"]), day.date())
                portfolio.open(
                    ticker=ticker,
                    entry_date=state.entry_date,
                    entry_price=state.entry_fill,
                    commission_bps=cfg.commission_bps,
                )
                slot_states[slot_id] = state
                slot_bars[slot_id] = bars
                taken.add(ticker)
                break

    # Force-close any still-open positions at their last available bar.
    for slot_id, state in list(slot_states.items()):
        if state is None:
            continue
        bars = slot_bars[slot_id]
        tail = bars.loc[bars.index > pd.Timestamp(state.entry_date)]
        if tail.empty:
            continue
        last_bar = tail.iloc[-1]
        last_date = tail.index[-1].date()
        fill = float(last_bar["close"]) * _slippage_factor(
            cfg.slippage_bps, buy=False
        )
        portfolio.close(
            ticker=state.ticker,
            exit_date=last_date,
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
    yf_symbols = list(dict.fromkeys(yf_by_tv.values()))

    # Fetch a window generous enough for lookback + hold + buffer
    start = (as_of_ts - pd.Timedelta(days=max(lookback * 2 + 30, 365))).date()
    end = (as_of_ts + pd.Timedelta(days=cfg.hold * 2 + 30)).date()
    price_panel = fetcher.fetch(yf_symbols, start, end)

    bars_by_tv = {tv: price_panel.get(yf_by_tv[tv], pd.DataFrame()) for tv in tv_symbols}

    selection, sel_warnings = select_candidates(
        bars_by_tv, entry_ast, as_of_ts, cfg.top, lookback, cfg
    )
    warnings.extend(sel_warnings)

    if selection.empty:
        # still build benchmark + empty curve for structural consistency
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

    # Build calendar: union of all trade dates + benchmark dates
    date_set: set[pd.Timestamp] = set()
    for t in trades:
        frame = bars_by_tv.get(t.ticker)
        if frame is None or frame.empty:
            continue
        dates = frame.loc[
            (frame.index >= pd.Timestamp(t.entry_date))
            & (frame.index <= pd.Timestamp(t.exit_date))
        ].index
        date_set.update(dates.tolist())
    if not date_set:
        # no trades made a real entry — fall back to a simple business calendar
        date_set.update(
            pd.date_range(
                as_of_ts, as_of_ts + pd.Timedelta(days=cfg.hold * 2), freq="B"
            ).tolist()
        )
    calendar = pd.DatetimeIndex(sorted(date_set))
    equity = build_equity_curve(calendar, trades, bars_by_tv, cfg.initial_capital)
    # residual cash in unused slots stays idle but included by build_equity_curve
    # because it starts from initial_capital and subtracts only what's deployed.

    benchmark = fetch_benchmark(cfg.benchmark, start, end, fetcher)
    benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()

    metrics = compute_metrics(equity, benchmark_aligned, trades, slot_count)

    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=equity,
        benchmark_curve=benchmark_aligned,
        metrics=metrics,
        warnings=warnings,
        selection=selection,
    )
