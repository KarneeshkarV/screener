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


def select_candidates(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast,
    as_of: pd.Timestamp,
    top_n: int,
    lookback_required: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate entry AST at ``as_of`` for each ticker and rank survivors.

    Returns (selection_df, warnings). selection_df columns:
    ``ticker, as_of_close, as_of_volume, as_of_dollar_vol, rank``.
    Rank is 1-based by descending dollar volume.
    """
    rows = []
    warnings: list[str] = []
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
    if not rows:
        return pd.DataFrame(
            columns=["ticker", "as_of_close", "as_of_volume", "as_of_dollar_vol", "rank"]
        ), warnings
    df = pd.DataFrame(rows).sort_values(
        "as_of_dollar_vol", ascending=False, kind="stable"
    ).reset_index(drop=True)
    df = df.head(top_n).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df, warnings


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
    if signal_idx + 1 >= len(bars):
        return _SimOutcome(
            trade=None,
            warning="no post-signal entry bar",
        )
    entry_bar = bars.iloc[signal_idx + 1]
    entry_idx = signal_idx + 1
    entry_open = float(entry_bar["open"])
    entry_fill = entry_open * _slippage_factor(cfg.slippage_bps, buy=True)
    entry_date = bars.index[entry_idx].date()

    # local portfolio of 1 slot for this single simulation — but engine uses
    # a shared Portfolio across tickers. Here we just compute exit event.
    # The caller threads fills into the shared Portfolio.

    stop_ref = None
    target_ref = None
    if cfg.stop_loss:
        stop_ref = entry_fill * (1.0 - cfg.stop_loss)
    if cfg.take_profit:
        target_ref = entry_fill * (1.0 + cfg.take_profit)

    peak = entry_fill
    exit_signal = None  # pre-compute exit_expr series up to end of bars
    if exit_ast is not None:
        try:
            exit_signal = evaluate(exit_ast, bars).fillna(False).astype(bool)
        except PineError as e:
            return _SimOutcome(trade=None, warning=f"exit eval failed: {e}")

    hold_limit_idx = entry_idx + cfg.hold  # time-exit at close of this bar

    for i in range(entry_idx + 1, len(bars)):
        bar = bars.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        bar_date = bars.index[i].date()

        # trailing stop reference uses peak AT START of this bar
        trail_ref = None
        if cfg.trailing_stop:
            trail_ref = peak * (1.0 - cfg.trailing_stop)

        stop_hit = stop_ref is not None and low <= stop_ref
        target_hit = target_ref is not None and high >= target_ref
        trail_hit = trail_ref is not None and low <= trail_ref

        # conflict resolution: any downside stop beats target (conservative)
        if stop_hit and target_hit:
            fill = stop_ref * _slippage_factor(cfg.slippage_bps, buy=False)
            return _SimOutcome(
                trade=_make_exit(
                    entry_date, entry_fill, bar_date, fill, "stop", signal_idx_bar=bars.index[signal_idx].date()
                ),
                warning=None,
            )
        if stop_hit:
            fill = stop_ref * _slippage_factor(cfg.slippage_bps, buy=False)
            return _SimOutcome(
                _make_exit(entry_date, entry_fill, bar_date, fill, "stop",
                           signal_idx_bar=bars.index[signal_idx].date()),
                None,
            )
        if trail_hit:
            fill = trail_ref * _slippage_factor(cfg.slippage_bps, buy=False)
            return _SimOutcome(
                _make_exit(entry_date, entry_fill, bar_date, fill, "trail",
                           signal_idx_bar=bars.index[signal_idx].date()),
                None,
            )
        if target_hit:
            fill = target_ref * _slippage_factor(cfg.slippage_bps, buy=False)
            return _SimOutcome(
                _make_exit(entry_date, entry_fill, bar_date, fill, "target",
                           signal_idx_bar=bars.index[signal_idx].date()),
                None,
            )

        # update peak AFTER stop checks, so first post-entry bar's trail ref
        # is entry_fill, not max(entry_fill, this_bar.high)
        if high > peak:
            peak = high

        if exit_signal is not None and bool(exit_signal.iloc[i]):
            fill = close * _slippage_factor(cfg.slippage_bps, buy=False)
            return _SimOutcome(
                _make_exit(entry_date, entry_fill, bar_date, fill, "exit_expr",
                           signal_idx_bar=bars.index[signal_idx].date()),
                None,
            )

        if i >= hold_limit_idx:
            fill = close * _slippage_factor(cfg.slippage_bps, buy=False)
            return _SimOutcome(
                _make_exit(entry_date, entry_fill, bar_date, fill, "time",
                           signal_idx_bar=bars.index[signal_idx].date()),
                None,
            )

    # ran off end of available data — close at last available bar
    last_bar = bars.iloc[-1]
    last_date = bars.index[-1].date()
    fill = float(last_bar["close"]) * _slippage_factor(cfg.slippage_bps, buy=False)
    return _SimOutcome(
        _make_exit(entry_date, entry_fill, last_date, fill, "eod",
                   signal_idx_bar=bars.index[signal_idx].date()),
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


def _resolve_universe(cfg: BacktestConfig) -> tuple[list[str], list[str]]:
    """Return (tv_symbols, warnings) for the configured universe."""
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
    # fallback: current-liquid scan via TradingView (survivorship-biased)
    warnings.append(
        "Using TradingView current-universe fallback — introduces survivorship "
        "bias. Pass --tickers or --universe-file for an as-of universe."
    )
    try:
        from screener.scanner import scan
        _total, df = scan(
            market=cfg.market,
            filters=[],
            limit=cfg.max_universe,
            order_by="volume",
        )
        tickers = [str(t) for t in df["name"].dropna().tolist()]
    except Exception as e:
        warnings.append(f"TradingView universe fetch failed: {e}")
        tickers = []
    return tickers, warnings


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
        bars_by_tv, entry_ast, as_of_ts, cfg.top, lookback
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

    slot_count = max(cfg.top, len(selection))
    portfolio = Portfolio(cfg.initial_capital, slot_count)
    for _, row in selection.iterrows():
        portfolio.assign(row["ticker"], int(row["rank"]), cfg.as_of)

    # simulate each selected ticker and record into portfolio
    for _, row in selection.iterrows():
        tv_ticker = row["ticker"]
        bars = bars_by_tv.get(tv_ticker)
        if bars is None or bars.empty:
            warnings.append(f"no data during sim: {tv_ticker}")
            continue
        history_mask = bars.index <= as_of_ts
        if not history_mask.any():
            warnings.append(f"no history at as_of: {tv_ticker}")
            continue
        signal_pos = int(np.where(history_mask)[0][-1])
        outcome = simulate_ticker(bars, signal_pos, cfg, exit_ast=exit_ast)
        if outcome.warning:
            warnings.append(f"{tv_ticker}: {outcome.warning}")
        if outcome.trade is None:
            continue
        partial = outcome.trade
        portfolio.open(
            ticker=tv_ticker,
            entry_date=partial.entry_date,
            entry_price=partial.entry_price,
            commission_bps=cfg.commission_bps,
        )
        # mark peak during the held window (for reporting consistency only)
        portfolio.close(
            ticker=tv_ticker,
            exit_date=partial.exit_date,
            exit_price=partial.exit_price,
            reason=partial.exit_reason,
            commission_bps=cfg.commission_bps,
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
