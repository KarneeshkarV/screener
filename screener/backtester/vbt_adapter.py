"""vectorbt-backed adapter for fast signal portfolio experiments.

``run_vbt`` keeps the heavyweight vectorbt import lazy and supports the
overlap where vectorbt's signal engine and this project's event engine can be
made numerically comparable: long-only entries from Pine-like boolean signals,
optional exits, hold caps, stop-loss, take-profit, trailing-stop, commission,
and fixed-bps slippage.

Feature gaps are explicit. HalfSpread, VolumeImpact, and Composite slippage
models, non-MOO entries, and partial exits require the event-driven engine
because their fills depend on path, liquidity, or tranche state that a simple
``Portfolio.from_signals`` call cannot represent faithfully. Callers that
request those features should catch ``UnsupportedVbtFeatureError`` and run the
core event engine instead.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional, cast

import pandas as pd

from screener.backtester.core import (
    _SlotState,
    _active_or_pending_tickers,
    _bar_index_on_or_before,
    _close_slot_at_day,
    _force_close_open_slots,
    _make_slot_state,
)
from screener.backtester.metrics import compute_metrics
from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.backtester.pine import PineError, evaluate, required_lookback
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.backtester.slippage import (
    CompositeSlippage,
    FixedBpsSlippage,
    HalfSpreadSlippage,
    VolumeImpactSlippage,
)


class UnsupportedVbtFeatureError(ValueError):
    """Raised when a requested feature must use the event-driven engine."""


def _validate_supported(cfg: BacktestConfig) -> None:
    model = cfg.slippage_model
    if model is not None and not isinstance(model, FixedBpsSlippage):
        unsupported = (HalfSpreadSlippage, VolumeImpactSlippage, CompositeSlippage)
        if isinstance(model, unsupported):
            raise UnsupportedVbtFeatureError(
                "vectorbt fast path supports FixedBpsSlippage only; "
                "use the event-driven engine for half-spread, volume-impact, "
                "or composite slippage"
            )
        raise UnsupportedVbtFeatureError(
            f"vectorbt fast path does not support slippage model {type(model).__name__}"
        )
    if cfg.entry_order_type != "moo":
        raise UnsupportedVbtFeatureError(
            "vectorbt fast path supports only MOO next-open entries"
        )
    if cfg.partial_exits:
        raise UnsupportedVbtFeatureError(
            "vectorbt fast path does not support partial exits; use the event engine"
        )


def _clean_bars(bars_by_ticker: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    for ticker, bars in bars_by_ticker.items():
        if bars is None or bars.empty:
            continue
        frame = bars.sort_index().copy()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        cleaned[ticker] = frame
    return cleaned


def _calendar(bars_by_ticker: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    days: set[pd.Timestamp] = set()
    for bars in bars_by_ticker.values():
        days.update(pd.Timestamp(day).normalize() for day in bars.index)
    return pd.DatetimeIndex(sorted(days))


def _series_panel(
    bars_by_ticker: dict[str, pd.DataFrame],
    field: str,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for ticker, bars in bars_by_ticker.items():
        data[ticker] = pd.to_numeric(bars[field], errors="coerce").reindex(index)
    return pd.DataFrame(data, index=index)


def _signal_panel(
    signals_by_ticker: dict[str, pd.Series],
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    data = {
        ticker: signal.reindex(index).fillna(False).astype(bool)
        for ticker, signal in signals_by_ticker.items()
    }
    return pd.DataFrame(data, index=index).fillna(False).astype(bool)


def _call_vectorbt_from_signals(
    cfg: BacktestConfig,
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_signals: dict[str, pd.Series],
    exit_signals: dict[str, pd.Series],
    index: pd.DatetimeIndex,
) -> None:
    """Invoke vectorbt lazily so import cost is paid only by ``run_vbt``."""
    vbt = cast(Any, import_module("vectorbt"))

    close = _series_panel(bars_by_ticker, "close", index).ffill()
    open_ = _series_panel(bars_by_ticker, "open", index).ffill()
    high = _series_panel(bars_by_ticker, "high", index).ffill()
    low = _series_panel(bars_by_ticker, "low", index).ffill()
    entries = (
        _signal_panel(entry_signals, index).shift(1, fill_value=False).astype(bool)
    )
    exits = _signal_panel(exit_signals, index) if exit_signals else entries & False

    price = close.copy()
    price = price.mask(entries, open_)
    stop_distance = cfg.stop_loss if cfg.stop_loss is not None else cfg.trailing_stop
    _ = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        price=price,
        open=open_,
        high=high,
        low=low,
        fees=cfg.commission_bps / 10_000.0,
        slippage=cfg.slippage_bps / 10_000.0,
        sl_stop=stop_distance,
        sl_trail=cfg.trailing_stop is not None,
        tp_stop=cfg.take_profit,
        init_cash=cfg.initial_capital,
        cash_sharing=True,
        freq="1D",
    )


def _evaluate_signals(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast: Any,
    exit_ast: Any,
    warnings: list[str],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    entry_signals: dict[str, pd.Series] = {}
    exit_signals: dict[str, pd.Series] = {}
    for ticker, bars in bars_by_ticker.items():
        try:
            entry_signals[ticker] = evaluate(entry_ast, bars).fillna(False).astype(bool)
        except PineError as exc:
            warnings.append(f"entry eval failed: {ticker}: {exc}")
            continue
        if exit_ast is not None:
            try:
                exit_signals[ticker] = (
                    evaluate(exit_ast, bars).fillna(False).astype(bool)
                )
            except PineError as exc:
                warnings.append(f"exit eval failed: {ticker}: {exc}")
    return entry_signals, exit_signals


def _candidate_rows_for_day(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_signals: dict[str, pd.Series],
    day: pd.Timestamp,
    lookback: int,
    *,
    exclude: set[str],
    opened_tickers: set[str],
    allow_reentry: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker, bars in bars_by_ticker.items():
        if ticker in exclude or (not allow_reentry and ticker in opened_tickers):
            continue
        signal = entry_signals.get(ticker)
        if signal is None or day not in signal.index or not bool(signal.loc[day]):
            continue
        signal_idx = _bar_index_on_or_before(bars, day)
        if signal_idx is None or signal_idx + 1 >= len(bars):
            continue
        if signal_idx + 1 < lookback:
            continue
        bar = bars.iloc[signal_idx]
        close = float(bar["close"])
        volume = float(bar["volume"])
        rows.append(
            {
                "ticker": ticker,
                "signal_idx": signal_idx,
                "as_of_dollar_vol": close * volume,
            }
        )
    rows.sort(key=lambda row: float(row["as_of_dollar_vol"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def _simulate_core_compatible_trades(
    cfg: BacktestConfig,
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_signals: dict[str, pd.Series],
    exit_ast: Any,
    lookback: int,
    warnings: list[str],
) -> list[Trade]:
    slot_count = max(int(cfg.top), 1)
    portfolio = Portfolio(cfg.initial_capital, slot_count)
    slot_states: dict[int, _SlotState | None] = {
        slot_id: None for slot_id in range(slot_count)
    }
    slot_bars: dict[int, pd.DataFrame] = {}
    opened_tickers: set[str] = set()

    for day in _calendar(bars_by_ticker):
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

        candidates = _candidate_rows_for_day(
            bars_by_ticker,
            entry_signals,
            day,
            lookback,
            exclude=_active_or_pending_tickers(slot_states),
            opened_tickers=opened_tickers,
            allow_reentry=cfg.allow_reentry,
        )
        if not candidates:
            continue

        for slot_id in free_slots:
            while candidates:
                row = candidates.pop(0)
                ticker = str(row["ticker"])
                if ticker in _active_or_pending_tickers(slot_states):
                    continue
                bars = bars_by_ticker[ticker]
                state, warn = _make_slot_state(
                    ticker=ticker,
                    bars=bars,
                    signal_idx=int(row["signal_idx"]),
                    cfg=cfg,
                    exit_ast=exit_ast,
                    rank=int(row["rank"]),
                )
                if state is None:
                    if warn:
                        warnings.append(f"{ticker}: {warn}")
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
                opened_tickers.add(ticker)
                break

    calendar = _calendar(bars_by_ticker)
    if len(calendar) > 0:
        _force_close_open_slots(
            slot_states=slot_states,
            slot_bars=slot_bars,
            cfg=cfg,
            portfolio=portfolio,
            end_ts=calendar[-1],
        )
    return portfolio.closed_trades()


def _benchmark_curve(
    cfg: BacktestConfig,
    bars_by_ticker: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    if cfg.benchmark in bars_by_ticker:
        benchmark = pd.to_numeric(
            bars_by_ticker[cfg.benchmark]["close"], errors="coerce"
        ).reindex(calendar, method="ffill")
        return benchmark.dropna()
    return pd.Series(cfg.initial_capital, index=calendar, dtype=float)


def run_vbt(
    cfg: BacktestConfig,
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast: Any,
    exit_ast: Optional[Any] = None,
) -> BacktestResult:
    """Run the vectorbt fast path and return a normal ``BacktestResult``.

    The adapter evaluates Pine-like entry and exit expressions into boolean
    signal panels and passes them to ``vectorbt.Portfolio.from_signals`` with
    fixed-bps fees/slippage. It then emits this project's ``Trade`` and equity
    objects using the same public accounting model as the event engine, which
    keeps reports and Monte Carlo inputs interchangeable.

    Unsupported feature coverage is intentional: HalfSpread, VolumeImpact, and
    Composite slippage, partial exits, and non-MOO entries should fall back to
    the event-driven engine. Stops with ``gap_fills=True`` are honored in the
    returned ledger by the core-compatible fill resolver, but exact vectorbt
    stop-record parity is not claimed for gap-through bars.
    """
    _validate_supported(cfg)
    warnings: list[str] = []
    clean = _clean_bars(bars_by_ticker)
    calendar = _calendar(clean)
    if not clean or len(calendar) == 0:
        equity = pd.Series(cfg.initial_capital, index=calendar, dtype=float)
        benchmark = _benchmark_curve(cfg, clean, calendar)
        metrics = compute_metrics(equity, benchmark, [], max(int(cfg.top), 1))
        return BacktestResult(
            config=cfg,
            trades=[],
            equity_curve=equity,
            benchmark_curve=benchmark,
            metrics=metrics,
            warnings=["no bars supplied to vectorbt fast path"],
            selection=pd.DataFrame(),
        )

    entry_signals, exit_signals = _evaluate_signals(
        clean, entry_ast, exit_ast, warnings
    )
    _call_vectorbt_from_signals(cfg, clean, entry_signals, exit_signals, calendar)

    lookback = required_lookback(entry_ast)
    if exit_ast is not None:
        lookback = max(lookback, required_lookback(exit_ast))
    trades = _simulate_core_compatible_trades(
        cfg, clean, entry_signals, exit_ast, lookback, warnings
    )

    date_set: set[pd.Timestamp] = set(calendar)
    for trade in trades:
        frame = clean.get(trade.ticker)
        if frame is None or frame.empty:
            continue
        dates = frame.loc[
            (frame.index >= pd.Timestamp(trade.entry_date))
            & (frame.index <= pd.Timestamp(trade.exit_date))
        ].index
        date_set.update(dates.tolist())
    result_calendar = pd.DatetimeIndex(sorted(date_set))
    equity = build_equity_curve(result_calendar, trades, clean, cfg.initial_capital)
    benchmark = _benchmark_curve(cfg, clean, result_calendar)
    metrics = compute_metrics(equity, benchmark, trades, max(int(cfg.top), 1))
    selection = pd.DataFrame(
        [
            {
                "ticker": trade.ticker,
                "signal_date": trade.signal_date,
                "rank": trade.rank,
                "role": "active",
            }
            for trade in trades
        ]
    )
    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=equity,
        benchmark_curve=benchmark,
        metrics=metrics,
        warnings=warnings,
        selection=selection,
    )
