"""Shared backtest primitives used by multiple backtest flows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from screener.backtester.data import PriceFetcher
from screener.backtester.models import BacktestConfig, ExitReason, Trade
from screener.backtester.pine import PineError, evaluate
from screener.backtester.portfolio import Portfolio
from screener.backtester.slippage import Side, apply_slippage


@dataclass
class _SimOutcome:
    trade: Optional[Trade]
    warning: Optional[str]


def _slippage_factor(bps: float, buy: bool) -> float:
    """Legacy helper kept for backwards compatibility."""
    delta = bps / 10_000.0
    return 1.0 + delta if buy else 1.0 - delta


def _apply_slip(
    ref_price: float,
    side: Side,
    cfg: BacktestConfig,
    *,
    shares: float = 0.0,
    adv_shares: float = 0.0,
    sigma_daily: float = 0.0,
) -> float:
    """Run ``cfg.slippage_model`` over a reference price."""
    model = cfg.slippage_model
    if model is None:
        return ref_price * _slippage_factor(cfg.slippage_bps, buy=(side == "buy"))
    return apply_slippage(
        model,
        ref_price,
        side,
        shares=shares,
        adv=adv_shares,
        sigma_daily=sigma_daily,
    )


def _trailing_liquidity(
    bars: pd.DataFrame, signal_idx: int, window: int = 20
) -> tuple[float, float]:
    """Return ``(adv_shares, sigma_daily)`` over trailing bars ending at ``signal_idx``."""
    if signal_idx < 0 or window <= 0:
        return 0.0, 0.0
    start = max(0, signal_idx - window + 1)
    window_bars = bars.iloc[start : signal_idx + 1]
    if window_bars.empty:
        return 0.0, 0.0
    vol = window_bars["volume"].astype(float)
    adv = float(vol.mean()) if vol.size else 0.0
    close = window_bars["close"].astype(float)
    if close.size < 2:
        sigma = 0.0
    else:
        rets = close.pct_change().dropna()
        sigma = float(rets.std()) if rets.size else 0.0
    if not np.isfinite(adv):
        adv = 0.0
    if not np.isfinite(sigma):
        sigma = 0.0
    return adv, sigma


def _passes_entry_filters(
    bars: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    cfg: BacktestConfig,
) -> tuple[bool, Optional[str]]:
    """Check min-price and liquidity filters against history up to ``as_of_ts``."""
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
    adv_shares: float = 0.0
    sigma_daily: float = 0.0
    partial_targets: tuple[float, ...] = ()
    partial_fractions: tuple[float, ...] = ()
    partial_fired: list[bool] = field(default_factory=list)


def _resolve_entry_fill(
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
) -> tuple[Optional[int], Optional[float], Optional[str]]:
    """Resolve entry bar index and reference fill price from order settings."""
    if signal_idx + 1 >= len(bars):
        return None, None, "no post-signal entry bar"
    order = cfg.entry_order_type
    if order == "moo":
        entry_idx = signal_idx + 1
        return entry_idx, float(bars.iloc[entry_idx]["open"]), None
    if order == "moc":
        entry_idx = signal_idx + 1
        return entry_idx, float(bars.iloc[entry_idx]["close"]), None
    if order == "limit":
        if cfg.entry_limit_bps is None:
            return None, None, "limit order requires entry_limit_bps"
        signal_close = float(bars.iloc[signal_idx]["close"])
        limit_price = signal_close * (1.0 - cfg.entry_limit_bps / 10_000.0)
        for i in range(signal_idx + 1, len(bars)):
            bar = bars.iloc[i]
            low = float(bar["low"])
            if low <= limit_price:
                ref = min(float(bar["open"]), limit_price)
                return i, ref, None
        return None, None, "limit order never filled in available window"
    return None, None, f"unknown entry_order_type: {order}"


def _make_slot_state(
    ticker: str,
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    exit_ast,
    rank: int,
) -> tuple[Optional[_SlotState], Optional[str]]:
    """Build the per-slot state used by both historical and rolling flows."""
    entry_idx, entry_ref, entry_warn = _resolve_entry_fill(bars, signal_idx, cfg)
    if entry_idx is None or entry_ref is None:
        return None, entry_warn
    adv_shares, sigma_daily = _trailing_liquidity(bars, signal_idx)
    entry_fill = _apply_slip(
        entry_ref, "buy", cfg, adv_shares=adv_shares, sigma_daily=sigma_daily
    )
    exit_signal = None
    if exit_ast is not None:
        try:
            exit_signal = evaluate(exit_ast, bars).fillna(False).astype(bool)
        except PineError as exc:
            return None, f"exit eval failed: {exc}"
    stop_ref = entry_fill * (1.0 - cfg.stop_loss) if cfg.stop_loss else None
    target_ref = entry_fill * (1.0 + cfg.take_profit) if cfg.take_profit else None
    partial_targets = tuple(
        entry_fill * (1.0 + pct) for pct, _frac in cfg.partial_exits
    )
    partial_fractions = tuple(frac for _pct, frac in cfg.partial_exits)
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
            adv_shares=adv_shares,
            sigma_daily=sigma_daily,
            partial_targets=partial_targets,
            partial_fractions=partial_fractions,
            partial_fired=[False] * len(partial_targets),
        ),
        None,
    )


def _resolve_stop_fill(bar_open: float, stop_ref: float, gap_fills: bool) -> float:
    """Reference price for a gap-aware stop-loss fill."""
    if gap_fills and bar_open <= stop_ref:
        return bar_open
    return stop_ref


def _resolve_target_fill(bar_open: float, target_ref: float, gap_fills: bool) -> float:
    """Reference price for a gap-aware take-profit fill."""
    if gap_fills and bar_open >= target_ref:
        return bar_open
    return target_ref


def _maybe_credit_dividends(
    portfolio: Portfolio,
    state: _SlotState,
    bars: pd.DataFrame,
    i: int,
    cfg: BacktestConfig,
) -> None:
    """Credit a cash dividend on an ex-date bar if the frame carries one."""
    if cfg.price_adjustment == "full" or "dividend" not in bars.columns:
        return
    try:
        div = float(bars.iloc[i]["dividend"])
    except (KeyError, TypeError, ValueError):
        return
    if not math.isfinite(div) or div <= 0:
        return
    portfolio.credit_dividends(state.ticker, div)


def _fire_partial_exits_at_bar(
    state: _SlotState,
    bars: pd.DataFrame,
    i: int,
    cfg: BacktestConfig,
    portfolio: Portfolio,
) -> None:
    """Close configured partial-exit tranches if their target prices trade."""
    if not state.partial_targets:
        return
    pos = portfolio.get_position(state.ticker)
    if pos is None or pos.shares <= 0:
        return
    bar = bars.iloc[i]
    bar_open = float(bar["open"])
    high = float(bar["high"])
    bar_date = bars.index[i].date()
    for tier_idx, target_price in enumerate(state.partial_targets):
        if state.partial_fired[tier_idx] or high < target_price:
            continue
        ref = _resolve_target_fill(bar_open, target_price, cfg.gap_fills)
        fill = _apply_slip(
            ref,
            "sell",
            cfg,
            adv_shares=state.adv_shares,
            sigma_daily=state.sigma_daily,
        )
        portfolio.partial_close(
            ticker=state.ticker,
            exit_date=bar_date,
            exit_price=fill,
            reason="target",
            fraction=state.partial_fractions[tier_idx],
            commission_bps=cfg.commission_bps,
        )
        state.partial_fired[tier_idx] = True
        if state.stop_ref is None or state.stop_ref < state.entry_fill:
            state.stop_ref = state.entry_fill


def _check_exit_at_bar(
    state: _SlotState,
    bars: pd.DataFrame,
    i: int,
    cfg: BacktestConfig,
) -> Optional[tuple[float, ExitReason]]:
    """Evaluate exit rules for ``state`` at ``bars[i]``."""
    bar = bars.iloc[i]
    bar_open = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])

    trail_ref = state.peak * (1.0 - cfg.trailing_stop) if cfg.trailing_stop else None
    stop_hit = state.stop_ref is not None and low <= state.stop_ref
    target_hit = state.target_ref is not None and high >= state.target_ref
    trail_hit = trail_ref is not None and low <= trail_ref

    def _slip_sell(ref: float) -> float:
        return _apply_slip(
            ref,
            "sell",
            cfg,
            adv_shares=state.adv_shares,
            sigma_daily=state.sigma_daily,
        )

    if stop_hit and target_hit:
        return (
            _slip_sell(_resolve_stop_fill(bar_open, state.stop_ref, cfg.gap_fills)),
            "stop",
        )
    if stop_hit:
        return (
            _slip_sell(_resolve_stop_fill(bar_open, state.stop_ref, cfg.gap_fills)),
            "stop",
        )
    if trail_hit:
        return (
            _slip_sell(_resolve_stop_fill(bar_open, trail_ref, cfg.gap_fills)),
            "trail",
        )
    if target_hit:
        return (
            _slip_sell(_resolve_target_fill(bar_open, state.target_ref, cfg.gap_fills)),
            "target",
        )

    if high > state.peak:
        state.peak = high

    if state.exit_signal is not None and bool(state.exit_signal.iloc[i]):
        return _slip_sell(close), "exit_expr"
    if i >= state.hold_limit_idx:
        return _slip_sell(close), "time"
    return None


def simulate_ticker(
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
    exit_ast=None,
) -> _SimOutcome:
    """Simulate a single long-only trade starting from the bar after ``signal_idx``."""
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
    fill = _apply_slip(
        float(last_bar["close"]),
        "sell",
        cfg,
        adv_shares=state.adv_shares,
        sigma_daily=state.sigma_daily,
    )
    return _SimOutcome(
        _make_exit(
            state.entry_date,
            state.entry_fill,
            bars.index[-1].date(),
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
    """Return a partial Trade with only price/date/reason fields set."""
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
    """Return ``(tv_symbols, warnings)`` for the configured universe."""
    warnings: list[str] = []

    def _cap(tickers: list[str]) -> list[str]:
        max_universe = int(cfg.max_universe)
        if max_universe <= 0 or len(tickers) <= max_universe:
            return tickers
        warnings.append(
            f"capped universe from {len(tickers)} to {max_universe} tickers"
        )
        return tickers[:max_universe]

    if cfg.tickers:
        return _cap(list(cfg.tickers)), warnings
    if cfg.universe_file:
        from pathlib import Path

        content = Path(cfg.universe_file).read_text()
        tickers = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return _cap(tickers), warnings
    raise ValueError(_NO_UNIVERSE_MSG)


def _prepare_strategy_bars(
    cfg: BacktestConfig,
    bars_by_tv: dict[str, pd.DataFrame],
    price_panel: dict[str, pd.DataFrame],
    tv_symbols: list[str],
    start: date,
    end: date,
    fetcher: PriceFetcher,
    warnings: list[str],
) -> tuple[dict[str, pd.DataFrame], int]:
    """Prepare strategy-specific derived bars, if needed."""
    lookback_floor = 0
    if cfg.strategy_name == "vivek_equity_tool":
        from screener.backtester.vivek_equity import (
            prepare_vivek_equity_tool_frame,
            required_history_bars,
        )

        lookback_floor = required_history_bars()
        return (
            {
                symbol: prepare_vivek_equity_tool_frame(bars)
                for symbol, bars in bars_by_tv.items()
            },
            lookback_floor,
        )

    if cfg.strategy_name != "rs_breakout":
        return bars_by_tv, lookback_floor

    from screener.rs_breakout import (
        india_symbol,
        prepare_backtest_frames,
        required_history_bars,
    )
    from screener.unusual_volume.delivery import load_delivery_panel

    lookback_floor = required_history_bars()
    benchmark_bars = price_panel.get(cfg.benchmark, pd.DataFrame())
    if benchmark_bars is None or benchmark_bars.empty:
        warnings.append(f"benchmark data unavailable for rs_breakout: {cfg.benchmark}")
        return bars_by_tv, lookback_floor

    delivery_panel = pd.DataFrame()
    if cfg.market == "india":
        history_days = max((pd.Timestamp(end) - pd.Timestamp(start)).days + 14, 40)
        try:
            delivery_panel = load_delivery_panel(
                [india_symbol(symbol) for symbol in tv_symbols],
                end,
                history_days=history_days,
            )
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            warnings.append(f"delivery panel unavailable for rs_breakout: {exc}")

    return (
        prepare_backtest_frames(
            bars_by_tv,
            benchmark_bars,
            market=cfg.market,
            delivery_panel=delivery_panel,
        ),
        lookback_floor,
    )


def _eligible_reserve_signal_idx(
    bars: pd.DataFrame,
    exit_day: pd.Timestamp,
    cfg: BacktestConfig,
    entry_ast,
    lookback: int,
) -> Optional[int]:
    """Return signal index if a reserve passes filters and entry AST on ``exit_day``."""
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


def _bar_index_on_or_before(bars: pd.DataFrame, day: pd.Timestamp) -> Optional[int]:
    mask = bars.index <= day
    if not mask.any():
        return None
    return int(np.where(mask)[0][-1])


def _active_or_pending_tickers(
    slot_states: dict[int, Optional[_SlotState]],
) -> set[str]:
    return {state.ticker for state in slot_states.values() if state is not None}


def _precompute_entry_signals(
    bars_by_ticker: dict[str, pd.DataFrame],
    entry_ast,
    warnings: list[str],
) -> dict[str, pd.Series]:
    signals: dict[str, pd.Series] = {}
    for ticker, bars in bars_by_ticker.items():
        if bars is None or bars.empty:
            continue
        try:
            signals[ticker] = evaluate(entry_ast, bars).fillna(False).astype(bool)
        except PineError as exc:
            warnings.append(f"entry eval failed: {ticker}: {exc}")
    return signals


def _close_slot_at_day(
    *,
    slot_id: int,
    state: _SlotState,
    bars: pd.DataFrame,
    day: pd.Timestamp,
    cfg: BacktestConfig,
    portfolio: Portfolio,
    slot_states: dict[int, Optional[_SlotState]],
) -> bool:
    """Process one slot for a day. Returns True when the slot becomes free."""
    if day not in bars.index:
        return False
    i = bars.index.get_loc(day)
    if isinstance(i, slice) or not isinstance(i, int):
        return False
    if i < state.entry_idx + 1:
        return False
    _maybe_credit_dividends(portfolio, state, bars, i, cfg)
    _fire_partial_exits_at_bar(state, bars, i, cfg, portfolio)
    if portfolio.get_position(state.ticker) is None:
        slot_states[slot_id] = None
        return True
    exit_ = _check_exit_at_bar(state, bars, i, cfg)
    if exit_ is None:
        return False
    fill, reason = exit_
    portfolio.close(
        ticker=state.ticker,
        exit_date=day.date(),
        exit_price=fill,
        reason=reason,
        commission_bps=cfg.commission_bps,
    )
    slot_states[slot_id] = None
    return True


def _force_close_open_slots(
    *,
    slot_states: dict[int, Optional[_SlotState]],
    slot_bars: dict[int, pd.DataFrame],
    cfg: BacktestConfig,
    portfolio: Portfolio,
    end_ts: pd.Timestamp,
) -> None:
    for slot_id, state in list(slot_states.items()):
        if state is None:
            continue
        bars = slot_bars[slot_id]
        tail = bars.loc[
            (bars.index > pd.Timestamp(state.entry_date)) & (bars.index <= end_ts)
        ]
        if tail.empty:
            tail = bars.loc[bars.index > pd.Timestamp(state.entry_date)]
        if tail.empty:
            continue
        last_bar = tail.iloc[-1]
        fill = _apply_slip(
            float(last_bar["close"]),
            "sell",
            cfg,
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
