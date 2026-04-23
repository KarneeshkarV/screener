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
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.data import PriceFetcher, fetch_benchmark
from screener.backtester.metrics import compute_metrics
from screener.backtester.models import BacktestConfig, BacktestResult, ExitReason, Trade
from screener.backtester.pine import (
    PineError,
    evaluate,
    parse,
    required_lookback,
)
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.backtester.slippage import Side, apply_slippage


@dataclass
class _SimOutcome:
    trade: Trade | None
    warning: str | None


def _apply_slip(
    ref_price: float,
    side: Side,
    cfg: BacktestConfig,
    *,
    shares: float = 0.0,
    adv_shares: float = 0.0,
    sigma_daily: float = 0.0,
) -> float:
    """Run ``cfg.slippage_model`` over a reference price.

    ``BacktestConfig.__post_init__`` guarantees ``slippage_model`` is set, so
    callers do not need to special-case a missing model.
    """
    return apply_slippage(
        cfg.slippage_model,
        ref_price,
        side,
        shares=shares,
        adv=adv_shares,
        sigma_daily=sigma_daily,
    )


def _trailing_liquidity(bars: pd.DataFrame, signal_idx: int, window: int = 20) -> tuple[float, float]:
    """Return ``(adv_shares, sigma_daily)`` over the ``window`` bars ending at
    ``signal_idx`` (inclusive). Any NaN/zero-bar edge cases collapse to 0 so
    volume-impact slippage degrades gracefully to zero rather than NaN."""
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
) -> tuple[bool, str | None]:
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
    cfg: BacktestConfig | None = None,
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
    stop_ref: float | None
    target_ref: float | None
    hold_limit_idx: int
    peak: float
    exit_signal: pd.Series | None
    adv_shares: float = 0.0
    sigma_daily: float = 0.0
    # Parallel arrays to ``cfg.partial_exits``. ``partial_targets`` holds the
    # absolute price at which each tier fires; ``partial_fired`` is True once
    # the tier has been closed out so it never fires twice.
    partial_targets: tuple[float, ...] = ()
    partial_fractions: tuple[float, ...] = ()
    partial_fired: list[bool] = field(default_factory=list)


def _resolve_entry_fill(
    bars: pd.DataFrame,
    signal_idx: int,
    cfg: BacktestConfig,
) -> tuple[int | None, float | None, str | None]:
    """Resolve the entry bar index and reference fill price for a new position
    based on ``cfg.entry_order_type``. Returns ``(entry_idx, ref_price,
    warning)``; ``entry_idx=None`` indicates the order was not fillable in the
    available data window.

      * ``moo``   — next bar's open.
      * ``moc``   — next bar's close.
      * ``limit`` — limit price = signal-bar close * (1 - entry_limit_bps/1e4);
                     first post-signal bar with low <= limit fills at
                     min(bar.open, limit); gap-through favours the buyer.
    """
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
) -> tuple[_SlotState | None, str | None]:
    """Build the mutable per-slot state used by both simulate_ticker and the
    event-driven loop. Returns (state, warning)."""
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
        except PineError as e:
            return None, f"exit eval failed: {e}"
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


def _resolve_stop_fill(
    bar_open: float, stop_ref: float, gap_fills: bool
) -> float:
    """Reference price for a stop-loss fill, gap-aware.

    When the bar *opens* at or below the stop, the trader cannot fill at the
    stop level — the opening print is the realistic fill and it is worse than
    ``stop_ref``. When the bar opens above the stop but trades through it
    intraday, the classical assumption (fill at ``stop_ref``) holds. Set
    ``gap_fills=False`` to reproduce the legacy "always fill at stop_ref"
    behaviour.
    """
    if gap_fills and bar_open <= stop_ref:
        return bar_open
    return stop_ref


def _resolve_target_fill(
    bar_open: float, target_ref: float, gap_fills: bool
) -> float:
    """Reference price for a take-profit fill, gap-aware.

    On a gap-up through the target, the trader fills at the bar open — better
    than ``target_ref``. Symmetric treatment to :func:`_resolve_stop_fill` to
    avoid biasing return distributions.
    """
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
    """Credit a cash dividend on an ex-date bar if the frame carries one.

    No-op under the legacy ``price_adjustment='full'`` regime (the yfinance
    auto_adjust path folds dividends into OHLC, so double-crediting would
    inflate returns). Also a no-op on synthetic bars that lack a ``dividend``
    column — all existing tests use synthetic frames.
    """
    if cfg.price_adjustment == "full":
        return
    if "dividend" not in bars.columns:
        return
    try:
        div = float(bars.iloc[i]["dividend"])
    except (KeyError, ValueError, TypeError):
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
    """Close tranches at pre-configured tiered price targets.

    Each tier ``(profit_fraction, shares_fraction)`` converts to an absolute
    price level at entry; when a bar's high reaches that level the sleeve is
    sold via ``portfolio.partial_close``. After the first tier fires we raise
    ``state.stop_ref`` to break-even (standard swing practice) so the
    remaining runner is risk-free. Remaining tiers continue to fire on later
    bars as the price advances.
    """
    if not state.partial_targets:
        return
    pos = portfolio.get_position(state.ticker)
    if pos is None or pos.shares <= 0:
        return
    bar = bars.iloc[i]
    bar_open = float(bar["open"])
    high = float(bar["high"])
    bar_date = bars.index[i].date()
    # Remaining-shares bookkeeping: each tier's fraction is interpreted as
    # ``fraction of the lot AT THE TIME THAT TIER FIRES``, so a ((1.0, 0.5),
    # (2.0, 0.5)) schedule closes half at 1R, then half of the remainder at
    # 2R (25% of the original lot).
    for tier_idx, target_price in enumerate(state.partial_targets):
        if state.partial_fired[tier_idx]:
            continue
        if high < target_price:
            continue
        # Gap-aware reference: if the bar opens beyond the tier, trader fills
        # at the better open — symmetric with full-target gap-up handling.
        ref = _resolve_target_fill(bar_open, target_price, cfg.gap_fills)
        fill = _apply_slip(
            ref,
            "sell",
            cfg,
            adv_shares=state.adv_shares,
            sigma_daily=state.sigma_daily,
        )
        frac = state.partial_fractions[tier_idx]
        portfolio.partial_close(
            ticker=state.ticker,
            exit_date=bar_date,
            exit_price=fill,
            reason="target",
            fraction=frac,
            commission_bps=cfg.commission_bps,
        )
        state.partial_fired[tier_idx] = True
        # After the first tier fires, raise stop to break-even (entry_fill)
        # so the runner is risk-free.
        if state.stop_ref is None or state.stop_ref < state.entry_fill:
            state.stop_ref = state.entry_fill


def _check_exit_at_bar(
    state: _SlotState,
    bars: pd.DataFrame,
    i: int,
    cfg: BacktestConfig,
) -> tuple[float, ExitReason] | None:
    """Evaluate exit rules for ``state`` at bars[i]. Returns (fill, reason) if
    the position exits on this bar, else None. Mutates ``state.peak`` in place
    after the stop/target/trail checks (matching the original ordering).

    Priority on same bar: stop+target→stop, then stop, then trail, then target
    (all use bar.low / bar.high); after that, peak is updated; then exit_expr
    and time exits fire on close.
    """
    bar = bars.iloc[i]
    bar_open = float(bar["open"])
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
            _slip_sell(
                _resolve_target_fill(bar_open, state.target_ref, cfg.gap_fills)
            ),
            "target",
        )

    # update peak AFTER stop checks (first post-entry bar's trail_ref uses
    # entry_fill, not max(entry_fill, this_bar.high))
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
    """Simulate a single long-only trade starting from the bar AFTER signal_idx.

    ``signal_idx`` is the positional index of the as-of/signal bar in ``bars``.
    Entry fills at bars[signal_idx + 1]['open'] (adverse slippage applied).
    Returns ``_SimOutcome`` with ``trade=None`` if no post-signal bar exists.

    **Scope:** this is the single-ticker reference simulator used by the test
    suite. The production path (:func:`run_backtest`) drives multi-slot
    simulation via :func:`_run_event_driven_sim`; they share per-bar logic
    through :func:`_check_exit_at_bar` and :func:`_make_slot_state`.
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


def _eligible_reserve_signal_idx(
    bars: pd.DataFrame,
    exit_day: pd.Timestamp,
    cfg: BacktestConfig,
    entry_ast,
    lookback: int,
) -> int | None:
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
    slot_states: dict[int, _SlotState | None] = {}
    slot_bars: dict[int, pd.DataFrame] = {}
    # Per-slot re-entry budget. A slot becomes eligible for same-ticker
    # re-entry when its position closes if ``reentries_left > 0``.
    reentries_left: dict[int, int] = {}
    pending_reentry: dict[int, str] = {}  # slot_id -> ticker awaiting re-entry

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
        reentries_left[slot_id] = cfg.max_reentries if cfg.allow_reentry else 0

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
        # Step 0: try to re-enter any slot whose ticker is awaiting a fresh
        # entry signal. Re-entry has priority over reserve rotation: a slot
        # "sticks" with its original ticker until its re-entry budget runs
        # out OR the entry signal doesn't re-fire within the horizon.
        if pending_reentry:
            for slot_id, ticker in list(pending_reentry.items()):
                bars = slot_bars.get(slot_id)
                if bars is None or bars.empty:
                    del pending_reentry[slot_id]
                    continue
                signal_idx = _eligible_reserve_signal_idx(
                    bars, day, cfg, entry_ast, lookback
                )
                if signal_idx is None:
                    continue
                # rank is preserved from initial assignment
                new_rank = portfolio.rank_of(ticker)
                state, warn = _make_slot_state(
                    ticker, bars, signal_idx, cfg, exit_ast, new_rank
                )
                if state is None:
                    if warn:
                        warnings.append(f"{ticker} re-entry: {warn}")
                    # Give up on re-entry if the entry signal triggered but
                    # the order could not be filled (e.g., no post-signal bar)
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

        # Step 1: check exits (and partial-target tiers) for every open slot.
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
            _maybe_credit_dividends(portfolio, state, bars, i, cfg)
            _fire_partial_exits_at_bar(state, bars, i, cfg, portfolio)
            # If partials closed the lot entirely (e.g. fractions summed to 1),
            # the oldest position is gone — skip the full-exit check.
            if portfolio.get_position(state.ticker) is None:
                slot_states[slot_id] = None
                freed.append(slot_id)
                if cfg.allow_reentry and reentries_left.get(slot_id, 0) > 0:
                    reentries_left[slot_id] -= 1
                    pending_reentry[slot_id] = state.ticker
                continue
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
            if cfg.allow_reentry and reentries_left.get(slot_id, 0) > 0:
                reentries_left[slot_id] -= 1
                pending_reentry[slot_id] = state.ticker

        if not cfg.reinvest or not freed:
            continue

        # Step 2: fill freed slots from the reserve queue (rank order).
        # Slots awaiting re-entry are held out of the queue so their original
        # ticker retains the slot until its re-entry budget expires.
        for slot_id in freed:
            if slot_id in pending_reentry:
                continue
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
        fill = _apply_slip(
            float(last_bar["close"]),
            "sell",
            cfg,
            adv_shares=state.adv_shares,
            sigma_daily=state.sigma_daily,
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
