"""Intraday options position backtester (Phase 4.2/4.3).

Where :mod:`screener.options.position_backtest` fills once per day at the EOD
chain, this engine walks the *intraday snapshot* time series served by an
:class:`~screener.options.history_provider.OptionsHistoryProvider` — for the
default backend that is the forward-capture contract store — and enters, marks,
and exits at snapshot timestamps.

Point-in-time rule
------------------
Snapshots for a session are processed in strict timestamp order. A decision at
snapshot time ``T`` only ever sees the chain observed at ``T`` (and state
accumulated from earlier snapshots); a later snapshot is never used to price an
earlier fill. Entries land at the first snapshot on/after ``entry_time`` and
exits at the snapshot that trips a target/stop/``exit_time``, otherwise the
position is flattened at session close (``session_end``) or at the last
recorded snapshot when the series ends early (``data_end``). Times are compared
in the market's local timezone (snapshots are stored naive UTC).

Phase 4.3 (mixed portfolios): an optional signed ``equity_hedge_qty`` of the
underlying is held for the life of each option position and marked from the
snapshot spot, so a delta hedge (or any static equity leg) nets into the same
portfolio P&L and equity curve.

Reuse: fill models, margin models, premium/mark helpers and the
:class:`~screener.options.bt_models.OptionPositionTrade` ledger type all come
from the EOD engine, so realism knobs (``fill_model``, ``slippage_bps``,
``margin_model`` …) behave identically here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from screener.markets import get_market
from screener.options.bt_models import (
    FillModel,
    LegFill,
    MarginModel,
    OptionPositionTrade,
    OptionsBacktestConfig,
)
from screener.options.history_provider import (
    OptionsHistoryProvider,
    default_history_provider,
)
from screener.options.models import OptionChain, OptionsMarket
from screener.options.position_backtest import (
    _entry_fill_price,
    _exit_fill_price,
    _find_contract,
    _leg_notional,
    _mark_price,
    _mtm_premium,
    _position_margin,
    _signed_premium,
)
from screener.options.structures import build_structure, select_expiry, select_strike

__all__ = [
    "IntradayOptionsBacktestConfig",
    "IntradayOptionsBacktestResult",
    "run_intraday_options_backtest",
]

# Regular-session close (market-local). No shared helper exposes close alone on
# OptionsMarket / Market, so pin the well-known equity-session closes here.
_SESSION_CLOSE_LOCAL: dict[str, time] = {
    "us": time(16, 0),
    "india": time(15, 30),
}
# Snapshots at/after close minus this epsilon are last-tradable (session_end).
_SESSION_END_EPSILON = timedelta(minutes=1)
# Last recorded snapshot more than this before scheduled close → data_end warn.
_DATA_END_EARLY_THRESHOLD = timedelta(minutes=30)


@dataclass(frozen=True)
class IntradayOptionsBacktestConfig:
    """Knobs for the intraday snapshot-driven options backtester."""

    tickers: tuple[str, ...]
    start: date
    end: date
    market: OptionsMarket = "us"
    structure: str = "long_call"
    strike_rule: str = "atm"
    expiry_rule: str = "front"
    width_pct: float = 0.05
    lots: int = 1
    # Session-local entry/exit clock. ``entry_time`` = first snapshot on/after
    # it (``None`` → the first snapshot); ``exit_time`` flattens at/after it.
    entry_time: time | None = None
    exit_time: time | None = None
    target_pct: float | None = None
    stop_pct: float | None = None
    # Realism knobs shared with the EOD engine.
    fill_model: FillModel = "legacy"
    slippage_bps: float = 0.0
    slippage_ticks: float = 0.0
    tick_size: float = 0.05
    commission_per_order: float = 0.0
    margin_model: MarginModel = "none"
    # Phase 4.3: signed underlying units held alongside each option position.
    equity_hedge_qty: float = 0.0
    initial_capital: float = 100_000.0
    # By default a symbol is entered at most once per session; set True to let a
    # strategy re-enter after an intraday exit (target/stop/exit_time).
    allow_reentry: bool = False


@dataclass
class IntradayOptionsBacktestResult:
    """Ledger + intraday equity/margin curves from an intraday run."""

    trades: list[OptionPositionTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    margin_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    peak_margin: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _OpenIntradayPosition:
    symbol: str
    structure: str
    entry_ts: datetime
    legs: list[LegFill]
    entry_premium: float
    gross_premium: float
    entry_costs: float
    entry_spot: float | None
    hedge_qty: float
    # Last *observed* mark per leg (not entry fallback). Updated whenever a
    # real quote is found so missing-chain snapshots keep stops/targets honest.
    last_marks: list[float]


def _fill_cfg(cfg: IntradayOptionsBacktestConfig) -> OptionsBacktestConfig:
    """Adapter carrying the realism knobs to the reused EOD helpers."""
    return OptionsBacktestConfig(
        tickers=cfg.tickers or ("X",),
        start=cfg.start,
        end=cfg.end,
        structure=cfg.structure,
        strike_rule=cfg.strike_rule,
        expiry_rule=cfg.expiry_rule,
        width_pct=cfg.width_pct,
        lots=cfg.lots,
        target_pct=cfg.target_pct,
        stop_pct=cfg.stop_pct,
        fill_model=cfg.fill_model,
        slippage_bps=cfg.slippage_bps,
        slippage_ticks=cfg.slippage_ticks,
        tick_size=cfg.tick_size,
        commission_per_order=cfg.commission_per_order,
        margin_model=cfg.margin_model,
        initial_capital=cfg.initial_capital,
    )


def _local_time(ts: datetime, tz: ZoneInfo) -> time:
    """Time-of-day of ``ts`` in the market timezone (naive ts assumed UTC)."""
    aware = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    return aware.astimezone(tz).timetz().replace(tzinfo=None)


def _session_close_local(market: str) -> time:
    return _SESSION_CLOSE_LOCAL.get(market, time(16, 0))


def _time_minus(t: time, delta: timedelta) -> time:
    """Subtract ``delta`` from a clock time (wraps within the same day)."""
    base = datetime(2000, 1, 1, t.hour, t.minute, t.second, t.microsecond)
    return (base - delta).time()


def _time_to_td(t: time) -> timedelta:
    return timedelta(
        hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond
    )


def _is_session_end_by_clock(local_t: time, market: str) -> bool:
    """True when local time is at/after scheduled close minus a small epsilon."""
    close = _session_close_local(market)
    threshold = _time_minus(close, _SESSION_END_EPSILON)
    return local_t >= threshold


def _minutes_before_close(local_t: time, market: str) -> float:
    close = _session_close_local(market)
    return (_time_to_td(close) - _time_to_td(local_t)).total_seconds() / 60.0


def _trading_days(start: date, end: date) -> list[date]:
    """Every weekday in ``[start, end]``; empty sessions are skipped downstream."""
    days: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _open_position(
    chain: OptionChain,
    cfg: IntradayOptionsBacktestConfig,
    fill_cfg: OptionsBacktestConfig,
    warnings: list[str],
) -> _OpenIntradayPosition | None:
    spec = build_structure(
        cfg.structure,
        strike_rule=cfg.strike_rule,
        expiry_rule=cfg.expiry_rule,
        width_pct=cfg.width_pct,
        lots=cfg.lots,
    )
    as_of_day = chain.as_of.date()
    legs: list[LegFill] = []
    for leg in spec.legs:
        expiry = select_expiry(chain, leg.expiry_rule, as_of_day)
        if expiry is None:
            warnings.append(f"{chain.underlying} {chain.as_of}: no expiry")
            return None
        contract = select_strike(
            chain, expiry, leg.right, leg.strike_rule, as_of=as_of_day
        )
        if contract is None:
            warnings.append(f"{chain.underlying} {chain.as_of}: no {leg.right} strike")
            return None
        price = _entry_fill_price(contract, leg.side, cfg=fill_cfg)
        if price is None or price <= 0:
            warnings.append(
                f"{chain.underlying} {chain.as_of}: no price for {leg.right}"
            )
            return None
        legs.append(
            LegFill(
                right=leg.right,
                strike=float(contract.strike),
                expiry=expiry,
                side=leg.side,
                lots=leg.lots,
                lot_size=float(contract.lot_size or 1.0),
                entry_price=price,
                entry_iv=contract.iv,
            )
        )
    entry_premium = _signed_premium(legs)
    gross = sum(_leg_notional(leg.entry_price, leg.lots, leg.lot_size) for leg in legs)
    return _OpenIntradayPosition(
        symbol=chain.underlying,
        structure=cfg.structure,
        entry_ts=chain.as_of,
        legs=legs,
        entry_premium=entry_premium,
        gross_premium=gross,
        entry_costs=len(legs) * cfg.commission_per_order,
        entry_spot=chain.spot,
        hedge_qty=cfg.equity_hedge_qty,
        last_marks=[float(leg.entry_price) for leg in legs],
    )


def _mark_legs(
    pos: _OpenIntradayPosition, chain: OptionChain
) -> tuple[list[float], list[str]]:
    """Mark each leg; fall back to last *observed* mark (not entry).

    Returns ``(marks, carried_labels)`` where ``carried_labels`` names legs
    whose mark was not observed on this snapshot (``right:strike``).
    """
    marks: list[float] = []
    carried: list[str] = []
    for i, leg in enumerate(pos.legs):
        contract = _find_contract(
            chain, right=leg.right, strike=leg.strike, expiry=leg.expiry
        )
        px = _mark_price(contract)
        if px is not None:
            observed = float(px)
            marks.append(observed)
            pos.last_marks[i] = observed
        else:
            marks.append(pos.last_marks[i])
            carried.append(f"{leg.right}:{leg.strike:g}")
    return marks, carried


def _hedge_pnl(pos: _OpenIntradayPosition, spot: float | None) -> float:
    """Mark-to-market P&L of the static equity hedge against the snapshot spot."""
    if not pos.hedge_qty or pos.entry_spot is None or spot is None:
        return 0.0
    return pos.hedge_qty * (spot - pos.entry_spot)


def _close_position(
    pos: _OpenIntradayPosition,
    chain: OptionChain,
    marks: list[float],
    *,
    exit_ts: datetime,
    exit_reason: str,
    fill_cfg: OptionsBacktestConfig,
    carried_marks: list[str] | None = None,
) -> OptionPositionTrade:
    from dataclasses import replace

    closed: list[LegFill] = []
    exit_costs = 0.0
    for leg, mark in zip(pos.legs, marks, strict=True):
        contract = _find_contract(
            chain, right=leg.right, strike=leg.strike, expiry=leg.expiry
        )
        exit_px = _exit_fill_price(contract, mark, leg.side, cfg=fill_cfg)
        exit_costs += fill_cfg.commission_per_order
        closed.append(replace(leg, exit_price=exit_px))

    exit_premium = _signed_premium(closed, use_exit=True)
    hedge_pnl = _hedge_pnl(pos, chain.spot)
    pnl = (exit_premium - pos.entry_premium) - pos.entry_costs - exit_costs + hedge_pnl
    gross = pos.gross_premium if pos.gross_premium > 0 else 1.0
    details: dict[str, object] = {
        "entry_ts": pos.entry_ts.isoformat(),
        "exit_ts": exit_ts.isoformat(),
        "entry_costs": pos.entry_costs,
        "exit_costs": exit_costs,
    }
    if pos.hedge_qty:
        details["equity_hedge_qty"] = pos.hedge_qty
        details["equity_hedge_pnl"] = hedge_pnl
    if carried_marks:
        details["carried_marks"] = list(carried_marks)
    return OptionPositionTrade(
        symbol=pos.symbol,
        structure=pos.structure,
        entry_date=pos.entry_ts.date(),
        exit_date=exit_ts.date(),
        legs=tuple(closed),
        entry_premium=pos.entry_premium,
        exit_premium=exit_premium,
        pnl=pnl,
        return_pct=(pnl / gross) * 100.0,
        exit_reason=exit_reason,  # type: ignore[arg-type]
        passed_filter=True,
        details=details,
    )


def _exit_reason(
    pos: _OpenIntradayPosition,
    marks: list[float],
    *,
    local_t: time,
    cfg: IntradayOptionsBacktestConfig,
    is_last_recorded: bool,
) -> str | None:
    mark_premium = _mtm_premium(pos.legs, marks)
    gross = pos.gross_premium if pos.gross_premium > 0 else 1.0
    ret = ((mark_premium - pos.entry_premium) / gross) * 100.0
    if cfg.stop_pct is not None and ret <= -abs(cfg.stop_pct):
        return "stop"
    if cfg.target_pct is not None and ret >= abs(cfg.target_pct):
        return "target"
    if cfg.exit_time is not None and local_t >= cfg.exit_time:
        return "time"
    # Clock-based session close first so a full day is not labeled data_end.
    if _is_session_end_by_clock(local_t, cfg.market):
        return "session_end"
    # Final recorded snapshot before scheduled close (dead/incomplete recorder).
    if is_last_recorded:
        return "data_end"
    return None


def _record_margin(
    result: IntradayOptionsBacktestResult,
    margin_points: list[tuple[datetime, float]],
    *,
    legs: list[LegFill],
    spot: float | None,
    as_of: datetime,
    fill_cfg: OptionsBacktestConfig,
) -> None:
    margin = _position_margin(legs, spot, as_of.date(), fill_cfg)
    margin_points.append((as_of, margin))
    result.peak_margin = max(result.peak_margin, margin)


def _build_portfolio_equity_curve(
    *,
    initial_capital: float,
    realized_events: list[tuple[datetime, float]],
    unrealized_events: list[tuple[datetime, str, float]],
    flat_events: list[tuple[datetime, str]],
) -> pd.Series:
    """Combine per-symbol realized deltas + unrealized overlays on a unique index.

    Portfolio equity at ``T`` = ``initial + realized-through-T + sum of each
    symbol's latest unrealized overlay at T``. Events at the same timestamp are
    applied together before emitting the point so multi-ticker runs never leave
    duplicate index labels.
    """
    if not realized_events and not unrealized_events and not flat_events:
        return pd.Series(dtype=float)

    # Event kinds: realized pnl credit, set-unrealized, clear-unrealized (flat/exit).
    events: list[tuple[datetime, int, str, float]] = []
    # Sort key priority within a timestamp: clear (0) / set unrealized (1) /
    # realized (2) so an exit that clears then realizes orders cleanly; flat
    # markers use clear with 0 unrealized.
    for ts, pnl in realized_events:
        events.append((ts, 2, "", pnl))
    for ts, symbol, unrealized in unrealized_events:
        events.append((ts, 1, symbol, unrealized))
    for ts, symbol in flat_events:
        events.append((ts, 0, symbol, 0.0))

    events.sort(key=lambda e: (e[0], e[1], e[2]))

    last_unrealized: dict[str, float] = defaultdict(float)
    realized = 0.0
    # Emit one equity value per distinct timestamp after applying all events at T.
    by_ts: dict[datetime, float] = {}
    i = 0
    n = len(events)
    while i < n:
        ts = events[i][0]
        while i < n and events[i][0] == ts:
            _, kind, symbol, value = events[i]
            if kind == 2:
                realized += value
            elif kind == 1:
                last_unrealized[symbol] = value
            else:  # clear / flat
                last_unrealized[symbol] = 0.0
            i += 1
        by_ts[ts] = initial_capital + realized + sum(last_unrealized.values())

    if not by_ts:
        return pd.Series(dtype=float)
    ordered = sorted(by_ts.items(), key=lambda kv: kv[0])
    return pd.Series(
        [v for _, v in ordered],
        index=pd.DatetimeIndex([ts for ts, _ in ordered]),
    )


def run_intraday_options_backtest(
    cfg: IntradayOptionsBacktestConfig,
    provider: OptionsHistoryProvider | None = None,
) -> IntradayOptionsBacktestResult:
    """Walk intraday snapshots per session and enter/mark/exit point-in-time.

    ``provider`` defaults to the forward-capture contract store for ``market``.
    Positions never carry overnight: open risk is flattened at scheduled session
    close (``session_end``) or at the last recorded snapshot when the series
    ends early (``data_end``). Each symbol is entered at most once per session
    unless ``allow_reentry`` is set.

    Multi-ticker equity is a true portfolio curve: realized P&L is global and
    unrealized marks are overlaid per symbol on the union timestamp index (no
    duplicate labels / sawtooth interleave).
    """
    provider = provider or default_history_provider(cfg.market)
    fill_cfg = _fill_cfg(cfg)
    tz = ZoneInfo(get_market(cfg.market).timezone)
    result = IntradayOptionsBacktestResult()

    # Equity construction inputs (combined after the walk — see H13).
    realized_events: list[tuple[datetime, float]] = []
    unrealized_events: list[tuple[datetime, str, float]] = []
    flat_events: list[tuple[datetime, str]] = []
    margin_points: list[tuple[datetime, float]] = []

    for day in _trading_days(cfg.start, cfg.end):
        for symbol in cfg.tickers:
            chains = sorted(provider.chains(symbol, day), key=lambda c: c.as_of)
            if not chains:
                continue
            pos: _OpenIntradayPosition | None = None
            entered_today = False
            n_chains = len(chains)
            for idx, chain in enumerate(chains):
                is_last_recorded = idx == n_chains - 1
                local_t = _local_time(chain.as_of, tz)
                at_session_end = _is_session_end_by_clock(local_t, cfg.market)

                if pos is None:
                    entry_ok = cfg.entry_time is None or local_t >= cfg.entry_time
                    # One entry per session unless re-entry is opted in.
                    if entered_today and not cfg.allow_reentry:
                        entry_ok = False
                    # Do not re-enter at/after the configured exit clock (churn).
                    if cfg.exit_time is not None and local_t >= cfg.exit_time:
                        entry_ok = False
                    # No new risk at/after scheduled close (clock, not list length).
                    if at_session_end:
                        entry_ok = False
                    # Need a later snapshot to mark/exit — skip orphan single-bar entries.
                    if is_last_recorded:
                        entry_ok = False
                    if entry_ok:
                        pos = _open_position(chain, cfg, fill_cfg, result.warnings)
                        entered_today = pos is not None
                        if pos is not None and cfg.margin_model != "none":
                            # Sample margin at the entry snapshot (peak includes open).
                            _record_margin(
                                result,
                                margin_points,
                                legs=pos.legs,
                                spot=chain.spot,
                                as_of=chain.as_of,
                                fill_cfg=fill_cfg,
                            )
                    flat_events.append((chain.as_of, symbol))
                    continue

                marks, carried = _mark_legs(pos, chain)
                reason = _exit_reason(
                    pos,
                    marks,
                    local_t=local_t,
                    cfg=cfg,
                    is_last_recorded=is_last_recorded,
                )
                if cfg.margin_model != "none":
                    _record_margin(
                        result,
                        margin_points,
                        legs=pos.legs,
                        spot=chain.spot,
                        as_of=chain.as_of,
                        fill_cfg=fill_cfg,
                    )

                if reason is None:
                    unrealized = (
                        _mtm_premium(pos.legs, marks)
                        - pos.entry_premium
                        + _hedge_pnl(pos, chain.spot)
                    )
                    unrealized_events.append((chain.as_of, symbol, unrealized))
                    continue

                if reason == "data_end":
                    early_min = _minutes_before_close(local_t, cfg.market)
                    if early_min > _DATA_END_EARLY_THRESHOLD.total_seconds() / 60.0:
                        result.warnings.append(
                            f"{symbol} {chain.as_of.date()}: data_end "
                            f"{early_min:.0f}m before scheduled close "
                            f"({_session_close_local(cfg.market).strftime('%H:%M')} local)"
                        )

                trade = _close_position(
                    pos,
                    chain,
                    marks,
                    exit_ts=chain.as_of,
                    exit_reason=reason,
                    fill_cfg=fill_cfg,
                    carried_marks=carried,
                )
                if carried:
                    result.warnings.append(
                        f"{symbol} {chain.as_of}: exit used carried marks for "
                        f"{', '.join(carried)} (not observed on exit snapshot)"
                    )
                result.trades.append(trade)
                realized_events.append((chain.as_of, trade.pnl))
                # Clear this symbol's unrealized overlay at the exit timestamp.
                flat_events.append((chain.as_of, symbol))
                pos = None

            # Belt-and-suspenders: never leave a position open past the session.
            if pos is not None:
                chain = chains[-1]
                local_t = _local_time(chain.as_of, tz)
                marks, carried = _mark_legs(pos, chain)
                reason = (
                    "session_end"
                    if _is_session_end_by_clock(local_t, cfg.market)
                    else "data_end"
                )
                if reason == "data_end":
                    early_min = _minutes_before_close(local_t, cfg.market)
                    if early_min > _DATA_END_EARLY_THRESHOLD.total_seconds() / 60.0:
                        result.warnings.append(
                            f"{symbol} {chain.as_of.date()}: data_end "
                            f"{early_min:.0f}m before scheduled close "
                            f"({_session_close_local(cfg.market).strftime('%H:%M')} local)"
                        )
                trade = _close_position(
                    pos,
                    chain,
                    marks,
                    exit_ts=chain.as_of,
                    exit_reason=reason,
                    fill_cfg=fill_cfg,
                    carried_marks=carried,
                )
                if carried:
                    result.warnings.append(
                        f"{symbol} {chain.as_of}: exit used carried marks for "
                        f"{', '.join(carried)} (not observed on exit snapshot)"
                    )
                result.trades.append(trade)
                realized_events.append((chain.as_of, trade.pnl))
                flat_events.append((chain.as_of, symbol))
                pos = None

    result.equity_curve = _build_portfolio_equity_curve(
        initial_capital=cfg.initial_capital,
        realized_events=realized_events,
        unrealized_events=unrealized_events,
        flat_events=flat_events,
    )
    if margin_points:
        # Sum concurrent per-symbol margin requirements at the same timestamp.
        margin_by_ts: dict[datetime, float] = defaultdict(float)
        for ts, m in margin_points:
            margin_by_ts[ts] += m
        ordered_m = sorted(margin_by_ts.items(), key=lambda kv: kv[0])
        result.margin_curve = pd.Series(
            [v for _, v in ordered_m],
            index=pd.DatetimeIndex([ts for ts, _ in ordered_m]),
        )
    return result
