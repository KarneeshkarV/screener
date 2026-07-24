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
position is flattened at the session's last snapshot (``session_end``). Times
are compared in the market's local timezone (snapshots are stored naive UTC).

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
    )


def _mark_legs(pos: _OpenIntradayPosition, chain: OptionChain) -> list[float]:
    marks: list[float] = []
    for leg in pos.legs:
        contract = _find_contract(
            chain, right=leg.right, strike=leg.strike, expiry=leg.expiry
        )
        px = _mark_price(contract)
        marks.append(float(px) if px is not None else leg.entry_price)
    return marks


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
    details = {
        "entry_ts": pos.entry_ts.isoformat(),
        "exit_ts": exit_ts.isoformat(),
        "entry_costs": pos.entry_costs,
        "exit_costs": exit_costs,
    }
    if pos.hedge_qty:
        details["equity_hedge_qty"] = pos.hedge_qty
        details["equity_hedge_pnl"] = hedge_pnl
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
    is_last: bool,
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
    if is_last:
        return "session_end"
    return None


def run_intraday_options_backtest(
    cfg: IntradayOptionsBacktestConfig,
    provider: OptionsHistoryProvider | None = None,
) -> IntradayOptionsBacktestResult:
    """Walk intraday snapshots per session and enter/mark/exit point-in-time.

    ``provider`` defaults to the forward-capture contract store for ``market``.
    Positions never carry overnight: whatever is open at a session's last
    snapshot is flattened there (``session_end``).
    """
    provider = provider or default_history_provider(cfg.market)
    fill_cfg = _fill_cfg(cfg)
    tz = ZoneInfo(get_market(cfg.market).timezone)
    result = IntradayOptionsBacktestResult()

    equity = cfg.initial_capital
    equity_points: list[tuple[datetime, float]] = []
    margin_points: list[tuple[datetime, float]] = []

    for day in _trading_days(cfg.start, cfg.end):
        for symbol in cfg.tickers:
            chains = sorted(provider.chains(symbol, day), key=lambda c: c.as_of)
            if not chains:
                continue
            pos: _OpenIntradayPosition | None = None
            for idx, chain in enumerate(chains):
                is_last = idx == len(chains) - 1
                local_t = _local_time(chain.as_of, tz)

                if pos is None:
                    entry_ok = cfg.entry_time is None or local_t >= cfg.entry_time
                    # Never open at the last snapshot (no bar left to exit on).
                    if entry_ok and not is_last:
                        pos = _open_position(chain, cfg, fill_cfg, result.warnings)
                    equity_points.append((chain.as_of, equity))
                    continue

                marks = _mark_legs(pos, chain)
                reason = _exit_reason(
                    pos, marks, local_t=local_t, cfg=cfg, is_last=is_last
                )
                if cfg.margin_model != "none":
                    margin = _position_margin(
                        pos.legs, chain.spot, chain.as_of.date(), fill_cfg
                    )
                    margin_points.append((chain.as_of, margin))
                    result.peak_margin = max(result.peak_margin, margin)

                if reason is None:
                    unrealized = (
                        _mtm_premium(pos.legs, marks)
                        - pos.entry_premium
                        + _hedge_pnl(pos, chain.spot)
                    )
                    equity_points.append((chain.as_of, equity + unrealized))
                    continue

                trade = _close_position(
                    pos,
                    chain,
                    marks,
                    exit_ts=chain.as_of,
                    exit_reason=reason,
                    fill_cfg=fill_cfg,
                )
                result.trades.append(trade)
                equity += trade.pnl
                equity_points.append((chain.as_of, equity))
                pos = None

    if equity_points:
        idx_e = pd.DatetimeIndex([ts for ts, _ in equity_points])
        result.equity_curve = pd.Series([v for _, v in equity_points], index=idx_e)
    if margin_points:
        idx_m = pd.DatetimeIndex([ts for ts, _ in margin_points])
        result.margin_curve = pd.Series([v for _, v in margin_points], index=idx_m)
    return result
