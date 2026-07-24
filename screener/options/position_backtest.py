"""India EOD options position-P&L backtester.

Point-in-time rule
------------------
An entry signal is evaluated on the underlying's day-D close (optionally with
forward-filled options-panel columns). The structure is opened at day-D+1
session EOD chain prices (``last``, with ``settle`` as fallback). This is
strictly causal: day-D chain data is never used to fill a day-D signal.

Phase 4.1 adds opt-in realism, all backward-compatible (the defaults reproduce
the historical output byte-for-byte): configurable fill models (mid / bid-ask
cross with bps+tick slippage and illiquid-strike spread proxying), short-option
margin models (India SPAN-like scenario grid, US Reg-T) with portfolio margin
utilisation tracking, explicit expiry settlement (intrinsic vs official settle,
cash vs physical with ITM-assignment metadata), and config-driven DTE/delta
rolls as a first-class same-session exit-and-reenter.

P&L always uses observed chain prices. Black-Scholes is used only inside
strike selection (delta targeting) via :mod:`screener.options.structures`.

``screen_criterion`` is a thin alias: criterion names map to panel-column
boolean Pine expressions (``CRITERION_EXPRS``). The full options screening
pipeline is **not** replayed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any, cast

import pandas as pd

from screener.backtester.data import PriceFetcher, build_price_fetcher, tv_to_yf
from screener.backtester.pine import PineError, evaluate, parse
from screener.options.backtest import (
    OPTION_EXPRESSION_FIELDS,
    merge_options_into_bars,
    referenced_options_fields,
)
from screener.options.bt_models import (
    ExitReason,
    LegFill,
    OptionPositionTrade,
    OptionsBacktestConfig,
    StructureSpec,
)
from screener.options.greeks import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from screener.options.metrics import _quote_price
from screener.options.models import OptionChain, OptionContract
from screener.options.nse_bhavcopy import load_bhavcopy_chains
from screener.options.structures import build_structure, select_expiry, select_strike
from screener.symbols import tv_to_nse
from screener.unusual_volume.nse_client import is_trading_day

LOG = logging.getLogger(__name__)

ChainLoader = Callable[[date, set[str]], dict[str, OptionChain]]

# Thin aliases: criterion name → panel-column boolean expression.
# Documented limitation: does not replay the full options screening pipeline.
CRITERION_EXPRS: dict[str, str] = {
    "unusual_options": "unusual_options_ratio >= 2",
    "high_iv_rank": "iv_rank >= 80",
    "low_iv_rank": "iv_rank <= 20",
    "bullish_oi_buildup": (
        "call_oi_change > 0 and put_oi_change > 0 and put_oi_change > call_oi_change"
    ),
}


@dataclass
class _OpenPosition:
    symbol: str
    structure: str
    signal_date: date
    entry_date: date
    legs: list[LegFill]
    entry_premium: float  # signed debit>0 / credit<0
    gross_premium: float  # sum of |leg notional| at entry
    entry_costs: float
    hold_days: int = 0
    last_mark_premium: float = 0.0
    margin: float = 0.0
    entry_spot: float | None = None


@dataclass
class OptionsBacktestResult:
    trades: list[OptionPositionTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    warnings: list[str] = field(default_factory=list)
    # Portfolio margin used per session (empty unless a margin model is set).
    margin_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    peak_margin: float = 0.0
    peak_margin_utilization: float = 0.0


def _mark_price(contract: OptionContract | None) -> float | None:
    if contract is None:
        return None
    px = _quote_price(contract)
    if px is not None and px > 0:
        return float(px)
    if contract.settle is not None and contract.settle > 0:
        return float(contract.settle)
    return None


def _find_contract(
    chain: OptionChain | None,
    *,
    right: str,
    strike: float,
    expiry: date,
) -> OptionContract | None:
    if chain is None:
        return None
    for contract in chain.contracts:
        if (
            contract.right == right
            and contract.expiry == expiry
            and abs(contract.strike - strike) < 1e-9
        ):
            return contract
    return None


def _slip_price(
    price: float, side: int, slippage_pct: float, *, opening: bool
) -> float:
    """Apply side-aware slippage.

    Opening a long (side=+1) or closing a short pays up; opening a short or
    closing a long receives less.
    """
    if opening:
        # Buy to open long / sell to open short
        if side > 0:
            return price * (1.0 + slippage_pct)
        return price * (1.0 - slippage_pct)
    # Closing: reverse of open
    if side > 0:
        return price * (1.0 - slippage_pct)
    return price * (1.0 + slippage_pct)


def _leg_notional(price: float, lots: int, lot_size: float) -> float:
    return float(price) * float(lots) * float(lot_size)


def _signed_premium(legs: list[LegFill], *, use_exit: bool = False) -> float:
    total = 0.0
    for leg in legs:
        px = leg.exit_price if use_exit else leg.entry_price
        if px is None:
            continue
        # side +1 long contributes +price (debit); -1 short contributes -price (credit)
        total += leg.side * _leg_notional(px, leg.lots, leg.lot_size)
    return total


def _mtm_premium(legs: list[LegFill], marks: list[float]) -> float:
    total = 0.0
    for leg, mark in zip(legs, marks, strict=True):
        total += leg.side * _leg_notional(mark, leg.lots, leg.lot_size)
    return total


def _intrinsic(right: str, strike: float, spot: float) -> float:
    if right == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


_DAYS_PER_YEAR = 365.0


def _india_risk_free_rate() -> float:
    """Risk-free rate for BS margin/greeks, per the package env convention."""
    raw = os.environ.get("SCREENER_INDIA_RISK_FREE_RATE")
    if raw is None:
        return 0.065
    try:
        return float(raw)
    except ValueError:
        return 0.065


# --- Fill models -----------------------------------------------------------


def _effective_quote(
    contract: OptionContract, cfg: OptionsBacktestConfig
) -> tuple[float, float] | None:
    """Return (bid, ask), proxying a spread when the leg has no live quotes.

    When quotes are absent the spread is proxied from settle/close dispersion,
    floored at ``illiquid_spread_pct`` of the mark, and centred on the mark.
    """
    bid, ask = contract.bid, contract.ask
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return float(bid), float(ask)
    mid = _mark_price(contract)
    if mid is None or mid <= 0:
        return None
    dispersion = 0.0
    if contract.settle is not None and contract.last is not None:
        dispersion = abs(float(contract.settle) - float(contract.last))
    spread = max(dispersion, cfg.illiquid_spread_pct * mid)
    half = spread / 2.0
    return max(mid - half, 0.0), mid + half


def _is_buying(side: int, *, opening: bool) -> bool:
    """True when the order buys (pays the ask): open-long or close-short."""
    return (opening and side > 0) or (not opening and side < 0)


def _fill_price(
    contract: OptionContract | None,
    side: int,
    *,
    opening: bool,
    cfg: OptionsBacktestConfig,
) -> float | None:
    """Fill price under the ``mid``/``cross`` fill models (never ``legacy``)."""
    mid = _mark_price(contract)
    if contract is None or mid is None or mid <= 0:
        return None
    buying = _is_buying(side, opening=opening)
    if cfg.fill_model == "mid":
        base = mid
    else:  # cross the spread
        quote = _effective_quote(contract, cfg)
        if quote is None:
            return None
        bid, ask = quote
        base = ask if buying else bid
    direction = 1.0 if buying else -1.0
    px = base * (1.0 + direction * cfg.slippage_bps / 10_000.0)
    px += direction * cfg.slippage_ticks * cfg.tick_size
    if cfg.slippage_pct:
        px = _slip_price(px, side, cfg.slippage_pct, opening=opening)
    return max(px, 0.0)


def _entry_fill_price(
    contract: OptionContract, side: int, *, cfg: OptionsBacktestConfig
) -> float | None:
    """Opening fill price; ``legacy`` reproduces the historical path exactly."""
    if cfg.fill_model == "legacy":
        raw = _mark_price(contract)
        if raw is None or raw <= 0:
            return None
        return _slip_price(raw, side, cfg.slippage_pct, opening=True)
    return _fill_price(contract, side, opening=True, cfg=cfg)


def _exit_fill_price(
    contract: OptionContract | None,
    mark: float,
    side: int,
    *,
    cfg: OptionsBacktestConfig,
) -> float:
    """Closing fill price; ``legacy`` (or a missing contract) slips the mark."""
    if cfg.fill_model != "legacy" and contract is not None:
        fp = _fill_price(contract, side, opening=False, cfg=cfg)
        if fp is not None:
            return fp
    return _slip_price(mark, side, cfg.slippage_pct, opening=False)


# --- Margin models ---------------------------------------------------------


def _leg_time_to_expiry(expiry: date, as_of: date) -> float:
    return max((expiry - as_of).days, 0) / _DAYS_PER_YEAR


def _leg_iv(leg: LegFill, spot: float, as_of: date, rate: float) -> float | None:
    if leg.entry_iv is not None and leg.entry_iv > 0:
        return float(leg.entry_iv)
    t = _leg_time_to_expiry(leg.expiry, as_of)
    if t <= 0 or leg.entry_price <= 0 or spot <= 0:
        return None
    return implied_volatility(leg.entry_price, spot, leg.strike, t, rate, leg.right)


def _leg_bs_price(
    leg: LegFill,
    spot: float,
    as_of: date,
    rate: float,
    iv: float,
    *,
    price_factor: float = 1.0,
    vol_shift: float = 0.0,
) -> float:
    t = _leg_time_to_expiry(leg.expiry, as_of)
    shocked = spot * price_factor
    if t <= 0:
        return _intrinsic(leg.right, leg.strike, shocked)
    vol = max(iv + vol_shift, 1e-6)
    price = black_scholes_price(shocked, leg.strike, t, rate, vol, leg.right)
    if price is None:
        return _intrinsic(leg.right, leg.strike, shocked)
    return price


def compute_span_margin(
    legs: list[LegFill], spot: float, as_of: date, cfg: OptionsBacktestConfig
) -> float:
    """India SPAN-like margin: worst-of price/vol scenario loss + exposure."""
    if spot <= 0:
        return 0.0
    rate = _india_risk_free_rate()
    ivs = [_leg_iv(leg, spot, as_of, rate) for leg in legs]
    price_factors = (
        1.0 - cfg.span_price_scan_pct,
        1.0,
        1.0 + cfg.span_price_scan_pct,
    )
    vol_shifts = (-cfg.span_vol_scan, 0.0, cfg.span_vol_scan)

    def _portfolio_value(pf: float, vs: float) -> float:
        total = 0.0
        for leg, iv in zip(legs, ivs, strict=True):
            if iv is None:
                px = _intrinsic(leg.right, leg.strike, spot * pf)
            else:
                px = _leg_bs_price(
                    leg, spot, as_of, rate, iv, price_factor=pf, vol_shift=vs
                )
            total += leg.side * _leg_notional(px, leg.lots, leg.lot_size)
        return total

    base = _portfolio_value(1.0, 0.0)
    worst = 0.0
    for pf in price_factors:
        for vs in vol_shifts:
            worst = min(worst, _portfolio_value(pf, vs) - base)
    scan_margin = max(-worst, 0.0)
    exposure = cfg.span_exposure_pct * sum(
        spot * leg.lots * leg.lot_size for leg in legs if leg.side < 0
    )
    return scan_margin + exposure


def compute_regt_margin(
    legs: list[LegFill], spot: float, cfg: OptionsBacktestConfig
) -> float:
    """US Reg-T margin: premium + max(pct*underlying - OTM, min_pct*strike)."""
    if spot <= 0:
        return 0.0
    total = 0.0
    for leg in legs:
        qty = leg.lots * leg.lot_size
        premium = leg.entry_price * qty
        if leg.side > 0:
            total += premium  # long options are paid for in full
            continue
        if leg.right == "call":
            otm = max(leg.strike - spot, 0.0)
        else:
            otm = max(spot - leg.strike, 0.0)
        per_unit = max(cfg.regt_pct * spot - otm, cfg.regt_min_pct * leg.strike)
        total += per_unit * qty + premium
    return total


def _position_margin(
    legs: list[LegFill],
    spot: float | None,
    as_of: date,
    cfg: OptionsBacktestConfig,
) -> float:
    if cfg.margin_model == "none" or spot is None or spot <= 0:
        return 0.0
    if cfg.margin_model == "span":
        return compute_span_margin(legs, spot, as_of, cfg)
    return compute_regt_margin(legs, spot, cfg)


def _net_delta(legs: list[LegFill], spot: float | None, as_of: date) -> float | None:
    """Position net delta (per lot) via BS; ``None`` if no leg prices out."""
    if spot is None or spot <= 0:
        return None
    rate = _india_risk_free_rate()
    total = 0.0
    seen = False
    for leg in legs:
        iv = _leg_iv(leg, spot, as_of, rate)
        t = _leg_time_to_expiry(leg.expiry, as_of)
        if iv is None or t <= 0:
            continue
        greeks = black_scholes_greeks(spot, leg.strike, t, rate, iv, leg.right)
        if greeks is None:
            continue
        seen = True
        total += leg.side * leg.lots * float(greeks["delta"])
    return total if seen else None


def _position_return_pct(
    entry_premium: float, mark_premium: float, gross_premium: float
) -> float:
    """Premium-relative MTM return in percent.

    Premiums are signed (debit > 0, credit < 0) so ``pnl = mark - entry``
    holds for both sides: a long paying 10 now worth 15 gains +5, and a
    short receiving 10 (entry -10) now worth 15 (mark -15) loses 5. The
    denominator is gross entry premium, not exchange margin.
    """
    if gross_premium <= 0:
        return 0.0
    pnl = mark_premium - entry_premium
    return (pnl / gross_premium) * 100.0


def _resolve_entry_expression(cfg: OptionsBacktestConfig) -> str:
    if cfg.screen_criterion:
        key = cfg.screen_criterion.strip().lower()
        if key not in CRITERION_EXPRS:
            raise ValueError(
                f"unknown screen criterion {cfg.screen_criterion!r}; expected one of "
                f"{', '.join(sorted(CRITERION_EXPRS))}"
            )
        return CRITERION_EXPRS[key]
    return cfg.entry_expr or "true"


def _trading_days(
    start: date,
    end: date,
    trading_day: Callable[[date], bool] = is_trading_day,
) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if trading_day(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _default_chain_loader(
    day: date, symbols: set[str], *, refresh: bool = False
) -> dict[str, OptionChain]:
    return load_bhavcopy_chains(day, symbols=symbols, refresh=refresh)


def _liquidity_ok(contract: OptionContract, cfg: OptionsBacktestConfig) -> bool:
    if cfg.min_oi > 0 and contract.oi < cfg.min_oi:
        return False
    if cfg.min_volume > 0 and contract.volume < cfg.min_volume:
        return False
    return True


def _open_structure(
    chain: OptionChain,
    structure: StructureSpec,
    *,
    as_of: date,
    cfg: OptionsBacktestConfig,
    warnings: list[str],
) -> list[LegFill] | None:
    fills: list[LegFill] = []
    for leg in structure.legs:
        expiry = select_expiry(chain, leg.expiry_rule, as_of)
        if expiry is None:
            warnings.append(
                f"{chain.underlying} {as_of}: no expiry for rule {leg.expiry_rule!r}"
            )
            return None
        contract = select_strike(chain, expiry, leg.right, leg.strike_rule, as_of=as_of)
        if contract is None:
            warnings.append(
                f"{chain.underlying} {as_of}: no strike for {leg.right} "
                f"{leg.strike_rule!r} exp {expiry}"
            )
            return None
        if not _liquidity_ok(contract, cfg):
            warnings.append(
                f"{chain.underlying} {as_of}: liquidity filter failed for "
                f"{contract.right} {contract.strike:g} {expiry}"
            )
            return None
        slipped = _entry_fill_price(contract, leg.side, cfg=cfg)
        if slipped is None or slipped <= 0:
            warnings.append(
                f"{chain.underlying} {as_of}: no price for "
                f"{contract.right} {contract.strike:g} {expiry}"
            )
            return None
        lot_size = float(contract.lot_size or 1.0)
        fills.append(
            LegFill(
                right=leg.right,
                strike=float(contract.strike),
                expiry=expiry,
                side=leg.side,
                lots=leg.lots,
                lot_size=lot_size,
                entry_price=slipped,
                exit_price=None,
                entry_iv=contract.iv,
            )
        )
    return fills


def _mark_legs(
    pos: _OpenPosition,
    chain: OptionChain | None,
    *,
    as_of: date,
    underlying_close: float | None,
    carry: Mapping[tuple[str, float, date], float],
) -> tuple[list[float], dict[tuple[str, float, date], float]]:
    """Return per-leg mark prices and an updated carry-forward map."""
    marks: list[float] = []
    updated = dict(carry)
    for leg in pos.legs:
        key = (leg.right, leg.strike, leg.expiry)
        contract = _find_contract(
            chain, right=leg.right, strike=leg.strike, expiry=leg.expiry
        )
        px = _mark_price(contract)
        if px is None and underlying_close is not None and leg.expiry <= as_of:
            # On/after expiry use intrinsic against underlying close.
            px = _intrinsic(leg.right, leg.strike, underlying_close)
        if px is None:
            px = updated.get(key)
        if px is None:
            px = leg.entry_price  # last resort
        else:
            updated[key] = px
        marks.append(float(px))
    return marks, updated


def _close_position(
    pos: _OpenPosition,
    marks: list[float],
    *,
    exit_date: date,
    exit_reason: ExitReason,
    cfg: OptionsBacktestConfig,
    at_expiry: bool = False,
    contracts: list[OptionContract | None] | None = None,
    extra_details: dict[str, Any] | None = None,
) -> OptionPositionTrade:
    leg_contracts = contracts or [None] * len(pos.legs)
    closed_legs: list[LegFill] = []
    exit_costs = 0.0
    for leg, mark, contract in zip(pos.legs, marks, leg_contracts, strict=True):
        if at_expiry:
            exit_px = mark  # intrinsic/settle, no extra slip
        else:
            exit_px = _exit_fill_price(contract, mark, leg.side, cfg=cfg)
        exit_costs += cfg.commission_per_order
        closed_legs.append(replace(leg, exit_price=exit_px))

    exit_premium = _signed_premium(closed_legs, use_exit=True)
    # pnl = mark_value change − entry costs − exit costs
    raw_pnl = exit_premium - pos.entry_premium
    pnl = raw_pnl - pos.entry_costs - exit_costs
    gross = pos.gross_premium if pos.gross_premium > 0 else 1.0
    return_pct = (pnl / gross) * 100.0
    details: dict[str, Any] = {
        "signal_date": pos.signal_date.isoformat(),
        "entry_costs": pos.entry_costs,
        "exit_costs": exit_costs,
        "hold_days": pos.hold_days,
    }
    if extra_details:
        details.update(extra_details)
    return OptionPositionTrade(
        symbol=pos.symbol,
        structure=pos.structure,
        entry_date=pos.entry_date,
        exit_date=exit_date,
        legs=tuple(closed_legs),
        entry_premium=pos.entry_premium,
        exit_premium=exit_premium,
        pnl=pnl,
        return_pct=return_pct,
        exit_reason=exit_reason,
        passed_filter=True,
        details=details,
    )


def _leg_contracts(
    pos: _OpenPosition, chain: OptionChain | None
) -> list[OptionContract | None]:
    return [
        _find_contract(chain, right=leg.right, strike=leg.strike, expiry=leg.expiry)
        for leg in pos.legs
    ]


def _settlement_marks(
    pos: _OpenPosition,
    chain: OptionChain | None,
    *,
    underlying_close: float | None,
) -> list[float]:
    """Expiry marks preferring the official per-contract settlement price."""
    marks: list[float] = []
    for leg in pos.legs:
        px: float | None = None
        contract = _find_contract(
            chain, right=leg.right, strike=leg.strike, expiry=leg.expiry
        )
        if contract is not None and contract.settle is not None and contract.settle > 0:
            px = float(contract.settle)
        if px is None:
            px = (
                _intrinsic(leg.right, leg.strike, underlying_close)
                if underlying_close is not None
                else 0.0
            )
        marks.append(px)
    return marks


def _expiry_details(
    pos: _OpenPosition,
    *,
    underlying_close: float | None,
    cfg: OptionsBacktestConfig,
) -> dict[str, Any]:
    """Settlement / assignment metadata recorded on expiry closes."""
    settlement = "physical" if cfg.physical_settlement else "cash"
    assigned: list[str] = []
    itm: list[str] = []
    if underlying_close is not None:
        for leg in pos.legs:
            if _intrinsic(leg.right, leg.strike, underlying_close) <= 0:
                continue
            tag = f"{leg.right}:{leg.strike:g}"
            itm.append(tag)
            if leg.side < 0:  # short ITM leg is assigned
                assigned.append(tag)
    return {
        "settlement": settlement,
        "settlement_price": cfg.settlement,
        "itm_legs": itm,
        "assigned_legs": assigned,
    }


def _check_exit(
    pos: _OpenPosition,
    marks: list[float],
    *,
    day: date,
    underlying_close: float | None,
    exit_signal: bool,
    cfg: OptionsBacktestConfig,
) -> tuple[ExitReason, bool] | None:
    """Return (reason, at_expiry) if the position should close today."""
    # Nearest leg expiry drives expiry / DTE checks.
    nearest_expiry = min(leg.expiry for leg in pos.legs)
    dte = (nearest_expiry - day).days

    if dte <= 0:
        # Force settle at intrinsic when we have a spot; marks already prefer settle.
        if underlying_close is not None:
            return "expiry", True
        return "expiry", True

    mark_premium = _mtm_premium(pos.legs, marks)
    ret = _position_return_pct(pos.entry_premium, mark_premium, pos.gross_premium)

    if cfg.stop_pct is not None and ret <= -abs(cfg.stop_pct):
        return "stop", False
    if cfg.target_pct is not None and ret >= abs(cfg.target_pct):
        return "target", False
    if cfg.roll_dte is not None and dte <= cfg.roll_dte:
        return "roll", False
    if cfg.roll_delta is not None:
        net = _net_delta(pos.legs, underlying_close, day)
        if net is not None and abs(net) >= cfg.roll_delta:
            return "roll", False
    if dte <= cfg.exit_dte:
        return "dte", False
    if exit_signal:
        return "exit_expr", False
    if cfg.max_hold is not None and pos.hold_days >= cfg.max_hold:
        return "time", False
    return None


def run_options_position_backtest(
    cfg: OptionsBacktestConfig,
    *,
    chain_loader: ChainLoader | None = None,
    price_fetcher: PriceFetcher | None = None,
    panel: pd.DataFrame | None = None,
    trading_day: Callable[[date], bool] = is_trading_day,
) -> OptionsBacktestResult:
    """Simulate option structures over ``cfg.start``..``cfg.end`` (India EOD)."""
    if cfg.end < cfg.start:
        raise ValueError("end must be on or after start")
    if cfg.market != "india":
        raise ValueError(
            "options position backtest currently supports market=india only"
        )

    structure = build_structure(
        cfg.structure,
        strike_rule=cfg.strike_rule,
        expiry_rule=cfg.expiry_rule,
        width_pct=cfg.width_pct,
        lots=cfg.lots,
    )
    entry_expr = _resolve_entry_expression(cfg)
    try:
        entry_ast = parse(entry_expr)
    except PineError as exc:
        raise ValueError(f"invalid entry expression: {exc}") from exc
    exit_ast = None
    if cfg.exit_expr:
        try:
            exit_ast = parse(cfg.exit_expr)
        except PineError as exc:
            raise ValueError(f"invalid exit expression: {exc}") from exc

    symbols = tuple(
        tv_to_nse(s, strip_suffix=True) for s in cfg.tickers if s and s.strip()
    )
    if not symbols:
        raise ValueError("at least one ticker is required")
    symbol_set = set(symbols)

    loader: ChainLoader
    if chain_loader is not None:
        loader = chain_loader
    else:

        def loader(day: date, syms: set[str]) -> dict[str, OptionChain]:
            return _default_chain_loader(day, syms, refresh=cfg.refresh)

    fetcher = price_fetcher or build_price_fetcher()
    # Warmup for Pine expressions (sma etc.): pull a bit of history.
    price_start = cfg.start - timedelta(days=60)
    yf_by_nse = {sym: tv_to_yf(sym, "india") for sym in symbols}
    raw_prices = fetcher.fetch(
        list(dict.fromkeys(yf_by_nse.values())), price_start, cfg.end
    )
    bars_by_tv: dict[str, pd.DataFrame] = {}
    for nse_sym, yf_sym in yf_by_nse.items():
        frame = raw_prices.get(yf_sym, pd.DataFrame())
        bars_by_tv[nse_sym] = frame

    opt_fields = referenced_options_fields(entry_ast, exit_ast)
    if opt_fields or cfg.screen_criterion:
        # Ensure criterion panel fields are requested even if not in entry_expr.
        if cfg.screen_criterion:
            opt_fields |= OPTION_EXPRESSION_FIELDS
        join = merge_options_into_bars(
            bars_by_tv, market="india", fields=opt_fields, panel=panel
        )
        bars_by_tv = join.bars_by_tv

    entry_signals: dict[str, pd.Series] = {}
    exit_signals: dict[str, pd.Series] = {}
    warnings: list[str] = []
    for sym, bars in bars_by_tv.items():
        if bars is None or bars.empty:
            warnings.append(f"{sym}: no underlying price history")
            continue
        try:
            entry_signals[sym] = evaluate(entry_ast, bars).fillna(False).astype(bool)
        except PineError as exc:
            warnings.append(f"{sym}: entry eval failed: {exc}")
            entry_signals[sym] = pd.Series(False, index=bars.index)
        if exit_ast is not None:
            try:
                exit_signals[sym] = evaluate(exit_ast, bars).fillna(False).astype(bool)
            except PineError as exc:
                warnings.append(f"{sym}: exit eval failed: {exc}")
                exit_signals[sym] = pd.Series(False, index=bars.index)

    # Structure used when rolling (exit + same-session re-enter into a further
    # expiry). Built once; only consulted when a roll rule is configured.
    roll_structure = build_structure(
        cfg.structure,
        strike_rule=cfg.strike_rule,
        expiry_rule=cfg.roll_expiry_rule,
        width_pct=cfg.width_pct,
        lots=cfg.lots,
    )

    days = _trading_days(cfg.start, cfg.end, trading_day=trading_day)
    open_positions: dict[str, _OpenPosition] = {}
    pending_entries: dict[str, date] = {}  # symbol → signal date
    carry_marks: dict[str, dict[tuple[str, float, date], float]] = {}
    trades: list[OptionPositionTrade] = []
    equity_points: list[tuple[date, float]] = []
    margin_points: list[tuple[date, float]] = []
    peak_margin = 0.0
    cash_pnl = 0.0  # realized pnl accumulator
    last_chains: dict[str, OptionChain] = {}

    def _underlying_close(sym: str, day: date) -> float | None:
        bars = bars_by_tv.get(sym)
        if bars is None or bars.empty:
            return None
        ts = pd.Timestamp(day)
        # Daily bars may be midnight-normalized.
        if ts in bars.index:
            return float(bars["close"].loc[ts])
        day_labels = cast(pd.DatetimeIndex, bars.index).normalize()
        day_rows = bars[day_labels == ts.normalize()]
        if day_rows.empty:
            return None
        return float(day_rows.iloc[-1]["close"])

    def _signal_on(sym: str, day: date, series_map: dict[str, pd.Series]) -> bool:
        series = series_map.get(sym)
        if series is None or series.empty:
            return False
        ts = pd.Timestamp(day)
        if ts in series.index:
            return bool(series.loc[ts])
        day_rows = series[
            cast(pd.DatetimeIndex, series.index).normalize() == ts.normalize()
        ]
        if day_rows.empty:
            return False
        return bool(day_rows.iloc[-1])

    def _open_equity() -> float:
        total = cash_pnl
        for pos in open_positions.values():
            total += pos.last_mark_premium - pos.entry_premium - pos.entry_costs
        return cfg.initial_capital + total

    def _current_margin() -> float:
        return sum(pos.margin for pos in open_positions.values())

    def _enter(
        sym: str,
        chain: OptionChain,
        struct: StructureSpec,
        *,
        signal_date: date,
        entry_date: date,
    ) -> bool:
        """Open ``struct`` for ``sym`` at ``chain``; False if it cannot fill."""
        fills = _open_structure(
            chain, struct, as_of=entry_date, cfg=cfg, warnings=warnings
        )
        if fills is None:
            return False
        spot = _underlying_close(sym, entry_date)
        if (spot is None or spot <= 0) and chain.spot:
            spot = float(chain.spot)
        margin = _position_margin(fills, spot, entry_date, cfg)
        if cfg.margin_cap_pct is not None:
            cap = cfg.initial_capital * cfg.margin_cap_pct
            if _current_margin() + margin > cap + 1e-9:
                warnings.append(
                    f"{sym} {entry_date}: margin cap reached "
                    f"({_current_margin():.0f}+{margin:.0f} > {cap:.0f})"
                )
                return False
        entry_premium = _signed_premium(fills)
        gross = sum(
            abs(_leg_notional(leg.entry_price, leg.lots, leg.lot_size)) for leg in fills
        )
        entry_costs = cfg.commission_per_order * len(fills)
        open_positions[sym] = _OpenPosition(
            symbol=sym,
            structure=struct.name,
            signal_date=signal_date,
            entry_date=entry_date,
            legs=fills,
            entry_premium=entry_premium,
            gross_premium=gross,
            entry_costs=entry_costs,
            hold_days=0,
            last_mark_premium=entry_premium,
            margin=margin,
            entry_spot=spot,
        )
        carry_marks[sym] = {
            (leg.right, leg.strike, leg.expiry): leg.entry_price for leg in fills
        }
        return True

    for day in days:
        try:
            chains = loader(day, symbol_set)
        except Exception as exc:  # noqa: BLE001 - archive gaps degrade per day
            LOG.warning("options chain unavailable for %s: %s", day, exc)
            warnings.append(f"{day}: chain load failed ({exc})")
            chains = {}

        if chains:
            last_chains.update(chains)

        # 1) Fill pending entries from prior-day signals at today's EOD chain.
        for sym, signal_date in list(pending_entries.items()):
            if sym in open_positions:
                del pending_entries[sym]
                continue
            chain = chains.get(sym)
            if chain is None:
                # Missing archive day: keep pending if within range.
                continue
            # Drop the pending entry whether it filled or was rejected (no
            # price / liquidity / margin cap) — the D+1 window has passed.
            _enter(sym, chain, structure, signal_date=signal_date, entry_date=day)
            del pending_entries[sym]

        # 2) MTM + exits for open positions.
        for sym, pos in list(open_positions.items()):
            u_close = _underlying_close(sym, day)
            chain = chains.get(sym)  # only today's chain for fresh marks
            marks, carry_marks[sym] = _mark_legs(
                pos,
                chain,
                as_of=day,
                underlying_close=u_close,
                carry=carry_marks.get(sym, {}),
            )
            pos.hold_days += 1
            pos.last_mark_premium = _mtm_premium(pos.legs, marks)

            exit_sig = (
                _signal_on(sym, day, exit_signals) if exit_ast is not None else False
            )
            # Don't exit on the entry day via exit_expr/time before marks settle.
            decision = _check_exit(
                pos,
                marks,
                day=day,
                underlying_close=u_close,
                exit_signal=exit_sig and day > pos.entry_date,
                cfg=cfg,
            )
            if decision is None:
                continue
            reason, at_expiry = decision
            extra_details: dict[str, Any] | None = None
            if at_expiry:
                if cfg.settlement == "settle":
                    marks = _settlement_marks(pos, chain, underlying_close=u_close)
                elif u_close is not None:
                    marks = [
                        _intrinsic(leg.right, leg.strike, u_close) for leg in pos.legs
                    ]
                if cfg.physical_settlement or cfg.settlement != "intrinsic":
                    extra_details = _expiry_details(
                        pos, underlying_close=u_close, cfg=cfg
                    )
            trade = _close_position(
                pos,
                marks,
                exit_date=day,
                exit_reason=reason,
                cfg=cfg,
                at_expiry=at_expiry,
                contracts=_leg_contracts(pos, chain),
                extra_details=extra_details,
            )
            trades.append(trade)
            cash_pnl += trade.pnl
            del open_positions[sym]
            carry_marks.pop(sym, None)
            # Roll is a first-class exit-and-reenter within the same session.
            if reason == "roll" and chain is not None:
                _enter(sym, chain, roll_structure, signal_date=day, entry_date=day)

        # 3) New signals (for D+1 entry) when flat.
        for sym in symbols:
            if sym in open_positions or sym in pending_entries:
                continue
            if _signal_on(sym, day, entry_signals):
                pending_entries[sym] = day

        if cfg.margin_model != "none":
            margin_used = _current_margin()
            margin_points.append((day, margin_used))
            peak_margin = max(peak_margin, margin_used)

        equity_points.append((day, _open_equity()))

    # Force-close remaining positions at end of range.
    if open_positions:
        last_day = days[-1] if days else cfg.end
        for sym, pos in list(open_positions.items()):
            chain = last_chains.get(sym)
            u_close = _underlying_close(sym, last_day)
            marks, _ = _mark_legs(
                pos,
                chain,
                as_of=last_day,
                underlying_close=u_close,
                carry=carry_marks.get(sym, {}),
            )
            trade = _close_position(
                pos,
                marks,
                exit_date=last_day,
                exit_reason="end",
                cfg=cfg,
                at_expiry=False,
                contracts=_leg_contracts(pos, chain),
            )
            trades.append(trade)
            cash_pnl += trade.pnl
            del open_positions[sym]
        if equity_points:
            equity_points[-1] = (equity_points[-1][0], cfg.initial_capital + cash_pnl)
        else:
            equity_points.append((last_day, cfg.initial_capital + cash_pnl))

    if equity_points:
        equity = pd.Series(
            {pd.Timestamp(d): v for d, v in equity_points}, dtype=float
        ).sort_index()
    else:
        equity = pd.Series(dtype=float)

    if margin_points:
        margin_curve = pd.Series(
            {pd.Timestamp(d): v for d, v in margin_points}, dtype=float
        ).sort_index()
    else:
        margin_curve = pd.Series(dtype=float)
    peak_margin_utilization = (
        peak_margin / cfg.initial_capital if cfg.initial_capital > 0 else 0.0
    )

    return OptionsBacktestResult(
        trades=trades,
        equity_curve=equity,
        warnings=warnings,
        margin_curve=margin_curve,
        peak_margin=peak_margin,
        peak_margin_utilization=peak_margin_utilization,
    )


__all__ = [
    "CRITERION_EXPRS",
    "ChainLoader",
    "OptionsBacktestResult",
    "compute_regt_margin",
    "compute_span_margin",
    "run_options_position_backtest",
]
