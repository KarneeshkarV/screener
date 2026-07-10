"""Market-agnostic option-chain signal math.

All functions operate only on :mod:`screener.options.models`; provider loading
stays outside this module so calculations remain deterministic and easy to
exercise with offline fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from statistics import median
from typing import Iterable

from screener.options.models import ChainMetrics, OptionChain, OptionContract

_MAX_IV = 5.0  # 500%; filters NSE/CBOE zero and garbage rows.
_NEAR_SPOT_PCT = 0.05


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return a finite ratio, collapsing an absent/zero leg to ``None``."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = float(numerator) / float(denominator)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _unexpired(chain: OptionChain) -> list[OptionContract]:
    as_of = chain.as_of.date()
    current = [contract for contract in chain.contracts if contract.expiry >= as_of]
    return current or list(chain.contracts)


def _contracts_for_expiry(
    contracts: Iterable[OptionContract], expiry: date | None
) -> list[OptionContract]:
    if expiry is None:
        return []
    return [contract for contract in contracts if contract.expiry == expiry]


def _front_expiries(chain: OptionChain) -> tuple[date | None, date | None]:
    expiries = sorted({contract.expiry for contract in _unexpired(chain)})
    front = expiries[0] if expiries else None
    next_expiry = expiries[1] if len(expiries) > 1 else None
    return front, next_expiry


def _max_pain(contracts: Iterable[OptionContract]) -> float | None:
    rows = list(contracts)
    strikes = sorted({contract.strike for contract in rows})
    if not strikes or not any(contract.oi > 0 for contract in rows):
        return None
    payouts: list[tuple[float, float]] = []
    for settlement in strikes:
        payout = 0.0
        for contract in rows:
            intrinsic = (
                max(settlement - contract.strike, 0.0)
                if contract.right == "call"
                else max(contract.strike - settlement, 0.0)
            )
            payout += intrinsic * contract.oi
        payouts.append((payout, settlement))
    return min(payouts, key=lambda item: (item[0], item[1]))[1]


def _oi_concentration(
    contracts: Iterable[OptionContract],
    *,
    spot: float | None,
    top_n: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    call_by_strike: dict[float, float] = defaultdict(float)
    put_by_strike: dict[float, float] = defaultdict(float)
    for contract in contracts:
        target = call_by_strike if contract.right == "call" else put_by_strike
        target[contract.strike] += contract.oi
    support_pool = {
        strike: oi
        for strike, oi in put_by_strike.items()
        if oi > 0 and (spot is None or strike <= spot)
    }
    resistance_pool = {
        strike: oi
        for strike, oi in call_by_strike.items()
        if oi > 0 and (spot is None or strike >= spot)
    }
    supports = tuple(
        strike
        for strike, _oi in sorted(
            support_pool.items(), key=lambda item: (-item[1], -item[0])
        )[:top_n]
    )
    resistances = tuple(
        strike
        for strike, _oi in sorted(
            resistance_pool.items(), key=lambda item: (-item[1], item[0])
        )[:top_n]
    )
    return supports, resistances


def _valid_iv(contract: OptionContract) -> bool:
    return (
        contract.iv is not None
        and 0 < contract.iv < _MAX_IV
        and (contract.oi > 0 or contract.volume > 0)
    )


def _nearest(
    contracts: Iterable[OptionContract],
    *,
    spot: float,
    right: str | None = None,
) -> OptionContract | None:
    rows = [
        contract
        for contract in contracts
        if (right is None or contract.right == right) and _valid_iv(contract)
    ]
    if not rows:
        return None
    return min(
        rows, key=lambda contract: (abs(contract.strike - spot), contract.strike)
    )


def _atm_iv(contracts: Iterable[OptionContract], spot: float | None) -> float | None:
    if spot is None:
        return None
    rows = list(contracts)
    call = _nearest(rows, spot=spot, right="call")
    put = _nearest(rows, spot=spot, right="put")
    values = [contract.iv for contract in (call, put) if contract and contract.iv]
    return float(median(values)) if values else None


def _iv_skew(contracts: Iterable[OptionContract], spot: float | None) -> float | None:
    rows = [contract for contract in contracts if _valid_iv(contract)]
    if spot is None or not rows:
        return None

    delta_calls = [
        contract
        for contract in rows
        if contract.right == "call" and contract.delta is not None
    ]
    delta_puts = [
        contract
        for contract in rows
        if contract.right == "put" and contract.delta is not None
    ]
    if delta_calls and delta_puts:
        call = min(
            delta_calls,
            key=lambda contract: abs(
                (contract.delta if contract.delta is not None else 0.0) - 0.25
            ),
        )
        put = min(
            delta_puts,
            key=lambda contract: abs(
                (contract.delta if contract.delta is not None else 0.0) + 0.25
            ),
        )
    else:
        # Sources without greeks use the nearest out-of-the-money strikes.
        calls = [c for c in rows if c.right == "call" and c.strike >= spot]
        puts = [c for c in rows if c.right == "put" and c.strike <= spot]
        if not calls or not puts:
            return None
        call = min(calls, key=lambda contract: contract.strike - spot)
        put = min(puts, key=lambda contract: spot - contract.strike)
    if call.iv is None or put.iv is None:
        return None
    return put.iv - call.iv


def _quote_price(contract: OptionContract | None) -> float | None:
    if contract is None:
        return None
    if contract.bid is not None and contract.ask is not None:
        if contract.bid > 0 or contract.ask > 0:
            return (contract.bid + contract.ask) / 2.0
    if contract.last is not None and contract.last > 0:
        return contract.last
    return None


def _implied_move_pct(
    contracts: Iterable[OptionContract], spot: float | None
) -> float | None:
    if spot is None or spot <= 0:
        return None
    rows = list(contracts)
    strikes = sorted({contract.strike for contract in rows})
    if not strikes:
        return None
    strike = min(strikes, key=lambda value: abs(value - spot))
    call = next((c for c in rows if c.right == "call" and c.strike == strike), None)
    put = next((c for c in rows if c.right == "put" and c.strike == strike), None)
    call_price = _quote_price(call)
    put_price = _quote_price(put)
    if call_price is None or put_price is None:
        return None
    return (call_price + put_price) / spot * 100.0


def _oi_change_totals(
    contracts: Iterable[OptionContract],
) -> tuple[float | None, float | None]:
    rows = list(contracts)
    call_values = [
        float(c.oi_change)
        for c in rows
        if c.right == "call" and c.oi_change is not None
    ]
    put_values = [
        float(c.oi_change) for c in rows if c.right == "put" and c.oi_change is not None
    ]
    return (
        sum(call_values) if call_values else None,
        sum(put_values) if put_values else None,
    )


def _near_spot_writing(
    contracts: Iterable[OptionContract], spot: float | None
) -> tuple[float | None, float | None]:
    rows = list(contracts)
    has_changes = any(contract.oi_change is not None for contract in rows)
    if spot is None or not has_changes:
        return None, None
    lower = spot * (1.0 - _NEAR_SPOT_PCT)
    upper = spot * (1.0 + _NEAR_SPOT_PCT)
    call_writing = 0.0
    put_writing = 0.0
    for contract in rows:
        if not (lower <= contract.strike <= upper):
            continue
        if contract.oi_change is None or contract.oi_change <= 0:
            continue
        if (
            contract.last is None
            or contract.previous_close is None
            or contract.last >= contract.previous_close
        ):
            continue
        if contract.right == "call":
            call_writing += contract.oi_change
        else:
            put_writing += contract.oi_change
    return call_writing, put_writing


def _notional_totals(
    contracts: Iterable[OptionContract], spot: float | None
) -> tuple[float | None, float | None]:
    rows = [contract for contract in contracts if contract.lot_size is not None]
    if spot is None or not rows:
        return None, None
    oi = sum(
        contract.oi
        * (contract.lot_size if contract.lot_size is not None else 0.0)
        * spot
        for contract in rows
    )
    volume = sum(
        contract.volume
        * (contract.lot_size if contract.lot_size is not None else 0.0)
        * spot
        for contract in rows
    )
    return oi, volume


def classify_oi_changes(chain: OptionChain) -> list[dict[str, object]]:
    """Classify per-contract day-over-day OI and option-premium changes."""
    rows: list[dict[str, object]] = []
    for contract in chain.contracts:
        premium_change: float | None = None
        if contract.last is not None and contract.previous_close not in (None, 0):
            prior = contract.previous_close or 0.0
            premium_change = (contract.last / prior - 1.0) * 100.0
        if contract.oi_change is None:
            label = "unknown"
        elif contract.oi_change == 0:
            label = "unchanged"
        elif premium_change is None:
            label = "new_oi" if contract.oi_change > 0 else "unwinding"
        elif contract.oi_change > 0 and premium_change > 0:
            label = "long_buildup"
        elif contract.oi_change > 0 and premium_change <= 0:
            label = "short_buildup"
        elif premium_change > 0:
            label = "short_covering"
        else:
            label = "long_unwinding"
        rows.append(
            {
                "underlying": chain.underlying,
                "expiry": contract.expiry,
                "strike": contract.strike,
                "right": contract.right,
                "oi_change": contract.oi_change,
                "premium_change_pct": premium_change,
                "classification": label,
            }
        )
    return rows


def compute_chain_metrics(
    chain: OptionChain, *, concentration_n: int = 3
) -> ChainMetrics:
    """Derive panel-ready signals from a normalized chain.

    PCR and aggregate OI/volume cover all unexpired contracts in the snapshot.
    Strike-sensitive metrics (max pain, ATM IV, skew, implied move) use the
    nearest expiry only; term structure compares that expiry with the next one.
    """
    contracts = _unexpired(chain)
    front_expiry, next_expiry = _front_expiries(chain)
    front = _contracts_for_expiry(contracts, front_expiry)
    next_rows = _contracts_for_expiry(contracts, next_expiry)

    calls = [contract for contract in contracts if contract.right == "call"]
    puts = [contract for contract in contracts if contract.right == "put"]
    call_oi = sum(contract.oi for contract in calls)
    put_oi = sum(contract.oi for contract in puts)
    call_volume = sum(contract.volume for contract in calls)
    put_volume = sum(contract.volume for contract in puts)
    pcr = safe_ratio(put_oi or None, call_oi or None)
    pcr_volume = safe_ratio(put_volume or None, call_volume or None)
    call_put = safe_ratio(call_oi or None, put_oi or None)

    max_pain = _max_pain(front)
    max_pain_distance = (
        (max_pain - chain.spot) / chain.spot * 100.0
        if max_pain is not None and chain.spot is not None
        else None
    )
    supports, resistances = _oi_concentration(
        front, spot=chain.spot, top_n=max(int(concentration_n), 0)
    )

    valid_ivs = [
        float(contract.iv)
        for contract in contracts
        if _valid_iv(contract) and contract.iv is not None
    ]
    median_iv = float(median(valid_ivs)) if valid_ivs else None
    front_atm_iv = _atm_iv(front, chain.spot)
    next_atm_iv = _atm_iv(next_rows, chain.spot)
    term_slope = (
        next_atm_iv - front_atm_iv
        if next_atm_iv is not None and front_atm_iv is not None
        else None
    )
    call_oi_change, put_oi_change = _oi_change_totals(contracts)
    call_writing, put_writing = _near_spot_writing(front, chain.spot)
    notional_oi, notional_volume = _notional_totals(contracts, chain.spot)

    return ChainMetrics(
        underlying=chain.underlying,
        market=chain.market,
        as_of=chain.as_of,
        source=chain.source,
        spot=chain.spot,
        contract_count=len(contracts),
        expiry_count=len({contract.expiry for contract in contracts}),
        front_expiry=front_expiry,
        next_expiry=next_expiry,
        call_oi=call_oi,
        put_oi=put_oi,
        call_volume=call_volume,
        put_volume=put_volume,
        pcr=pcr,
        pcr_volume=pcr_volume,
        call_put_oi_ratio=call_put,
        max_pain_strike=max_pain,
        max_pain_distance_pct=max_pain_distance,
        support_strikes=supports,
        resistance_strikes=resistances,
        atm_iv=front_atm_iv,
        median_iv=median_iv,
        put_call_iv_skew=_iv_skew(front, chain.spot),
        term_structure_slope=term_slope,
        implied_move_pct=_implied_move_pct(front, chain.spot),
        call_oi_change=call_oi_change,
        put_oi_change=put_oi_change,
        oi_chg_ratio=safe_ratio(put_oi_change, call_oi_change),
        call_writing_near_spot=call_writing,
        put_writing_near_spot=put_writing,
        notional_oi=notional_oi,
        notional_volume=notional_volume,
    )


__all__ = ["classify_oi_changes", "compute_chain_metrics", "safe_ratio"]
