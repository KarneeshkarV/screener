"""Expiry/strike selection and the option-structure registry.

Strike selection may use Black-Scholes greeks for delta targeting; P&L always
uses observed chain prices, never model prices.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any

from screener.options.bt_models import LegSpec, StructureSpec
from screener.options.greeks import black_scholes_greeks, implied_volatility
from screener.options.metrics import _quote_price
from screener.options.models import OptionChain, OptionContract, OptionRight

LOG = logging.getLogger(__name__)

_DAYS_PER_YEAR = 365.0
_RISK_FREE = 0.0
StructureBuilder = Callable[..., StructureSpec]


def select_expiry(
    chain: OptionChain,
    rule: str,
    as_of: date,
) -> date | None:
    """Pick an expiry from ``chain`` under ``rule`` with DTE ≥ 1.

    Rules: ``front``, ``next``, or ``dte:<n>`` (nearest expiry with DTE ≥ n).
    """
    candidates = sorted(e for e in chain.expiries if (e - as_of).days >= 1)
    if not candidates:
        return None
    normalized = rule.strip().lower()
    if normalized == "front":
        return candidates[0]
    if normalized == "next":
        return candidates[1] if len(candidates) > 1 else candidates[0]
    if normalized.startswith("dte:"):
        try:
            min_dte = int(normalized.split(":", 1)[1])
        except ValueError:
            return None
        eligible = [e for e in candidates if (e - as_of).days >= min_dte]
        if not eligible:
            return candidates[-1]
        return min(eligible, key=lambda e: abs((e - as_of).days - min_dte))
    raise ValueError(f"unknown expiry rule {rule!r}")


def _contracts_for(
    chain: OptionChain, expiry: date, right: OptionRight
) -> list[OptionContract]:
    return [
        c
        for c in chain.contracts
        if c.expiry == expiry and c.right == right and _quote_price(c) is not None
    ]


def _nearest_strike(
    contracts: list[OptionContract], target: float
) -> OptionContract | None:
    if not contracts:
        return None
    return min(contracts, key=lambda c: (abs(c.strike - target), c.strike))


def _bs_delta(
    contract: OptionContract,
    *,
    spot: float,
    as_of: date,
) -> float | None:
    price = _quote_price(contract)
    if price is None or spot <= 0:
        return None
    dte = (contract.expiry - as_of).days
    if dte < 1:
        return None
    t = dte / _DAYS_PER_YEAR
    iv = contract.iv
    if iv is None or iv <= 0:
        iv = implied_volatility(price, spot, contract.strike, t, _RISK_FREE, contract.right)
    if iv is None or iv <= 0:
        return None
    greeks = black_scholes_greeks(
        spot, contract.strike, t, _RISK_FREE, iv, contract.right
    )
    if greeks is None:
        return None
    return float(greeks["delta"])


def select_strike(
    chain: OptionChain,
    expiry: date,
    right: OptionRight,
    rule: str,
    *,
    as_of: date | None = None,
) -> OptionContract | None:
    """Select a contract by strike rule.

    Rules:
    * ``atm`` — nearest strike to spot
    * ``moneyness:<±pct>`` — nearest to ``spot * (1 + pct)`` (pct as decimal, e.g. 0.05)
    * ``delta:<abs>`` — nearest |delta| via IV solve + BS delta; falls back to
      moneyness with a warning when no IV solves
    """
    contracts = _contracts_for(chain, expiry, right)
    if not contracts:
        return None
    spot = chain.spot
    if spot is None or spot <= 0:
        # Fall back to median strike as a crude ATM proxy.
        strikes = sorted(c.strike for c in contracts)
        spot = strikes[len(strikes) // 2]

    normalized = rule.strip().lower()
    as_of_day = as_of or chain.as_of.date()

    if normalized == "atm":
        return _nearest_strike(contracts, float(spot))

    if normalized.startswith("moneyness:"):
        try:
            pct = float(normalized.split(":", 1)[1])
        except ValueError:
            return None
        return _nearest_strike(contracts, float(spot) * (1.0 + pct))

    if normalized.startswith("delta:"):
        try:
            target = abs(float(normalized.split(":", 1)[1]))
        except ValueError:
            return None
        scored: list[tuple[float, OptionContract]] = []
        for contract in contracts:
            delta = _bs_delta(contract, spot=float(spot), as_of=as_of_day)
            if delta is None:
                continue
            scored.append((abs(abs(delta) - target), contract))
        if scored:
            scored.sort(key=lambda item: (item[0], item[1].strike))
            return scored[0][1]
        # Fall back to OTM moneyness roughly matching the target delta band.
        fallback_pct = 0.05 if target >= 0.4 else 0.10
        if right == "put":
            fallback_pct = -fallback_pct
        LOG.warning(
            "delta strike selection failed for %s %s %s; falling back to moneyness:%g",
            chain.underlying,
            expiry,
            right,
            fallback_pct,
        )
        return _nearest_strike(contracts, float(spot) * (1.0 + fallback_pct))

    raise ValueError(f"unknown strike rule {rule!r}")


def build_structure(
    name: str,
    *,
    strike_rule: str = "atm",
    expiry_rule: str = "front",
    width_pct: float = 0.05,
    lots: int = 1,
) -> StructureSpec:
    """Resolve a named structure into concrete leg specs."""
    key = name.strip().lower()
    builder = STRUCTURE_BUILDERS.get(key)
    if builder is None:
        raise KeyError(
            f"unknown structure {name!r}; expected one of "
            f"{', '.join(sorted(STRUCTURE_BUILDERS))}"
        )
    return builder(
        strike_rule=strike_rule,
        expiry_rule=expiry_rule,
        width_pct=width_pct,
        lots=lots,
    )


def _single(
    name: str,
    right: OptionRight,
    side: int,
    *,
    strike_rule: str,
    expiry_rule: str,
    width_pct: float,
    lots: int,
) -> StructureSpec:
    del width_pct
    return StructureSpec(
        name=name,
        legs=(
            LegSpec(
                right=right,
                side=side,
                lots=lots,
                strike_rule=strike_rule,
                expiry_rule=expiry_rule,
            ),
        ),
    )


def _straddle(
    name: str,
    side: int,
    *,
    strike_rule: str,
    expiry_rule: str,
    width_pct: float,
    lots: int,
) -> StructureSpec:
    del width_pct
    return StructureSpec(
        name=name,
        legs=(
            LegSpec(
                right="call",
                side=side,
                lots=lots,
                strike_rule=strike_rule,
                expiry_rule=expiry_rule,
            ),
            LegSpec(
                right="put",
                side=side,
                lots=lots,
                strike_rule=strike_rule,
                expiry_rule=expiry_rule,
            ),
        ),
    )


def _strangle(
    *,
    strike_rule: str,
    expiry_rule: str,
    width_pct: float,
    lots: int,
) -> StructureSpec:
    del strike_rule  # wings are defined by width around spot
    w = abs(width_pct)
    return StructureSpec(
        name="strangle",
        legs=(
            LegSpec(
                right="call",
                side=1,
                lots=lots,
                strike_rule=f"moneyness:{w:g}",
                expiry_rule=expiry_rule,
            ),
            LegSpec(
                right="put",
                side=1,
                lots=lots,
                strike_rule=f"moneyness:{-w:g}",
                expiry_rule=expiry_rule,
            ),
        ),
    )


def _vertical(
    name: str,
    right: OptionRight,
    long_side_near: bool,
    *,
    strike_rule: str,
    expiry_rule: str,
    width_pct: float,
    lots: int,
) -> StructureSpec:
    """Build a two-leg vertical.

    ``long_side_near`` True → long the near (strike_rule) leg and short the far
    wing; False → short near / long far (credit spreads).
    """
    w = abs(width_pct)
    if right == "call":
        far_rule = f"moneyness:{w:g}"
    else:
        far_rule = f"moneyness:{-w:g}"
    near_side = 1 if long_side_near else -1
    far_side = -near_side
    return StructureSpec(
        name=name,
        legs=(
            LegSpec(
                right=right,
                side=near_side,
                lots=lots,
                strike_rule=strike_rule,
                expiry_rule=expiry_rule,
            ),
            LegSpec(
                right=right,
                side=far_side,
                lots=lots,
                strike_rule=far_rule,
                expiry_rule=expiry_rule,
            ),
        ),
    )


def _kw(fn: Callable[..., StructureSpec], **fixed: Any) -> StructureBuilder:
    def builder(**kwargs: Any) -> StructureSpec:
        return fn(**fixed, **kwargs)

    return builder


STRUCTURE_BUILDERS: dict[str, StructureBuilder] = {
    "long_call": _kw(_single, name="long_call", right="call", side=1),
    "long_put": _kw(_single, name="long_put", right="put", side=1),
    "short_call": _kw(_single, name="short_call", right="call", side=-1),
    "short_put": _kw(_single, name="short_put", right="put", side=-1),
    "straddle": _kw(_straddle, name="straddle", side=1),
    "short_straddle": _kw(_straddle, name="short_straddle", side=-1),
    "strangle": _strangle,
    "bull_call_spread": _kw(
        _vertical, name="bull_call_spread", right="call", long_side_near=True
    ),
    "bear_put_spread": _kw(
        _vertical, name="bear_put_spread", right="put", long_side_near=True
    ),
    "bull_put_credit": _kw(
        _vertical, name="bull_put_credit", right="put", long_side_near=False
    ),
    "bear_call_credit": _kw(
        _vertical, name="bear_call_credit", right="call", long_side_near=False
    ),
}


def available_structures() -> list[str]:
    return sorted(STRUCTURE_BUILDERS)


__all__ = [
    "STRUCTURE_BUILDERS",
    "available_structures",
    "build_structure",
    "select_expiry",
    "select_strike",
]
