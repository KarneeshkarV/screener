"""Expiry/strike selection and structure registry."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from screener.options.models import OptionChain, OptionContract
from screener.options.structures import (
    available_structures,
    build_structure,
    select_expiry,
    select_strike,
)


def _contract(**overrides) -> OptionContract:
    values = {
        "symbol": "ABC260731C00100000",
        "underlying": "ABC",
        "expiry": date(2026, 7, 31),
        "strike": 100.0,
        "right": "call",
        "oi": 100.0,
        "oi_change": 10.0,
        "volume": 20.0,
        "iv": 0.25,
        "bid": 4.0,
        "ask": 6.0,
        "last": 5.0,
        "previous_close": 4.0,
        "lot_size": 10.0,
        "as_of": datetime(2026, 7, 10, tzinfo=UTC),
        "source": "fixture",
    }
    values.update(overrides)
    return OptionContract(**values)


def _chain(*contracts: OptionContract, spot: float = 100.0) -> OptionChain:
    return OptionChain(
        underlying="abc",
        market="us",
        spot=spot,
        as_of=date(2026, 7, 10),
        source="fixture",
        contracts=tuple(contracts),
    )


def test_select_expiry_front_next_dte():
    front = date(2026, 7, 17)
    nxt = date(2026, 7, 24)
    far = date(2026, 8, 28)
    chain = _chain(
        _contract(expiry=front, strike=100),
        _contract(expiry=nxt, strike=100, symbol="B"),
        _contract(expiry=far, strike=100, symbol="C"),
    )
    as_of = date(2026, 7, 10)
    assert select_expiry(chain, "front", as_of) == front
    assert select_expiry(chain, "next", as_of) == nxt
    assert select_expiry(chain, "dte:20", as_of) == far
    # DTE ≥ 1: same-day expiry excluded.
    assert select_expiry(chain, "front", front) == nxt


def test_select_strike_atm_and_moneyness():
    chain = _chain(
        _contract(strike=95, last=8.0, bid=7.5, ask=8.5),
        _contract(strike=100, last=5.0, bid=4.5, ask=5.5, symbol="ATM"),
        _contract(strike=105, last=3.0, bid=2.5, ask=3.5, symbol="OTM"),
        spot=100.0,
    )
    atm = select_strike(chain, date(2026, 7, 31), "call", "atm")
    assert atm is not None and atm.strike == 100.0
    otm = select_strike(chain, date(2026, 7, 31), "call", "moneyness:0.05")
    assert otm is not None and otm.strike == 105.0
    itm = select_strike(chain, date(2026, 7, 31), "call", "moneyness:-0.05")
    assert itm is not None and itm.strike == 95.0


def test_select_strike_delta_path_and_fallback():
    # Priced chain around spot 100 with enough spread for IV solve.
    contracts = []
    for strike, last in ((90, 12.0), (100, 5.5), (110, 2.0), (120, 0.8)):
        contracts.append(
            _contract(
                strike=strike,
                last=last,
                bid=last - 0.2,
                ask=last + 0.2,
                iv=None,
                symbol=f"C{strike}",
                expiry=date(2026, 8, 28),
            )
        )
    chain = _chain(*contracts, spot=100.0)
    picked = select_strike(
        chain,
        date(2026, 8, 28),
        "call",
        "delta:0.50",
        as_of=date(2026, 7, 10),
    )
    assert picked is not None
    assert picked.strike in {90.0, 100.0, 110.0, 120.0}

    # Unsolvable prices (zero last, no bid/ask) → moneyness fallback.
    bare = _chain(
        _contract(strike=100, last=None, bid=None, ask=None, iv=None, symbol="X"),
        _contract(
            strike=105,
            last=None,
            bid=None,
            ask=None,
            iv=None,
            symbol="Y",
        ),
        spot=100.0,
    )
    # No quote prices → select_strike returns None (no candidates).
    assert select_strike(bare, date(2026, 7, 31), "call", "delta:0.25") is None


def test_structure_leg_signs():
    expected = {
        "long_call": (("call", 1),),
        "long_put": (("put", 1),),
        "short_call": (("call", -1),),
        "short_put": (("put", -1),),
        "straddle": (("call", 1), ("put", 1)),
        "short_straddle": (("call", -1), ("put", -1)),
        "strangle": (("call", 1), ("put", 1)),
        "bull_call_spread": (("call", 1), ("call", -1)),
        "bear_put_spread": (("put", 1), ("put", -1)),
        "bull_put_credit": (("put", -1), ("put", 1)),
        "bear_call_credit": (("call", -1), ("call", 1)),
    }
    for name, legs in expected.items():
        spec = build_structure(name, width_pct=0.05, lots=2)
        assert spec.name == name
        assert len(spec.legs) == len(legs)
        for leg, (right, side) in zip(spec.legs, legs, strict=True):
            assert leg.right == right
            assert leg.side == side
            assert leg.lots == 2


def test_available_structures_covers_registry():
    names = available_structures()
    assert "long_call" in names
    assert "straddle" in names
    assert "bull_call_spread" in names
    with pytest.raises(KeyError, match="unknown structure"):
        build_structure("not_a_real_structure")
