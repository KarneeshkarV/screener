"""Regressions for anchored vertical wings and NSE notional-OI units."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from screener.options.bt_models import LegFill, OptionsBacktestConfig
from screener.options.metrics import compute_chain_metrics
from screener.options.models import OptionChain, OptionContract
from screener.options.nse_bhavcopy import normalize_bhavcopy_options
from screener.options.position_backtest import _open_structure
from screener.options.structures import build_structure, select_strike

EXPIRY = date(2026, 7, 31)
AS_OF = date(2026, 7, 10)
SPOT = 100.0
STRIKES = (85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0)
CALL_DELTAS = {
    85.0: 0.90,
    90.0: 0.78,
    95.0: 0.63,
    100.0: 0.50,
    105.0: 0.37,
    110.0: 0.25,
    115.0: 0.15,
    120.0: 0.08,
}
PUT_DELTAS = {
    85.0: -0.10,
    90.0: -0.25,
    95.0: -0.38,
    100.0: -0.50,
    105.0: -0.62,
    110.0: -0.75,
    115.0: -0.85,
    120.0: -0.92,
}


def _contract(**overrides) -> OptionContract:
    values = {
        "symbol": "ABC260731C00100000",
        "underlying": "ABC",
        "expiry": EXPIRY,
        "strike": 100.0,
        "right": "call",
        "oi": 1000.0,
        "oi_change": 10.0,
        "volume": 20.0,
        "iv": 0.25,
        "last": 5.0,
        "previous_close": 4.5,
        "lot_size": 500.0,
        "as_of": datetime(2026, 7, 10, tzinfo=UTC),
        "source": "fixture",
    }
    values.update(overrides)
    return OptionContract(**values)


def _chain(*contracts: OptionContract, spot: float = SPOT) -> OptionChain:
    return OptionChain(
        underlying="ABC",
        market="india",
        spot=spot,
        as_of=AS_OF,
        source="fixture",
        contracts=tuple(contracts),
    )


def _ladder_chain() -> OptionChain:
    """Both rights at every strike, priced OTM so every leg has a quote."""
    contracts: list[OptionContract] = []
    for strike in STRIKES:
        call_last = max(SPOT - strike, 0.0) + 3.0
        put_last = max(strike - SPOT, 0.0) + 3.0
        contracts.append(
            _contract(
                symbol=f"C{strike:g}",
                strike=strike,
                right="call",
                last=call_last,
                delta=CALL_DELTAS[strike],
            )
        )
        contracts.append(
            _contract(
                symbol=f"P{strike:g}",
                strike=strike,
                right="put",
                last=put_last,
                delta=PUT_DELTAS[strike],
            )
        )
    return _chain(*contracts)


def _open(
    name: str,
    strike_rule: str,
    *,
    width_pct: float = 0.05,
    chain: OptionChain | None = None,
) -> tuple[list[LegFill] | None, list[str]]:
    spec = build_structure(name, strike_rule=strike_rule, width_pct=width_pct, lots=1)
    cfg = OptionsBacktestConfig(tickers=("ABC",), start=AS_OF, end=EXPIRY)
    warnings: list[str] = []
    fills = _open_structure(
        chain or _ladder_chain(), spec, as_of=AS_OF, cfg=cfg, warnings=warnings
    )
    return fills, warnings


@pytest.mark.parametrize(
    ("name", "strike_rule", "near", "far"),
    [
        # Puts: the long wing must sit strictly below the short near leg.
        ("bull_put_credit", "moneyness:-0.05", 95.0, 90.0),
        ("bull_put_credit", "moneyness:-0.10", 90.0, 85.0),
        ("bull_put_credit", "delta:0.25", 90.0, 85.0),
        ("bear_put_spread", "moneyness:-0.05", 95.0, 90.0),
        ("bear_put_spread", "delta:0.25", 90.0, 85.0),
        # Calls: the short wing must sit strictly above the long near leg.
        ("bull_call_spread", "moneyness:0.05", 105.0, 110.0),
        ("bull_call_spread", "moneyness:0.10", 110.0, 115.0),
        ("bull_call_spread", "delta:0.25", 110.0, 115.0),
        ("bear_call_credit", "moneyness:0.05", 105.0, 110.0),
        ("bear_call_credit", "moneyness:0.10", 110.0, 115.0),
        ("bear_call_credit", "delta:0.25", 110.0, 115.0),
    ],
)
def test_vertical_wing_is_anchored_beyond_the_near_strike(
    name: str, strike_rule: str, near: float, far: float
) -> None:
    fills, warnings = _open(name, strike_rule)
    assert fills is not None, warnings
    near_leg, far_leg = fills
    assert near_leg.strike == pytest.approx(near)
    assert far_leg.strike == pytest.approx(far)
    if near_leg.right == "call":
        assert far_leg.strike > near_leg.strike
    else:
        assert far_leg.strike < near_leg.strike
    assert far_leg.strike != near_leg.strike


def test_atm_vertical_wing_still_targets_width_off_the_near_strike() -> None:
    # ATM keeps behaving as before: near at spot, wing ~width_pct away.
    call_fills, call_warnings = _open("bull_call_spread", "atm")
    assert call_fills is not None, call_warnings
    assert [leg.strike for leg in call_fills] == [100.0, 105.0]

    put_fills, put_warnings = _open("bull_put_credit", "atm")
    assert put_fills is not None, put_warnings
    assert [leg.strike for leg in put_fills] == [100.0, 95.0]


def test_vertical_fails_cleanly_when_no_wing_strike_exists() -> None:
    # Near leg at the chain's highest strike: a call wing has nowhere to go.
    chain = _chain(
        *[c for c in _ladder_chain().contracts if c.strike in {95.0, 100.0, 105.0}]
    )
    fills, warnings = _open("bull_call_spread", "moneyness:0.05", chain=chain)
    assert fills is None
    assert any("anchor:0.05" in warning for warning in warnings)
    assert any("no strike" in warning for warning in warnings)

    # Same for a put wing below the chain's lowest strike.
    low_chain = _chain(
        *[c for c in _ladder_chain().contracts if c.strike in {90.0, 95.0, 100.0}]
    )
    fills, warnings = _open("bull_put_credit", "moneyness:-0.10", chain=low_chain)
    assert fills is None
    assert any("no strike" in warning for warning in warnings)


def test_select_strike_anchor_rule_requires_an_existing_wing() -> None:
    chain = _ladder_chain()
    assert (
        select_strike(chain, EXPIRY, "call", "anchor:0.05", anchor_strike=120.0) is None
    )
    assert (
        select_strike(chain, EXPIRY, "put", "anchor:-0.05", anchor_strike=85.0) is None
    )
    # No anchor strike, or a zero-width anchor, cannot form a spread.
    assert select_strike(chain, EXPIRY, "call", "anchor:0.05") is None
    assert select_strike(chain, EXPIRY, "call", "anchor:0", anchor_strike=100.0) is None
    picked = select_strike(chain, EXPIRY, "call", "anchor:0.05", anchor_strike=100.0)
    assert picked is not None and picked.strike == 105.0


def _udiff_frame() -> pd.DataFrame:
    """A minimal UDiff-shaped frame: OI in shares, volume in contracts."""
    return pd.DataFrame(
        [
            {
                "TradDt": "2026-07-10",
                "FinInstrmTp": "STO",
                "TckrSymb": "ABC",
                "XpryDt": "2026-08-28",
                "StrkPric": 100.0,
                "OptnTp": "CE",
                "FinInstrmNm": "ABC26AUG100CE",
                "ClsPric": 5.0,
                "LastPric": 5.0,
                "PrvsClsgPric": 4.5,
                "UndrlygPric": SPOT,
                "SttlmPric": 5.0,
                "OpnIntrst": 5000,
                "ChngInOpnIntrst": 500,
                "TtlTradgVol": 7,
                "NewBrdLotQty": 500,
            },
            {
                "TradDt": "2026-07-10",
                "FinInstrmTp": "STO",
                "TckrSymb": "ABC",
                "XpryDt": "2026-08-28",
                "StrkPric": 100.0,
                "OptnTp": "PE",
                "FinInstrmNm": "ABC26AUG100PE",
                "ClsPric": 4.5,
                "LastPric": 4.5,
                "PrvsClsgPric": 5.0,
                "UndrlygPric": SPOT,
                "SttlmPric": 4.5,
                "OpnIntrst": 2500,
                "ChngInOpnIntrst": -250,
                "TtlTradgVol": 3,
                "NewBrdLotQty": 500,
            },
        ]
    )


def test_bhavcopy_notional_oi_does_not_multiply_lot_size_twice() -> None:
    chain = normalize_bhavcopy_options(_udiff_frame(), as_of=AS_OF, derive_iv=False)[
        "ABC"
    ]
    # Raw OI stays in underlying units for PCR / OI-change math.
    assert [contract.oi for contract in chain.contracts] == [5000.0, 2500.0]
    assert all(contract.oi_unit == "units" for contract in chain.contracts)
    assert all(contract.lot_size == 500 for contract in chain.contracts)

    metrics = compute_chain_metrics(chain)
    # OI is already shares: 7500 shares * 100 spot.
    assert metrics.notional_oi == pytest.approx(7500 * SPOT)
    # Volume is contracts: 10 contracts * 500 lot * 100 spot.
    assert metrics.notional_volume == pytest.approx(10 * 500 * SPOT)


def test_notional_oi_still_scales_contracts_unit_sources_by_lot_size() -> None:
    contract = _contract(oi=10.0, oi_unit="contracts", volume=5.0, lot_size=500.0)
    assert OptionContract.model_fields["oi_unit"].default == "contracts"
    metrics = compute_chain_metrics(_chain(contract))
    assert metrics.notional_oi == pytest.approx(10 * 500 * SPOT)
    assert metrics.notional_volume == pytest.approx(5 * 500 * SPOT)
