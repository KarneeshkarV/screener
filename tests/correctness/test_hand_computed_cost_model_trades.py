"""Hand-computed end-to-end statutory-cost trade oracles.

These use the production FillModel, Portfolio, DayLoop, and force-close path.
Expected values below are formulas from costs.py, not values copied from a run.
"""

from __future__ import annotations

from datetime import date

import pytest

from screener.backtester.models import BacktestConfig
from tests.backtest_helpers import simulate_single_ticker
from tests.correctness.fixtures.explicit_bars import bars_s1_buy_and_hold

TOL = 1e-6


def _cfg(**overrides) -> BacktestConfig:
    values = dict(
        market="us",
        as_of=date(2024, 1, 5),
        hold=100,
        top=1,
        entry_expr="close > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=10_000.0,
        benchmark="SPY",
        tickers=("TEST",),
        gap_fills=True,
    )
    values.update(overrides)
    return BacktestConfig(**values)


# signal_idx=3, entry reference/fill=100, eod exit fill=115.
# India buy rate = STT .001 + stamp .00015 + exchange .0000297 +
#                  SEBI .000001 + GST(.18 * (.0000297 + .000001)) + IPFT .000001
#                = .001187226.
# shares = 10_000 / (100 * 1.001187226) = 99.8814181834.
# Buy fees total 11.8581716666, so entry cost is exactly 10_000.
# Sell rate = .001037226 (the same stack without stamp duty).
# Sell fees = shares * 115 * .001037226 = 11.9139544435.
# Residual cash after the round trip = shares * 115 - sell fees
#                                    = 11_474.4491366493.


def test_india_cost_model_hand_computed_trade_cash_shares_and_fees():
    out = simulate_single_ticker(bars_s1_buy_and_hold(), 3, _cfg(cost_model="india"))
    trade = out.trade
    assert trade is not None

    buy_rate = (
        0.001
        + 0.00015
        + 0.0000297
        + 0.000001
        + 0.18 * (0.0000297 + 0.000001)
        + 0.000001
    )
    sell_rate = buy_rate - 0.00015
    shares = 10_000.0 / (100.0 * (1.0 + buy_rate))
    buy_notional = shares * 100.0
    sell_notional = shares * 115.0

    assert trade.shares == pytest.approx(shares, abs=TOL)
    assert trade.entry_cost == pytest.approx(10_000.0, abs=TOL)
    assert out.fees_paid["stt"] == pytest.approx(
        buy_notional * 0.001 + sell_notional * 0.001, abs=TOL
    )
    assert out.fees_paid["stamp_duty"] == pytest.approx(buy_notional * 0.00015, abs=TOL)
    assert out.fees_paid["exchange_txn"] == pytest.approx(
        (buy_notional + sell_notional) * 0.0000297, abs=TOL
    )
    assert trade.exit_value == pytest.approx(sell_notional * (1.0 - sell_rate), abs=TOL)
    assert out.cash == pytest.approx(11_474.4491366493, abs=TOL)


# Vested buy brokerage is 0.25% below its $35 cap.
# shares = 10_000 / (100 * 1.0025) = 99.7506234414.
# Buy brokerage = shares * 100 * .0025 = 24.9376558603.
# Sell proceeds = shares * 115 = 11_471.3216957606.
# Sell fees are brokerage .25%, SEC .00206%, and TAF $0.000195/share.
# Residual cash = 11_471.3216957606 - 28.6783042394 - 0.2363092269
#                 - 0.0194513716 = 11_442.3876309227.


def test_us_vested_cost_model_hand_computed_trade_cash_shares_and_fees():
    out = simulate_single_ticker(
        bars_s1_buy_and_hold(), 3, _cfg(cost_model="us_vested")
    )
    trade = out.trade
    assert trade is not None

    shares = 10_000.0 / (100.0 * 1.0025)
    buy_notional = shares * 100.0
    sell_notional = shares * 115.0
    expected_brokerage = buy_notional * 0.0025 + sell_notional * 0.0025
    expected_sec = sell_notional * 0.0000206
    expected_taf = shares * 0.000195

    assert trade.shares == pytest.approx(shares, abs=TOL)
    assert trade.entry_cost == pytest.approx(10_000.0, abs=TOL)
    assert out.fees_paid["brokerage"] == pytest.approx(expected_brokerage, abs=TOL)
    assert out.fees_paid["sec_fee"] == pytest.approx(expected_sec, abs=TOL)
    assert out.fees_paid["taf"] == pytest.approx(expected_taf, abs=TOL)
    assert trade.exit_value == pytest.approx(
        sell_notional
        - expected_brokerage
        + buy_notional * 0.0025
        - expected_sec
        - expected_taf,
        abs=TOL,
    )
    assert out.cash == pytest.approx(11_442.3876309227, abs=TOL)
