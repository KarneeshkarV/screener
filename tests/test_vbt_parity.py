from __future__ import annotations

import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.pine import parse
from screener.backtester.vbt_adapter import run_vbt

from tests.conftest import make_bars
from tests.backtester_synthetic import (
    STRATEGY_FIXTURES,
    fixture_config,
    run_core_portfolio_path,
    synthetic_ohlcv_panel,
)


def _risk_cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=make_bars(n=8).index[0].date(),
        hold=5,
        top=1,
        entry_expr="entry_flag > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        allow_reentry=False,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


@pytest.mark.parametrize(
    ("name", "cfg_overrides", "spikes", "expected_reason", "expected_exit_price"),
    [
        ("hold_cap", {"hold": 2}, {}, "time", None),
        (
            "stop_loss",
            {"stop_loss": 0.05},
            {
                1: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
                2: {"open": 100.0, "high": 100.5, "low": 94.0, "close": 96.0},
            },
            "stop",
            95.0,
        ),
        (
            "take_profit",
            {"take_profit": 0.10},
            {
                1: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
                2: {"open": 100.0, "high": 111.0, "low": 99.5, "close": 108.0},
            },
            "target",
            110.0,
        ),
        (
            "trailing_stop",
            {"trailing_stop": 0.10},
            {
                1: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
                2: {"open": 100.0, "high": 120.0, "low": 99.5, "close": 118.0},
                3: {"open": 118.0, "high": 119.0, "low": 107.0, "close": 109.0},
            },
            "trail",
            108.0,
        ),
    ],
    ids=["hold_cap", "stop_loss", "take_profit", "trailing_stop"],
)
def test_vbt_fast_path_honors_core_risk_controls(
    name,
    cfg_overrides,
    spikes,
    expected_reason,
    expected_exit_price,
):
    bars = make_bars(n=8, spikes=spikes)
    bars["entry_flag"] = 0.0
    bars.iloc[0, bars.columns.get_loc("entry_flag")] = 1.0
    cfg = _risk_cfg(**cfg_overrides)

    result = run_vbt(cfg, {"AAA": bars}, parse(cfg.entry_expr))

    assert name
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == expected_reason
    if expected_exit_price is not None:
        assert trade.exit_price == pytest.approx(expected_exit_price)


@pytest.mark.parametrize("fixture", STRATEGY_FIXTURES, ids=lambda item: item.name)
def test_vbt_matches_core_path_on_overlapping_features(fixture):
    panel = synthetic_ohlcv_panel()
    cfg = fixture_config(fixture)
    core_result = run_core_portfolio_path(cfg, panel)
    vbt_result = run_vbt(
        cfg,
        panel,
        parse(cfg.entry_expr),
        parse(cfg.exit_expr) if cfg.exit_expr else None,
    )

    core_trades = sorted(core_result.trades, key=lambda trade: trade.ticker)
    vbt_trades = sorted(vbt_result.trades, key=lambda trade: trade.ticker)
    assert len(vbt_trades) == len(core_trades)

    for core_trade, vbt_trade in zip(core_trades, vbt_trades):
        assert vbt_trade.ticker == core_trade.ticker
        assert vbt_trade.entry_date == core_trade.entry_date
        assert vbt_trade.exit_date == core_trade.exit_date
        assert vbt_trade.exit_reason == core_trade.exit_reason
        assert vbt_trade.entry_price == pytest.approx(core_trade.entry_price, abs=1e-6)
        assert vbt_trade.exit_price == pytest.approx(core_trade.exit_price, abs=1e-6)

    core_terminal = float(core_result.equity_curve.iloc[-1])
    vbt_terminal = float(vbt_result.equity_curve.iloc[-1])
    assert abs(vbt_terminal / core_terminal - 1.0) <= 0.0005
    for metric in ("sharpe", "cagr", "max_drawdown"):
        assert vbt_result.metrics[metric] == pytest.approx(
            core_result.metrics[metric],
            abs=1e-4,
        )


# Skip catalog for explicit vectorbt coverage gaps:
# - gap_fills=True stop/target resolution is path-dependent in
#   screener/backtester/core.py::_resolve_stop_fill and _resolve_target_fill.
# - partial exits mutate tranche state in
#   screener/backtester/core.py::_fire_partial_exits_at_bar.
# - HalfSpread/VolumeImpact/Composite slippage require liquidity inputs in
#   screener/backtester/core.py::_apply_slip.
@pytest.mark.parametrize(
    "feature",
    [
        pytest.param(
            "gap_fills",
            marks=pytest.mark.skip(
                reason="gap_fills=True stop/target parity is path-dependent in core.py"
            ),
        ),
        pytest.param(
            "partial_exits",
            marks=pytest.mark.skip(
                reason="partial exits are tranche-stateful in core.py"
            ),
        ),
        pytest.param(
            "non_fixed_slippage",
            marks=pytest.mark.skip(
                reason="non-FixedBps slippage depends on liquidity inputs in core.py"
            ),
        ),
    ],
)
def test_vbt_parity_gap_catalog(feature):
    raise AssertionError(f"coverage gap should be skipped explicitly: {feature}")
