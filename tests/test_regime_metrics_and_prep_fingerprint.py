"""Regression tests for two backtester accounting/reuse-key bugs.

1. ``compute_regime_metrics`` must count positions, not ledger rows, and must
   report total return (dividends included) like ``hit_rate`` does.  A
   ``--partial-exit`` position emits one ``Trade`` per scale-out tranche, and
   ``Trade.return_pct`` is capital-only, so reading it directly double-counted
   tranches and called dividend winners losers.
2. ``_preparation_fingerprint`` must not serialize book-only fields.  A custom
   protocol-conforming ``SlippageModel`` is not a Pydantic model, so dumping
   ``slippage_model`` raised ``PydanticSerializationError``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from screener.backtester.metrics import compute_regime_metrics
from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.rolling_simulation import _preparation_fingerprint

_SIGNAL_DAY = pd.Timestamp("2023-12-28")
_ENTRY_DAY = pd.Timestamp("2023-12-29")


def _bull_benchmark() -> pd.Series:
    """A curve that ends deep in the ``bull`` regime (200-bar warmup)."""
    return pd.Series(
        np.r_[np.linspace(100.0, 200.0, 260), [201.0]],
        index=pd.bdate_range("2023-01-02", periods=261),
    )


def _trade(**overrides: Any) -> Trade:
    fields: dict[str, Any] = dict(
        ticker="AAA",
        rank=1,
        signal_date=_SIGNAL_DAY,
        entry_date=_ENTRY_DAY,
        exit_date=pd.Timestamp("2024-01-05"),
        entry_price=100.0,
        exit_price=101.0,
        exit_reason="time",
        shares=10.0,
        entry_cost=1000.0,
        exit_value=1010.0,
        pnl=10.0,
        return_pct=0.01,
        dividend_income=0.0,
        open_seq=0,
    )
    fields.update(overrides)
    return Trade(**fields)


def test_regime_metrics_collapse_partial_exit_tranches_into_one_position():
    # Two scale-out tranches of one lot: (ticker, open_seq, entry_date).
    first = _trade(open_seq=1, pnl=100.0, entry_cost=1000.0, return_pct=0.10)
    runner = _trade(open_seq=1, pnl=-20.0, entry_cost=500.0, return_pct=-0.04)

    metrics = compute_regime_metrics(_bull_benchmark(), [first, runner])

    assert metrics["regime_bull_trades"] == 1
    # Position total return: (100 - 20) / (1000 + 500), not a mean of tranche
    # ``return_pct`` values and not two separate trades.
    assert metrics["regime_bull_avg_return"] == pytest.approx(80.0 / 1500.0)
    assert metrics["regime_bull_win_rate"] == 1.0


def test_regime_metrics_use_dividend_inclusive_total_return_for_win_rate():
    # Capital return is negative, but the dividend makes the position a winner,
    # exactly as ``positive_pnl_rate`` (hit_rate) scores it.
    dividend_winner = _trade(pnl=20.0, entry_cost=1000.0, return_pct=-0.02)

    metrics = compute_regime_metrics(_bull_benchmark(), [dividend_winner])

    assert metrics["regime_bull_trades"] == 1
    assert metrics["regime_bull_win_rate"] == 1.0
    assert metrics["regime_bull_avg_return"] == pytest.approx(0.02)


class _CustomSlippage:
    """Protocol-conforming slippage model that is not a Pydantic model."""

    def __init__(self, bps: float) -> None:
        self.bps = bps

    def adverse_fraction(
        self,
        side: str,
        shares: float,
        adv: float,
        sigma_daily: float,
        half_spread: float = 0.0,
    ) -> float:
        return self.bps / 10_000.0


def _cfg(**overrides: Any) -> BacktestConfig:
    defaults: dict[str, Any] = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=10,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def test_preparation_fingerprint_ignores_custom_slippage_models():
    baseline = _preparation_fingerprint(_cfg())

    first = _preparation_fingerprint(_cfg(slippage_model=_CustomSlippage(5.0)))
    second = _preparation_fingerprint(_cfg(slippage_model=_CustomSlippage(40.0)))

    assert "slippage_model" not in first
    assert first == second == baseline
