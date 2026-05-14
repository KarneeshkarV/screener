from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.engine import simulate_ticker
from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.pine import evaluate, parse

from tests.backtester_synthetic import (
    STRATEGY_FIXTURES,
    first_signal_index,
    fixture_config,
    run_core_portfolio_path,
    synthetic_ohlcv_panel,
)


def _manual_trade(
    *,
    ticker: str,
    rank: int,
    cfg: BacktestConfig,
    bars: pd.DataFrame,
) -> Trade:
    assert cfg.stop_loss is None
    assert cfg.take_profit is None
    assert cfg.trailing_stop is None
    signal_idx = first_signal_index(bars, cfg.entry_expr)
    entry_idx = signal_idx + 1
    if entry_idx >= len(bars):
        raise AssertionError("fixture must contain a post-signal entry bar")

    slip = cfg.slippage_bps / 10_000.0
    commission = cfg.commission_bps / 10_000.0
    entry_price = float(bars.iloc[entry_idx]["open"]) * (1.0 + slip)
    exit_signal = (
        evaluate(parse(cfg.exit_expr), bars).fillna(False).astype(bool)
        if cfg.exit_expr
        else pd.Series(False, index=bars.index)
    )

    exit_idx = len(bars) - 1
    exit_reason = "eod"
    for i in range(entry_idx + 1, len(bars)):
        if bool(exit_signal.iloc[i]):
            exit_idx = i
            exit_reason = "exit_expr"
            break
        if i >= entry_idx + cfg.hold:
            exit_idx = i
            exit_reason = "time"
            break
    exit_price = float(bars.iloc[exit_idx]["close"]) * (1.0 - slip)

    slot_capital = cfg.initial_capital / cfg.top
    gross_per_share = entry_price * (1.0 + commission)
    shares = slot_capital / gross_per_share
    entry_cost = shares * entry_price * (1.0 + commission)
    exit_value = shares * exit_price * (1.0 - commission)
    pnl = exit_value - entry_cost
    return Trade(
        ticker=ticker,
        rank=rank,
        signal_date=bars.index[signal_idx].date(),
        entry_date=bars.index[entry_idx].date(),
        entry_price=entry_price,
        exit_date=bars.index[exit_idx].date(),
        exit_price=exit_price,
        exit_reason=exit_reason,
        shares=shares,
        entry_cost=entry_cost,
        exit_value=exit_value,
        pnl=pnl,
        return_pct=pnl / entry_cost,
    )


def _manual_equity_curve(
    calendar: pd.DatetimeIndex,
    trades: list[Trade],
    panel: dict[str, pd.DataFrame],
    initial_capital: float,
) -> pd.Series:
    events: list[tuple[pd.Timestamp, int, int, Trade]] = []
    for seq, trade in enumerate(trades):
        events.append((pd.Timestamp(trade.entry_date), 1, seq, trade))
        events.append((pd.Timestamp(trade.exit_date), 0, seq, trade))
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    cash = float(initial_capital)
    open_positions: dict[int, Trade] = {}
    equity = pd.Series(0.0, index=calendar, dtype=float)
    event_idx = 0
    for day in calendar:
        while event_idx < len(events) and events[event_idx][0] <= day:
            _event_day, kind, seq, trade = events[event_idx]
            if kind == 1:
                cash -= trade.entry_cost
                open_positions[seq] = trade
            else:
                open_positions.pop(seq, None)
                cash += trade.exit_value
            event_idx += 1

        mark_to_market = 0.0
        for trade in open_positions.values():
            frame = panel[trade.ticker]
            price = float(frame.loc[day, "close"])
            mark_to_market += trade.shares * price
        equity.loc[day] = cash + mark_to_market
    return equity


@pytest.mark.parametrize("fixture", STRATEGY_FIXTURES, ids=lambda item: item.name)
def test_core_simulate_ticker_matches_manual_oracle(fixture):
    panel = synthetic_ohlcv_panel()
    cfg = fixture_config(fixture)
    exit_ast = parse(cfg.exit_expr) if cfg.exit_expr else None
    expected_trades = [
        _manual_trade(ticker=ticker, rank=rank, cfg=cfg, bars=bars)
        for rank, (ticker, bars) in enumerate(panel.items(), 1)
    ]

    for expected in expected_trades:
        bars = panel[expected.ticker]
        signal_idx = first_signal_index(bars, cfg.entry_expr)
        outcome = simulate_ticker(bars, signal_idx, cfg, exit_ast=exit_ast)
        assert outcome.trade is not None
        assert outcome.trade.entry_date == expected.entry_date
        assert outcome.trade.exit_date == expected.exit_date
        assert outcome.trade.exit_reason == expected.exit_reason
        assert outcome.trade.entry_price == pytest.approx(expected.entry_price)
        assert outcome.trade.exit_price == pytest.approx(expected.exit_price)

    core_result = run_core_portfolio_path(cfg, panel)
    core_trades = sorted(core_result.trades, key=lambda trade: trade.ticker)
    expected_sorted = sorted(expected_trades, key=lambda trade: trade.ticker)
    for core_trade, expected in zip(core_trades, expected_sorted):
        assert core_trade.entry_cost == pytest.approx(expected.entry_cost)
        assert core_trade.exit_value == pytest.approx(expected.exit_value)
        assert core_trade.shares == pytest.approx(expected.shares)
        assert core_trade.pnl == pytest.approx(expected.pnl)
        assert core_trade.return_pct == pytest.approx(expected.return_pct)

    calendar = pd.DatetimeIndex(
        sorted({day for bars in panel.values() for day in bars.index})
    )
    expected_equity = _manual_equity_curve(
        calendar,
        expected_trades,
        panel,
        cfg.initial_capital,
    )
    assert np.allclose(
        core_result.equity_curve.to_numpy(dtype=float),
        expected_equity.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-9,
    )
