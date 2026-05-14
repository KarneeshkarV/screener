from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from screener.backtester.engine import simulate_ticker
from screener.backtester.metrics import compute_metrics
from screener.backtester.models import BacktestConfig, BacktestResult
from screener.backtester.pine import evaluate, parse
from screener.backtester.portfolio import Portfolio, build_equity_curve


@dataclass(frozen=True)
class StrategyFixture:
    name: str
    entry_expr: str
    exit_expr: str | None
    hold: int


STRATEGY_FIXTURES = (
    StrategyFixture("buy_and_hold", "entry_day0 > 0", None, 9999),
    StrategyFixture(
        "sma_cross", "close > sma(close, 20)", "close < sma(close, 20)", 9999
    ),
    StrategyFixture(
        "rsi_meanreversion",
        "rsi(close, 14) < 30",
        "rsi(close, 14) > 70",
        9999,
    ),
)


def synthetic_ohlcv_panel() -> dict[str, pd.DataFrame]:
    """Pinned deterministic OHLCV panel: 3 tickers, one date range, no network."""
    return {
        "AAA": _ticker_frame(offset=0.0, volume=300_000.0),
        "BBB": _ticker_frame(offset=5.0, volume=200_000.0),
        "CCC": _ticker_frame(offset=-3.0, volume=100_000.0),
    }


def _ticker_frame(offset: float, volume: float) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=90)
    close = pd.Series(
        np.concatenate(
            [
                np.full(20, 100.0 + offset),
                np.linspace(101.0 + offset, 125.0 + offset, 15),
                np.linspace(124.0 + offset, 82.0 + offset, 20),
                np.linspace(83.0 + offset, 128.0 + offset, 20),
                np.full(15, 128.0 + offset),
            ]
        ),
        index=index,
        dtype=float,
    )
    open_ = close.shift(1).fillna(close.iloc[0] - 0.5)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([open_, close], axis=1).min(axis=1) - 1.0
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": float(volume),
        },
        index=index,
    )
    frame["entry_day0"] = 0.0
    frame.iloc[0, frame.columns.get_loc("entry_day0")] = 1.0
    return frame


def fixture_config(fixture: StrategyFixture) -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=date(2024, 1, 2),
        hold=fixture.hold,
        top=3,
        entry_expr=fixture.entry_expr,
        exit_expr=fixture.exit_expr,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=2.0,
        commission_bps=1.0,
        initial_capital=90_000.0,
        benchmark="SPY",
        allow_reentry=False,
        reinvest=False,
    )


def first_signal_index(bars: pd.DataFrame, expr: str) -> int:
    signal = evaluate(parse(expr), bars).fillna(False).astype(bool)
    matches = np.flatnonzero(signal.to_numpy(dtype=bool))
    if matches.size == 0:
        raise AssertionError(f"no signal for expression {expr!r}")
    return int(matches[0])


def run_core_portfolio_path(
    cfg: BacktestConfig,
    panel: dict[str, pd.DataFrame],
) -> BacktestResult:
    """Use core.simulate_ticker for fills, then Portfolio for shares/equity."""
    exit_ast = parse(cfg.exit_expr) if cfg.exit_expr else None
    outcomes = []
    for rank, (ticker, bars) in enumerate(panel.items(), 1):
        signal_idx = first_signal_index(bars, cfg.entry_expr)
        outcome = simulate_ticker(bars, signal_idx, cfg, exit_ast=exit_ast)
        if outcome.trade is None:
            raise AssertionError(outcome.warning or f"no trade for {ticker}")
        outcomes.append((ticker, rank, outcome.trade))

    portfolio = Portfolio(cfg.initial_capital, slot_count=cfg.top)
    for ticker, rank, trade in sorted(outcomes, key=lambda item: item[2].entry_date):
        portfolio.assign(ticker, rank, trade.signal_date)
        portfolio.open(
            ticker,
            trade.entry_date,
            trade.entry_price,
            cfg.commission_bps,
        )
    for ticker, _rank, trade in sorted(outcomes, key=lambda item: item[2].exit_date):
        portfolio.close(
            ticker,
            trade.exit_date,
            trade.exit_price,
            trade.exit_reason,
            cfg.commission_bps,
        )

    trades = portfolio.closed_trades()
    calendar = pd.DatetimeIndex(
        sorted({day for bars in panel.values() for day in bars.index})
    )
    equity = build_equity_curve(calendar, trades, panel, cfg.initial_capital)
    benchmark = pd.Series(cfg.initial_capital, index=calendar, dtype=float)
    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=equity,
        benchmark_curve=benchmark,
        metrics=compute_metrics(equity, benchmark, trades, cfg.top),
    )
