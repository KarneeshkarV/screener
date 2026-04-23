from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

import pandas as pd


ExitReason = Literal["stop", "target", "trail", "time", "exit_expr", "eod"]


@dataclass(frozen=True)
class BacktestConfig:
    market: str
    as_of: date
    hold: int
    top: int
    entry_expr: str
    exit_expr: Optional[str]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    trailing_stop: Optional[float]
    slippage_bps: float
    commission_bps: float
    initial_capital: float
    benchmark: str
    tickers: Optional[tuple[str, ...]] = None
    universe_file: Optional[str] = None
    max_universe: int = 200
    min_price: Optional[float] = None
    min_avg_dollar_volume: Optional[float] = None
    avg_dollar_volume_window: int = 20
    reserve_multiple: int = 3
    reinvest: bool = True


@dataclass
class Position:
    ticker: str
    entry_date: date
    entry_fill: float
    shares: float
    slot_capital: float
    peak_price: float


@dataclass
class Trade:
    ticker: str
    rank: int
    signal_date: date
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_reason: ExitReason
    shares: float
    entry_cost: float   # total cash out at entry (shares*entry_price + commission)
    exit_value: float   # total cash in at exit (shares*exit_price - commission)
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: list[Trade]
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    metrics: dict
    warnings: list[str] = field(default_factory=list)
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)
