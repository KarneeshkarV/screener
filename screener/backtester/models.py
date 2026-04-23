from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

from screener.backtester.slippage import FixedBpsSlippage, SlippageModel

ExitReason = Literal["stop", "target", "trail", "time", "exit_expr", "eod"]


@dataclass(frozen=True)
class BacktestConfig:
    market: str
    as_of: date
    hold: int
    top: int
    entry_expr: str
    exit_expr: str | None
    stop_loss: float | None
    take_profit: float | None
    trailing_stop: float | None
    slippage_bps: float
    commission_bps: float
    initial_capital: float
    benchmark: str
    tickers: tuple[str, ...] | None = None
    universe_file: str | None = None
    max_universe: int = 200
    min_price: float | None = None
    min_avg_dollar_volume: float | None = None
    avg_dollar_volume_window: int = 20
    reserve_multiple: int = 3
    reinvest: bool = True
    # Slippage is pluggable: default is a FixedBpsSlippage built from
    # ``slippage_bps`` for backwards compatibility. Richer models
    # (HalfSpread, VolumeImpact, Composite) live in ``screener.backtester.slippage``.
    slippage_model: SlippageModel | None = None
    # Gap-aware stop / target fills. When True (default going forward) a bar
    # that *opens* through the stop fills at the open (worse than stop_ref);
    # symmetric for gap-ups through a target. False reproduces legacy behaviour.
    gap_fills: bool = True
    # Entry order type. ``moo`` = next-bar open (legacy); ``moc`` = next-bar
    # close; ``limit`` = wait for bar whose low <= limit_price, fill at
    # ``min(bar.open, limit_price)``.
    entry_order_type: Literal["moo", "moc", "limit"] = "moo"
    entry_limit_bps: float | None = None
    # Per-ticker trade lifecycle. Default preserves the historical
    # "one trade per ticker" behavior. Opt-in via allow_reentry + caps.
    allow_reentry: bool = False
    max_reentries: int = 0
    max_concurrent_per_ticker: int = 1
    # Scale-out tuple: each entry is (r_multiple, fraction_of_position).
    # e.g. ((1.0, 0.5),) closes half at +1R and holds the rest.
    partial_exits: tuple[tuple[float, float], ...] = ()
    # Price-adjustment regime for signal evaluation and fills.
    #   ``full``        — legacy, yfinance auto_adjust=True.
    #   ``splits_only`` — split-adjusted OHLC, dividends as explicit cash.
    #   ``none``        — raw OHLC, no adjustment.
    price_adjustment: Literal["full", "splits_only", "none"] = "full"

    def __post_init__(self) -> None:
        if self.slippage_model is None:
            object.__setattr__(
                self, "slippage_model", FixedBpsSlippage(self.slippage_bps)
            )


@dataclass
class Position:
    ticker: str
    entry_date: date
    entry_fill: float
    shares: float
    slot_capital: float
    peak_price: float
    dividend_income: float = 0.0


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
    # Cash dividends received while the position was held. Excluded from
    # ``return_pct`` for backwards compatibility with existing reports;
    # exposed as a separate field so total-return can be computed when the
    # ``splits_only`` price-adjustment regime is in use.
    dividend_income: float = 0.0


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: list[Trade]
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    metrics: dict
    warnings: list[str] = field(default_factory=list)
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)
