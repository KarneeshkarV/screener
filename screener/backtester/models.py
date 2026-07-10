from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screener.backtester.slippage import FixedBpsSlippage, SlippageModel


ExitReason = Literal["stop", "target", "trail", "time", "exit_expr", "eod"]

# Supported bar intervals. "1d" is the default daily bar; the rest are intraday
# bars sourced from yfinance. A ``date | datetime`` union is used on all the
# temporal fields below so the daily path keeps emitting plain ``date`` objects
# (byte-for-byte identical to the pre-intraday engine) while intraday runs carry
# full timestamps.
SUPPORTED_INTERVALS = ("1d", "1h", "30m", "15m", "5m", "1m")


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    market: str
    as_of: date | datetime
    hold: int
    top: int
    # Bar interval for the whole run. "1d" reproduces the daily engine exactly;
    # intraday values thread through data fetch, the simulation calendar and the
    # metrics annualization factor.
    interval: str = "1d"
    entry_expr: str
    exit_expr: Optional[str]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    trailing_stop: Optional[float]
    slippage_bps: float
    commission_bps: float
    initial_capital: float
    benchmark: str
    strategy_name: Optional[str] = None
    tickers: Optional[tuple[str, ...]] = None
    universe_file: Optional[str] = None
    # Point-in-time universe membership: (symbol, first_eligible_date) pairs.
    # Entry signals before a symbol's date are suppressed; symbols not listed
    # are always eligible. Open positions are never force-closed.
    membership_added: tuple[tuple[str, date], ...] = ()
    # Benchmark-regime entry gate (rolling backtest): when non-empty, entry
    # signals are suppressed on days whose benchmark trend regime (see
    # ``screener.regime.classify_regimes``) is not in this set. Days with an
    # 'unknown' (warmup) regime are also suppressed.
    regime_filter: tuple[str, ...] = ()
    # Earnings blackout entry gate (rolling backtest): when set to N, suppress
    # entry signals on any calendar day within N days BEFORE (and including) a
    # known earnings date for that ticker. ``None`` disables the gate. Tickers
    # with no known earnings dates remain eligible (a warning is recorded).
    earnings_blackout_days: int | None = None
    # Optional rolling-backtest fundamental enrichment. When configured, dated
    # fundamental columns are merged into each ticker's bars before entry/exit
    # expressions are evaluated.
    fundamentals_provider: Optional[str] = None
    fundamental_fields: tuple[str, ...] = ()
    fundamental_lag_days: int = 1
    max_universe: int = 200
    min_price: Optional[float] = None
    min_avg_dollar_volume: Optional[float] = None
    avg_dollar_volume_window: int = 20
    reserve_multiple: int = 3
    reinvest: bool = True
    # Slippage is pluggable: default is a FixedBpsSlippage built from
    # ``slippage_bps`` for backwards compatibility. Richer models
    # (HalfSpread, VolumeImpact, Composite) live in ``screener.backtester.slippage``.
    slippage_model: Optional[SlippageModel] = None
    # Gap-aware stop / target fills. When True (default going forward) a bar
    # that *opens* through the stop fills at the open (worse than stop_ref);
    # symmetric for gap-ups through a target. False reproduces legacy behaviour.
    gap_fills: bool = True
    # Entry order type. ``moo`` = next-bar open (legacy); ``moc`` = next-bar
    # close; ``limit`` = wait for bar whose low <= limit_price, fill at
    # ``min(bar.open, limit_price)``.
    entry_order_type: Literal["moo", "moc", "limit"] = "moo"
    entry_limit_bps: Optional[float] = None
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

    @model_validator(mode="before")
    @classmethod
    def _default_slippage(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("slippage_model") is None:
            bps = data.get("slippage_bps", 0.0)
            data["slippage_model"] = FixedBpsSlippage(bps=bps)
        return data

    @field_validator("interval")
    @classmethod
    def _validate_interval(cls, value: str) -> str:
        if value not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"unsupported interval {value!r}; expected one of "
                f"{', '.join(SUPPORTED_INTERVALS)}"
            )
        return value


class Position(BaseModel):
    ticker: str
    entry_date: date | datetime
    entry_fill: float
    shares: float
    slot_capital: float
    peak_price: float
    dividend_income: float = 0.0


class Trade(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    rank: int
    signal_date: date | datetime
    entry_date: date | datetime
    entry_price: float
    exit_date: date | datetime
    exit_price: float
    exit_reason: ExitReason
    shares: float
    entry_cost: float  # total cash out at entry (shares*entry_price + commission)
    exit_value: float  # total cash in at exit (shares*exit_price - commission)
    pnl: float
    return_pct: float
    # Cash dividends received while the position was held. Excluded from
    # ``return_pct`` for backwards compatibility with existing reports;
    # exposed as a separate field so total-return can be computed when the
    # ``splits_only`` price-adjustment regime is in use.
    dividend_income: float = 0.0


class BacktestResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: BacktestConfig
    trades: list[Trade]
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    metrics: dict
    warnings: list[str] = Field(default_factory=list)
    selection: pd.DataFrame = Field(default_factory=pd.DataFrame)
