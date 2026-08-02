from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional, TypeAlias, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screener.backtester.slippage import (
    CompositeSlippage,
    EstimatedHalfSpreadSlippage,
    FixedBpsSlippage,
    SlippageModel,
)
from screener.ledger import ExitReason, Trade as LifecycleTrade


# Supported bar intervals. "1d" is the default daily bar; the rest are intraday
# bars sourced from yfinance. A ``date | datetime`` union is used on all the
# temporal fields below so the daily path keeps emitting plain ``date`` objects
# (byte-for-byte identical to the pre-intraday engine) while intraday runs carry
# full timestamps.
SUPPORTED_INTERVALS = ("1d", "1h", "30m", "15m", "5m", "1m")


class BacktestConfig(BaseModel):
    """Backtest request as a single flat set of fields.

    Every knob lives directly on the config; construction and attribute access
    are flat throughout (``cfg.hold``, ``cfg.top``, ``cfg.interval``). The
    optimization grid and ``model_copy`` updates address these same field names.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    market: str
    as_of: date | datetime
    benchmark: str

    # Universe
    tickers: Optional[tuple[str, ...]] = None
    universe_file: Optional[str] = None
    membership_added: tuple[tuple[str, date], ...] = ()
    membership_windows: tuple[tuple[str, date, date | None], ...] = ()
    dynamic_universe_size: int | None = None
    dynamic_universe_lookback: int = Field(default=60, ge=2)
    dynamic_universe_rebalance: Literal["daily", "weekly", "monthly", "quarterly"] = (
        "monthly"
    )
    max_universe: int = 200

    # Signals
    entry_expr: str
    exit_expr: Optional[str]
    strategy_name: Optional[str] = None
    regime_filter: tuple[str, ...] = ()
    # Earnings blackout entry gate (rolling backtest): when set to N, suppress
    # entry signals on any calendar day within N days BEFORE (and including) a
    # known earnings date for that ticker. ``None`` disables the gate. Tickers
    # with no known earnings dates remain eligible (a warning is recorded).
    earnings_blackout_days: int | None = None
    fundamentals_provider: Optional[str] = None
    fundamental_fields: tuple[str, ...] = ()
    fundamental_lag_days: int = 1
    # Cross-sectional sector neutralization of ``rank_score`` inside the rolling
    # candidate builder. When True and a factor score matrix exists, scores are
    # z-scored within each sector group per day before ranking.
    sector_neutral: bool = False

    # Data
    interval: str = "1d"
    price_adjustment: Literal["full", "splits_only", "none"] = "full"
    # Force positions flat on the last bar of each trading session so intraday
    # runs never hold overnight. Requires an intraday interval.
    intraday_only: bool = False

    # Execution
    hold: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    trailing_stop: Optional[float]
    slippage_bps: float
    commission_bps: float
    slippage_model: Optional[SlippageModel] = None
    # Statutory fee model name (cash impact, not fill-price). ``flat`` uses
    # ``commission_bps`` exactly as before; ``india`` applies NSE delivery fees;
    # ``us_vested`` applies the Vested/DriveWealth US equity fee stack.
    cost_model: Literal["flat", "india", "us_vested"] = "flat"
    # When True, compute Corwin-Schultz half-spread from bar high/low and feed
    # it into the fill-layer slippage stack as ``half_spread``.
    spread_proxy: bool = False
    gap_fills: bool = True
    entry_order_type: Literal["moo", "moc", "limit"] = "moo"
    entry_limit_bps: Optional[float] = None
    partial_exits: tuple[tuple[float, float], ...] = ()

    # Portfolio
    top: int
    initial_capital: float
    min_price: Optional[float] = None
    min_avg_dollar_volume: Optional[float] = None
    avg_dollar_volume_window: int = 20
    reserve_multiple: int = 3
    reinvest: bool = True
    allow_reentry: bool = False
    max_reentries: int = 0
    max_concurrent_per_ticker: int = 1
    # Rule-based per-entry sizing (see ``screener.backtester.sizing``).
    # ``equal_slot`` reproduces the legacy fixed-slot budget exactly; every
    # other rule sizes DOWN from that slot ceiling, never above it.
    sizing_rule: str = "equal_slot"
    sizing_risk_pct: float = Field(default=0.01, gt=0.0)
    sizing_position_pct: float = Field(default=0.10, gt=0.0)
    sizing_atr_window: int = Field(default=14, gt=0)
    sizing_atr_multiple: float = Field(default=2.0, gt=0.0)
    sizing_vol_window: int = Field(default=20, gt=1)

    @field_validator("interval")
    @classmethod
    def _validate_interval(cls, value: str) -> str:
        if value not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"unsupported interval {value!r}; expected one of "
                f"{', '.join(SUPPORTED_INTERVALS)}"
            )
        return value

    @field_validator("sizing_rule")
    @classmethod
    def _validate_sizing_rule(cls, value: str) -> str:
        from screener.backtester.sizing import available_sizing_rules

        if value not in available_sizing_rules():
            raise ValueError(
                f"unknown sizing rule {value!r}; expected one of "
                f"{', '.join(available_sizing_rules())}"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _default_slippage(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("slippage_model") is None:
            data = dict(data)
            data["slippage_model"] = FixedBpsSlippage(bps=data.get("slippage_bps", 0.0))
        return data

    @model_validator(mode="after")
    def _validate_after(self) -> BacktestConfig:
        if self.intraday_only and self.interval == "1d":
            raise ValueError("intraday_only requires an intraday interval (got '1d')")

        def consumes_estimated_spread(model: SlippageModel) -> bool:
            if isinstance(model, EstimatedHalfSpreadSlippage):
                return True
            if isinstance(model, CompositeSlippage):
                return any(consumes_estimated_spread(item) for item in model.models)
            return False

        consumes = consumes_estimated_spread(cast(SlippageModel, self.slippage_model))
        if self.spread_proxy != consumes:
            raise ValueError(
                "spread_proxy and EstimatedHalfSpreadSlippage must be enabled together"
            )

        if self.sizing_rule == "fixed_risk" and (
            self.stop_loss is None or self.stop_loss <= 0
        ):
            raise ValueError("sizing rule 'fixed_risk' requires a positive stop_loss")
        return self


class Position(BaseModel):
    ticker: str
    entry_date: date | datetime
    entry_fill: float
    shares: float
    slot_capital: float
    peak_price: float
    dividend_income: float = 0.0


class EquityLedgerTrade(LifecycleTrade):
    """Equity accounting extension of the neutral trade lifecycle."""

    ticker: str
    rank: int
    signal_date: date | datetime
    entry_price: float
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


# Public compatibility alias. The implementation lives in neutral ledger
# vocabulary as ``EquityLedgerTrade`` so it does not collide with research
# round trips.
Trade: TypeAlias = EquityLedgerTrade


class BacktestResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: BacktestConfig
    trades: list[Trade]
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    metrics: dict
    warnings: list[str] = Field(default_factory=list)
    selection: pd.DataFrame = Field(default_factory=pd.DataFrame)
