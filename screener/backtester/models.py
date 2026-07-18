from __future__ import annotations

from datetime import date, datetime
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Optional, Self, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screener.backtester.slippage import (
    CompositeSlippage,
    EstimatedHalfSpreadSlippage,
    FixedBpsSlippage,
    SlippageModel,
)


ExitReason = Literal["stop", "target", "trail", "time", "exit_expr", "eod"]

# Supported bar intervals. "1d" is the default daily bar; the rest are intraday
# bars sourced from yfinance. A ``date | datetime`` union is used on all the
# temporal fields below so the daily path keeps emitting plain ``date`` objects
# (byte-for-byte identical to the pre-intraday engine) while intraday runs carry
# full timestamps.
SUPPORTED_INTERVALS = ("1d", "1h", "30m", "15m", "5m", "1m")


class UniversePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tickers: Optional[tuple[str, ...]] = None
    universe_file: Optional[str] = None
    membership_added: tuple[tuple[str, date], ...] = ()
    max_universe: int = 200


class SignalPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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


class DataPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    interval: str = "1d"
    price_adjustment: Literal["full", "splits_only", "none"] = "full"

    @field_validator("interval")
    @classmethod
    def _validate_interval(cls, value: str) -> str:
        if value not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"unsupported interval {value!r}; expected one of "
                f"{', '.join(SUPPORTED_INTERVALS)}"
            )
        return value


class ExecutionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

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

    @model_validator(mode="before")
    @classmethod
    def _default_slippage(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("slippage_model") is None:
            data = dict(data)
            data["slippage_model"] = FixedBpsSlippage(bps=data.get("slippage_bps", 0.0))
        return data

    @model_validator(mode="after")
    def _validate_spread_proxy_consumer(self) -> ExecutionPolicy:
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
        return self


class PortfolioPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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


class BacktestConfig(BaseModel):
    """Backtest request composed from focused immutable policy objects.

    Flat construction and attribute access remain supported at the public
    boundary while callers migrate. Internally, ownership is explicit through
    ``universe``, ``signals``, ``data``, ``execution``, and ``portfolio``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    market: str
    as_of: date | datetime
    benchmark: str
    universe: UniversePolicy = Field(default_factory=UniversePolicy)
    signals: SignalPolicy
    data: DataPolicy = Field(default_factory=DataPolicy)
    execution: ExecutionPolicy
    portfolio: PortfolioPolicy

    _GROUP_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "universe": tuple(UniversePolicy.model_fields),
        "signals": tuple(SignalPolicy.model_fields),
        "data": tuple(DataPolicy.model_fields),
        "execution": tuple(ExecutionPolicy.model_fields),
        "portfolio": tuple(PortfolioPolicy.model_fields),
    }
    _LEGACY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        field for fields in _GROUP_FIELDS.values() for field in fields
    )
    # Remove this flat shim once tests and the optimization grid construct and
    # update the nested policy models directly.

    @model_validator(mode="before")
    @classmethod
    def _group_flat_input(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for group, fields in cls._GROUP_FIELDS.items():
            legacy = {field: data.pop(field) for field in fields if field in data}
            if not legacy:
                continue
            current = data.get(group)
            if isinstance(current, BaseModel):
                grouped = current.model_dump()
            elif isinstance(current, Mapping):
                grouped = dict(current)
            else:
                grouped = {}
            grouped.update(legacy)
            data[group] = grouped
        return data

    @model_validator(mode="after")
    def _validate_sizing_dependencies(self) -> BacktestConfig:
        if self.portfolio.sizing_rule == "fixed_risk" and (
            self.execution.stop_loss is None or self.execution.stop_loss <= 0
        ):
            raise ValueError("sizing rule 'fixed_risk' requires a positive stop_loss")
        return self

    def to_flat_dict(self) -> dict[str, Any]:
        """Return the historical flat configuration shape for external seams."""
        execution = self.execution.model_dump(exclude={"slippage_model"})
        execution["slippage_model"] = self.execution.slippage_model
        return {
            "market": self.market,
            "as_of": self.as_of,
            "benchmark": self.benchmark,
            **self.universe.model_dump(),
            **self.signals.model_dump(),
            **self.data.model_dump(),
            **execution,
            **self.portfolio.model_dump(),
        }

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Preserve flat optimization/test updates while policies are nested."""
        if update and self._LEGACY_FIELDS.intersection(update):
            data = self.to_flat_dict()
            data.update(update)
            return self.__class__.model_validate(data)
        return super().model_copy(update=dict(update) if update else None, deep=deep)

    @property
    def tickers(self) -> Optional[tuple[str, ...]]:
        return self.universe.tickers

    @property
    def universe_file(self) -> Optional[str]:
        return self.universe.universe_file

    @property
    def membership_added(self) -> tuple[tuple[str, date], ...]:
        return self.universe.membership_added

    @property
    def max_universe(self) -> int:
        return self.universe.max_universe

    @property
    def entry_expr(self) -> str:
        return self.signals.entry_expr

    @property
    def exit_expr(self) -> Optional[str]:
        return self.signals.exit_expr

    @property
    def strategy_name(self) -> Optional[str]:
        return self.signals.strategy_name

    @property
    def regime_filter(self) -> tuple[str, ...]:
        return self.signals.regime_filter

    @property
    def earnings_blackout_days(self) -> Optional[int]:
        return self.signals.earnings_blackout_days

    @property
    def sector_neutral(self) -> bool:
        return self.signals.sector_neutral

    @property
    def fundamentals_provider(self) -> Optional[str]:
        return self.signals.fundamentals_provider

    @property
    def fundamental_fields(self) -> tuple[str, ...]:
        return self.signals.fundamental_fields

    @property
    def fundamental_lag_days(self) -> int:
        return self.signals.fundamental_lag_days

    @property
    def interval(self) -> str:
        return self.data.interval

    @property
    def price_adjustment(self) -> Literal["full", "splits_only", "none"]:
        return self.data.price_adjustment

    @property
    def hold(self) -> int:
        return self.execution.hold

    @property
    def stop_loss(self) -> Optional[float]:
        return self.execution.stop_loss

    @property
    def take_profit(self) -> Optional[float]:
        return self.execution.take_profit

    @property
    def trailing_stop(self) -> Optional[float]:
        return self.execution.trailing_stop

    @property
    def slippage_bps(self) -> float:
        return self.execution.slippage_bps

    @property
    def commission_bps(self) -> float:
        return self.execution.commission_bps

    @property
    def slippage_model(self) -> Optional[SlippageModel]:
        return self.execution.slippage_model

    @property
    def cost_model(self) -> Literal["flat", "india", "us_vested"]:
        return self.execution.cost_model

    @property
    def spread_proxy(self) -> bool:
        return self.execution.spread_proxy

    @property
    def gap_fills(self) -> bool:
        return self.execution.gap_fills

    @property
    def entry_order_type(self) -> Literal["moo", "moc", "limit"]:
        return self.execution.entry_order_type

    @property
    def entry_limit_bps(self) -> Optional[float]:
        return self.execution.entry_limit_bps

    @property
    def partial_exits(self) -> tuple[tuple[float, float], ...]:
        return self.execution.partial_exits

    @property
    def top(self) -> int:
        return self.portfolio.top

    @property
    def initial_capital(self) -> float:
        return self.portfolio.initial_capital

    @property
    def min_price(self) -> Optional[float]:
        return self.portfolio.min_price

    @property
    def min_avg_dollar_volume(self) -> Optional[float]:
        return self.portfolio.min_avg_dollar_volume

    @property
    def avg_dollar_volume_window(self) -> int:
        return self.portfolio.avg_dollar_volume_window

    @property
    def reserve_multiple(self) -> int:
        return self.portfolio.reserve_multiple

    @property
    def reinvest(self) -> bool:
        return self.portfolio.reinvest

    @property
    def allow_reentry(self) -> bool:
        return self.portfolio.allow_reentry

    @property
    def max_reentries(self) -> int:
        return self.portfolio.max_reentries

    @property
    def max_concurrent_per_ticker(self) -> int:
        return self.portfolio.max_concurrent_per_ticker


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
