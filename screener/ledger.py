"""Neutral contracts for completed trades and event-study summaries.

The concrete ``Trade`` base deliberately contains only lifecycle fields whose
meaning and units are shared across every backtest subsystem.  Accounting,
research, event, and multi-leg option fields belong to their subsystem
extensions rather than as nullable fields here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

TradeTime = date | datetime | pd.Timestamp
ExitReason = Literal[
    "stop",
    "target",
    "trail",
    "time",
    "exit_expr",
    "eod",
    "session",
    "expiry",
    "dte",
    "end",
]


class Trade(BaseModel):
    """Completed trade lifecycle shared by all trade extensions."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    entry_date: TradeTime
    exit_date: TradeTime


@runtime_checkable
class EventTradeSummary(Protocol):
    """Completed event trade fields consumed by event-study aggregation."""

    @property
    def entry_date(self) -> date: ...

    @property
    def exit_date(self) -> date: ...

    @property
    def return_pct(self) -> float: ...

    @property
    def passed_filter(self) -> bool: ...


def positive_return_rate_pct(returns: np.ndarray) -> float:
    """Return the percentage-point share of positive reported trade returns."""
    return round(float((returns > 0).mean()) * 100, 2)


def trade_return_sharpe_by_holding_period(
    returns: np.ndarray, average_holding_days: float
) -> float:
    """Annualize per-trade returns using their average holding period.

    This is intentionally distinct from an equity-curve Sharpe ratio because
    its inputs are completed trades rather than a daily portfolio return series.
    """
    if len(returns) <= 1 or np.std(returns) == 0 or average_holding_days <= 0:
        return 0.0
    average_annualized = np.mean(returns) / average_holding_days * 252
    std_annualized = np.std(returns) / np.sqrt(average_holding_days) * np.sqrt(252)
    return round(
        float(average_annualized / std_annualized) if std_annualized > 0 else 0.0,
        4,
    )


def _aggregate_event_fees(trades: Sequence[EventTradeSummary]) -> dict[str, float]:
    """Sum fee components stored in event-trade details."""
    if not trades:
        return {}
    first_details = getattr(trades[0], "details", None) or {}
    run_total = first_details.get("fees_paid_total")
    if isinstance(run_total, dict) and run_total:
        return {str(k): float(v) for k, v in run_total.items() if float(v) != 0.0}

    out: dict[str, float] = {}
    for trade in trades:
        details = getattr(trade, "details", None) or {}
        fees = details.get("fees") or {}
        if not isinstance(fees, dict):
            continue
        for name, amount in fees.items():
            amt = float(amount)
            if amt > 0.0:
                out[str(name)] = out.get(str(name), 0.0) + amt
    return out


def compute_event_trade_summary(
    trades: Sequence[EventTradeSummary], strategy: str = ""
) -> dict[str, str | int | float]:
    """Compute event-study statistics without depending on an event subsystem."""
    if not trades:
        return {
            "total_events": 0,
            "trades_taken": 0,
            "strategy": strategy,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "total_return_pct": 0.0,
            "max_winner_pct": 0.0,
            "max_loser_pct": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
            "sharpe_approx": 0.0,
            "total_fees": 0.0,
        }

    taken = [trade for trade in trades if trade.passed_filter]
    fees_paid = _aggregate_event_fees(trades)
    fee_fields: dict[str, float] = {
        "total_fees": round(float(sum(fees_paid.values())), 6),
    }
    for name, amount in fees_paid.items():
        fee_fields[f"fee_{name}"] = round(float(amount), 6)

    if not taken:
        return {
            "total_events": len(trades),
            "trades_taken": 0,
            "strategy": strategy,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "total_return_pct": 0.0,
            "max_winner_pct": 0.0,
            "max_loser_pct": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
            "sharpe_approx": 0.0,
            **fee_fields,
        }

    returns = np.array([trade.return_pct for trade in taken])
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    holding_days = [(trade.exit_date - trade.entry_date).days for trade in taken]
    average_holding = float(np.mean(holding_days)) if holding_days else 0.0
    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_events": len(trades),
        "trades_taken": len(taken),
        "strategy": strategy,
        "win_rate": positive_return_rate_pct(returns),
        "avg_return_pct": round(float(returns.mean()), 4),
        "median_return_pct": round(float(np.median(returns)), 4),
        "total_return_pct": round(float(returns.sum()), 4),
        "max_winner_pct": round(float(returns.max()), 4),
        "max_loser_pct": round(float(returns.min()), 4),
        "profit_factor": round(profit_factor, 4),
        "avg_holding_days": round(average_holding, 2),
        "sharpe_approx": trade_return_sharpe_by_holding_period(
            returns, average_holding
        ),
        **fee_fields,
    }


__all__ = [
    "EventTradeSummary",
    "ExitReason",
    "Trade",
    "TradeTime",
    "compute_event_trade_summary",
    "positive_return_rate_pct",
    "trade_return_sharpe_by_holding_period",
]
