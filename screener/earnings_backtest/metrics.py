"""Metrics for earnings event studies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from screener.earnings_backtest.models import EventTradeSummary


def _empty_summary(total_events: int, strategy: str) -> dict[str, str | int | float]:
    return {
        "total_events": total_events,
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
    }


def compute_backtest_summary(
    trades: Sequence[EventTradeSummary], strategy: str = ""
) -> dict[str, str | int | float]:
    """Compute aggregate statistics for any executed earnings-event trade."""
    if not trades:
        return _empty_summary(0, strategy)

    taken = [trade for trade in trades if trade.passed_filter]
    if not taken:
        return _empty_summary(len(trades), strategy)

    returns = np.array([trade.return_pct for trade in taken])
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    holding_days = [(trade.exit_date - trade.entry_date).days for trade in taken]
    avg_holding = float(np.mean(holding_days)) if holding_days else 0.0

    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        avg_annualized = np.mean(returns) / avg_holding * 252 if avg_holding > 0 else 0
        std_annualized = (
            np.std(returns) / np.sqrt(avg_holding) * np.sqrt(252)
            if avg_holding > 0
            else 1
        )
        sharpe = round(
            avg_annualized / std_annualized if std_annualized > 0 else 0.0, 4
        )

    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_events": len(trades),
        "trades_taken": len(taken),
        "strategy": strategy,
        "win_rate": round(float((returns > 0).mean()) * 100, 2),
        "avg_return_pct": round(float(returns.mean()), 4),
        "median_return_pct": round(float(np.median(returns)), 4),
        "total_return_pct": round(float(returns.sum()), 4),
        "max_winner_pct": round(float(returns.max()), 4) if len(returns) > 0 else 0.0,
        "max_loser_pct": round(float(returns.min()), 4) if len(returns) > 0 else 0.0,
        "profit_factor": round(profit_factor, 4),
        "avg_holding_days": round(avg_holding, 2),
        "sharpe_approx": sharpe,
    }


__all__ = ["compute_backtest_summary"]
