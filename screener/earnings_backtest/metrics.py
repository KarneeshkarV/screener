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
        "total_fees": 0.0,
    }


def _aggregate_fees(trades: Sequence[EventTradeSummary]) -> dict[str, float]:
    """Sum per-component fees from trade details (run-level or per-trade)."""
    if not trades:
        return {}
    # Prefer the run-level totals stamped by the engine when present.
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
            if amt <= 0.0:
                continue
            out[str(name)] = out.get(str(name), 0.0) + amt
    return out


def compute_backtest_summary(
    trades: Sequence[EventTradeSummary], strategy: str = ""
) -> dict[str, str | int | float]:
    """Compute aggregate statistics for any executed earnings-event trade."""
    if not trades:
        return _empty_summary(0, strategy)

    taken = [trade for trade in trades if trade.passed_filter]
    fees_paid = _aggregate_fees(trades)
    fee_fields: dict[str, float] = {
        "total_fees": round(float(sum(fees_paid.values())), 6),
    }
    for name, amount in fees_paid.items():
        fee_fields[f"fee_{name}"] = round(float(amount), 6)

    if not taken:
        empty = _empty_summary(len(trades), strategy)
        empty.update(fee_fields)
        return empty

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
        **fee_fields,
    }


__all__ = ["compute_backtest_summary"]
