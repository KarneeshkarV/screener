"""Optimizer-specific metric adapters."""

from __future__ import annotations

import numpy as np

from screener.backtester.metrics import (
    deflated_sharpe,
    periods_per_year_for_interval,
)
from screener.backtester.models import BacktestResult


def risk_adjusted_return(result: BacktestResult) -> float:
    total = float(result.metrics.get("total_return", 0.0))
    dd = abs(float(result.metrics.get("max_drawdown", 0.0)))
    if dd == 0.0:
        return total
    return total / dd


def optimization_metrics(
    result: BacktestResult, *, n_trials: int = 1
) -> dict[str, float]:
    """Result metrics as the optimizer reports them.

    ``n_trials`` is how many parameter configurations the search evaluated.
    The engines compute every result as a single trial, so the Deflated Sharpe
    is applied here — this is the only layer that knows the size of the search
    that the result was selected from. At one trial there is nothing to deflate
    and no ``dsr`` key is produced, so the report cannot show an undeflated
    number under a deflated name.
    """
    values = dict(result.metrics)
    if n_trials > 1:
        values["dsr"] = deflated_sharpe(
            result.equity_curve,
            n_trials=n_trials,
            periods_per_year=periods_per_year_for_interval(result.config.interval),
        )
        values["dsr_trials"] = float(n_trials)
    if "hit_rate" in values:
        values.setdefault("win_rate", float(values["hit_rate"]))
    values.setdefault("risk_adjusted_return", risk_adjusted_return(result))
    values.setdefault("trade_count", float(len(result.trades)))
    return values


def score_result(result: BacktestResult, metric: str) -> float:
    metrics = optimization_metrics(result)
    score = float(metrics.get(metric, 0.0))
    if np.isnan(score):
        return float("-inf")
    return score
