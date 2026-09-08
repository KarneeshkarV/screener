"""Optimizer-specific metric adapters."""

from __future__ import annotations

import numpy as np

from screener.backtester.metrics import deflated_sharpe
from screener.backtester.models import BacktestResult


def risk_adjusted_return(result: BacktestResult) -> float:
    total = float(result.metrics.get("total_return", 0.0))
    dd = abs(float(result.metrics.get("max_drawdown", 0.0)))
    if dd == 0.0:
        return total
    return total / dd


def optimization_metrics(result: BacktestResult, n_trials: int = 1) -> dict[str, float]:
    """Result metrics plus the optimizer's own derived fields.

    ``n_trials`` is the number of configurations the surrounding search
    evaluated. The engines cannot know it - each backtest is one point in a
    grid it has never seen - so the Deflated Sharpe is raised here, where the
    grid size is known, rather than inside ``compute_metrics``. At one trial
    nothing is deflated and no ``dsr`` key is emitted, because DSR degenerates
    to PSR there and reporting it would imply a correction that never ran.
    """
    values = dict(result.metrics)
    if "hit_rate" in values:
        values.setdefault("win_rate", float(values["hit_rate"]))
    values.setdefault("risk_adjusted_return", risk_adjusted_return(result))
    values.setdefault("trade_count", float(len(result.trades)))
    if n_trials > 1 and not result.equity_curve.empty:
        values["dsr"] = deflated_sharpe(result.equity_curve, n_trials=n_trials)
        values["dsr_trials"] = float(n_trials)
    return values


def score_result(result: BacktestResult, metric: str) -> float:
    metrics = optimization_metrics(result)
    score = float(metrics.get(metric, 0.0))
    if np.isnan(score):
        return float("-inf")
    return score
