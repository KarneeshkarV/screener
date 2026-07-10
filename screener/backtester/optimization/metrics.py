"""Optimizer-specific metric adapters."""

from __future__ import annotations

import numpy as np

from screener.backtester.models import BacktestResult


def risk_adjusted_return(result: BacktestResult) -> float:
    total = float(result.metrics.get("total_return", 0.0))
    dd = abs(float(result.metrics.get("max_drawdown", 0.0)))
    if dd == 0.0:
        return total
    return total / dd


def optimization_metrics(result: BacktestResult) -> dict[str, float]:
    values = dict(result.metrics)
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
