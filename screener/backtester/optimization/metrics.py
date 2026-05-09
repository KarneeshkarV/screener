"""Trading metrics used by parameter optimization."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestResult, Trade

TRADING_DAYS_PER_YEAR = 252


def equity_returns(equity: pd.Series) -> pd.Series:
    if equity.empty or len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.astype(float).pct_change().dropna()


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = equity_returns(equity)
    if returns.empty:
        return 0.0
    vol = float(returns.std(ddof=0))
    if vol == 0.0 or not np.isfinite(vol):
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def profit_factor(trades: Iterable[Trade]) -> float:
    trades = list(trades)
    gross_profit = sum(float(t.pnl) for t in trades if t.pnl > 0)
    gross_loss = abs(sum(float(t.pnl) for t in trades if t.pnl < 0))
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def maximum_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.astype(float).cummax()
    drawdown = (equity.astype(float) - peak) / peak
    return float(drawdown.min()) if not drawdown.empty else 0.0


def win_rate(trades: Iterable[Trade]) -> float:
    trades = list(trades)
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl > 0) / len(trades)


def expectancy(trades: Iterable[Trade]) -> float:
    trades = list(trades)
    if not trades:
        return 0.0
    return float(np.mean([float(t.return_pct) for t in trades]))


def calmar_ratio(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2 or float(equity.iloc[0]) <= 0:
        return 0.0
    years = max(len(equity) / TRADING_DAYS_PER_YEAR, 1e-9)
    total_return = float(equity.iloc[-1] / equity.iloc[0])
    cagr = total_return ** (1.0 / years) - 1.0
    dd = abs(maximum_drawdown(equity))
    if dd == 0.0:
        return float("inf") if cagr > 0 else 0.0
    return cagr / dd


def risk_adjusted_return(result: BacktestResult) -> float:
    total = float(result.metrics.get("total_return", 0.0))
    dd = abs(
        float(result.metrics.get("max_drawdown", maximum_drawdown(result.equity_curve)))
    )
    if dd == 0.0:
        return total
    return total / dd


def optimization_metrics(result: BacktestResult) -> dict[str, float]:
    values = dict(result.metrics)
    values.update(
        {
            "sharpe": sharpe_ratio(result.equity_curve),
            "profit_factor": profit_factor(result.trades),
            "max_drawdown": maximum_drawdown(result.equity_curve),
            "win_rate": win_rate(result.trades),
            "expectancy": expectancy(result.trades),
            "calmar": calmar_ratio(result.equity_curve),
            "risk_adjusted_return": risk_adjusted_return(result),
            "trade_count": float(len(result.trades)),
        }
    )
    return values


def score_result(result: BacktestResult, metric: str) -> float:
    metrics = optimization_metrics(result)
    score = float(metrics.get(metric, 0.0))
    if np.isnan(score):
        return float("-inf")
    return score
