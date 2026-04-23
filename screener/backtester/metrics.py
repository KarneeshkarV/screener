"""Backtest performance metrics.

All metrics derive from the portfolio equity curve and the aligned benchmark.
Alpha/beta use a simple OLS fit via ``numpy.polyfit`` — no sklearn.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from screener.backtester.models import Trade

TRADING_DAYS_PER_YEAR = 252


def _daily_returns(equity: pd.Series) -> pd.Series:
    if equity.empty or len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change().dropna()


def _cagr(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return 0.0
    years = max(len(equity) / TRADING_DAYS_PER_YEAR, 1e-9)
    return (end / start) ** (1.0 / years) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min()) if not dd.empty else 0.0


def _sharpe(daily: pd.Series, rf: float = 0.0) -> float:
    if daily.empty or daily.std(ddof=0) == 0:
        return 0.0
    excess = daily - rf / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / excess.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _vol_annual(daily: pd.Series) -> float:
    if daily.empty:
        return 0.0
    return float(daily.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _alpha_beta(daily: pd.Series, bench_daily: pd.Series) -> tuple[float, float]:
    if daily.empty or bench_daily.empty:
        return 0.0, 0.0
    aligned = pd.concat([daily, bench_daily], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0, 0.0
    x = aligned.iloc[:, 1].to_numpy()
    y = aligned.iloc[:, 0].to_numpy()
    if x.std() == 0:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    # annualize alpha (intercept is per-day)
    return float(intercept * TRADING_DAYS_PER_YEAR), float(slope)


def _exposure(
    equity_index: pd.DatetimeIndex, trades: Iterable[Trade], slot_count: int
) -> float:
    trades = list(trades)
    if not trades or len(equity_index) == 0:
        return 0.0
    open_count = pd.Series(0, index=equity_index, dtype=int)
    for t in trades:
        entry = pd.Timestamp(t.entry_date)
        exit_ = pd.Timestamp(t.exit_date)
        mask = (equity_index >= entry) & (equity_index <= exit_)
        open_count.loc[mask] += 1
    return float(open_count.mean() / max(slot_count, 1))


def _invested_return(trades: Iterable[Trade]) -> float:
    """Capital-deployed-only total return.

    Ignores idle cash in the denominator: sums realized PnL across all closed
    trades and divides by total capital that actually touched the market
    (sum of entry_cost). Exposes the dead-cash gap even when the engine's
    reinvestment path is on — a low ratio of equity_return to invested_return
    indicates a large share of capital sat idle.
    """
    trades = list(trades)
    total_cost = sum(float(t.entry_cost) for t in trades)
    total_pnl = sum(float(t.pnl) for t in trades)
    if total_cost <= 0:
        return 0.0
    return total_pnl / total_cost


def compute_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    trades: list[Trade],
    slot_count: int,
) -> dict:
    daily = _daily_returns(equity)
    bench_daily = _daily_returns(benchmark) if not benchmark.empty else pd.Series(dtype=float)
    total_return = (
        float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        if len(equity) >= 2 and equity.iloc[0] > 0
        else 0.0
    )
    alpha, beta = _alpha_beta(daily, bench_daily)
    hit_rate = (
        float(sum(1 for t in trades if t.pnl > 0) / len(trades))
        if trades
        else 0.0
    )
    bench_return = (
        float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
        if len(benchmark) >= 2 and benchmark.iloc[0] > 0
        else 0.0
    )
    return {
        "total_return": total_return,
        "cagr": _cagr(equity),
        "vol_annual": _vol_annual(daily),
        "sharpe": _sharpe(daily),
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": hit_rate,
        "alpha_annual": alpha,
        "beta": beta,
        "exposure": _exposure(equity.index, trades, slot_count),
        "benchmark_return": bench_return,
        "trade_count": len(trades),
        "invested_return": _invested_return(trades),
    }
