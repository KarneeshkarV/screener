"""Monte Carlo bootstrap analysis for completed backtest results."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from screener.backtester.metrics import TRADING_DAYS_PER_YEAR
from screener.backtester.models import BacktestResult


MonteCarloMethod = Literal["bootstrap_trades", "bootstrap_returns", "block_bootstrap"]


class MonteCarloResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sim_equity_curves: pd.DataFrame
    sim_metrics: pd.DataFrame
    confidence: dict[str, dict[str, float]]
    prob_of_loss: float
    prob_breach_max_dd: float


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _cagr(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0.0:
        return 0.0
    years = max(len(equity) / TRADING_DAYS_PER_YEAR, 1e-9)
    return float((end / start) ** (1.0 / years) - 1.0)


def _sharpe(equity: pd.Series) -> float:
    daily = equity.pct_change().dropna()
    if daily.empty or float(daily.std(ddof=0)) == 0.0:
        return 0.0
    return float(daily.mean() / daily.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _metrics_for_curve(equity: pd.Series) -> dict[str, float]:
    terminal_return = (
        float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        if len(equity) >= 2 and float(equity.iloc[0]) > 0.0
        else 0.0
    )
    daily = equity.pct_change().dropna()
    return {
        "terminal_return": terminal_return,
        "cagr": _cagr(equity),
        "sharpe": _sharpe(equity),
        "max_dd": _max_drawdown(equity),
        "vol": float(daily.std(ddof=0)) if not daily.empty else 0.0,
    }


def _flat_curve(index: pd.DatetimeIndex, initial_capital: float) -> pd.Series:
    return pd.Series(initial_capital, index=index, dtype=float)


def _bootstrap_trade_curve(
    result: BacktestResult,
    rng: np.random.Generator,
) -> pd.Series:
    index = pd.DatetimeIndex(result.equity_curve.index)
    initial = float(result.config.initial_capital)
    if index.empty:
        return pd.Series(dtype=float)
    if not result.trades:
        return _flat_curve(index, initial)

    pnls = np.array([float(trade.pnl) for trade in result.trades], dtype=float)
    sampled = rng.choice(pnls, size=len(pnls), replace=True)
    exit_dates = sorted(pd.Timestamp(trade.exit_date) for trade in result.trades)
    curve = _flat_curve(index, initial)
    for pnl, exit_date in zip(sampled, exit_dates):
        loc = int(index.searchsorted(exit_date, side="left"))
        if loc >= len(index):
            loc = len(index) - 1
        curve.iloc[loc:] = curve.iloc[loc:] + float(pnl)
    return curve


def _bootstrap_return_curve(
    result: BacktestResult,
    rng: np.random.Generator,
) -> pd.Series:
    index = pd.DatetimeIndex(result.equity_curve.index)
    initial = float(result.config.initial_capital)
    if index.empty:
        return pd.Series(dtype=float)
    returns = result.equity_curve.pct_change().dropna().to_numpy(dtype=float)
    if returns.size == 0:
        return _flat_curve(index, initial)
    sampled = rng.choice(returns, size=max(len(index) - 1, 0), replace=True)
    values = np.concatenate(([initial], initial * np.cumprod(1.0 + sampled)))
    return pd.Series(values, index=index, dtype=float)


def _block_bootstrap_curve(
    result: BacktestResult,
    rng: np.random.Generator,
    block_size: int,
) -> pd.Series:
    index = pd.DatetimeIndex(result.equity_curve.index)
    initial = float(result.config.initial_capital)
    if index.empty:
        return pd.Series(dtype=float)
    returns = result.equity_curve.pct_change().dropna().to_numpy(dtype=float)
    target = max(len(index) - 1, 0)
    if returns.size == 0 or target == 0:
        return _flat_curve(index, initial)
    blocks: list[np.ndarray] = []
    max_start = max(int(returns.size) - block_size, 0)
    while sum(block.size for block in blocks) < target:
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        blocks.append(returns[start : start + block_size])
    sampled = np.concatenate(blocks)[:target]
    values = np.concatenate(([initial], initial * np.cumprod(1.0 + sampled)))
    return pd.Series(values, index=index, dtype=float)


def _confidence(metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    confidence: dict[str, dict[str, float]] = {}
    for source, label in [
        ("terminal_return", "terminal_return"),
        ("max_dd", "max_dd"),
        ("sharpe", "sharpe"),
    ]:
        values = (
            metrics[source].to_numpy(dtype=float) if source in metrics else np.array([])
        )
        if values.size == 0:
            confidence[label] = {"p05": 0.0, "p50": 0.0, "p95": 0.0}
            continue
        confidence[label] = {
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    return confidence


def run_monte_carlo(
    result: BacktestResult,
    n_sims: int = 1000,
    method: MonteCarloMethod = "bootstrap_trades",
    block_size: int = 5,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo bootstraps over trades or daily equity returns."""
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = np.random.default_rng(seed)
    curves: dict[str, pd.Series] = {}
    rows: list[dict[str, float]] = []
    for sim_idx in range(n_sims):
        if method == "bootstrap_trades":
            curve = _bootstrap_trade_curve(result, rng)
        elif method == "bootstrap_returns":
            curve = _bootstrap_return_curve(result, rng)
        elif method == "block_bootstrap":
            curve = _block_bootstrap_curve(result, rng, block_size)
        else:
            raise ValueError(f"unknown Monte Carlo method: {method}")

        col = f"sim_{sim_idx:04d}"
        curves[col] = curve
        row = _metrics_for_curve(curve)
        row["sim"] = float(sim_idx)
        rows.append(row)

    sim_equity_curves = pd.DataFrame(curves)
    sim_metrics = pd.DataFrame(rows).set_index("sim")
    conf = _confidence(sim_metrics)
    terminal = (
        sim_metrics["terminal_return"].to_numpy(dtype=float)
        if "terminal_return" in sim_metrics
        else np.array([])
    )
    max_dd = (
        sim_metrics["max_dd"].to_numpy(dtype=float)
        if "max_dd" in sim_metrics
        else np.array([])
    )
    realized_max_dd = _max_drawdown(result.equity_curve)
    prob_of_loss = float(np.mean(terminal < 0.0)) if terminal.size else 0.0
    prob_breach = float(np.mean(max_dd < realized_max_dd)) if max_dd.size else 0.0
    return MonteCarloResult(
        sim_equity_curves=sim_equity_curves,
        sim_metrics=sim_metrics,
        confidence=conf,
        prob_of_loss=prob_of_loss,
        prob_breach_max_dd=prob_breach,
    )
