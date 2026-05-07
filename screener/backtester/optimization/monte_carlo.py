"""Monte Carlo stress testing for completed trade lists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from screener.backtester.models import Trade


@dataclass(frozen=True)
class MonteCarloResult:
    iterations: int
    seed: int
    initial_capital: float
    median_return: float
    return_p05: float
    return_p95: float
    median_drawdown: float
    drawdown_p05: float
    worst_drawdown: float
    probability_of_profit: float
    risk_of_ruin: float


def _drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def simulate_monte_carlo(
    trades: Sequence[Trade],
    *,
    iterations: int = 5000,
    initial_capital: float = 100_000.0,
    seed: int = 42,
    ruin_threshold: float = 0.5,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    returns = np.array([float(t.return_pct) for t in trades], dtype=float)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if returns.size == 0:
        return MonteCarloResult(
            iterations=iterations,
            seed=seed,
            initial_capital=initial_capital,
            median_return=0.0,
            return_p05=0.0,
            return_p95=0.0,
            median_drawdown=0.0,
            drawdown_p05=0.0,
            worst_drawdown=0.0,
            probability_of_profit=0.0,
            risk_of_ruin=0.0,
        )

    terminal_returns: list[float] = []
    drawdowns: list[float] = []
    ruin_count = 0
    ruin_level = initial_capital * ruin_threshold
    sample_size = int(returns.size)
    for _ in range(iterations):
        sampled = rng.choice(returns, size=sample_size, replace=True)
        equity = initial_capital * np.cumprod(1.0 + sampled)
        terminal_returns.append(float(equity[-1] / initial_capital - 1.0))
        dd = _drawdown(np.concatenate(([initial_capital], equity)))
        drawdowns.append(dd)
        if float(equity.min()) <= ruin_level:
            ruin_count += 1

    terminal = np.array(terminal_returns)
    dds = np.array(drawdowns)
    return MonteCarloResult(
        iterations=iterations,
        seed=seed,
        initial_capital=initial_capital,
        median_return=float(np.median(terminal)),
        return_p05=float(np.percentile(terminal, 5)),
        return_p95=float(np.percentile(terminal, 95)),
        median_drawdown=float(np.median(dds)),
        drawdown_p05=float(np.percentile(dds, 5)),
        worst_drawdown=float(dds.min()),
        probability_of_profit=float(np.mean(terminal > 0)),
        risk_of_ruin=float(ruin_count / iterations),
    )
