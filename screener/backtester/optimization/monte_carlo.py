"""Monte Carlo stress testing for trade lists and equity curves."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from screener.backtester.models import Trade

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class MonteCarloResult(BaseModel):
    model_config = ConfigDict(frozen=True)

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


class EquityMonteCarloResult(BaseModel):
    """Block-bootstrap stress test of a backtest's daily equity curve.

    Reported separately from :class:`MonteCarloResult` because the two resample
    different things. The trade bootstrap chains completed trades one after the
    other, which is only faithful to a single-slot strategy; a rolling backtest
    holds ``top`` positions at once, so its trades overlap in time and chaining
    them compounds a portfolio that never existed. Resampling the equity curve
    instead keeps the concurrency, the position sizing and the cash drag that
    the simulation actually produced.
    """

    model_config = ConfigDict(frozen=True)

    iterations: int
    seed: int
    block: int
    bars: int
    initial_capital: float
    median_return: float
    return_p05: float
    return_p95: float
    median_drawdown: float
    drawdown_p05: float
    worst_drawdown: float
    probability_of_profit: float
    risk_of_ruin: float


def _empty_equity_result(
    *, iterations: int, seed: int, block: int, initial_capital: float
) -> EquityMonteCarloResult:
    return EquityMonteCarloResult(
        iterations=iterations,
        seed=seed,
        block=block,
        bars=0,
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


def simulate_equity_monte_carlo(
    equity: "pd.Series",
    *,
    iterations: int = 5000,
    block: int = 20,
    seed: int = 42,
    ruin_threshold: float = 0.5,
) -> EquityMonteCarloResult:
    """Resample the equity curve's bar returns in blocks and restate the risk.

    Uses a *circular* block bootstrap: each synthetic path is built from
    ``ceil(bars / block)`` runs of ``block`` consecutive returns, drawn from
    uniformly random start points and wrapping past the end of the series. The
    wrap is what makes every observed return equally likely to be drawn; a
    non-circular version under-samples the tail of the curve. Blocks (rather
    than single days) preserve the short-horizon autocorrelation that drives
    drawdown, so the drawdown percentiles stay honest instead of collapsing
    toward the i.i.d. case.

    Iterations are looped rather than vectorized: a fully vectorized draw
    allocates ``iterations x bars`` indices at once, which is hundreds of MB on
    a multi-year daily run.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if block <= 0:
        raise ValueError("block must be positive")
    initial_capital = float(equity.iloc[0]) if len(equity) else 0.0
    if initial_capital <= 0:
        raise ValueError("equity curve must start above zero")

    returns = equity.pct_change().dropna().to_numpy(dtype=float)
    n = int(returns.size)
    if n == 0:
        return _empty_equity_result(
            iterations=iterations,
            seed=seed,
            block=block,
            initial_capital=initial_capital,
        )

    rng = np.random.default_rng(seed)
    span = min(block, n)
    draws = -(-n // span)  # ceil division: blocks needed to cover the window
    offsets = np.arange(span)
    ruin_level = initial_capital * ruin_threshold

    terminal_returns = np.empty(iterations, dtype=float)
    drawdowns = np.empty(iterations, dtype=float)
    ruin_count = 0
    for i in range(iterations):
        starts = rng.integers(0, n, size=draws)
        index = (starts[:, None] + offsets) % n
        sampled = returns[index.reshape(-1)[:n]]
        equity_path = initial_capital * np.cumprod(1.0 + sampled)
        terminal_returns[i] = equity_path[-1] / initial_capital - 1.0
        drawdowns[i] = _drawdown(np.concatenate(([initial_capital], equity_path)))
        if float(equity_path.min()) <= ruin_level:
            ruin_count += 1

    return EquityMonteCarloResult(
        iterations=iterations,
        seed=seed,
        block=span,
        bars=n,
        initial_capital=initial_capital,
        median_return=float(np.median(terminal_returns)),
        return_p05=float(np.percentile(terminal_returns, 5)),
        return_p95=float(np.percentile(terminal_returns, 95)),
        median_drawdown=float(np.median(drawdowns)),
        drawdown_p05=float(np.percentile(drawdowns, 5)),
        worst_drawdown=float(drawdowns.min()),
        probability_of_profit=float(np.mean(terminal_returns > 0)),
        risk_of_ruin=float(ruin_count / iterations),
    )


def equity_monte_carlo_metrics(result: EquityMonteCarloResult) -> dict[str, float]:
    """Flatten the result into ``mc_``-prefixed keys for a backtest metrics dict.

    The keys are declared in ``metrics._RESULT_VIEW_ORDER``, so the terminal
    table and the HTML tear-sheet both render them with no renderer edit.
    """
    return {
        "mc_iterations": result.iterations,
        "mc_block": result.block,
        "mc_median_return": result.median_return,
        "mc_return_p05": result.return_p05,
        "mc_return_p95": result.return_p95,
        "mc_median_drawdown": result.median_drawdown,
        "mc_drawdown_p05": result.drawdown_p05,
        "mc_worst_drawdown": result.worst_drawdown,
        "mc_probability_of_profit": result.probability_of_profit,
        "mc_risk_of_ruin": result.risk_of_ruin,
    }
