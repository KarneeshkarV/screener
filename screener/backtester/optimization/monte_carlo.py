"""Monte Carlo stress testing for trade lists and equity curves."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
    # Carried so a reader can tell which threshold produced ``risk_of_ruin``;
    # the number is meaningless without it.
    ruin_threshold: float
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
    *, iterations: int, seed: int, ruin_threshold: float, initial_capital: float
) -> EquityMonteCarloResult:
    return EquityMonteCarloResult(
        iterations=iterations,
        seed=seed,
        # No bars, so no block was ever drawn. Reporting the requested block
        # here would claim a span the run did not use, which the normal path
        # caps at the number of bars.
        block=0,
        bars=0,
        ruin_threshold=ruin_threshold,
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


@dataclass(frozen=True)
class EquityMonteCarloPaths:
    """The raw draws behind an :class:`EquityMonteCarloResult`.

    ``terminal_returns`` and ``drawdowns`` hold one entry per iteration, so any
    distribution plotted from them is exact. ``paths`` is capped instead:
    ``iterations x bars`` equity levels is ~100 MB of float64 on a long daily
    run, and no browser draws 5,000 lines. The cap keeps the first iterations
    rather than a random subset, so the retained sample is reproducible from
    ``seed`` alone.
    """

    initial_capital: float
    paths: np.ndarray
    terminal_returns: np.ndarray
    drawdowns: np.ndarray


def _empty_equity_monte_carlo_paths(
    *, iterations: int, seed: int, ruin_threshold: float, initial_capital: float
) -> tuple[EquityMonteCarloResult, EquityMonteCarloPaths]:
    empty = np.empty(0, dtype=float)
    return (
        _empty_equity_result(
            iterations=iterations,
            seed=seed,
            ruin_threshold=ruin_threshold,
            initial_capital=initial_capital,
        ),
        EquityMonteCarloPaths(
            initial_capital=initial_capital,
            paths=np.empty((0, 0), dtype=np.float32),
            terminal_returns=empty,
            drawdowns=empty,
        ),
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
    """
    result, _ = simulate_equity_monte_carlo_paths(
        equity,
        iterations=iterations,
        block=block,
        seed=seed,
        ruin_threshold=ruin_threshold,
        keep_paths=0,
    )
    return result


def simulate_equity_monte_carlo_paths(
    equity: "pd.Series",
    *,
    iterations: int = 5000,
    block: int = 20,
    seed: int = 42,
    ruin_threshold: float = 0.5,
    keep_paths: int = 1000,
) -> tuple[EquityMonteCarloResult, EquityMonteCarloPaths]:
    """Run the bootstrap and hand back the draws as well as the summary.

    Same simulation as :func:`simulate_equity_monte_carlo`. This variant exists
    so a report can plot the fan of simulated curves and the outcome
    distributions, which the summary percentiles alone cannot show.

    Iterations are looped rather than vectorized: a fully vectorized draw
    allocates ``iterations x bars`` indices at once, which is hundreds of MB on
    a multi-year daily run.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if block <= 0:
        raise ValueError("block must be positive")
    if keep_paths < 0:
        raise ValueError("keep_paths must not be negative")
    if not 0.0 < ruin_threshold <= 1.0:
        raise ValueError("ruin_threshold must be in (0, 1]")
    if len(equity) == 0:
        return _empty_equity_monte_carlo_paths(
            iterations=iterations,
            seed=seed,
            ruin_threshold=ruin_threshold,
            initial_capital=0.0,
        )

    initial_capital = float(equity.iloc[0])
    if initial_capital <= 0:
        raise ValueError("equity curve must start above zero")

    returns = equity.pct_change().dropna().to_numpy(dtype=float)
    n = int(returns.size)
    if n == 0:
        return _empty_equity_monte_carlo_paths(
            iterations=iterations,
            seed=seed,
            ruin_threshold=ruin_threshold,
            initial_capital=initial_capital,
        )

    rng = np.random.default_rng(seed)
    span = min(block, n)
    draws = -(-n // span)  # ceil division: blocks needed to cover the window
    offsets = np.arange(span)
    ruin_level = initial_capital * ruin_threshold

    terminal_returns = np.empty(iterations, dtype=float)
    drawdowns = np.empty(iterations, dtype=float)
    kept = min(keep_paths, iterations)
    # float32 halves the retained sample; a chart cannot resolve more precision.
    stored = np.empty((kept, n), dtype=np.float32)
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
        if i < kept:
            stored[i] = equity_path

    result = EquityMonteCarloResult(
        iterations=iterations,
        seed=seed,
        block=span,
        bars=n,
        ruin_threshold=ruin_threshold,
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
    paths = EquityMonteCarloPaths(
        initial_capital=initial_capital,
        paths=stored,
        terminal_returns=terminal_returns,
        drawdowns=drawdowns,
    )
    return result, paths


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
        "mc_ruin_threshold": result.ruin_threshold,
    }
