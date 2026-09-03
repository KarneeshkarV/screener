"""Monte Carlo stress testing for trade lists and equity curves."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from screener.backtester.metrics import bar_returns
from screener.backtester.models import Trade

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class MonteCarloOutcome(BaseModel):
    """The numbers every Monte Carlo run reports, whatever it resampled.

    Shared by the trade bootstrap and the equity bootstrap so a metric cannot
    be added to one and silently missed by the other; each subclass adds only
    the fields that describe *its* draw.
    """

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

    @classmethod
    def undefined(cls, **fields: Any) -> Self:
        """Build the result of a run that had nothing to resample.

        Every outcome is NaN rather than zero, because zero is a claim. These
        metrics are merged into the backtest's own, so a zeroed run prints
        "MC Probability of Profit 0.00%" and "MC Risk of Ruin 0.00%" beside
        the realized run's genuine return, asserting a measured 0% chance of
        profit for a simulation that never drew a path. NaN is what
        ``format_result_value`` already renders as "-" and what ``_json_safe``
        already writes as ``null``, so both surfaces say "no distribution".
        """
        return cls(
            median_return=math.nan,
            return_p05=math.nan,
            return_p95=math.nan,
            median_drawdown=math.nan,
            drawdown_p05=math.nan,
            worst_drawdown=math.nan,
            probability_of_profit=math.nan,
            risk_of_ruin=math.nan,
            **fields,
        )


class MonteCarloResult(MonteCarloOutcome):
    """Bootstrap of a trade list, resampling completed trades with replacement."""


class MonteCarloArgumentError(ValueError):
    """A rejected simulation argument, tagged with the argument it names.

    The bounds live here and not in a second hand-maintained copy inside the
    ``backtest-monte-carlo`` command, so the command has to restate them in
    its own vocabulary: the user typed ``--block``, not ``block``. Carrying
    ``field`` and ``problem`` separately is what makes that restatement a
    lookup instead of a string-prefix match on the message, which silently
    reverts to the engine's wording the first time a message is reworded.
    """

    def __init__(self, field: str, problem: str) -> None:
        super().__init__(f"{field} {problem}")
        self.field = field
        self.problem = problem

    def named(self, name: str) -> str:
        """Restate the problem using the caller's name for the argument."""
        return f"{name} {self.problem}"


def validate_equity_monte_carlo_flags(
    *,
    iterations: int,
    block: int,
    seed: int,
    keep_paths: int,
    ruin_threshold: float,
) -> None:
    """Reject an out-of-range simulation argument.

    Split out of the simulation so a caller that runs something expensive
    first can check the arguments before paying for it: the CLI runs a full
    rolling backtest, and a bad flag must not surface as a traceback after
    minutes of work.
    """
    if iterations <= 0:
        raise MonteCarloArgumentError("iterations", "must be positive")
    if block <= 0:
        raise MonteCarloArgumentError("block", "must be positive")
    if seed < 0:
        raise MonteCarloArgumentError("seed", "must not be negative")
    if keep_paths < 0:
        raise MonteCarloArgumentError("keep_paths", "must not be negative")
    if not 0.0 < ruin_threshold <= 1.0:
        raise MonteCarloArgumentError(
            "ruin_threshold", "must be a fraction of starting capital in (0, 1]"
        )


def _validate_equity_curve(equity: "pd.Series") -> None:
    """Reject a curve the bootstrap cannot resample into a real distribution.

    Checked across the whole curve, not just its first bar. A hole or a zero
    anywhere turns one bar return into inf or NaN, and ``np.cumprod`` then
    poisons every path drawn through it: the metrics come back NaN, the table
    renders them as "-" and the JSON payload carries a bare NaN token. Naming
    the offending bar is the difference between a fixable report and a silent
    one.
    """
    values = equity.to_numpy(dtype=float)
    holes = np.flatnonzero(~np.isfinite(values))
    if holes.size:
        first = int(holes[0])
        raise ValueError(
            f"equity curve must be finite at every bar: "
            f"{values[first]} at position {first}"
        )
    non_positive = np.flatnonzero(values <= 0.0)
    if non_positive.size:
        first = int(non_positive[0])
        raise ValueError(
            f"equity curve must be positive at every bar: "
            f"{values[first]} at position {first}"
        )


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
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if seed < 0:
        raise ValueError("seed must not be negative")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    rng = np.random.default_rng(seed)
    returns = np.array([float(t.return_pct) for t in trades], dtype=float)
    if returns.size == 0:
        return MonteCarloResult.undefined(
            iterations=iterations,
            seed=seed,
            initial_capital=initial_capital,
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


class EquityMonteCarloResult(MonteCarloOutcome):
    """Block-bootstrap stress test of a backtest's daily equity curve.

    Reported separately from :class:`MonteCarloResult` because the two resample
    different things. The trade bootstrap chains completed trades one after the
    other, which is only faithful to a single-slot strategy; a rolling backtest
    holds ``top`` positions at once, so its trades overlap in time and chaining
    them compounds a portfolio that never existed. Resampling the equity curve
    instead keeps the concurrency, the position sizing and the cash drag that
    the simulation actually produced.
    """

    block: int
    bars: int
    # Carried so a reader can tell which threshold produced ``risk_of_ruin``;
    # the number is meaningless without it.
    ruin_threshold: float


def _empty_equity_result(
    *, iterations: int, seed: int, ruin_threshold: float, initial_capital: float
) -> EquityMonteCarloResult:
    return EquityMonteCarloResult.undefined(
        iterations=iterations,
        seed=seed,
        # No bars, so no block was ever drawn. Reporting the requested block
        # here would claim a span the run never resampled.
        block=0,
        bars=0,
        ruin_threshold=ruin_threshold,
        initial_capital=initial_capital,
    )


# The percentile bands a fan chart draws around the simulated paths.
_BAND_PERCENTILES = (5, 50, 95)
# Cap on the buffer the bands are computed from. 25M float32 cells is ~100 MB,
# and ``np.percentile`` allocates a sort copy on top of it. The default run,
# 5,000 iterations x 2,520 daily bars, is 12.6M cells, so it stays exact; only
# a longer or larger request falls back to a stride.
_BAND_CELL_BUDGET = 25_000_000


# ``eq=False``: every field is a numpy array, so the generated ``__eq__``
# would compare field tuples and raise "truth value of an array ... is
# ambiguous", and the ``frozen=True`` ``__hash__`` built from the same tuple
# would raise "unhashable type: numpy.ndarray". Inheriting object identity
# instead keeps ``a == b``, ``x in [...]``, a dict key and ``lru_cache`` from
# blowing up on a container whose fields have no elementwise truth value.
@dataclass(frozen=True, eq=False)
class EquityMonteCarloPaths:
    """The raw draws behind an :class:`EquityMonteCarloResult`.

    ``terminal_returns`` and ``drawdowns`` hold one entry per iteration, so any
    distribution plotted from them is exact. ``paths`` is capped instead:
    ``iterations x bars`` equity levels is ~100 MB of float64 on a long daily
    run, and no browser draws 5,000 lines. The cap keeps the first iterations
    rather than a random subset, so the retained sample is reproducible from
    ``seed`` alone.

    ``bands`` is *not* derived from that capped sample, which is the whole
    reason it is computed here rather than in the renderer. Percentiles taken
    over the retained subset answer a different question from the ones in the
    summary table, and at ``keep_paths=1`` all three collapse onto the single
    stored path while the table still reports a spread. These are taken over
    every iteration the budget allows, so the chart and the table agree.

    ``bands`` holds equity *levels*, shaped ``(len(band_percentiles), bars + 1)``.
    Column 0 is ``initial_capital`` for every band, so band column ``i`` lines
    up with bar ``i`` of the realized curve. ``band_iterations`` records how
    many iterations went into them, which equals ``iterations`` unless the
    budget forced a stride.
    """

    initial_capital: float
    paths: np.ndarray
    terminal_returns: np.ndarray
    drawdowns: np.ndarray
    band_percentiles: tuple[int, ...]
    bands: np.ndarray
    band_iterations: int


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
            band_percentiles=_BAND_PERCENTILES,
            # No bars, so there is no column to hold a band level. The row
            # count still matches ``band_percentiles`` so a reader can index
            # by band without checking for the empty case first.
            bands=np.empty((len(_BAND_PERCENTILES), 0), dtype=float),
            band_iterations=0,
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
    validate_equity_monte_carlo_flags(
        iterations=iterations,
        block=block,
        seed=seed,
        keep_paths=keep_paths,
        ruin_threshold=ruin_threshold,
    )
    if len(equity) == 0:
        return _empty_equity_monte_carlo_paths(
            iterations=iterations,
            seed=seed,
            ruin_threshold=ruin_threshold,
            initial_capital=0.0,
        )

    _validate_equity_curve(equity)
    initial_capital = float(equity.iloc[0])
    # The curve is finite and positive at every bar, so no return is dropped
    # here and ``bars`` is always ``len(equity) - 1``. A report can therefore
    # plot a simulated path against the realized curve on one bar axis without
    # the two ending at different bars.
    returns = bar_returns(equity).to_numpy(dtype=float)
    n = int(returns.size)
    if n < 2:
        # One return resamples to itself however it is blocked, so there is no
        # distribution to report. Handled like an absent curve rather than as
        # an error: a window this short is a thin backtest, not a bad flag.
        return _empty_equity_monte_carlo_paths(
            iterations=iterations,
            seed=seed,
            ruin_threshold=ruin_threshold,
            initial_capital=initial_capital,
        )
    if block >= n:
        # A circular block as long as the series is a rotation of it, and the
        # product of a rotated return series is the series' own product, so
        # every path would land on the identical terminal return. Capping the
        # block silently, as this used to, published that point mass as a
        # p05/p95 spread and made a mistyped block read as certainty.
        raise MonteCarloArgumentError(
            "block",
            f"must be shorter than the {n} resampled bars "
            f"({len(equity)} bars of equity); a block that long rotates the "
            f"whole curve, so every path repeats the realized result",
        )

    rng = np.random.default_rng(seed)
    draws = -(-n // block)  # ceil division: blocks needed to cover the window
    offsets = np.arange(block)
    ruin_level = initial_capital * ruin_threshold

    terminal_returns = np.empty(iterations, dtype=float)
    drawdowns = np.empty(iterations, dtype=float)
    kept = min(keep_paths, iterations)
    # The bands come from every iteration, not from the retained sample, so
    # they agree with the summary percentiles. Only a run past the cell budget
    # strides, and ``band_iterations`` then says how many it really used.
    band_rows = min(iterations, max(1, _BAND_CELL_BUDGET // n))
    band_stride = -(-iterations // band_rows)
    band_iterations = -(-iterations // band_stride)
    # float32 halves both buffers; a chart cannot resolve more precision.
    band_buffer = np.empty((band_iterations, n), dtype=np.float32)
    # At ``band_stride == 1`` -- the default run, and anything else inside the
    # cell budget -- the band buffer already holds every path in iteration
    # order, so the retained sample is exactly its first ``kept`` rows. Writing
    # them a second time inside the loop copies ~10 MB for nothing. Sliced
    # after the loop instead, and copied rather than viewed so the far larger
    # band buffer is freed when this function returns.
    stored = None if band_stride == 1 else np.empty((kept, n), dtype=np.float32)
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
        if stored is not None and i < kept:
            stored[i] = equity_path
        if i % band_stride == 0:
            band_buffer[i // band_stride] = equity_path

    if stored is None:
        stored = band_buffer[:kept].copy()

    bands = np.empty((len(_BAND_PERCENTILES), n + 1), dtype=float)
    # Every path starts at the capital the real run started with, so bar 0 is
    # that level for all three bands rather than a percentile of nothing.
    bands[:, 0] = initial_capital
    bands[:, 1:] = np.percentile(band_buffer, _BAND_PERCENTILES, axis=0)

    result = EquityMonteCarloResult(
        iterations=iterations,
        seed=seed,
        block=block,
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
        band_percentiles=_BAND_PERCENTILES,
        bands=bands,
        band_iterations=band_iterations,
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
