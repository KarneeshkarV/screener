"""Monte Carlo simulation for backtest trade resampling."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from screener.backtester.metrics import compute_metrics
from screener.backtester.models import Trade
from screener.backtester.portfolio import build_equity_curve


TRADING_DAYS_PER_YEAR = 252


def _trade_returns(trades: list[Trade]) -> np.ndarray:
    """Return array of per-trade return_pct."""
    return np.array([t.return_pct for t in trades], dtype=float)


def _rebuild_equity_from_returns(
    returns: np.ndarray,
    initial_capital: float,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Rebuild equity by sequentially applying returns."""
    equity = np.empty(len(returns) + 1, dtype=float)
    equity[0] = initial_capital
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    # Stretch to calendar length (simple replication for visual alignment)
    if len(equity) >= len(calendar):
        aligned = equity[: len(calendar)]
    else:
        aligned = np.concatenate(
            [equity, np.full(len(calendar) - len(equity), equity[-1])]
        )
    return pd.Series(aligned, index=calendar[: len(aligned)], dtype=float)


def _rebuild_equity_from_trades(
    trades: list[Trade],
    calendar: pd.DatetimeIndex,
    bars_by_tv: dict[str, pd.DataFrame],
    initial_capital: float,
) -> pd.Series:
    """Rebuild equity curve from a reordered list of trades."""
    if not trades:
        return pd.Series(initial_capital, index=calendar, dtype=float)
    return build_equity_curve(calendar, trades, bars_by_tv, initial_capital)


@dataclass
class MonteCarloResult:
    method: str
    n_runs: int
    max_drawdowns: np.ndarray
    final_equities: np.ndarray
    probabilities_of_profit: float
    var_95: float
    cvar_95: float
    median_max_dd: float
    p95_max_dd: float
    median_final_equity: float
    p05_final_equity: float
    p95_final_equity: float
    median_sharpe: float
    percentile_breakdown: dict = field(default_factory=dict)


class MonteCarloSimulator:
    """Resample backtest trades to estimate distributions of key metrics."""

    def __init__(
        self,
        trades: list[Trade],
        initial_capital: float,
        calendar: Optional[pd.DatetimeIndex] = None,
        bars_by_tv: Optional[dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.trades = list(trades)
        self.initial_capital = float(initial_capital)
        self.calendar = calendar
        self.bars_by_tv = bars_by_tv or {}
        if self.calendar is None:
            # Build a minimal calendar from trade dates
            dates: set[pd.Timestamp] = set()
            for t in self.trades:
                dates.add(pd.Timestamp(t.entry_date))
                dates.add(pd.Timestamp(t.exit_date))
            if dates:
                self.calendar = pd.DatetimeIndex(sorted(dates))
            else:
                self.calendar = pd.DatetimeIndex([])

    def trade_shuffle(
        self,
        n_runs: int = 1000,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """Randomize trade order and rebuild equity curves."""
        rng = np.random.default_rng(seed)
        max_dds = np.empty(n_runs, dtype=float)
        final_equities = np.empty(n_runs, dtype=float)
        sharpes = np.empty(n_runs, dtype=float)

        for i in range(n_runs):
            shuffled = list(self.trades)
            rng.shuffle(shuffled)
            equity = _rebuild_equity_from_trades(
                shuffled, self.calendar, self.bars_by_tv, self.initial_capital
            )
            max_dds[i] = self._max_dd(equity)
            final_equities[i] = float(equity.iloc[-1]) if len(equity) else self.initial_capital
            sharpes[i] = self._sharpe(equity)

        return self._summarize("trade_shuffle", n_runs, max_dds, final_equities, sharpes)

    def returns_bootstrap(
        self,
        n_runs: int = 1000,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """Bootstrap sample trade returns with replacement."""
        rng = np.random.default_rng(seed)
        returns = _trade_returns(self.trades)
        if returns.size == 0:
            zeros = np.full(n_runs, self.initial_capital)
            return self._summarize("returns_bootstrap", n_runs, np.zeros(n_runs), zeros, np.zeros(n_runs))

        max_dds = np.empty(n_runs, dtype=float)
        final_equities = np.empty(n_runs, dtype=float)
        sharpes = np.empty(n_runs, dtype=float)

        for i in range(n_runs):
            sample = rng.choice(returns, size=returns.size, replace=True)
            equity = _rebuild_equity_from_returns(
                sample, self.initial_capital, self.calendar
            )
            max_dds[i] = self._max_dd(equity)
            final_equities[i] = float(equity.iloc[-1]) if len(equity) else self.initial_capital
            sharpes[i] = self._sharpe(equity)

        return self._summarize("returns_bootstrap", n_runs, max_dds, final_equities, sharpes)

    def block_bootstrap(
        self,
        block_size: int = 20,
        n_runs: int = 1000,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """Block-bootstrap to preserve serial correlation."""
        rng = np.random.default_rng(seed)
        returns = _trade_returns(self.trades)
        n = returns.size
        if n == 0:
            zeros = np.full(n_runs, self.initial_capital)
            return self._summarize("block_bootstrap", n_runs, np.zeros(n_runs), zeros, np.zeros(n_runs))

        max_dds = np.empty(n_runs, dtype=float)
        final_equities = np.empty(n_runs, dtype=float)
        sharpes = np.empty(n_runs, dtype=float)

        for i in range(n_runs):
            blocks_needed = int(np.ceil(n / block_size))
            blocks: list[np.ndarray] = []
            for _ in range(blocks_needed):
                start = rng.integers(0, max(n - block_size + 1, 1))
                blocks.append(returns[start : start + block_size])
            sample = np.concatenate(blocks)[:n]
            equity = _rebuild_equity_from_returns(
                sample, self.initial_capital, self.calendar
            )
            max_dds[i] = self._max_dd(equity)
            final_equities[i] = float(equity.iloc[-1]) if len(equity) else self.initial_capital
            sharpes[i] = self._sharpe(equity)

        return self._summarize("block_bootstrap", n_runs, max_dds, final_equities, sharpes)

    @staticmethod
    def _max_dd(equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return float(dd.min())

    @staticmethod
    def _sharpe(equity: pd.Series) -> float:
        if equity.empty or len(equity) < 2:
            return 0.0
        daily = equity.pct_change().dropna()
        if daily.empty or daily.std(ddof=0) == 0:
            return 0.0
        return float(daily.mean() / daily.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))

    def _summarize(
        self,
        method: str,
        n_runs: int,
        max_dds: np.ndarray,
        final_equities: np.ndarray,
        sharpes: np.ndarray,
    ) -> MonteCarloResult:
        prob_profit = float(np.mean(final_equities > self.initial_capital))
        var_95 = float(np.percentile(final_equities, 5))
        cvar_95 = float(final_equities[final_equities <= var_95].mean()) if np.any(final_equities <= var_95) else var_95
        return MonteCarloResult(
            method=method,
            n_runs=n_runs,
            max_drawdowns=max_dds,
            final_equities=final_equities,
            probabilities_of_profit=prob_profit,
            var_95=var_95,
            cvar_95=cvar_95,
            median_max_dd=float(np.percentile(max_dds, 50)),
            p95_max_dd=float(np.percentile(max_dds, 95)),
            median_final_equity=float(np.percentile(final_equities, 50)),
            p05_final_equity=float(np.percentile(final_equities, 5)),
            p95_final_equity=float(np.percentile(final_equities, 95)),
            median_sharpe=float(np.percentile(sharpes, 50)),
            percentile_breakdown={
                "max_drawdown": {
                    "p5": float(np.percentile(max_dds, 5)),
                    "p25": float(np.percentile(max_dds, 25)),
                    "p50": float(np.percentile(max_dds, 50)),
                    "p75": float(np.percentile(max_dds, 75)),
                    "p95": float(np.percentile(max_dds, 95)),
                },
                "final_equity": {
                    "p5": float(np.percentile(final_equities, 5)),
                    "p25": float(np.percentile(final_equities, 25)),
                    "p50": float(np.percentile(final_equities, 50)),
                    "p75": float(np.percentile(final_equities, 75)),
                    "p95": float(np.percentile(final_equities, 95)),
                },
                "sharpe": {
                    "p5": float(np.percentile(sharpes, 5)),
                    "p25": float(np.percentile(sharpes, 25)),
                    "p50": float(np.percentile(sharpes, 50)),
                    "p75": float(np.percentile(sharpes, 75)),
                    "p95": float(np.percentile(sharpes, 95)),
                },
            },
        )


def print_monte_carlo(result: MonteCarloResult) -> None:
    console = Console()
    console.print(
        Panel.fit(
            f"[bold]Monte Carlo[/bold]  method=[cyan]{result.method}[/cyan]  "
            f"runs=[green]{result.n_runs}[/green]"
        )
    )

    table = Table(title="Distributions", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row(
        "Median Max Drawdown",
        f"{result.median_max_dd * 100:.2f}%",
    )
    table.add_row(
        "95th Percentile Max Drawdown",
        f"{result.p95_max_dd * 100:.2f}%",
    )
    table.add_row(
        "Median Final Equity",
        f"{result.median_final_equity:,.2f}",
    )
    table.add_row(
        "5th Percentile Final Equity",
        f"{result.p05_final_equity:,.2f}",
    )
    table.add_row(
        "95th Percentile Final Equity",
        f"{result.p95_final_equity:,.2f}",
    )
    table.add_row(
        "Probability of Profit",
        f"{result.probabilities_of_profit:.1%}",
    )
    table.add_row(
        "Value at Risk (95%)",
        f"{result.var_95:,.2f}",
    )
    table.add_row(
        "Conditional VaR (95%)",
        f"{result.cvar_95:,.2f}",
    )
    table.add_row(
        "Median Sharpe",
        f"{result.median_sharpe:.3f}",
    )
    console.print(table)


def run_monte_carlo_from_result(
    result: "screener.backtester.models.BacktestResult",
    method: Literal["shuffle", "returns", "block"],
    n_runs: int = 1000,
    block_size: int = 20,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """Convenience wrapper that builds a simulator from a BacktestResult."""
    sim = MonteCarloSimulator(
        trades=result.trades,
        initial_capital=result.config.initial_capital,
        calendar=result.equity_curve.index,
        bars_by_tv={},
    )
    if method == "shuffle":
        return sim.trade_shuffle(n_runs=n_runs, seed=seed)
    if method == "returns":
        return sim.returns_bootstrap(n_runs=n_runs, seed=seed)
    return sim.block_bootstrap(block_size=block_size, n_runs=n_runs, seed=seed)


@click.command(name="monte-carlo")
@click.option(
    "--backtest-results",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSON file containing trades + equity curve.",
)
@click.option(
    "--method",
    type=click.Choice(["shuffle", "returns", "block"]),
    default="shuffle",
)
@click.option("--runs", type=int, default=1000, show_default=True)
@click.option("--block-size", type=int, default=20, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--json-out", default=None, help="Write result JSON to path.")
def monte_carlo(backtest_results, method, runs, block_size, seed, json_out):
    """Run Monte Carlo simulation on a saved backtest result."""
    import pathlib

    data = json.loads(pathlib.Path(backtest_results).read_text())
    trades = [Trade(**t) for t in data.get("trades", [])]
    initial_capital = float(data.get("initial_capital", 100_000.0))
    calendar = pd.DatetimeIndex(
        pd.to_datetime(data.get("calendar", []))
    )

    sim = MonteCarloSimulator(
        trades=trades,
        initial_capital=initial_capital,
        calendar=calendar,
        bars_by_tv={},
    )
    if method == "shuffle":
        result = sim.trade_shuffle(n_runs=runs, seed=seed)
    elif method == "returns":
        result = sim.returns_bootstrap(n_runs=runs, seed=seed)
    else:
        result = sim.block_bootstrap(block_size=block_size, n_runs=runs, seed=seed)

    print_monte_carlo(result)

    if json_out:
        out = {
            "method": result.method,
            "n_runs": result.n_runs,
            "median_max_dd": result.median_max_dd,
            "p95_max_dd": result.p95_max_dd,
            "median_final_equity": result.median_final_equity,
            "p05_final_equity": result.p05_final_equity,
            "p95_final_equity": result.p95_final_equity,
            "probability_of_profit": result.probabilities_of_profit,
            "var_95": result.var_95,
            "cvar_95": result.cvar_95,
            "median_sharpe": result.median_sharpe,
            "percentile_breakdown": result.percentile_breakdown,
        }
        pathlib.Path(json_out).write_text(json.dumps(out, indent=2, default=str))
