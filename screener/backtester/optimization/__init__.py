"""Parameter optimization tools for the backtester."""
from screener.backtester.optimization.grid import GridSearchResult, grid_search, parameter_combinations
from screener.backtester.optimization.metrics import (
    calmar_ratio,
    expectancy,
    maximum_drawdown,
    profit_factor,
    risk_adjusted_return,
    sharpe_ratio,
    win_rate,
)
from screener.backtester.optimization.monte_carlo import MonteCarloResult, simulate_monte_carlo
from screener.backtester.optimization.walk_forward import (
    WalkForwardResult,
    WalkForwardSummary,
    WalkForwardWindow,
    generate_walk_forward_windows,
    walk_forward_optimize,
)

__all__ = [
    "GridSearchResult",
    "MonteCarloResult",
    "WalkForwardResult",
    "WalkForwardSummary",
    "WalkForwardWindow",
    "calmar_ratio",
    "expectancy",
    "generate_walk_forward_windows",
    "grid_search",
    "maximum_drawdown",
    "parameter_combinations",
    "profit_factor",
    "risk_adjusted_return",
    "sharpe_ratio",
    "simulate_monte_carlo",
    "walk_forward_optimize",
    "win_rate",
]
