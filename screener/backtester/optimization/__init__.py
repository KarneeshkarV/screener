"""Parameter optimization tools for the backtester."""

from screener.backtester.optimization.grid import (
    GridSearchResult,
    grid_search,
    parameter_combinations,
)
from screener.backtester.optimization.metrics import risk_adjusted_return
from screener.backtester.optimization.monte_carlo import (
    EquityMonteCarloPaths,
    EquityMonteCarloResult,
    MonteCarloResult,
    equity_monte_carlo_metrics,
    simulate_equity_monte_carlo,
    simulate_equity_monte_carlo_paths,
    simulate_monte_carlo,
)
from screener.backtester.optimization.research_report import (
    compute_parameter_stability,
    run_research_report,
)
from screener.backtester.optimization.walk_forward import (
    WalkForwardResult,
    WalkForwardSummary,
    WalkForwardWindow,
    generate_walk_forward_windows,
    walk_forward_optimize,
)

__all__ = [
    "EquityMonteCarloPaths",
    "EquityMonteCarloResult",
    "GridSearchResult",
    "MonteCarloResult",
    "WalkForwardResult",
    "WalkForwardSummary",
    "WalkForwardWindow",
    "compute_parameter_stability",
    "equity_monte_carlo_metrics",
    "generate_walk_forward_windows",
    "grid_search",
    "parameter_combinations",
    "risk_adjusted_return",
    "run_research_report",
    "simulate_equity_monte_carlo",
    "simulate_equity_monte_carlo_paths",
    "simulate_monte_carlo",
    "walk_forward_optimize",
]
