"""Parameter optimization tools for the backtester."""

from screener.backtester.optimization.cache import (
    FrozenInputIdentity,
    build_frozen_input_identity,
    hash_price_frames,
    universe_membership_identity_from_config,
)
from screener.backtester.optimization.grid import (
    GridSearchResult,
    grid_search,
    parameter_combinations,
)
from screener.backtester.optimization.metrics import risk_adjusted_return
from screener.backtester.optimization.trials import (
    TrialSearchStats,
    default_experiment_id,
    load_trial_search_stats,
    record_trial,
    trial_config_identity,
)
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
    OOS_EVIDENCE_CRITERIA,
    WalkForwardResult,
    WalkForwardSummary,
    WalkForwardWindow,
    generate_walk_forward_windows,
    train_result_eligible,
    walk_forward_optimize,
)

__all__ = [
    "EquityMonteCarloPaths",
    "EquityMonteCarloResult",
    "FrozenInputIdentity",
    "GridSearchResult",
    "MonteCarloResult",
    "TrialSearchStats",
    "WalkForwardResult",
    "WalkForwardSummary",
    "WalkForwardWindow",
    "build_frozen_input_identity",
    "compute_parameter_stability",
    "default_experiment_id",
    "equity_monte_carlo_metrics",
    "OOS_EVIDENCE_CRITERIA",
    "generate_walk_forward_windows",
    "grid_search",
    "hash_price_frames",
    "load_trial_search_stats",
    "parameter_combinations",
    "record_trial",
    "risk_adjusted_return",
    "run_research_report",
    "simulate_equity_monte_carlo",
    "simulate_equity_monte_carlo_paths",
    "simulate_monte_carlo",
    "train_result_eligible",
    "trial_config_identity",
    "universe_membership_identity_from_config",
    "walk_forward_optimize",
]
