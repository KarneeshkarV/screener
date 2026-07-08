"""Fast vectorbt parameter sweeps for strategy exploration.

This module is for **fast exploration**, not validation. Results may diverge
from ``backtest-rolling`` because the following are **not** modeled here:

- slot allocation / position sizing rules
- partial exits
- dividends
- custom slippage models (``slippage=0.0`` hard-coded)
- commissions / fees (``fees=0.0`` hard-coded)
- trailing stops, stop-loss, take-profit (custom engine supports all three)

Entries/exits are shifted by one bar and filled at the next bar's **open** to
match the custom engine's MOO (market-on-open) semantics — so there is no
same-bar look-ahead, but residual differences vs MOC-style exits still apply.

Always validate promising parameter combinations with ``backtest-rolling``
before drawing conclusions.

The heavy sweep implementation lives under ``screener.backtester.vbt``. This
module remains the public CLI/API compatibility surface for existing imports.
"""

from screener.backtester.vbt.config import (
    DEFAULT_BBANDS_STD,
    DEFAULT_BBANDS_WINDOWS,
    DEFAULT_BREAKOUT_WINDOWS,
    DEFAULT_KELTNER_MULT,
    DEFAULT_KELTNER_WINDOWS,
    DEFAULT_OBV_EMA_WINDOWS,
    DEFAULT_RSI_PERIOD,
    DEFAULT_RSI_THRESHOLDS,
    DEFAULT_SUPERTREND_MULT,
    DEFAULT_SUPERTREND_PERIODS,
    DEFAULT_VOL_MA_WINDOW,
    DEFAULT_VOL_MULTIPLIER,
    DEFAULT_MACD_FAST,
    DEFAULT_MACD_SLOW,
    DEFAULT_MACD_SIGNAL,
    DISCLAIMER,
    HL_INDICATORS,
    INITIAL_CAPITAL_DEFAULT,
    MetricName,
    StrategyBuilder,
    StrategyName,
    VALID_INDICATORS,
    VOLUME_INDICATORS,
    WALK_FORWARD_DAYS_PER_MONTH,
    WALK_FORWARD_OOS_COLUMNS,
)
from screener.backtester.vbt.indicators import (
    STRATEGY_BUILDERS,
    IndicatorCombo,
    SignalCache,
    _atr_wilder,
    _build_indicator_signal_panels,
    _fixed_hold_exits,
    _fixed_hold_exits_np,
    _ma_for_window,
    _obv,
    _rsi_wilder,
    _sma,
    _supertrend_signals_np,
    iter_indicator_combos,
    iter_param_combos,
    sma_crossover_signals,
)
from screener.backtester.vbt.panels import (
    _build_column_panel,
    build_close_panel,
    build_high_panel,
    build_low_panel,
    build_open_panel,
    build_volume_panel,
)
from screener.backtester.vbt.render import (
    _fmt_int_or_dash,
    print_results_table,
    print_walk_forward_sweep_table,
)
from screener.backtester.vbt.sweep import (
    _combo_metric,
    _portfolio_chunk_metrics,
    _require_vectorbt,
    _scalar_metric,
    rank_results,
    run_combo_backtest,
)
from screener.backtester.vbt.walk_forward import (
    WalkForwardSweepSummary,
    _single_combo_sweep_kwargs,
    _slice_panel,
    parse_walk_forward,
)
from screener.backtester.vbt.cli import (
    parse_indicator_list,
    parse_int_list,
    run_parameter_sweep,
    run_walk_forward_sweep,
    vbt_sweep,
)

__all__ = [
    "DISCLAIMER",
    "DEFAULT_BBANDS_STD",
    "DEFAULT_BBANDS_WINDOWS",
    "DEFAULT_BREAKOUT_WINDOWS",
    "DEFAULT_KELTNER_MULT",
    "DEFAULT_KELTNER_WINDOWS",
    "DEFAULT_OBV_EMA_WINDOWS",
    "DEFAULT_RSI_PERIOD",
    "DEFAULT_RSI_THRESHOLDS",
    "DEFAULT_SUPERTREND_MULT",
    "DEFAULT_SUPERTREND_PERIODS",
    "DEFAULT_VOL_MA_WINDOW",
    "DEFAULT_VOL_MULTIPLIER",
    "DEFAULT_MACD_FAST",
    "DEFAULT_MACD_SLOW",
    "DEFAULT_MACD_SIGNAL",
    "HL_INDICATORS",
    "INITIAL_CAPITAL_DEFAULT",
    "IndicatorCombo",
    "MetricName",
    "STRATEGY_BUILDERS",
    "SignalCache",
    "StrategyBuilder",
    "StrategyName",
    "VALID_INDICATORS",
    "VOLUME_INDICATORS",
    "WALK_FORWARD_DAYS_PER_MONTH",
    "WALK_FORWARD_OOS_COLUMNS",
    "WalkForwardSweepSummary",
    "_atr_wilder",
    "_build_column_panel",
    "_build_indicator_signal_panels",
    "_combo_metric",
    "_fixed_hold_exits",
    "_fixed_hold_exits_np",
    "_fmt_int_or_dash",
    "_ma_for_window",
    "_obv",
    "_portfolio_chunk_metrics",
    "_require_vectorbt",
    "_rsi_wilder",
    "_scalar_metric",
    "_single_combo_sweep_kwargs",
    "_slice_panel",
    "_sma",
    "_supertrend_signals_np",
    "build_close_panel",
    "build_high_panel",
    "build_low_panel",
    "build_open_panel",
    "build_volume_panel",
    "iter_indicator_combos",
    "iter_param_combos",
    "parse_indicator_list",
    "parse_int_list",
    "parse_walk_forward",
    "print_results_table",
    "print_walk_forward_sweep_table",
    "rank_results",
    "run_combo_backtest",
    "run_parameter_sweep",
    "run_walk_forward_sweep",
    "sma_crossover_signals",
    "vbt_sweep",
]
