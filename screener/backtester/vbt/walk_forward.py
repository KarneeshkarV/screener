"""Walk-forward helpers for vectorbt parameter sweeps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import click
import numpy as np
import pandas as pd

from screener.backtester.optimization.walk_forward import (
    WalkForwardWindow,
    _parameter_stability,
)
from screener.backtester.vbt.config import (
    INITIAL_CAPITAL_DEFAULT,
    MetricName,
    WALK_FORWARD_OOS_COLUMNS,
)
from screener.backtester.vbt.sweep import rank_results, run_parameter_sweep


def parse_walk_forward(raw: str) -> tuple[int, int]:
    """Parse ``--walk-forward TRAIN_MONTHS:TEST_MONTHS`` (e.g. ``12:3``)."""
    parts = raw.split(":")
    if len(parts) != 2:
        raise click.UsageError(
            "--walk-forward expects TRAIN_MONTHS:TEST_MONTHS, e.g. 12:3."
        )
    try:
        train_months, test_months = (int(p.strip()) for p in parts)
    except ValueError as exc:
        raise click.UsageError(
            "--walk-forward expects integer months, e.g. 12:3."
        ) from exc
    if train_months <= 0 or test_months <= 0:
        raise click.UsageError("--walk-forward months must be positive.")
    return train_months, test_months


def _single_combo_sweep_kwargs(
    indicator: str, fast: int, slow: int, hold: int
) -> dict[str, Any]:
    """Sweep kwargs that reproduce exactly one ``(indicator, fast, slow, hold)``."""
    base: dict[str, Any] = {
        "indicators": [indicator],
        "fast_values": [fast],
        "slow_values": [slow],
        "hold_values": [hold],
    }
    if indicator in ("sma", "ema", "sma_rsi", "macd"):
        # macd ignores fast/slow lists (fixed defaults); hold drives the combo.
        return base
    window_kwarg = {
        "breakout": "breakout_windows",
        "vol_breakout": "breakout_windows",
        "breakout_rsi": "breakout_windows",
        "bbands": "bbands_windows",
        "supertrend": "supertrend_periods",
        "keltner": "keltner_windows",
        "rsi": "rsi_thresholds",
        "obv_trend": "obv_ema_windows",
    }.get(indicator)
    if window_kwarg is None:
        raise ValueError(f"Unknown indicator: {indicator!r}")
    base[window_kwarg] = [fast]
    return base


def _slice_panel(
    panel: pd.DataFrame | None, start: date, end: date
) -> pd.DataFrame | None:
    if panel is None:
        return None
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return panel.loc[(panel.index >= s) & (panel.index <= e)]


@dataclass(frozen=True)
class WalkForwardSweepSummary:
    """Per-window walk-forward results plus aggregate diagnostics."""

    metric: str
    windows: pd.DataFrame
    aggregate_is_score: float
    aggregate_oos_score: float
    efficiency: float
    parameter_stability: float
    aggregate_oos_metrics: dict[str, float]


def run_walk_forward_sweep(
    close: pd.DataFrame,
    *,
    windows: list[WalkForwardWindow],
    metric: MetricName,
    grid: dict[str, Any],
    open_: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
    sweep_fn: Callable[..., pd.DataFrame] | None = None,
) -> WalkForwardSweepSummary:
    """Walk-forward validation of the vbt sweep grid.

    For each window: run the full sweep ``grid`` on the training slice, pick
    the in-sample winner by ``metric``, then evaluate exactly those params on
    the out-of-sample test slice. Aggregate OOS metrics are trade-weighted;
    walk-forward efficiency is mean(OOS metric) / |mean(IS metric)|.
    """
    fn = sweep_fn if sweep_fn is not None else run_parameter_sweep
    rows: list[dict[str, Any]] = []
    is_scores: list[float] = []
    oos_scores: list[float] = []
    param_sets: list[dict[str, Any]] = []
    weighted: dict[str, float] = {}
    weights: dict[str, float] = {}
    total_oos_trades = 0

    for window in windows:
        train_close = _slice_panel(close, window.train_start, window.train_end)
        test_close = _slice_panel(close, window.test_start, window.test_end)
        if (
            train_close is None
            or train_close.empty
            or test_close is None
            or test_close.empty
        ):
            continue
        train_df = fn(
            train_close,
            open_=_slice_panel(open_, window.train_start, window.train_end),
            high=_slice_panel(high, window.train_start, window.train_end),
            low=_slice_panel(low, window.train_start, window.train_end),
            volume=_slice_panel(volume, window.train_start, window.train_end),
            initial_capital=initial_capital,
            **grid,
        )
        ranked = rank_results(train_df, metric)
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        is_score = float(best[metric])
        if not np.isfinite(is_score):
            continue
        indicator = str(best["indicator"]) if "indicator" in ranked.columns else "sma"
        fast = int(best["fast"])
        try:
            slow_float = float(best["slow"])
            slow = int(slow_float) if np.isfinite(slow_float) else 0
        except (KeyError, TypeError, ValueError):
            slow = 0
        hold = int(best["hold"])
        test_df = fn(
            test_close,
            open_=_slice_panel(open_, window.test_start, window.test_end),
            high=_slice_panel(high, window.test_start, window.test_end),
            low=_slice_panel(low, window.test_start, window.test_end),
            volume=_slice_panel(volume, window.test_start, window.test_end),
            initial_capital=initial_capital,
            **_single_combo_sweep_kwargs(indicator, fast, slow, hold),
        )
        if test_df.empty:
            continue
        oos = test_df.iloc[0]
        oos_score = float(oos[metric])
        oos_trades = int(oos["trades"])
        row: dict[str, Any] = {
            "train_start": window.train_start,
            "train_end": window.train_end,
            "test_start": window.test_start,
            "test_end": window.test_end,
            "indicator": indicator,
            "fast": fast,
            "slow": best["slow"],
            "hold": hold,
            "is_score": is_score,
            "oos_score": oos_score,
        }
        for col in WALK_FORWARD_OOS_COLUMNS:
            row[f"oos_{col}"] = float(oos[col])
        row["oos_trades"] = oos_trades
        rows.append(row)
        is_scores.append(is_score)
        oos_scores.append(oos_score if np.isfinite(oos_score) else 0.0)
        param_sets.append(
            {"indicator": indicator, "fast": fast, "slow": slow, "hold": hold}
        )
        total_oos_trades += oos_trades
        weight = max(oos_trades, 1)
        for col in WALK_FORWARD_OOS_COLUMNS:
            value = float(oos[col])
            if np.isfinite(value):
                weighted[col] = weighted.get(col, 0.0) + value * weight
                weights[col] = weights.get(col, 0.0) + weight

    aggregate_oos_metrics = {col: weighted[col] / weights[col] for col in weighted}
    aggregate_oos_metrics["trades"] = float(total_oos_trades)
    is_avg = float(np.mean(is_scores)) if is_scores else 0.0
    oos_avg = float(np.mean(oos_scores)) if oos_scores else 0.0
    efficiency = oos_avg / max(abs(is_avg), 1e-9) if is_scores else 0.0
    return WalkForwardSweepSummary(
        metric=metric,
        windows=pd.DataFrame(rows),
        aggregate_is_score=is_avg,
        aggregate_oos_score=oos_avg,
        efficiency=efficiency,
        parameter_stability=_parameter_stability(param_sets),
        aggregate_oos_metrics=aggregate_oos_metrics,
    )
