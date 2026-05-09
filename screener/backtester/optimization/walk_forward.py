"""Walk-forward optimization over rolling train/test windows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screener.backtester.data import PriceFetcher
from screener.backtester.engine import run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.optimization.grid import GridSearchResult, grid_search
from screener.backtester.optimization.metrics import optimization_metrics


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class WalkForwardResult:
    window: WalkForwardWindow
    best_train: GridSearchResult
    test_metrics: dict[str, float]
    test_trade_count: int


@dataclass(frozen=True)
class WalkForwardSummary:
    windows: list[WalkForwardResult]
    stability_score: float
    aggregate_metrics: dict[str, float]
    overfit_flag: bool
    train_test_score_ratio: float


def generate_walk_forward_windows(
    start_date: date,
    end_date: date,
    *,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[WalkForwardWindow]:
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be positive")
    step = step_days or test_days
    if step <= 0:
        raise ValueError("step_days must be positive")
    windows: list[WalkForwardWindow] = []
    cursor = pd.Timestamp(start_date)
    final = pd.Timestamp(end_date)
    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)
        if test_end > final:
            break
        windows.append(
            WalkForwardWindow(
                train_start=train_start.date(),
                train_end=train_end.date(),
                test_start=test_start.date(),
                test_end=test_end.date(),
            )
        )
        cursor = cursor + pd.Timedelta(days=step)
    return windows


def _parameter_stability(param_sets: list[dict[str, Any]]) -> float:
    if len(param_sets) <= 1:
        return 1.0
    scores: list[float] = []
    keys = sorted({key for params in param_sets for key in params})
    for key in keys:
        values = [params.get(key) for params in param_sets]
        numeric = [float(v) for v in values if isinstance(v, (int, float))]
        if len(numeric) == len(values):
            arr = np.array(numeric, dtype=float)
            denom = max(float(np.mean(np.abs(arr))), 1e-9)
            scores.append(max(0.0, 1.0 - float(np.std(arr) / denom)))
        else:
            unique = len(set(values))
            scores.append(1.0 - ((unique - 1) / max(len(values) - 1, 1)))
    return float(np.mean(scores)) if scores else 1.0


def walk_forward_optimize(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    parameter_grid: dict[str, list[Any]],
    *,
    start_date: date,
    end_date: date,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
    metric: str = "sharpe",
    min_trades: int = 1,
    max_workers: int | None = None,
    cache_path: Path | str | None = None,
    overfit_ratio: float = 2.0,
) -> WalkForwardSummary:
    windows = generate_walk_forward_windows(
        start_date,
        end_date,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    results: list[WalkForwardResult] = []
    test_trade_counts = 0
    aggregate_weight = 0
    weighted_metrics: dict[str, float] = {}
    train_scores: list[float] = []
    test_scores: list[float] = []

    for idx, window in enumerate(windows):
        window_cache = None
        if cache_path:
            base = Path(cache_path)
            window_cache = base.with_name(f"{base.stem}_wf_{idx}{base.suffix}")
        ranked = grid_search(
            cfg,
            fetcher,
            parameter_grid,
            metric=metric,
            top_n=1,
            min_trades=min_trades,
            max_workers=max_workers,
            cache_path=window_cache,
            runner="rolling",
            start_date=window.train_start,
            end_date=window.train_end,
        )
        if not ranked:
            continue
        best = ranked[0]
        test_cfg = replace(cfg, **best.params)
        test_result = run_rolling_backtest(
            test_cfg,
            fetcher,
            start_date=window.test_start,
            end_date=window.test_end,
        )
        metrics = optimization_metrics(test_result)
        count = len(test_result.trades)
        results.append(
            WalkForwardResult(
                window=window,
                best_train=best,
                test_metrics=metrics,
                test_trade_count=count,
            )
        )
        test_trade_counts += count
        weight = max(count, 1)
        aggregate_weight += weight
        for key, value in metrics.items():
            weighted_metrics[key] = (
                weighted_metrics.get(key, 0.0) + float(value) * weight
            )
        train_scores.append(float(best.score))
        test_scores.append(float(metrics.get(metric, 0.0)))

    if test_trade_counts:
        aggregate = {
            key: value / max(aggregate_weight, 1)
            for key, value in weighted_metrics.items()
        }
    else:
        aggregate = {}
    params = [result.best_train.params for result in results]
    stability = _parameter_stability(params)
    train_avg = float(np.mean(train_scores)) if train_scores else 0.0
    test_avg = float(np.mean(test_scores)) if test_scores else 0.0
    ratio = train_avg / max(abs(test_avg), 1e-9) if train_avg > 0 else 0.0
    return WalkForwardSummary(
        windows=results,
        stability_score=stability,
        aggregate_metrics=aggregate,
        overfit_flag=bool(train_avg > 0 and ratio >= overfit_ratio),
        train_test_score_ratio=ratio,
    )
