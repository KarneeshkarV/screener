"""Walk-forward optimization over rolling train/test windows.

Daily interval only. Train/test/step lengths are calendar days. Out-of-sample
metrics come from one chronological combined portfolio path that starts at the
first generated test boundary (with a single previous-session initial-capital
anchor). Each fold closes flat at the test boundary (the rolling engine already
force-closes open slots at end with configured costs). Ending capital carries
into the next fold, which restarts flat. Gaps between test windows are filled
with idle cash on the session calendar when available.

Engine same-bar equity reconciliation depends on PR156; this module does not
edit portfolio accounting.

Minimum OOS evidence (see :data:`OOS_EVIDENCE_CRITERIA`):
- every generated fold must produce a finite, error-free eligible train score
  (NaN, inf, and trade-count below ``max(1, min_trades)`` are rejected)
- combined OOS test trades >= ``max(1, min_trades)``
- positive finite chronological OOS equity with at least 2 returns
- selected OOS objective metric is present and finite
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from screener.backtester.data import PriceFetcher, fetch_benchmark
from screener.backtester.metrics import (
    compute_cost_metrics,
    compute_metrics,
    periods_per_year_for_interval,
)
from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.backtester.optimization.grid import (
    GridSearchResult,
    grid_search,
    parameter_combinations,
)
from screener.backtester.optimization.metrics import optimization_metrics
from screener.backtester.rolling_simulation import run_rolling_backtest

FOLD_BOUNDARY_POLICY = "close_flat_at_fold_boundary_with_configured_costs"
CAPITAL_POLICY = "carry_ending_capital_restart_flat"
SUPPORTED_INTERVAL = "1d"
# Same-bar force-close equity accounting lives in portfolio/engine (PR156).
ENGINE_SAME_BAR_DEPENDENCY = "PR156"

OOS_EVIDENCE_CRITERIA = (
    "Out-of-sample evidence requires: (1) every generated fold has a finite "
    "error-free eligible train score (NaN/inf and trade_count < max(1, min_trades) "
    "rejected; a missing eligible fold counts as insufficient); (2) combined OOS "
    "test trades >= max(1, min_trades); (3) positive finite chronological OOS "
    "equity with at least 2 returns; (4) selected OOS objective is present and "
    "finite. These are minimum data-integrity checks, not statistical proof of "
    "alpha or live-trading approval. Daily interval only; train/test/step lengths "
    "are calendar days."
)


class WalkForwardWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_start: date
    train_end: date
    test_start: date
    test_end: date


class WalkForwardResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: WalkForwardWindow
    best_train: GridSearchResult
    test_metrics: dict[str, float]
    test_trade_count: int


class WalkForwardSummary(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    windows: list[WalkForwardResult]
    stability_score: float
    aggregate_metrics: dict[str, float]
    overfit_flag: bool
    train_test_score_ratio: float
    oos_trades: tuple[Trade, ...] = Field(default=(), exclude=True)
    oos_equity: pd.Series | None = Field(default=None, exclude=True)
    insufficient_data: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)
    fold_boundary_policy: str = FOLD_BOUNDARY_POLICY
    capital_policy: str = CAPITAL_POLICY


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
    if step_days is None:
        step = test_days
    else:
        step = step_days
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
    _assert_non_overlapping_test_windows(windows)
    return windows


def _assert_non_overlapping_test_windows(windows: list[WalkForwardWindow]) -> None:
    for prev, curr in zip(windows, windows[1:]):
        if curr.test_start <= prev.test_end:
            raise ValueError(
                "overlapping walk-forward test windows are not allowed: "
                f"{prev.test_start}..{prev.test_end} overlaps "
                f"{curr.test_start}..{curr.test_end}"
            )


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


def evidence_trade_floor(min_trades: int) -> int:
    """Hard floor on trade evidence; ``min_trades=0`` still requires at least 1."""
    return max(1, int(min_trades))


def train_result_eligible(result: GridSearchResult, *, min_trades: int = 1) -> bool:
    """True when a train-fold grid result may select parameters for OOS test."""
    if result.error is not None:
        return False
    if int(result.trade_count) < evidence_trade_floor(min_trades):
        return False
    try:
        score = float(result.score)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(score))


def oos_evidence_adequate(
    *,
    equity: pd.Series | None,
    trade_count: int,
    min_trades: int,
    missing_eligible_folds: int,
    generated_folds: int,
    oos_objective: float | None = None,
) -> bool:
    """Return whether combined OOS evidence meets the documented minimum."""
    if generated_folds <= 0:
        return False
    if missing_eligible_folds > 0:
        return False
    if trade_count < evidence_trade_floor(min_trades):
        return False
    if oos_objective is None or not math.isfinite(float(oos_objective)):
        return False
    if equity is None or equity.empty:
        return False
    values = equity.to_numpy(dtype=float)
    if values.size < 3:
        # Need at least 2 bar returns (len(equity) - 1 >= 2).
        return False
    if not np.isfinite(values).all():
        return False
    if (values <= 0.0).any():
        return False
    return True


def require_daily_walk_forward_scope(
    cfg: BacktestConfig,
    parameter_grid: dict[str, list[Any]] | None = None,
) -> None:
    """Reject non-daily configs and grid interval overrides before any work."""
    if cfg.interval != SUPPORTED_INTERVAL:
        raise ValueError(
            "walk-forward supports daily interval '1d' only "
            f"(got {cfg.interval!r}); train/test/step lengths are calendar days"
        )
    if parameter_grid and "interval" in parameter_grid:
        bad = [v for v in parameter_grid["interval"] if v != SUPPORTED_INTERVAL]
        if bad:
            raise ValueError(
                "walk-forward rejects non-daily parameter_grid interval values "
                f"{bad!r}; daily '1d' only"
            )


def _session_calendar(
    fetcher: PriceFetcher,
    cfg: BacktestConfig,
    start_date: date,
    end_date: date,
) -> pd.DatetimeIndex:
    series = fetch_benchmark(cfg.benchmark, start_date, end_date, fetcher)
    if not series.empty:
        return pd.DatetimeIndex(pd.DatetimeIndex(series.index).normalize().unique())
    return pd.DatetimeIndex(pd.bdate_range(start_date, end_date))


def _oos_anchor_timestamp(
    calendar: pd.DatetimeIndex,
    first_test_start: pd.Timestamp,
) -> tuple[pd.Timestamp, bool]:
    """One session before the first test boundary.

    When no prior benchmark session exists, use the previous business day and
    mark it synthetic so reports can document the anchor.
    """
    prior = calendar[calendar < first_test_start]
    if len(prior):
        return pd.Timestamp(prior[-1]), False
    synthetic = pd.Timestamp(first_test_start) - pd.offsets.BDay(1)
    return pd.Timestamp(synthetic), True


def _prepare_daily_equity(equity: pd.Series) -> pd.Series:
    """Keep daily fold equity as-is; reject corrupt chronology."""
    if equity.empty:
        return equity.astype(float)
    out = equity.astype(float).copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.has_duplicates:
        raise ValueError("fold equity curve has duplicated timestamps")
    if not bool(idx.is_monotonic_increasing):
        raise ValueError("fold equity curve is not monotonic increasing")
    out.index = idx
    return out


def _append_idle_cash(
    pieces: list[pd.Series],
    *,
    calendar: pd.DatetimeIndex,
    after: pd.Timestamp,
    until: pd.Timestamp,
    capital: float,
) -> pd.Timestamp:
    """Append flat capital on sessions in (after, until). Returns last ts used."""
    idle_idx = calendar[(calendar > after) & (calendar < until)]
    if len(idle_idx) == 0:
        return after
    pieces.append(pd.Series(float(capital), index=idle_idx, dtype=float))
    return pd.Timestamp(idle_idx[-1])


def _append_idle_range(
    pieces: list[pd.Series],
    *,
    calendar: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    after: pd.Timestamp | None,
    capital: float,
) -> pd.Timestamp | None:
    """Append flat capital on sessions in [start, end], after ``after`` if set."""
    idle_idx = calendar[(calendar >= start) & (calendar <= end)]
    if after is not None:
        idle_idx = idle_idx[idle_idx > after]
    if len(idle_idx) == 0:
        return after
    pieces.append(pd.Series(float(capital), index=idle_idx, dtype=float))
    return pd.Timestamp(idle_idx[-1])


def _combine_equity_pieces(pieces: list[pd.Series]) -> pd.Series:
    if not pieces:
        return pd.Series(dtype=float)
    combined = pd.concat(pieces)
    if combined.index.has_duplicates:
        raise ValueError("combined OOS equity has duplicated timestamps")
    if not bool(combined.index.is_monotonic_increasing):
        raise ValueError("combined OOS equity is not monotonic increasing")
    return pd.Series(combined.astype(float))


def _sum_fold_fee_metrics(
    results: list[WalkForwardResult],
) -> dict[str, float]:
    fees_paid: dict[str, float] = {}
    for result in results:
        for key, value in result.test_metrics.items():
            if key.startswith("fee_") and key not in {
                "fees_pct_capital",
                "fees_pct_net_pnl",
            }:
                name = key[len("fee_") :]
                fees_paid[name] = fees_paid.get(name, 0.0) + float(value)
            elif key == "total_fees" and not any(
                k.startswith("fee_")
                and k not in {"fees_pct_capital", "fees_pct_net_pnl"}
                for k in result.test_metrics
            ):
                fees_paid["total"] = fees_paid.get("total", 0.0) + float(value)
    return fees_paid


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
    require_daily_walk_forward_scope(cfg, parameter_grid)
    trade_floor = evidence_trade_floor(min_trades)
    windows = generate_walk_forward_windows(
        start_date,
        end_date,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    results: list[WalkForwardResult] = []
    train_scores: list[float] = []
    oos_trades: list[Trade] = []
    equity_pieces: list[pd.Series] = []
    missing_eligible_folds = 0
    skipped_fold_details: list[dict[str, Any]] = []
    initial_capital = float(cfg.initial_capital)
    capital = initial_capital
    last_equity_ts: pd.Timestamp | None = None
    oos_anchor: dict[str, Any] | None = None
    calendar = _session_calendar(fetcher, cfg, start_date, end_date)
    n_combos = len(parameter_combinations(parameter_grid))

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
            top_n=n_combos,
            min_trades=trade_floor,
            max_workers=max_workers,
            cache_path=window_cache,
            runner="rolling",
            start_date=window.train_start,
            end_date=window.train_end,
        )
        # An infinite score can rank first while a lower result is eligible.
        best = next(
            (
                row
                for row in ranked
                if train_result_eligible(row, min_trades=min_trades)
            ),
            ranked[0] if ranked else None,
        )
        test_start_ts = pd.Timestamp(window.test_start)
        test_end_ts = pd.Timestamp(window.test_end)

        if oos_anchor is None and windows:
            anchor_ts, synthetic = _oos_anchor_timestamp(calendar, test_start_ts)
            equity_pieces.append(
                pd.Series([initial_capital], index=pd.DatetimeIndex([anchor_ts]))
            )
            last_equity_ts = anchor_ts
            oos_anchor = {
                "timestamp": str(anchor_ts.date()),
                "equity": initial_capital,
                "synthetic": synthetic,
                "note": (
                    "synthetic previous business day; no prior benchmark session"
                    if synthetic
                    else "previous benchmark session before first test boundary"
                ),
            }
        elif last_equity_ts is not None:
            last_equity_ts = _append_idle_cash(
                equity_pieces,
                calendar=calendar,
                after=last_equity_ts,
                until=test_start_ts,
                capital=capital,
            )

        if best is None or not train_result_eligible(best, min_trades=min_trades):
            missing_eligible_folds += 1
            skipped_score: float | None
            if best is None:
                skipped_score = None
            else:
                try:
                    raw_score = float(best.score)
                    skipped_score = raw_score if math.isfinite(raw_score) else None
                except (TypeError, ValueError):
                    skipped_score = None
            skipped_fold_details.append(
                {
                    "window": window.model_dump(mode="json"),
                    "reason": "no_finite_eligible_train_score",
                    "train_score": skipped_score,
                    "train_trade_count": None if best is None else best.trade_count,
                    "train_error": None if best is None else best.error,
                }
            )
            last_equity_ts = _append_idle_range(
                equity_pieces,
                calendar=calendar,
                start=test_start_ts,
                end=test_end_ts,
                after=last_equity_ts,
                capital=capital,
            )
            continue

        test_cfg = cfg.model_copy(
            update={**best.params, "initial_capital": float(capital)}
        )
        test_result = run_rolling_backtest(
            test_cfg,
            fetcher,
            start_date=window.test_start,
            end_date=window.test_end,
        )
        # Engine force-closes open slots at end_ts with configured costs
        # (fold_boundary_policy). Next fold restarts flat at carried capital.
        # Do not rescale fold equity: first-bar fees/PnL must remain intact.
        fold_equity = _prepare_daily_equity(test_result.equity_curve)
        if fold_equity.empty:
            last_equity_ts = _append_idle_range(
                equity_pieces,
                calendar=calendar,
                start=test_start_ts,
                end=test_end_ts,
                after=last_equity_ts,
                capital=capital,
            )
        else:
            first_bar = pd.Timestamp(fold_equity.index[0])
            last_equity_ts = _append_idle_range(
                equity_pieces,
                calendar=calendar,
                start=test_start_ts,
                end=first_bar - pd.Timedelta(days=1),
                after=last_equity_ts,
                capital=capital,
            )
            if last_equity_ts is not None and first_bar <= last_equity_ts:
                raise ValueError(
                    f"fold equity overlaps prior OOS timestamps at {first_bar.date()}"
                )
            equity_pieces.append(fold_equity)
            last_equity_ts = pd.Timestamp(fold_equity.index[-1])
            capital = float(fold_equity.iloc[-1])
            last_equity_ts = _append_idle_range(
                equity_pieces,
                calendar=calendar,
                start=last_equity_ts + pd.Timedelta(days=1),
                end=test_end_ts,
                after=last_equity_ts,
                capital=capital,
            )

        fold_metrics = optimization_metrics(test_result)
        count = len(test_result.trades)
        oos_trades.extend(test_result.trades)
        results.append(
            WalkForwardResult(
                window=window,
                best_train=best,
                test_metrics=fold_metrics,
                test_trade_count=count,
            )
        )
        train_scores.append(float(best.score))

    combined_equity = _combine_equity_pieces(equity_pieces)
    # Bound OOS to the generated test span (anchor through final test end).
    if windows and not combined_equity.empty:
        final_test_end = pd.Timestamp(windows[-1].test_end)
        combined_equity = combined_equity[combined_equity.index <= final_test_end]

    fold_tops = [
        int(result.best_train.params.get("top", cfg.top)) for result in results
    ]
    exposure_reason: str | None = None
    if not fold_tops:
        slot_count = max(int(cfg.top), 1)
    elif len(set(fold_tops)) == 1:
        slot_count = max(fold_tops[0], 1)
    else:
        slot_count = max(int(cfg.top), 1)
        exposure_reason = (
            "exposure unavailable: selected top varies across folds "
            f"{sorted(set(fold_tops))}"
        )

    benchmark = fetch_benchmark(cfg.benchmark, start_date, end_date, fetcher)
    if not combined_equity.empty and not benchmark.empty:
        benchmark_aligned = benchmark.reindex(combined_equity.index, method="ffill")
    elif not combined_equity.empty:
        benchmark_aligned = pd.Series(
            initial_capital,
            index=combined_equity.index,
            dtype=float,
        )
    else:
        benchmark_aligned = pd.Series(dtype=float)

    if not combined_equity.empty:
        combined_metrics = compute_metrics(
            combined_equity,
            benchmark_aligned,
            list(oos_trades),
            slot_count,
            periods_per_year=periods_per_year_for_interval(SUPPORTED_INTERVAL),
        )
        if exposure_reason is not None:
            combined_metrics["exposure"] = float("nan")
        fees_paid = _sum_fold_fee_metrics(results)
        if fees_paid:
            net_pnl = float(sum(float(t.pnl) for t in oos_trades))
            combined_metrics.update(
                compute_cost_metrics(fees_paid, initial_capital, net_pnl)
            )
        aggregate = optimization_metrics(
            BacktestResult(
                config=cfg,
                trades=list(oos_trades),
                equity_curve=combined_equity,
                benchmark_curve=benchmark_aligned,
                metrics=combined_metrics,
            )
        )
    else:
        aggregate = {}

    params = [result.best_train.params for result in results]
    stability = _parameter_stability(params)
    train_avg = float(np.mean(train_scores)) if train_scores else 0.0
    oos_objective_raw = aggregate.get(metric)
    try:
        oos_objective = (
            float(oos_objective_raw) if oos_objective_raw is not None else None
        )
    except (TypeError, ValueError):
        oos_objective = None
    if oos_objective is not None and not math.isfinite(oos_objective):
        oos_objective = None
    # Aggregate train/test ratio uses the combined OOS objective, not mean fold.
    if train_avg > 0 and oos_objective is not None:
        ratio = train_avg / max(abs(oos_objective), 1e-9)
    else:
        ratio = 0.0

    adequate = oos_evidence_adequate(
        equity=combined_equity if not combined_equity.empty else None,
        trade_count=len(oos_trades),
        min_trades=min_trades,
        missing_eligible_folds=missing_eligible_folds,
        generated_folds=len(windows),
        oos_objective=oos_objective,
    )
    evidence = {
        "criteria": OOS_EVIDENCE_CRITERIA,
        "generated_folds": len(windows),
        "eligible_folds": len(results),
        "missing_eligible_folds": missing_eligible_folds,
        "skipped_folds": skipped_fold_details,
        "oos_trade_count": len(oos_trades),
        "min_trades": min_trades,
        "trade_floor": trade_floor,
        "oos_equity_bars": int(len(combined_equity)),
        "oos_return_count": max(int(len(combined_equity)) - 1, 0),
        "oos_objective_metric": metric,
        "oos_objective_finite": oos_objective is not None,
        "adequate": adequate,
        "fold_boundary_policy": FOLD_BOUNDARY_POLICY,
        "capital_policy": CAPITAL_POLICY,
        "interval": SUPPORTED_INTERVAL,
        "oos_anchor": oos_anchor,
        "exposure_unavailable_reason": exposure_reason,
        "engine_same_bar_dependency": ENGINE_SAME_BAR_DEPENDENCY,
    }

    return WalkForwardSummary(
        windows=results,
        stability_score=stability,
        aggregate_metrics=aggregate,
        overfit_flag=bool(train_avg > 0 and ratio >= overfit_ratio),
        train_test_score_ratio=ratio,
        oos_trades=tuple(oos_trades),
        oos_equity=combined_equity if not combined_equity.empty else None,
        insufficient_data=not adequate,
        evidence=evidence,
        fold_boundary_policy=FOLD_BOUNDARY_POLICY,
        capital_policy=CAPITAL_POLICY,
    )
