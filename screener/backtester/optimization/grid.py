"""Exhaustive grid search for backtest parameters."""

from __future__ import annotations

import itertools
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from screener.backtester.data import PriceFetcher
from screener.backtester.historical import run_backtest
from screener.backtester.metrics import (
    SharpeMoments,
    apply_trial_context_to_metrics,
    dsr_unavailable_reason,
    periods_per_year_for_interval,
    sharpe_moments_from_equity,
)
from screener.backtester.models import BacktestConfig, BacktestResult
from screener.backtester.optimization.cache import (
    FrozenInputIdentity,
    ResultCache,
    make_cache_key,
)
from screener.backtester.optimization.metrics import optimization_metrics, score_result
from screener.backtester.optimization.trials import (
    classify_trial_status,
    default_experiment_id,
    load_trial_search_stats,
    record_trial,
    trial_config_identity,
)
from screener.backtester.rolling_simulation import (
    PreparedRollingBacktest,
    _preparation_fingerprint,
    prepare_rolling_backtest,
    run_prepared_rolling_backtest,
    run_rolling_backtest,
)

RunnerName = Literal["historical", "rolling"]

LOG = logging.getLogger(__name__)


class GridSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    params: dict[str, Any]
    score: float
    metrics: dict[str, float]
    trade_count: int
    cached: bool = False
    error: str | None = None
    dsr_unavailable_reason: str | None = None


def parameter_combinations(
    parameter_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    keys = list(parameter_grid)
    values = [parameter_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _fingerprint_key(cfg: BacktestConfig) -> str:
    return json.dumps(_preparation_fingerprint(cfg), sort_keys=True, default=str)


def _group_params_by_preparation(
    cfg: BacktestConfig, pending: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """Group configs that share a rolling preparation fingerprint."""
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for params in pending:
        key = _fingerprint_key(cfg.model_copy(update=params))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(params)
    return [groups[key] for key in order]


def _prep_homogeneous_chunks(
    groups: list[list[dict[str, Any]]], worker_count: int
) -> list[list[dict[str, Any]]]:
    """Split each prep group across workers; each chunk stays homogeneous.

    Homogeneous chunks keep prepare-once semantics under parallel execution and
    preserve score/trade parity with the sequential path. Interruption still
    re-raises ``KeyboardInterrupt`` after workers unwind; ATTEMPTED rows written
    before submit remain as search-exposure evidence.
    """
    chunks: list[list[dict[str, Any]]] = []
    for group in groups:
        workers = min(max(worker_count, 1), len(group))
        for i in range(workers):
            part = group[i::workers]
            if part:
                chunks.append(part)
    return chunks


def _moments_from_result(
    result: BacktestResult, cfg: BacktestConfig
) -> SharpeMoments | None:
    return sharpe_moments_from_equity(
        result.equity_curve,
        periods_per_year=periods_per_year_for_interval(cfg.interval),
    )


def _trial_key_for(
    cfg: BacktestConfig,
    params: dict[str, Any],
    *,
    runner: RunnerName,
    start_date: date | None,
    end_date: date | None,
    metric: str,
    min_trades: int,
) -> str:
    return trial_config_identity(
        cfg.model_copy(update=params),
        runner=runner,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        min_trades=min_trades,
    )


def _run_one(
    cfg: BacktestConfig,
    params: dict[str, Any],
    fetcher: PriceFetcher,
    runner: RunnerName,
    start_date: date | None,
    end_date: date | None,
    metric: str,
    min_trades: int,
) -> tuple[GridSearchResult, SharpeMoments | None]:
    test_cfg = cfg.model_copy(update=params)
    if runner == "rolling":
        if start_date is None or end_date is None:
            raise ValueError("rolling grid search requires start_date and end_date")
        result = run_rolling_backtest(
            test_cfg,
            fetcher,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        result = run_backtest(test_cfg, fetcher)
    metrics = optimization_metrics(result)
    trade_count = len(result.trades)
    score = score_result(result, metric) if trade_count >= min_trades else float("-inf")
    return (
        GridSearchResult(
            params=params,
            score=score,
            metrics=metrics,
            trade_count=trade_count,
        ),
        _moments_from_result(result, test_cfg),
    )


def _run_one_prepared(
    cfg: BacktestConfig,
    params: dict[str, Any],
    prepared: PreparedRollingBacktest,
    metric: str,
    min_trades: int,
) -> tuple[GridSearchResult, SharpeMoments | None]:
    test_cfg = cfg.model_copy(update=params)
    result = run_prepared_rolling_backtest(prepared, test_cfg)
    metrics = optimization_metrics(result)
    trade_count = len(result.trades)
    score = score_result(result, metric) if trade_count >= min_trades else float("-inf")
    return (
        GridSearchResult(
            params=params,
            score=score,
            metrics=metrics,
            trade_count=trade_count,
        ),
        _moments_from_result(result, test_cfg),
    )


def _error_result(params: dict[str, Any], exc: BaseException) -> GridSearchResult:
    return GridSearchResult(
        params=params,
        score=float("-inf"),
        metrics={},
        trade_count=0,
        error=str(exc),
    )


def _run_one_safe(args: tuple[Any, ...]) -> tuple[GridSearchResult, dict | None]:
    cfg, params, fetcher, runner, start_date, end_date, metric, min_trades = args
    try:
        result, moments = _run_one(
            cfg, params, fetcher, runner, start_date, end_date, metric, min_trades
        )
        return result, None if moments is None else moments.as_dict()
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — process-pool worker: surface any failure
        return _error_result(params, exc), None


def _run_chunk_safe(
    args: tuple[Any, ...],
) -> list[tuple[GridSearchResult, dict | None]]:
    """Run a prep-homogeneous chunk, preparing rolling data only once."""
    (
        cfg,
        params_chunk,
        fetcher,
        runner,
        start_date,
        end_date,
        metric,
        min_trades,
    ) = args
    prepared: PreparedRollingBacktest | None = None
    if runner == "rolling" and params_chunk:
        if start_date is None or end_date is None:
            return [
                (
                    GridSearchResult(
                        params=params,
                        score=float("-inf"),
                        metrics={},
                        trade_count=0,
                        error="rolling grid search requires start_date and end_date",
                    ),
                    None,
                )
                for params in params_chunk
            ]
        try:
            prepared = prepare_rolling_backtest(
                cfg.model_copy(update=params_chunk[0]),
                fetcher,
                start_date=start_date,
                end_date=end_date,
            )
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001 - individual runs preserve error details
            prepared = None

    out: list[tuple[GridSearchResult, dict | None]] = []
    for params in params_chunk:
        test_cfg = cfg.model_copy(update=params)
        if prepared is not None and prepared.supports(test_cfg):
            try:
                result, moments = _run_one_prepared(
                    cfg, params, prepared, metric, min_trades
                )
                out.append((result, None if moments is None else moments.as_dict()))
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve grid result contract
                out.append((_error_result(params, exc), None))
        else:
            out.append(
                _run_one_safe(
                    (
                        cfg,
                        params,
                        fetcher,
                        runner,
                        start_date,
                        end_date,
                        metric,
                        min_trades,
                    )
                )
            )
    prepared = None
    return out


def _from_cache(
    record: dict[str, Any],
) -> tuple[GridSearchResult, SharpeMoments | None]:
    moments_raw = record.get("moments")
    moments = None if not moments_raw else SharpeMoments.from_dict(moments_raw)
    return (
        GridSearchResult(
            params=record["params"],
            score=float(record["score"]),
            metrics={k: float(v) for k, v in record.get("metrics", {}).items()},
            trade_count=int(record.get("trade_count", 0)),
            cached=True,
            error=record.get("error"),
        ),
        moments,
    )


def _persist_trial(
    *,
    experiment_id: str,
    result: GridSearchResult,
    min_trades: int,
    trial_db_path: Path | None,
    trial_key: str,
) -> None:
    status = classify_trial_status(
        error=result.error,
        score=result.score,
        trade_count=result.trade_count,
        min_trades=min_trades,
    )
    record_trial(
        experiment_id,
        result.params,
        status=status,
        score=result.score,
        metrics=result.metrics if result.metrics else None,
        trade_count=result.trade_count,
        trial_key=trial_key,
        error=result.error,
        db_path=trial_db_path,
    )


def _mark_attempted(
    *,
    experiment_id: str,
    params: dict[str, Any],
    trial_key: str,
    trial_db_path: Path | None,
) -> None:
    """Register search exposure before execution so interruption still counts."""
    record_trial(
        experiment_id,
        params,
        status="attempted",
        score=None,
        metrics=None,
        trade_count=0,
        trial_key=trial_key,
        error=None,
        db_path=trial_db_path,
    )


def _score_after_dsr_correction(
    *,
    metric: str,
    metrics: dict[str, float],
    prior_score: float,
    trade_count: int,
    min_trades: int,
    error: str | None,
) -> float:
    """Return ranking score after search-level DSR rewrite.

    When ``metric == 'dsr'``, the returned score must reflect the corrected DSR
    (never the original single-trial objective). Non-finite corrected DSR
    becomes ``-inf``.
    """
    if error:
        return float("-inf")
    if trade_count < min_trades:
        return float("-inf")
    if metric != "dsr":
        return prior_score
    value = float(metrics.get("dsr", float("nan")))
    if not math.isfinite(value):
        return float("-inf")
    return value


def _apply_search_dsr(
    rows: list[tuple[GridSearchResult, SharpeMoments | None]],
    *,
    experiment_id: str,
    trial_db_path: Path | None,
    n_trials_effective: int | None,
    metric: str,
    min_trades: int,
) -> list[GridSearchResult]:
    stats = load_trial_search_stats(
        experiment_id,
        db_path=trial_db_path,
        n_trials_effective=n_trials_effective,
    )
    n_trials = (
        stats.n_trials_effective
        if stats.n_trials_effective is not None
        else stats.n_trials_nominal
    )
    if n_trials < 1:
        n_trials = 1
    dispersion = stats.sr_trial_std_annual
    applied: list[GridSearchResult] = []
    for result, moments in rows:
        if result.error or not result.metrics:
            applied.append(
                GridSearchResult(
                    params=result.params,
                    score=result.score,
                    metrics=result.metrics,
                    trade_count=result.trade_count,
                    cached=result.cached,
                    error=result.error,
                    dsr_unavailable_reason=(
                        "missing_moments" if not result.error else None
                    ),
                )
            )
            continue
        metrics = apply_trial_context_to_metrics(
            result.metrics,
            moments,
            n_trials=n_trials,
            sr_trial_std_annual=dispersion if n_trials > 1 else None,
        )
        reason = dsr_unavailable_reason(
            moments,
            n_trials=n_trials,
            sr_trial_std_annual=dispersion if n_trials > 1 else None,
        )
        score = _score_after_dsr_correction(
            metric=metric,
            metrics=metrics,
            prior_score=result.score,
            trade_count=result.trade_count,
            min_trades=min_trades,
            error=result.error,
        )
        applied.append(
            GridSearchResult(
                params=result.params,
                score=score,
                metrics=metrics,
                trade_count=result.trade_count,
                cached=result.cached,
                error=result.error,
                dsr_unavailable_reason=reason,
            )
        )
    return applied


def grid_search(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    parameter_grid: dict[str, list[Any]],
    *,
    metric: str = "sharpe",
    top_n: int = 10,
    min_trades: int = 1,
    max_workers: int | None = None,
    cache_path: Path | str | None = None,
    runner: RunnerName = "historical",
    start_date: date | None = None,
    end_date: date | None = None,
    frozen_input_identity: FrozenInputIdentity | None = None,
    experiment_id: str | None = None,
    n_trials_effective: int | None = None,
    trial_db_path: Path | str | None = None,
) -> list[GridSearchResult]:
    """Exhaustive parameter search with optional identity-aware caching.

    Cross-run result reuse requires ``frozen_input_identity``. When it is
    omitted, prior cache rows are not read or written (safe default). The trial
    register still records every attempt before ``top_n`` truncation.

    Reuse the same explicit ``experiment_id`` and ``trial_db_path`` across
    iterative searches when rejected families must accumulate under one history.
    """
    if cache_path is not None and frozen_input_identity is None:
        LOG.warning(
            "cache_path requested without frozen_input_identity; "
            "result cache disabled (ordinary CLI has no input identity)"
        )
    cache = ResultCache(cache_path if frozen_input_identity is not None else None)
    reuse = frozen_input_identity is not None
    trial_path = Path(trial_db_path) if trial_db_path else None
    exp_id = experiment_id or default_experiment_id(
        cfg,
        parameter_grid,
        runner=runner,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        min_trades=min_trades,
    )

    combos = parameter_combinations(parameter_grid)
    if not combos:
        return []
    if n_trials_effective is None:
        n_trials_effective = len(combos)
    rows: list[tuple[GridSearchResult, SharpeMoments | None]] = []
    pending: list[dict[str, Any]] = []

    def _key_for(params: dict[str, Any]) -> str:
        return _trial_key_for(
            cfg,
            params,
            runner=runner,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
            min_trades=min_trades,
        )

    for params in combos:
        if reuse:
            assert frozen_input_identity is not None
            key = make_cache_key(
                cfg,
                params,
                runner=runner,
                start_date=start_date,
                end_date=end_date,
                metric=metric,
                min_trades=min_trades,
                frozen_input_identity=frozen_input_identity,
                code_fp=cache.code_fp,
                n_trials=n_trials_effective,
            )
            hit = cache.get(key)
            if hit is not None:
                cached_row = _from_cache(hit)
                rows.append(cached_row)
                _persist_trial(
                    experiment_id=exp_id,
                    result=cached_row[0],
                    min_trades=min_trades,
                    trial_db_path=trial_path,
                    trial_key=_key_for(params),
                )
                continue
        pending.append(params)

    def _store(
        result: GridSearchResult, moments: SharpeMoments | None
    ) -> tuple[GridSearchResult, SharpeMoments | None]:
        _persist_trial(
            experiment_id=exp_id,
            result=result,
            min_trades=min_trades,
            trial_db_path=trial_path,
            trial_key=_key_for(result.params),
        )
        # Cache deterministic outcomes only. Never persist transient failures.
        # Store pre-search metrics/moments; post-search DSR rewrite happens later
        # for both fresh and cached rows.
        if reuse and frozen_input_identity is not None and result.error is None:
            key = make_cache_key(
                cfg,
                result.params,
                runner=runner,
                start_date=start_date,
                end_date=end_date,
                metric=metric,
                min_trades=min_trades,
                frozen_input_identity=frozen_input_identity,
                code_fp=cache.code_fp,
                n_trials=n_trials_effective,
            )
            cache.put(
                key,
                params=result.params,
                score=result.score,
                metrics=result.metrics,
                trade_count=result.trade_count,
                moments=None if moments is None else moments.as_dict(),
            )
        return result, moments

    # Register ATTEMPTED before execution so interruption records exposure.
    for params in pending:
        _mark_attempted(
            experiment_id=exp_id,
            params=params,
            trial_key=_key_for(params),
            trial_db_path=trial_path,
        )

    try:
        if pending and (max_workers or 1) != 1:
            worker_count = min(int(max_workers or 1), len(pending))
            groups = _group_params_by_preparation(cfg, pending)
            chunks = _prep_homogeneous_chunks(groups, worker_count)
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        _run_chunk_safe,
                        (
                            cfg,
                            chunk,
                            fetcher,
                            runner,
                            start_date,
                            end_date,
                            metric,
                            min_trades,
                        ),
                    ): chunk
                    for chunk in chunks
                }
                for future in as_completed(futures):
                    for result, moments_dict in future.result():
                        moments = (
                            None
                            if moments_dict is None
                            else SharpeMoments.from_dict(moments_dict)
                        )
                        rows.append(_store(result, moments))
        else:
            groups = _group_params_by_preparation(cfg, pending) if pending else []
            for group in groups:
                prepared: PreparedRollingBacktest | None = None
                if runner == "rolling":
                    if start_date is None or end_date is None:
                        raise ValueError(
                            "rolling grid search requires start_date and end_date"
                        )
                    try:
                        prepared = prepare_rolling_backtest(
                            cfg.model_copy(update=group[0]),
                            fetcher,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception:  # noqa: BLE001 - per-run fallback preserves errors
                        prepared = None
                for params in group:
                    test_cfg = cfg.model_copy(update=params)
                    if prepared is not None and prepared.supports(test_cfg):
                        try:
                            result, moments = _run_one_prepared(
                                cfg, params, prepared, metric, min_trades
                            )
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            result, moments = _error_result(params, exc), None
                    else:
                        try:
                            result, moments_dict = _run_one_safe(
                                (
                                    cfg,
                                    params,
                                    fetcher,
                                    runner,
                                    start_date,
                                    end_date,
                                    metric,
                                    min_trades,
                                )
                            )
                            moments = (
                                None
                                if moments_dict is None
                                else SharpeMoments.from_dict(moments_dict)
                            )
                        except KeyboardInterrupt:
                            raise
                    rows.append(_store(result, moments))
                prepared = None
    except KeyboardInterrupt:
        raise

    applied = _apply_search_dsr(
        rows,
        experiment_id=exp_id,
        trial_db_path=trial_path,
        n_trials_effective=n_trials_effective,
        metric=metric,
        min_trades=min_trades,
    )
    return sorted(applied, key=lambda item: item.score, reverse=True)[:top_n]


def _cache_key(
    cfg: BacktestConfig,
    params: dict[str, Any],
    *,
    runner: RunnerName,
    start_date: date | None,
    end_date: date | None,
    metric: str,
    min_trades: int,
    frozen_input_identity: FrozenInputIdentity | None = None,
    n_trials: int = 1,
) -> str:
    """Compatibility helper around :func:`make_cache_key`.

    Without ``frozen_input_identity``, uses a deterministic unknown placeholder.
    That key is not authentic input identity and must not be treated as reusable
    across real price snapshots.
    """
    if frozen_input_identity is None:
        identity = FrozenInputIdentity(
            price_snapshot_hash="unknown",
            universe_membership_identity="unknown",
            source="unauthenticated",
        )
    else:
        identity = frozen_input_identity
    return make_cache_key(
        cfg,
        params,
        runner=runner,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        min_trades=min_trades,
        frozen_input_identity=identity,
        n_trials=n_trials,
    )
