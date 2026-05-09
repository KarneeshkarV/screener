"""Exhaustive grid search for backtest parameters."""

from __future__ import annotations

import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np

from screener.backtester.data import PriceFetcher
from screener.backtester.engine import run_backtest, run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.optimization.metrics import optimization_metrics, score_result

RunnerName = Literal["historical", "rolling"]


@dataclass(frozen=True)
class GridSearchResult:
    params: dict[str, Any]
    score: float
    metrics: dict[str, float]
    trade_count: int
    cached: bool = False
    error: str | None = None


def parameter_combinations(
    parameter_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    keys = list(parameter_grid)
    values = [parameter_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return str(value)


def _stable_fingerprint(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_stable_fingerprint(item) for item in value]
    if isinstance(value, list):
        return [_stable_fingerprint(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _stable_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value):
        return {
            "__class__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": _stable_fingerprint(asdict(value)),
        }
    return {
        "__class__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "repr": repr(value),
    }


def _config_fingerprint(cfg: BacktestConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["slippage_model"] = _stable_fingerprint(cfg.slippage_model)
    return data


def _cache_key(
    cfg: BacktestConfig,
    params: dict[str, Any],
    *,
    runner: RunnerName,
    start_date: date | None,
    end_date: date | None,
    metric: str,
    min_trades: int,
) -> str:
    payload = {
        "config": _config_fingerprint(cfg),
        "params": params,
        "runner": runner,
        "start_date": start_date,
        "end_date": end_date,
        "metric": metric,
        "min_trades": min_trades,
    }
    raw = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_cache(path: Path | None, cache: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True, default=_json_default))
    tmp.replace(path)


def _run_one(
    cfg: BacktestConfig,
    params: dict[str, Any],
    fetcher: PriceFetcher,
    runner: RunnerName,
    start_date: date | None,
    end_date: date | None,
    metric: str,
    min_trades: int,
) -> GridSearchResult:
    test_cfg = replace(cfg, **params)
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
    return GridSearchResult(
        params=params,
        score=score,
        metrics=metrics,
        trade_count=trade_count,
    )


def _run_one_safe(args: tuple[Any, ...]) -> GridSearchResult:
    cfg, params, fetcher, runner, start_date, end_date, metric, min_trades = args
    try:
        return _run_one(
            cfg, params, fetcher, runner, start_date, end_date, metric, min_trades
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — process-pool worker: surface any failure as a ranked result, never crash the whole grid
        return GridSearchResult(
            params=params,
            score=float("-inf"),
            metrics={},
            trade_count=0,
            error=str(exc),
        )


def _from_cache(record: dict[str, Any]) -> GridSearchResult:
    return GridSearchResult(
        params=record["params"],
        score=float(record["score"]),
        metrics={k: float(v) for k, v in record.get("metrics", {}).items()},
        trade_count=int(record.get("trade_count", 0)),
        cached=True,
        error=record.get("error"),
    )


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
) -> list[GridSearchResult]:
    cache_file = Path(cache_path) if cache_path else None
    cache = _load_cache(cache_file)
    combos = parameter_combinations(parameter_grid)
    results: list[GridSearchResult] = []
    pending: list[dict[str, Any]] = []

    for params in combos:
        key = _cache_key(
            cfg,
            params,
            runner=runner,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
            min_trades=min_trades,
        )
        if key in cache:
            results.append(_from_cache(cache[key]))
        else:
            pending.append(params)

    args = [
        (cfg, params, fetcher, runner, start_date, end_date, metric, min_trades)
        for params in pending
    ]
    try:
        if args and (max_workers or 1) != 1:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_one_safe, arg): arg[1] for arg in args}
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    key = _cache_key(
                        cfg,
                        result.params,
                        runner=runner,
                        start_date=start_date,
                        end_date=end_date,
                        metric=metric,
                        min_trades=min_trades,
                    )
                    cache[key] = asdict(result)
                    _save_cache(cache_file, cache)
        else:
            for arg in args:
                result = _run_one_safe(arg)
                results.append(result)
                key = _cache_key(
                    cfg,
                    result.params,
                    runner=runner,
                    start_date=start_date,
                    end_date=end_date,
                    metric=metric,
                    min_trades=min_trades,
                )
                cache[key] = asdict(result)
                _save_cache(cache_file, cache)
    except KeyboardInterrupt:
        _save_cache(cache_file, cache)
        raise

    return sorted(results, key=lambda item: item.score, reverse=True)[:top_n]
