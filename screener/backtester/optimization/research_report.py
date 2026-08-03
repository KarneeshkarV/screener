"""One-command research report: grid → walk-forward → Monte Carlo.

Orchestration only — all signal/math work is delegated to
:func:`grid_search`, :func:`walk_forward_optimize`, and
:func:`simulate_monte_carlo`. Callers inject a :class:`PriceFetcher` so the
same price cache is reused across stages.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from screener.backtester.data import PriceFetcher
from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.optimization.grid import (
    GridSearchResult,
    grid_search,
    parameter_combinations,
)
from screener.backtester.optimization.monte_carlo import (
    MonteCarloResult,
    simulate_monte_carlo,
)
from screener.backtester.optimization.reporting import (
    GRID_IN_SAMPLE_DISCLAIMER,
    write_json_report,
    write_research_html_report,
)
from screener.backtester.optimization.walk_forward import (
    WalkForwardSummary,
    walk_forward_optimize,
)
from screener.backtester.rolling_simulation import run_rolling_backtest

# Relative score range below which a parameter optimum is treated as a plateau
# rather than a narrow spike (range / max(|best|, eps)).
_PLATEAU_RANGE_FRACTION = 0.15


def compute_parameter_stability(
    results: Sequence[GridSearchResult],
    parameter_grid: dict[str, list[Any]],
    *,
    metric: str = "sharpe",
) -> list[dict[str, Any]]:
    """Per-parameter metric spread across grid values.

    For each parameter in ``parameter_grid``, group completed (error-free)
    results by that parameter's value and take the best objective score among
    combos sharing that value. Report min/max/mean/std/range of those
    best-per-value scores plus a coarse ``shape`` label:

    * ``plateau`` — range of best-per-value scores is small relative to |best|
    * ``spike`` — one value stands out (wide range)
    * ``flat`` — fewer than two evaluated values
    * ``empty`` — no usable results for this parameter
    """
    usable = [r for r in results if r.error is None and r.score != float("-inf")]
    summaries: list[dict[str, Any]] = []
    for name, values in parameter_grid.items():
        best_by_value: dict[str, float] = {}
        for value in values:
            matching = [
                float(r.score)
                for r in usable
                if _param_equal(r.params.get(name), value)
            ]
            if matching:
                best_by_value[_value_key(value)] = max(matching)
        scores = list(best_by_value.values())
        if not scores:
            summaries.append(
                {
                    "parameter": name,
                    "values_evaluated": 0,
                    "best_value": None,
                    "best_score": None,
                    "score_min": None,
                    "score_max": None,
                    "score_mean": None,
                    "score_std": None,
                    "score_range": None,
                    "shape": "empty",
                    "by_value": {},
                }
            )
            continue

        score_min = min(scores)
        score_max = max(scores)
        score_mean = float(statistics.fmean(scores))
        score_std = float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0
        score_range = score_max - score_min
        best_value_key = max(best_by_value, key=best_by_value.get)  # type: ignore[arg-type]
        best_score = best_by_value[best_value_key]
        if len(scores) < 2:
            shape = "flat"
        else:
            denom = max(abs(best_score), 1e-9)
            shape = (
                "plateau"
                if (score_range / denom) <= _PLATEAU_RANGE_FRACTION
                else "spike"
            )
        summaries.append(
            {
                "parameter": name,
                "values_evaluated": len(scores),
                "best_value": _parse_value_key(best_value_key),
                "best_score": best_score,
                "score_min": score_min,
                "score_max": score_max,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_range": score_range,
                "shape": shape,
                "by_value": dict(best_by_value),
            }
        )
    return summaries


def _param_equal(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) < 1e-12
        except (TypeError, ValueError):
            return bool(left == right)
    return bool(left == right)


def _value_key(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _parse_value_key(key: str) -> Any:
    if key == "null":
        return None
    try:
        if "." in key or "e" in key.lower():
            return float(key)
        return int(key)
    except ValueError:
        return key


def _banner(console: Console, title: str) -> float:
    console.print()
    console.rule(f"[bold]{title}[/bold]")
    return time.perf_counter()


def _stage_done(console: Console, started: float) -> float:
    elapsed = time.perf_counter() - started
    console.print(f"[dim]Stage completed in {elapsed:.2f}s[/dim]")
    return elapsed


def _fixed_param_grid(params: dict[str, Any]) -> dict[str, list[Any]]:
    return {key: [value] for key, value in params.items()}


def _degradation_ratio(is_metric: float, oos_metric: float) -> float:
    """OOS-vs-IS degradation: 0 means no loss, 1 means OOS fully wiped IS gain.

    Defined as ``1 - oos/is`` when ``is > 0``; when IS is non-positive the ratio
    is 0 if OOS >= IS else 1 (cannot interpret relative degradation).
    """
    if is_metric > 0:
        return float(1.0 - (oos_metric / is_metric))
    if oos_metric >= is_metric:
        return 0.0
    return 1.0


def _verdict(
    *,
    overfit_flag: bool,
    degradation: float,
    mc_return_p05: float,
    oos_metric: float,
) -> str:
    if overfit_flag or degradation >= 0.75:
        return "FAIL: severe IS→OOS degradation / overfit risk"
    if mc_return_p05 < 0 and oos_metric <= 0:
        return "FAIL: weak OOS and left-tail MC returns negative"
    if degradation >= 0.40 or mc_return_p05 < 0:
        return "CAUTION: material degradation or negative MC 5th-percentile"
    return "PASS: OOS holds reasonably vs IS; MC left tail acceptable"


def run_research_report(
    cfg: BacktestConfig,
    fetcher: PriceFetcher,
    parameter_grid: dict[str, list[Any]],
    *,
    start_date: date,
    end_date: date,
    train_days: int = 252,
    test_days: int = 63,
    step_days: int | None = 63,
    metric: str = "sharpe",
    min_trades: int = 1,
    max_workers: int | None = 1,
    cache_path: Path | str | None = None,
    mc_iterations: int = 1000,
    mc_seed: int = 42,
    ruin_threshold: float = 0.5,
    top_n: int = 10,
    out_path: Path | str,
    console: Console | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run grid → walk-forward → Monte Carlo and write JSON/HTML reports.

    Prices are read only through ``fetcher`` (caller should reuse one instance).
    Returns the JSON payload written to ``<out>.json``.
    """
    console = console or Console()
    out = Path(out_path)
    timings: dict[str, float] = {}

    # ── Stage 1: grid / parameter stability ──────────────────────────────
    started = _banner(console, "Stage 1/3 — Grid search & parameter stability")
    if progress:
        progress("grid")
    n_combos = max(len(parameter_combinations(parameter_grid)), 1)
    grid_results = grid_search(
        cfg,
        fetcher,
        parameter_grid,
        metric=metric,
        top_n=max(int(top_n), n_combos),
        min_trades=min_trades,
        max_workers=max_workers,
        cache_path=cache_path,
        runner="rolling",
        start_date=start_date,
        end_date=end_date,
    )
    stability = compute_parameter_stability(grid_results, parameter_grid, metric=metric)
    best = grid_results[0] if grid_results else None
    best_params: dict[str, Any] = dict(best.params) if best is not None else {}
    is_metric = float(best.score) if best is not None else 0.0
    console.print(
        f"Combos evaluated (returned): {len(grid_results)}  "
        f"Best params: {best_params}  "
        f"Best {metric}: {is_metric:.4f}"
    )
    for row in stability:
        console.print(
            f"  stability[{row['parameter']}]: shape={row['shape']} "
            f"range={row['score_range']} best_value={row['best_value']}"
        )
    timings["grid_seconds"] = _stage_done(console, started)

    # ── Stage 2: walk-forward on best params ─────────────────────────────
    started = _banner(console, "Stage 2/3 — Walk-forward on best params")
    if progress:
        progress("walk_forward")
    if best_params:
        wf_grid = _fixed_param_grid(best_params)
        wf_cache = None
        if cache_path:
            base = Path(cache_path)
            wf_cache = base.with_name(f"{base.stem}_research_wf{base.suffix}")
        walk_forward = walk_forward_optimize(
            cfg,
            fetcher,
            wf_grid,
            start_date=start_date,
            end_date=end_date,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            metric=metric,
            min_trades=min_trades,
            max_workers=max_workers,
            cache_path=wf_cache,
        )
    else:
        walk_forward = WalkForwardSummary(
            windows=[],
            stability_score=1.0,
            aggregate_metrics={},
            overfit_flag=False,
            train_test_score_ratio=0.0,
        )
    oos_metric = float(walk_forward.aggregate_metrics.get(metric, 0.0))
    # Prefer mean IS train score from WF windows when available (true IS on
    # train folds); fall back to full-sample grid best score.
    if walk_forward.windows:
        is_from_wf = float(
            sum(w.best_train.score for w in walk_forward.windows)
            / len(walk_forward.windows)
        )
    else:
        is_from_wf = is_metric
    degradation = _degradation_ratio(is_from_wf, oos_metric)
    console.print(
        f"Windows: {len(walk_forward.windows)}  "
        f"IS {metric}: {is_from_wf:.4f}  "
        f"OOS {metric}: {oos_metric:.4f}  "
        f"degradation: {degradation:.3f}  "
        f"overfit_flag: {walk_forward.overfit_flag}"
    )
    timings["walk_forward_seconds"] = _stage_done(console, started)

    # ── Stage 3: Monte Carlo on best-param trade list ────────────────────
    started = _banner(console, "Stage 3/3 — Monte Carlo bootstrap")
    if progress:
        progress("monte_carlo")
    trades: list[Trade] = []
    trade_source = "none"
    if best_params:
        # Prefer the OOS trades collected by walk-forward; otherwise fall back
        # to a full-period run (mirrors validate on a full ledger).
        if walk_forward.windows:
            trade_source = "walk_forward_oos"
            trades = list(walk_forward.oos_trades)
        else:
            trade_source = "full_period"
            full_cfg = cfg.model_copy(update=best_params)
            full_result = run_rolling_backtest(
                full_cfg,
                fetcher,
                start_date=start_date,
                end_date=end_date,
            )
            trades = list(full_result.trades)
    monte_carlo: MonteCarloResult = simulate_monte_carlo(
        trades,
        iterations=int(mc_iterations),
        seed=int(mc_seed),
        initial_capital=float(cfg.initial_capital),
        ruin_threshold=float(ruin_threshold),
    )
    console.print(
        f"Trades ({trade_source}): {len(trades)}  "
        f"MC p05 return: {monte_carlo.return_p05:.4f}  "
        f"P(profit): {monte_carlo.probability_of_profit:.3f}  "
        f"risk_of_ruin: {monte_carlo.risk_of_ruin:.3f}"
    )
    timings["monte_carlo_seconds"] = _stage_done(console, started)

    verdict = _verdict(
        overfit_flag=walk_forward.overfit_flag,
        degradation=degradation,
        mc_return_p05=float(monte_carlo.return_p05),
        oos_metric=oos_metric,
    )

    payload: dict[str, Any] = {
        "config": {
            "market": cfg.market,
            "strategy_name": cfg.strategy_name,
            "entry_expr": cfg.entry_expr,
            "exit_expr": cfg.exit_expr,
            "start_date": start_date,
            "end_date": end_date,
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days if step_days is not None else test_days,
            "metric": metric,
            "min_trades": min_trades,
            "mc_iterations": mc_iterations,
            "mc_seed": mc_seed,
            "parameter_grid": parameter_grid,
            "tickers": list(cfg.tickers) if cfg.tickers else None,
            "universe_file": cfg.universe_file,
            "initial_capital": cfg.initial_capital,
        },
        "grid": {
            "warning": GRID_IN_SAMPLE_DISCLAIMER,
            "results": [r.model_dump(mode="json") for r in grid_results],
            "best_params": best_params,
            "best_score": is_metric if best is not None else None,
            "stability": stability,
        },
        "walk_forward": walk_forward.model_dump(mode="json"),
        "monte_carlo": {
            **monte_carlo.model_dump(mode="json"),
            "trade_count": len(trades),
            "trade_source": trade_source,
        },
        "summary": {
            "best_params": best_params,
            "is_metric": is_from_wf,
            "oos_metric": oos_metric,
            "degradation": degradation,
            "train_test_score_ratio": walk_forward.train_test_score_ratio,
            "overfit_flag": walk_forward.overfit_flag,
            "mc_return_p05": float(monte_carlo.return_p05),
            "mc_median_return": float(monte_carlo.median_return),
            "mc_probability_of_profit": float(monte_carlo.probability_of_profit),
            "verdict": verdict,
        },
        "timings": timings,
    }

    # --out reports/foo → reports/foo.json + reports/foo.html
    stem_path = out.with_suffix("") if out.suffix in {".json", ".html"} else out
    json_path = Path(str(stem_path) + ".json")
    html_path = Path(str(stem_path) + ".html")

    write_json_report(payload, json_path)
    write_research_html_report(payload, html_path)
    console.print(f"[green]Wrote[/green] {json_path}")
    console.print(f"[green]Wrote[/green] {html_path}")

    _print_final_summary(console, payload["summary"], metric=metric)
    return payload


def _print_final_summary(
    console: Console, summary: dict[str, Any], *, metric: str
) -> None:
    table = Table(
        title="Research Report Summary", show_header=True, header_style="bold"
    )
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Best params", str(summary.get("best_params")))
    table.add_row(f"IS {metric}", f"{float(summary.get('is_metric', 0.0)):.4f}")
    table.add_row(f"OOS {metric}", f"{float(summary.get('oos_metric', 0.0)):.4f}")
    table.add_row("Degradation", f"{float(summary.get('degradation', 0.0)):.3f}")
    table.add_row(
        "MC 5th-pct return", f"{float(summary.get('mc_return_p05', 0.0)):.4f}"
    )
    table.add_row("Verdict", str(summary.get("verdict", "")))
    console.print()
    console.print(table)
    console.print(f"[bold]{summary.get('verdict', '')}[/bold]")
