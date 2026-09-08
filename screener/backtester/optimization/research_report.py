"""One-command research report: grid → walk-forward → Monte Carlo.

Orchestration only - all signal/math work is delegated to
:func:`grid_search`, :func:`walk_forward_optimize`, and
:func:`simulate_equity_monte_carlo`. Callers inject a :class:`PriceFetcher` so the
same price cache is reused across stages.

The full-period grid stage is descriptive only (parameter stability). Walk-forward
selects parameters inside each training fold from the original complete grid.
Monte Carlo resamples the combined walk-forward OOS equity curve with the equity
block bootstrap; full-period ledgers are never substituted for absent OOS.

Daily interval only. Explicit evidence thresholds are minimum data-integrity
checks, not statistical proof of alpha or live-trading approval.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.data import PriceFetcher
from screener.backtester.models import BacktestConfig
from screener.backtester.optimization.grid import (
    GridSearchResult,
    grid_search,
    parameter_combinations,
)
from screener.backtester.optimization.monte_carlo import (
    EquityMonteCarloResult,
    simulate_equity_monte_carlo,
    validate_equity_monte_carlo_flags,
)
from screener.backtester.optimization.reporting import (
    GRID_IN_SAMPLE_DISCLAIMER,
    write_json_report,
    write_research_html_report,
)
from screener.backtester.optimization.walk_forward import (
    OOS_EVIDENCE_CRITERIA,
    require_daily_walk_forward_scope,
    train_result_eligible,
    walk_forward_optimize,
)

# Relative score range below which a parameter optimum is treated as a plateau
# rather than a narrow spike (range / max(|best|, eps)).
_PLATEAU_RANGE_FRACTION = 0.15

FULL_PERIOD_GRID_NOTE = (
    "Full-period grid metrics are descriptive only. Walk-forward selects "
    "parameters on each training fold from the original complete grid and never "
    "reuses the full-period winner as OOS evidence."
)

EVIDENCE_INTEGRITY_NOTE = (
    "Explicit evidence thresholds are minimum data-integrity checks, not "
    "statistical proof of alpha or live-trading approval."
)

INSUFFICIENT_DATA_VERDICT = (
    "INSUFFICIENT DATA: missing or inadequate out-of-sample evidence"
)

PASS_VERDICT = (
    "PASS: descriptive checks passed (OOS vs IS and MC left tail within "
    "configured thresholds); not validated alpha or live approval"
)


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

    * ``plateau`` - range of best-per-value scores is small relative to |best|
    * ``spike`` - one value stands out (wide range)
    * ``flat`` - fewer than two evaluated values
    * ``empty`` - no usable results for this parameter
    """
    usable: list[GridSearchResult] = []
    for result in results:
        if result.error is not None:
            continue
        try:
            score = float(result.score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        usable.append(result)
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


def _degradation_ratio(is_metric: float, oos_metric: float) -> float:
    """OOS-vs-IS degradation: 0 means no loss, 1 means OOS fully wiped IS gain.

    Defined as ``1 - oos/is`` when ``is > 0``; when IS is non-positive the ratio
    is 0 if OOS >= IS else 1 (cannot interpret relative degradation).
    """
    if not math.isfinite(is_metric) or not math.isfinite(oos_metric):
        return 1.0
    if is_metric > 0:
        return float(1.0 - (oos_metric / is_metric))
    if oos_metric >= is_metric:
        return 0.0
    return 1.0


def _verdict(
    *,
    insufficient_data: bool,
    overfit_flag: bool,
    degradation: float,
    mc_return_p05: float,
    oos_metric: float | None,
) -> str:
    if insufficient_data:
        return INSUFFICIENT_DATA_VERDICT
    if oos_metric is None or not math.isfinite(float(oos_metric)):
        return INSUFFICIENT_DATA_VERDICT
    if overfit_flag or degradation >= 0.75:
        return "FAIL: severe IS→OOS degradation / overfit risk"
    if mc_return_p05 < 0 and oos_metric <= 0:
        return "FAIL: weak OOS and left-tail MC returns negative"
    if degradation >= 0.40 or mc_return_p05 < 0:
        return "CAUTION: material degradation or negative MC 5th-percentile"
    return PASS_VERDICT


def _descriptive_grid_best(
    results: Sequence[GridSearchResult],
    *,
    min_trades: int = 1,
) -> GridSearchResult | None:
    """Best finite eligible full-period result for descriptive display only."""
    for result in results:
        if train_result_eligible(result, min_trades=min_trades):
            return result
    return None


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
    mc_block: int = 20,
    ruin_threshold: float = 0.5,
    top_n: int = 10,
    out_path: Path | str,
    console: Console | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run grid → walk-forward → Monte Carlo and write JSON/HTML reports.

    Prices are read only through ``fetcher`` (caller should reuse one instance).
    Returns the JSON payload written to ``<out>.json``.

    Evidence gate (see :data:`OOS_EVIDENCE_CRITERIA`): inadequate OOS yields
    ``INSUFFICIENT DATA`` rather than PASS/FAIL/CAUTION. Daily interval only.
    """
    console = console or Console()
    out = Path(out_path)
    timings: dict[str, float] = {}

    require_daily_walk_forward_scope(cfg, parameter_grid)
    # Validate MC flags before any expensive stage, including empty-evidence paths.
    validate_equity_monte_carlo_flags(
        iterations=int(mc_iterations),
        block=int(mc_block),
        seed=int(mc_seed),
        keep_paths=0,
        ruin_threshold=float(ruin_threshold),
    )

    # ── Stage 1: descriptive full-period grid / parameter stability ──────
    started = _banner(
        console, "Stage 1/3 — Full-period grid (descriptive) & parameter stability"
    )
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
    best = _descriptive_grid_best(grid_results, min_trades=min_trades)
    best_params: dict[str, Any] = dict(best.params) if best is not None else {}
    descriptive_score = _finite_or_none(best.score) if best is not None else None
    console.print(
        f"Combos evaluated (returned): {len(grid_results)}  "
        f"Descriptive best params: {best_params}  "
        f"Descriptive best {metric}: "
        f"{descriptive_score if descriptive_score is not None else 'n/a'}"
    )
    console.print(f"[dim]{FULL_PERIOD_GRID_NOTE}[/dim]")
    for row in stability:
        console.print(
            f"  stability[{row['parameter']}]: shape={row['shape']} "
            f"range={row['score_range']} best_value={row['best_value']}"
        )
    timings["grid_seconds"] = _stage_done(console, started)

    # ── Stage 2: walk-forward on the original complete grid ──────────────
    started = _banner(
        console, "Stage 2/3 — Walk-forward (per-fold grid on original parameters)"
    )
    if progress:
        progress("walk_forward")
    wf_cache = None
    if cache_path:
        base = Path(cache_path)
        wf_cache = base.with_name(f"{base.stem}_research_wf{base.suffix}")
    walk_forward = walk_forward_optimize(
        cfg,
        fetcher,
        parameter_grid,
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
    oos_metric = _finite_or_none(walk_forward.aggregate_metrics.get(metric))
    # IS metric is the mean of eligible per-fold train scores only.
    # Full-period grid scores are never used as IS or OOS substitutes.
    if walk_forward.windows:
        is_from_wf = float(
            sum(w.best_train.score for w in walk_forward.windows)
            / len(walk_forward.windows)
        )
    else:
        is_from_wf = float("nan")
    degradation = (
        _degradation_ratio(is_from_wf, oos_metric)
        if math.isfinite(is_from_wf) and oos_metric is not None
        else 1.0
    )
    console.print(
        f"Windows eligible: {len(walk_forward.windows)}  "
        f"insufficient_data: {walk_forward.insufficient_data}  "
        f"IS {metric}: "
        f"{is_from_wf if math.isfinite(is_from_wf) else 'n/a'}  "
        f"OOS {metric}: "
        f"{oos_metric if oos_metric is not None else 'n/a'}  "
        f"degradation: {degradation:.3f}  "
        f"overfit_flag: {walk_forward.overfit_flag}"
    )
    timings["walk_forward_seconds"] = _stage_done(console, started)

    # ── Stage 3: equity block bootstrap on combined OOS equity only ──────
    started = _banner(console, "Stage 3/3 — Monte Carlo (OOS equity block bootstrap)")
    if progress:
        progress("monte_carlo")
    oos_equity = walk_forward.oos_equity
    equity_bars = int(len(oos_equity)) if isinstance(oos_equity, pd.Series) else 0
    n_returns = max(equity_bars - 1, 0)
    mc_method = "equity_block_bootstrap"
    mc_reasons: list[str] = []
    insufficient_data = bool(walk_forward.insufficient_data)
    if oos_metric is None:
        insufficient_data = True
        mc_reasons.append("selected OOS objective missing or non-finite")

    can_run_mc = (
        oos_equity is not None
        and not oos_equity.empty
        and not insufficient_data
        and n_returns >= 2
    )
    if can_run_mc and int(mc_block) >= n_returns:
        insufficient_data = True
        can_run_mc = False
        mc_reasons.append(
            f"mc_block ({int(mc_block)}) >= available OOS returns ({n_returns})"
        )

    if can_run_mc:
        assert oos_equity is not None  # narrowed for type checkers
        mc_source = "walk_forward_oos_equity"
        monte_carlo: EquityMonteCarloResult = simulate_equity_monte_carlo(
            oos_equity,
            iterations=int(mc_iterations),
            block=int(mc_block),
            seed=int(mc_seed),
            ruin_threshold=float(ruin_threshold),
        )
    else:
        # Do not bootstrap idle-cash or missing OOS paths; report an explicit empty source.
        mc_source = "none"
        if not mc_reasons:
            if walk_forward.insufficient_data:
                mc_reasons.append("walk-forward OOS evidence inadequate")
            else:
                mc_reasons.append("OOS equity unavailable for block bootstrap")
        monte_carlo = EquityMonteCarloResult.zeroed(
            iterations=int(mc_iterations),
            seed=int(mc_seed),
            initial_capital=float(cfg.initial_capital),
            block=0,
            bars=0,
            ruin_threshold=float(ruin_threshold),
        )
    console.print(
        f"OOS equity bars: {equity_bars}  "
        f"OOS trades: {len(walk_forward.oos_trades)}  "
        f"MC source: {mc_source}  method: {mc_method}  "
        f"MC p05 return: {monte_carlo.return_p05:.4f}  "
        f"P(profit): {monte_carlo.probability_of_profit:.3f}  "
        f"risk_of_ruin: {monte_carlo.risk_of_ruin:.3f}"
    )
    timings["monte_carlo_seconds"] = _stage_done(console, started)

    verdict = _verdict(
        insufficient_data=insufficient_data,
        overfit_flag=walk_forward.overfit_flag,
        degradation=degradation,
        mc_return_p05=float(monte_carlo.return_p05),
        oos_metric=oos_metric,
    )

    # Prefer fold-selected params when available; else descriptive full-period.
    summary_best_params = (
        dict(walk_forward.windows[-1].best_train.params)
        if walk_forward.windows
        else best_params
    )

    wf_payload = walk_forward.model_dump(mode="json")
    # oos_equity / oos_trades are excluded; expose a compact equity fingerprint.
    if walk_forward.oos_equity is not None and not walk_forward.oos_equity.empty:
        eq = walk_forward.oos_equity
        wf_payload["oos_equity_meta"] = {
            "bars": int(len(eq)),
            "start": str(eq.index[0].date())
            if hasattr(eq.index[0], "date")
            else str(eq.index[0]),
            "end": str(eq.index[-1].date())
            if hasattr(eq.index[-1], "date")
            else str(eq.index[-1]),
            "start_equity": _finite_or_none(eq.iloc[0]),
            "end_equity": _finite_or_none(eq.iloc[-1]),
        }
    # Non-finite aggregate metrics become null in JSON via write_json_report;
    # keep selected-objective evidence explicit when missing.
    if oos_metric is None:
        wf_payload.setdefault("evidence", {})
        if isinstance(wf_payload["evidence"], dict):
            wf_payload["evidence"]["oos_objective_finite"] = False
            wf_payload["evidence"]["adequate"] = False
        wf_payload["insufficient_data"] = True

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
            "interval": cfg.interval,
            "mc_iterations": mc_iterations,
            "mc_seed": mc_seed,
            "mc_block": mc_block,
            "parameter_grid": parameter_grid,
            "tickers": list(cfg.tickers) if cfg.tickers else None,
            "universe_file": cfg.universe_file,
            "initial_capital": cfg.initial_capital,
            "oos_evidence_criteria": OOS_EVIDENCE_CRITERIA,
            "evidence_integrity_note": EVIDENCE_INTEGRITY_NOTE,
        },
        "grid": {
            "warning": f"{GRID_IN_SAMPLE_DISCLAIMER} {FULL_PERIOD_GRID_NOTE}",
            "role": "descriptive_full_period_only",
            "results": [r.model_dump(mode="json") for r in grid_results],
            "best_params": best_params,
            "best_score": descriptive_score,
            "stability": stability,
        },
        "walk_forward": wf_payload,
        "monte_carlo": {
            **monte_carlo.model_dump(mode="json"),
            "source": mc_source,
            "method": mc_method,
            "equity_bars": equity_bars,
            "oos_return_count": n_returns,
            "trade_count": len(walk_forward.oos_trades),
            "requested_block": int(mc_block),
            "reason": "; ".join(mc_reasons) if mc_reasons else None,
            # Compatibility alias: previous payloads used trade_source.
            "trade_source": mc_source,
        },
        "summary": {
            "best_params": summary_best_params,
            "is_metric": is_from_wf if math.isfinite(is_from_wf) else None,
            "oos_metric": oos_metric if not insufficient_data else None,
            "degradation": degradation if not insufficient_data else None,
            "train_test_score_ratio": walk_forward.train_test_score_ratio,
            "overfit_flag": walk_forward.overfit_flag,
            "insufficient_data": insufficient_data,
            "oos_evidence_criteria": OOS_EVIDENCE_CRITERIA,
            "evidence_integrity_note": EVIDENCE_INTEGRITY_NOTE,
            "mc_return_p05": float(monte_carlo.return_p05),
            "mc_median_return": float(monte_carlo.median_return),
            "mc_probability_of_profit": float(monte_carlo.probability_of_profit),
            "mc_source": mc_source,
            "mc_method": mc_method,
            "mc_reason": "; ".join(mc_reasons) if mc_reasons else None,
            "verdict": verdict,
        },
        "timings": timings,
        "evidence_integrity_note": EVIDENCE_INTEGRITY_NOTE,
        "engine_same_bar_dependency": "PR156",
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
    is_metric = summary.get("is_metric")
    oos_metric = summary.get("oos_metric")
    table.add_row(
        f"IS {metric}",
        f"{float(is_metric):.4f}" if isinstance(is_metric, (int, float)) else "n/a",
    )
    table.add_row(
        f"OOS {metric}",
        f"{float(oos_metric):.4f}" if isinstance(oos_metric, (int, float)) else "n/a",
    )
    degradation = summary.get("degradation")
    table.add_row(
        "Degradation",
        f"{float(degradation):.3f}" if isinstance(degradation, (int, float)) else "n/a",
    )
    table.add_row(
        "MC 5th-pct return", f"{float(summary.get('mc_return_p05', 0.0)):.4f}"
    )
    table.add_row("MC source", str(summary.get("mc_source", "")))
    table.add_row("MC method", str(summary.get("mc_method", "")))
    table.add_row("Insufficient data", str(summary.get("insufficient_data", False)))
    table.add_row("Verdict", str(summary.get("verdict", "")))
    console.print()
    console.print(table)
    console.print(f"[bold]{summary.get('verdict', '')}[/bold]")
    note = summary.get("evidence_integrity_note")
    if note:
        console.print(f"[dim]{note}[/dim]")
