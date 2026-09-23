#!/usr/bin/env python
"""Veto tests for India: lag, costs, and both sizing books.

    uv run python scripts/run_india_veto.py

Writes JSON under reports/all_strategies_mc/veto/.
Each job prepares once, then runs equal_slot (fixed) and
reinvested_equal_slot (compounding). Research, not advice.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "all_strategies_mc" / "veto"
ARCHIVE = ROOT / "reports" / "all_strategies_mc" / "veto_pre_fix"
UNIVERSE_CONFIG = ROOT / "universes.yaml"
END_DATE = date(2026, 9, 4)
YEARS = (5, 3, 2, 1)
WORKERS = 3

sys.path.insert(0, str(ROOT / "scripts"))
from run_all_strategies_monte_carlo import (  # noqa: E402
    FAMILIES,
    METRIC_KEYS,
    compact_metrics,
    configure_fmp_sources,
    expression_strategy_names,
    jsonable,
    serialize_equity,
    write_html,
)

SIZING = (
    ("equal_slot", "fixed"),
    ("reinvested_equal_slot", "compounding"),
)


def window_start(years: int) -> date:
    return END_DATE - timedelta(days=365 * int(years))


def make_request(strategy: str, years: int, extra: dict[str, Any]) -> Any:
    from screener.backtester.workflow import BacktestRequest

    payload: dict[str, Any] = {
        "mode": "rolling",
        "context_obj": None,
        "market": "india",
        "hold": 20,
        "top": 10,
        "entry_expr": None,
        "exit_expr": None,
        "strategy_name": strategy,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "cost_model": "flat",
        "initial_capital": 100_000.0,
        "benchmark": "^NSEI",
        "tickers": None,
        "universe_file": None,
        "max_universe": 0,
        "min_price": None,
        "min_avg_dollar_volume": None,
        "adv_window": 20,
        "slippage_model": "fixed",
        "half_spread_bps": 0.0,
        "vol_impact_k": 0.1,
        "no_gap_fills": False,
        "entry_order": "moo",
        "entry_limit_bps": None,
        "partial_exit_args": (),
        "price_adjustment": "full",
        "interval": "1d",
        "output_csv": False,
        "report_path": None,
        "open_report": False,
        "sizing_rule": "equal_slot",
        "compounding": True,
        "sizing_risk_pct": 0.01,
        "sizing_position_pct": 0.10,
        "sizing_atr_window": 14,
        "sizing_atr_multiple": 2.0,
        "sizing_vol_window": 20,
        "intraday_only": False,
        "start_arg": datetime.combine(window_start(years), datetime.min.time()),
        "end_arg": datetime.combine(END_DATE, datetime.min.time()),
        "years": years,
        "universe": "nifty500_pit",
        "universe_config": UNIVERSE_CONFIG,
        "point_in_time": True,
        "point_in_time_was_explicit": True,
        "compare_reinvestment": False,
        "fundamentals_provider": "fmp",
    }
    payload.update(extra)
    return BacktestRequest(**payload)


VARIANTS = (
    ("baseline", {}),
    ("lag60", {"fundamental_lag_days": 60}),
    (
        "lag60_india_slip10",
        {
            "fundamental_lag_days": 60,
            "cost_model": "india",
            "slippage_bps": 10.0,
        },
    ),
)


def record_path(strategy: str, years: int, variant: str, sizing: str) -> Path:
    return OUT / f"india__{strategy}__{years}y__{variant}__{sizing}.json"


def job_complete(strategy: str, years: int, variant: str) -> bool:
    paths = [record_path(strategy, years, variant, sizing) for sizing, _ in SIZING]
    if not all(path.exists() for path in paths):
        return False
    for path in paths:
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            return False
        if row.get("sizing") not in {sizing for sizing, _ in SIZING}:
            return False
    return True


def archive_stale_veto() -> int:
    """Move pre-fix JSON out of the live veto folder."""
    if not OUT.exists():
        return 0
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in list(OUT.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            dest = ARCHIVE / path.name
            path.replace(dest)
            moved += 1
            continue
        if row.get("sizing") in {sizing for sizing, _ in SIZING}:
            continue
        dest = ARCHIVE / path.name
        if dest.exists():
            dest = ARCHIVE / f"{path.stem}__archived.json"
        path.replace(dest)
        moved += 1
    return moved


def _book_record(
    *,
    strategy: str,
    years: int,
    variant: str,
    extra: dict[str, Any],
    sizing: str,
    sizing_label: str,
    run: Any,
    result: Any,
    compact: dict[str, Any],
    mc_error: str | None,
    elapsed: float,
    error: str | None,
) -> dict[str, Any]:
    return {
        "run_id": f"india__{strategy}__{years}y__{variant}__{sizing}",
        "market": "india",
        "strategy": strategy,
        "family": FAMILIES.get(strategy, "other"),
        "universe": "nifty500_pit",
        "benchmark": "^NSEI",
        "years": years,
        "variant": variant,
        "sizing": sizing,
        "sizing_label": sizing_label,
        "compounding": True,
        "start": None if run is None else run.start_date.isoformat(),
        "end": None if run is None else run.end_date.isoformat(),
        "hold": 20,
        "top": 10,
        "fundamental_lag_days": extra.get("fundamental_lag_days"),
        "cost_model": extra.get("cost_model", "flat"),
        "slippage_bps": extra.get("slippage_bps", 0.0),
        "metrics": compact,
        "equity": [] if result is None else serialize_equity(result.equity_curve),
        "warnings": [] if result is None else list(result.warnings)[:12],
        "mc_error": mc_error,
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
    }


def run_one(strategy: str, years: int, variant: str, extra: dict[str, Any]) -> dict[str, Any]:
    from screener.backtester.optimization.monte_carlo import (
        equity_monte_carlo_metrics,
        simulate_equity_monte_carlo,
    )
    from screener.backtester.rolling_simulation import (
        prepare_rolling_backtest,
        run_prepared_rolling_backtest,
    )
    from screener.backtester.workflow import resolve_backtest_run

    run_id = f"india__{strategy}__{years}y__{variant}"
    if job_complete(strategy, years, variant):
        return {"run_id": run_id, "status": "skipped", "elapsed": 0.0, "error": None}
    t0 = time.time()
    configure_fmp_sources()
    error = None
    try:
        request = make_request(strategy, years, extra)
        run = resolve_backtest_run(request)
        assert run.start_date is not None and run.end_date is not None
        prepared = prepare_rolling_backtest(
            run.config,
            run.price_fetcher,
            start_date=run.start_date,
            end_date=run.end_date,
            fundamental_fetcher=run.fundamental_fetcher,
        )
        for sizing, sizing_label in SIZING:
            book_error = None
            mc_error = None
            compact = {k: None for k in METRIC_KEYS}
            compact["excess_return"] = None
            result = None
            try:
                config = run.config.model_copy(update={"sizing_rule": sizing})
                result = run_prepared_rolling_backtest(prepared, config)
                metrics = dict(result.metrics)
                try:
                    mc = simulate_equity_monte_carlo(
                        result.equity_curve,
                        iterations=5000,
                        block=20,
                        seed=42,
                        ruin_threshold=0.5,
                    )
                    metrics.update(equity_monte_carlo_metrics(mc))
                except Exception as exc:  # noqa: BLE001
                    mc_error = f"{type(exc).__name__}: {exc}"
                compact = compact_metrics(metrics)
            except Exception as exc:  # noqa: BLE001
                book_error = f"{type(exc).__name__}: {exc}"
                error = error or book_error
            record = _book_record(
                strategy=strategy,
                years=years,
                variant=variant,
                extra=extra,
                sizing=sizing,
                sizing_label=sizing_label,
                run=run,
                result=result,
                compact=compact,
                mc_error=mc_error,
                elapsed=time.time() - t0,
                error=book_error,
            )
            record_path(strategy, years, variant, sizing).write_text(
                json.dumps(jsonable(record), indent=2, sort_keys=True)
            )
            del result
            gc.collect()
        del prepared
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        for sizing, sizing_label in SIZING:
            record = _book_record(
                strategy=strategy,
                years=years,
                variant=variant,
                extra=extra,
                sizing=sizing,
                sizing_label=sizing_label,
                run=None,
                result=None,
                compact={k: None for k in METRIC_KEYS} | {"excess_return": None},
                mc_error=None,
                elapsed=time.time() - t0,
                error=error,
            )
            record_path(strategy, years, variant, sizing).write_text(
                json.dumps(jsonable(record), indent=2, sort_keys=True)
            )
    return {
        "run_id": run_id,
        "status": "error" if error else "ok",
        "elapsed": round(time.time() - t0, 1),
        "error": error,
    }


def load_veto_rows() -> list[dict[str, Any]]:
    rows = []
    if not OUT.exists():
        return rows
    for path in sorted(OUT.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def rebuild_page(planned: int) -> None:
    site = ROOT / "reports" / "all_strategies_mc"
    write_html(site, load_veto_rows(), total_planned=planned)


def write_status(payload: dict[str, Any]) -> None:
    site = ROOT / "reports" / "all_strategies_mc"
    site.mkdir(parents=True, exist_ok=True)
    body = {
        **payload,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    (site / "veto_status.json").write_text(json.dumps(body, indent=2))


def job_run(payload: dict[str, Any]) -> dict[str, Any]:
    return run_one(
        payload["strategy"],
        payload["years"],
        payload["variant"],
        payload["extra"],
    )


def main() -> int:
    configure_fmp_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    moved = archive_stale_veto()
    strategies = expression_strategy_names()
    print(
        f"strategies {len(strategies)}; archived stale {moved}; "
        f"sizing {[s for s, _ in SIZING]}",
        flush=True,
    )
    jobs = [
        {
            "strategy": strategy,
            "years": years,
            "variant": variant,
            "extra": extra,
        }
        for strategy in strategies
        for years in YEARS
        for variant, extra in VARIANTS
    ]
    pending = [
        job
        for job in jobs
        if not job_complete(job["strategy"], job["years"], job["variant"])
    ]
    planned_rows = len(jobs) * len(SIZING)
    print(
        f"planned jobs {len(jobs)}; books {planned_rows}; "
        f"pending {len(pending)}; workers {WORKERS}",
        flush=True,
    )
    write_status(
        {
            "state": "running",
            "planned_jobs": len(jobs),
            "planned_rows": planned_rows,
            "pending": len(pending),
            "done": len(jobs) - len(pending),
            "failed": 0,
        }
    )
    rebuild_page(planned_rows)
    failed = 0
    done = len(jobs) - len(pending)
    if pending:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(job_run, job): job for job in pending}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    job = futures[future]
                    result = {
                        "run_id": (
                            f"india__{job['strategy']}__{job['years']}y__{job['variant']}"
                        ),
                        "status": "error",
                        "elapsed": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                done += 1
                if result.get("status") == "error":
                    failed += 1
                print(
                    f"[{done}/{len(jobs)}] {result['run_id']} {result['status']} "
                    f"{result.get('elapsed', 0)}s",
                    flush=True,
                )
                if result.get("error"):
                    print(str(result["error"])[:400], flush=True)
                write_status(
                    {
                        "state": "running",
                        "planned_jobs": len(jobs),
                        "planned_rows": planned_rows,
                        "pending": len(jobs) - done,
                        "done": done,
                        "failed": failed,
                        "last": result,
                    }
                )
                if done % 4 == 0 or result.get("status") == "error":
                    rebuild_page(planned_rows)
    rebuild_page(planned_rows)
    write_status(
        {
            "state": "done",
            "planned_jobs": len(jobs),
            "planned_rows": planned_rows,
            "pending": 0,
            "done": done,
            "failed": failed,
        }
    )
    print(
        f"done. failed={failed} rows={planned_rows} "
        f"page=reports/all_strategies_mc/index.html",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
