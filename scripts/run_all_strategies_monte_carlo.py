#!/usr/bin/env python
"""Rolling backtest + equity Monte Carlo for every expression strategy.

Writes per-run JSON under ``reports/all_strategies_mc/runs/`` and rebuilds
``reports/all_strategies_mc/index.html`` as each run finishes.

    uv run python scripts/run_all_strategies_monte_carlo.py
    uv run python scripts/run_all_strategies_monte_carlo.py --report-only
    uv run python scripts/run_all_strategies_monte_carlo.py --workers 4

Callable-only plugins have no entry/exit expressions and are skipped.
Research, not financial advice.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "reports" / "all_strategies_mc"
UNIVERSE_CONFIG = ROOT / "universes.yaml"
END_DATE = date(2026, 9, 4)
YEARS = 2
HOLD = 20
TOP = 10
MC_ITERATIONS = 5000
MC_BLOCK = 20
MC_SEED = 42
MC_RUIN = 0.5
PRICE_PROVIDER = "fmp"
FUNDAMENTALS_PROVIDER = "fmp"
CALLABLE_ONLY = ("bb_pattern", "heikin_ashi", "rsi_pattern", "shooting_star")

MARKETS: dict[str, dict[str, Any]] = {
    "us": {
        "market": "us",
        "universe": "sp500",
        "universe_config": None,
        "benchmark": "SPY",
    },
    "india": {
        "market": "india",
        "universe": "nifty500_pit",
        "universe_config": UNIVERSE_CONFIG,
        "benchmark": "^NSEI",
    },
}

FAMILIES = {
    "awesome_oscillator": "oscillator",
    "bb_breakout": "breakout",
    "breakout": "breakout",
    "donchian_breakout": "breakout",
    "ema150_200_revenue_up_3q": "trend",
    "ema_stack": "trend",
    "ema_stack_lowvol": "trend",
    "ema_trend": "trend",
    "ha_momentum": "momentum",
    "low_volatility": "momentum",
    "ma_cross": "trend",
    "ma_cross_regime": "trend",
    "ma_cross_st_entry": "trend",
    "ma_cross_st_exit": "trend",
    "macd_oscillator": "oscillator",
    "macd_rsi": "oscillator",
    "mark_minervini": "minervini",
    "minervini_growth_in": "minervini",
    "minervini_growth_us": "minervini",
    "minervini_leaders": "minervini",
    "minervini_pro_in": "minervini",
    "minervini_pro_us": "minervini",
    "mom_lowvol_combo": "momentum",
    "momentum_12_1": "momentum",
    "momentum_12_1_ema10": "momentum",
    "momentum_12_1_riskadj": "momentum",
    "momentum_12_1_trend": "momentum",
    "mq_in1": "minervini",
    "mq_in2": "minervini",
    "mq_in3": "minervini",
    "mq_us1": "minervini",
    "mq_us2": "minervini",
    "mq_us3": "minervini",
    "parabolic_sar": "breakout",
    "rs_breakout": "momentum",
    "rs_momentum_regime": "momentum",
    "rsi_ema": "oscillator",
    "rsi_reversion": "oscillator",
    "supertrend": "breakout",
    "supertrend_flip": "breakout",
    "supertrend_rsi": "breakout",
    "vivek_equity_tool": "breakout",
}

METRIC_KEYS = (
    "starting_equity",
    "final_equity",
    "total_return",
    "cagr",
    "vol_annual",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "hit_rate",
    "alpha_annual",
    "beta",
    "exposure",
    "benchmark_return",
    "trade_count",
    "unique_tickers",
    "median_trade_return",
    "avg_trade_return",
    "profit_factor",
    "expectancy",
    "winning_trades",
    "losing_trades",
    "mc_iterations",
    "mc_block",
    "mc_median_return",
    "mc_return_p05",
    "mc_return_p95",
    "mc_median_drawdown",
    "mc_drawdown_p05",
    "mc_worst_drawdown",
    "mc_probability_of_profit",
    "mc_risk_of_ruin",
    "mc_ruin_threshold",
)


def configure_fmp_sources() -> None:
    from screener.config import load_env_file

    load_env_file()
    os.environ["SCREENER_PRICE_PROVIDER"] = PRICE_PROVIDER


def window_start(years: int = YEARS, end: date = END_DATE) -> date:
    return end - timedelta(days=365 * int(years))


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def run_id_for(market: str, strategy: str) -> str:
    return f"{market}__{strategy}__{YEARS}y"


def run_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "runs" / f"{run_id}.json"


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {key: jsonable(metrics.get(key)) for key in METRIC_KEYS}
    bench = metrics.get("benchmark_return")
    total = metrics.get("total_return")
    if isinstance(bench, (int, float)) and isinstance(total, (int, float)):
        if math.isfinite(float(bench)) and math.isfinite(float(total)):
            out["excess_return"] = float(total) - float(bench)
        else:
            out["excess_return"] = None
    else:
        out["excess_return"] = None
    return out


def serialize_equity(series: Any) -> list[list[Any]]:
    if series is None or getattr(series, "empty", True):
        return []
    points: list[list[Any]] = []
    for ts, val in series.items():
        try:
            number = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        if hasattr(ts, "date"):
            day = ts.date().isoformat()
        else:
            day = str(ts)[:10]
        points.append([day, round(number, 2)])
    return points


def make_request(market_key: str, strategy: str) -> Any:
    from screener.backtester.workflow import BacktestRequest

    spec = MARKETS[market_key]
    payload: dict[str, Any] = {
        "mode": "rolling",
        "context_obj": None,
        "market": spec["market"],
        "hold": HOLD,
        "top": TOP,
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
        "benchmark": spec["benchmark"],
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
        "start_arg": datetime.combine(window_start(), datetime.min.time()),
        "end_arg": datetime.combine(END_DATE, datetime.min.time()),
        "years": YEARS,
        "universe": spec["universe"],
        "universe_config": spec["universe_config"],
        "point_in_time": True,
        "point_in_time_was_explicit": True,
        "compare_reinvestment": False,
        "fundamentals_provider": FUNDAMENTALS_PROVIDER,
    }
    return BacktestRequest(**payload)


def expression_strategy_names() -> list[str]:
    from screener.strategies.expressions import NAMED_STRATEGIES

    return sorted(NAMED_STRATEGIES)


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """One market + strategy: rolling backtest, then equity-curve Monte Carlo."""
    configure_fmp_sources()
    market = str(payload["market"])
    strategy = str(payload["strategy"])
    out_dir = Path(payload["out_dir"])
    run_id = run_id_for(market, strategy)
    path = run_path(out_dir, run_id)
    t0 = time.time()
    if path.exists() and not payload.get("force"):
        return {
            "run_id": run_id,
            "market": market,
            "strategy": strategy,
            "status": "skipped",
            "elapsed": 0.0,
            "error": None,
        }
    error = None
    record: dict[str, Any] | None = None
    try:
        from screener.backtester.optimization.monte_carlo import (
            equity_monte_carlo_metrics,
            simulate_equity_monte_carlo,
        )
        from screener.backtester.rolling_simulation import (
            prepare_rolling_backtest,
            run_prepared_rolling_backtest,
        )
        from screener.backtester.workflow import resolve_backtest_run

        request = make_request(market, strategy)
        run = resolve_backtest_run(request)
        assert run.start_date is not None and run.end_date is not None
        prepared = prepare_rolling_backtest(
            run.config,
            run.price_fetcher,
            start_date=run.start_date,
            end_date=run.end_date,
            fundamental_fetcher=run.fundamental_fetcher,
        )
        result = run_prepared_rolling_backtest(prepared, run.config)
        metrics = dict(result.metrics)
        mc_error = None
        try:
            mc = simulate_equity_monte_carlo(
                result.equity_curve,
                iterations=MC_ITERATIONS,
                block=MC_BLOCK,
                seed=MC_SEED,
                ruin_threshold=MC_RUIN,
            )
            metrics.update(equity_monte_carlo_metrics(mc))
        except Exception as exc:  # noqa: BLE001 - keep the realized run
            mc_error = f"{type(exc).__name__}: {exc}"
        compact = compact_metrics(metrics)
        record = {
            "run_id": run_id,
            "market": market,
            "strategy": strategy,
            "family": FAMILIES.get(strategy, "other"),
            "years": YEARS,
            "start": run.start_date.isoformat(),
            "end": run.end_date.isoformat(),
            "hold": HOLD,
            "top": TOP,
            "sizing_rule": "equal_slot",
            "universe": MARKETS[market]["universe"],
            "benchmark": MARKETS[market]["benchmark"],
            "price_provider": PRICE_PROVIDER,
            "fundamentals_provider": FUNDAMENTALS_PROVIDER,
            "point_in_time": True,
            "universe_note": run.universe_note,
            "metrics": compact,
            "equity": serialize_equity(result.equity_curve),
            "warnings": list(result.warnings)[:12],
            "mc_error": mc_error,
            "error": None,
            "elapsed_seconds": round(time.time() - t0, 3),
            "generated": datetime.now().isoformat(timespec="seconds"),
        }
        del prepared
        del result
        gc.collect()
    except Exception as exc:  # noqa: BLE001 - batch must continue
        error = f"{type(exc).__name__}: {exc}"
        record = {
            "run_id": run_id,
            "market": market,
            "strategy": strategy,
            "family": FAMILIES.get(strategy, "other"),
            "years": YEARS,
            "start": window_start().isoformat(),
            "end": END_DATE.isoformat(),
            "hold": HOLD,
            "top": TOP,
            "universe": MARKETS[market]["universe"],
            "benchmark": MARKETS[market]["benchmark"],
            "metrics": {},
            "equity": [],
            "warnings": [],
            "mc_error": None,
            "error": f"{error}\n{traceback.format_exc()[-1500:]}",
            "elapsed_seconds": round(time.time() - t0, 3),
            "generated": datetime.now().isoformat(timespec="seconds"),
        }
    assert record is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(record), indent=2, sort_keys=True))
    return {
        "run_id": run_id,
        "market": market,
        "strategy": strategy,
        "status": "error" if error else "ok",
        "elapsed": round(time.time() - t0, 1),
        "error": error,
    }


def load_runs(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        return rows
    for path in sorted(runs_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def write_status(out_dir: Path, payload: dict[str, Any]) -> None:
    payload = {
        **payload,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "status.json").write_text(json.dumps(payload, indent=2))


def build_html(rows: list[dict[str, Any]], *, total_planned: int) -> str:
    slim: list[dict[str, Any]] = []
    for row in rows:
        slim.append(
            {
                "run_id": row.get("run_id"),
                "market": row.get("market"),
                "strategy": row.get("strategy"),
                "family": row.get("family") or FAMILIES.get(str(row.get("strategy")), "other"),
                "universe": row.get("universe"),
                "benchmark": row.get("benchmark"),
                "start": row.get("start"),
                "end": row.get("end"),
                "error": row.get("error"),
                "mc_error": row.get("mc_error"),
                "years": row.get("years"),
                "variant": row.get("variant") or "baseline",
                "sizing": row.get("sizing") or row.get("sizing_rule") or "equal_slot",
                "sizing_label": row.get("sizing_label")
                or (
                    "compounding"
                    if (row.get("sizing") or row.get("sizing_rule"))
                    == "reinvested_equal_slot"
                    else "fixed"
                ),
                "metrics": row.get("metrics") or {},
                "equity": (row.get("equity") or [])[::5] or row.get("equity") or [],
                "elapsed_seconds": row.get("elapsed_seconds"),
            }
        )
    data = {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "years": YEARS,
        "hold": HOLD,
        "top": TOP,
        "end": END_DATE.isoformat(),
        "mc_iterations": MC_ITERATIONS,
        "mc_block": MC_BLOCK,
        "mc_seed": MC_SEED,
        "planned": total_planned,
        "callable_skipped": list(CALLABLE_ONLY),
        "runs": slim,
    }
    payload = json.dumps(data, separators=(",", ":"))
    return _HTML.replace("__DATA__", payload)


def write_html(out_dir: Path, rows: list[dict[str, Any]], total_planned: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for row in rows:
        metrics = row.get("metrics") or {}
        summary_rows.append(
            {
                "market": row.get("market"),
                "strategy": row.get("strategy"),
                "family": row.get("family"),
                "years": row.get("years"),
                "variant": row.get("variant") or "baseline",
                "sizing": row.get("sizing") or row.get("sizing_rule") or "equal_slot",
                "sizing_label": row.get("sizing_label"),
                "error": bool(row.get("error")),
                **{key: metrics.get(key) for key in METRIC_KEYS},
                "excess_return": metrics.get("excess_return"),
            }
        )
    html = build_html(rows, total_planned=total_planned)
    (out_dir / "index.html").write_text(html)
    (out_dir / "compare.html").write_text(html)
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary_rows), indent=2))


def warm_market(market: str) -> str:
    configure_fmp_sources()
    from screener.backtester.rolling_simulation import prepare_rolling_backtest
    from screener.backtester.workflow import resolve_backtest_run

    request = make_request(market, "bb_breakout")
    run = resolve_backtest_run(request)
    assert run.start_date is not None and run.end_date is not None
    prepared = prepare_rolling_backtest(
        run.config,
        run.price_fetcher,
        start_date=run.start_date,
        end_date=run.end_date,
        fundamental_fetcher=run.fundamental_fetcher,
    )
    n_tickers = len(prepared.bars_by_tv)
    n_days = len(prepared.master_dates)
    note = run.universe_note or ""
    del prepared
    gc.collect()
    return f"{market}: {n_tickers} tickers, {n_days} session days. {note}"


def planned_jobs(strategies: list[str], markets: list[str]) -> list[dict[str, Any]]:
    return [
        {"market": market, "strategy": strategy}
        for market in markets
        for strategy in strategies
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--markets",
        default="us,india",
        help="Comma-separated market keys: us, india.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Run one strategy name instead of the full registry.",
    )
    args = parser.parse_args(argv)

    markets = [item.strip() for item in args.markets.split(",") if item.strip()]
    for market in markets:
        if market not in MARKETS:
            parser.error(f"unknown market {market!r}. known: {sorted(MARKETS)}")

    configure_fmp_sources()
    strategies = (
        [args.strategy] if args.strategy else expression_strategy_names()
    )
    jobs = planned_jobs(strategies, markets)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)

    if args.report_only:
        rows = load_runs(out_dir)
        write_html(out_dir, rows, total_planned=len(jobs))
        print(f"wrote {out_dir / 'index.html'} from {len(rows)} runs")
        return 0

    pending = []
    for job in jobs:
        path = run_path(out_dir, run_id_for(job["market"], job["strategy"]))
        if args.force or not path.exists():
            pending.append({**job, "out_dir": str(out_dir), "force": bool(args.force)})

    write_status(
        out_dir,
        {
            "state": "running",
            "planned": len(jobs),
            "pending": len(pending),
            "done": len(jobs) - len(pending),
            "failed": 0,
            "current": None,
        },
    )
    write_html(out_dir, load_runs(out_dir), total_planned=len(jobs))

    if pending:
        print("warming price caches...", flush=True)
        for market in markets:
            print(warm_market(market), flush=True)

    done = len(jobs) - len(pending)
    failed = 0
    print(f"queued {len(pending)} / {len(jobs)} runs, workers={args.workers}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(run_one, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "run_id": run_id_for(job["market"], job["strategy"]),
                        "market": job["market"],
                        "strategy": job["strategy"],
                        "status": "error",
                        "elapsed": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                done += 1
                if result.get("status") == "error":
                    failed += 1
                print(
                    f"[{done}/{len(jobs)}] {result['market']} {result['strategy']} "
                    f"{result['status']} {result['elapsed']}s",
                    flush=True,
                )
                write_status(
                    out_dir,
                    {
                        "state": "running",
                        "planned": len(jobs),
                        "pending": len(jobs) - done,
                        "done": done,
                        "failed": failed,
                        "current": f"{result['market']}/{result['strategy']}",
                        "last": result,
                    },
                )
                write_html(out_dir, load_runs(out_dir), total_planned=len(jobs))

    rows = load_runs(out_dir)
    write_html(out_dir, rows, total_planned=len(jobs))
    write_status(
        out_dir,
        {
            "state": "done",
            "planned": len(jobs),
            "pending": 0,
            "done": done,
            "failed": failed,
            "current": None,
        },
    )
    print(f"done. {done} recorded, {failed} failed. page: {out_dir / 'index.html'}", flush=True)
    return 0 if failed == 0 else 1


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
  <meta http-equiv="Pragma" content="no-cache">
  <title>Strategy ledger v4 · fixed vs compounding</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inconsolata:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --paper: #efe6d2;
      --sheet: #f7f1e3;
      --ink: #1a1410;
      --mute: #6b5e4e;
      --rule: #c9ba9a;
      --green: #1d6b4f;
      --red: #b42318;
      --amber: #b45309;
      --us: #1e4d7b;
      --india: #9a3412;
      --chip: #e4d8be;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; background: var(--paper); color: var(--ink); }
    body {
      font-family: Inconsolata, ui-monospace, monospace;
      min-height: 100vh;
      background-image:
        linear-gradient(180deg, rgba(26,20,16,.04), transparent 220px),
        repeating-linear-gradient(0deg, transparent, transparent 27px, rgba(26,20,16,.045) 28px);
    }
    button, select, input { font: inherit; color: inherit; background: var(--sheet); border: 1px solid var(--rule); }
    button { cursor: pointer; padding: 6px 10px; }
    button.active, .tab.active { background: var(--ink); color: var(--paper); border-color: var(--ink); }
    a { color: var(--us); }
    .wrap { max-width: 1440px; margin: 0 auto; padding: 28px 18px 80px; }
    .mast { display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end; border-bottom: 2px solid var(--ink); padding-bottom: 16px; }
    .kicker { letter-spacing: .22em; font-size: 11px; text-transform: uppercase; color: var(--amber); }
    h1 {
      font-family: Fraunces, serif;
      font-size: clamp(2.2rem, 5vw, 4rem);
      line-height: .92;
      margin: 8px 0 0;
      font-weight: 700;
      letter-spacing: -.03em;
    }
    .stamp {
      border: 2px solid var(--red);
      color: var(--red);
      padding: 8px 12px;
      transform: rotate(-7deg);
      font-size: 11px;
      letter-spacing: .08em;
      text-transform: uppercase;
      align-self: start;
    }
    .lede { max-width: 68ch; color: var(--mute); margin: 14px 0 0; line-height: 1.5; }
    .stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px; background: var(--rule); border: 1px solid var(--rule); margin: 22px 0; }
    .stat { background: var(--sheet); padding: 12px 14px; }
    .stat b { display: block; font-family: Fraunces, serif; font-size: 1.45rem; }
    .stat span { color: var(--mute); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 16px 0; }
    .tabs { display: flex; gap: 6px; }
    .tab { padding: 6px 12px; }
    input[type=search] { min-width: 220px; padding: 6px 10px; }
    .charts { display: grid; grid-template-columns: 1fr; gap: 18px; margin: 18px 0 24px; }
    .panel {
      background: var(--sheet);
      border: 1px solid var(--rule);
      padding: 12px 12px 10px;
      position: relative;
    }
    .panel h2 { font-family: Fraunces, serif; font-size: 1.05rem; margin: 0; font-weight: 500; }
    .panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 8px; }
    .hint { color: var(--mute); font-size: 11px; }
    .chart-box { position: relative; }
    canvas { width: 100%; height: 420px; display: block; background: #fbf7ee; cursor: crosshair; }
    .legend { display: flex; flex-wrap: wrap; gap: 6px 12px; margin-top: 8px; min-height: 22px; }
    .legend button { border: 0; background: transparent; padding: 0; display: flex; align-items: center; gap: 6px; font-size: 12px; }
    .legend i { width: 14px; height: 3px; display: inline-block; }
    .legend .on { font-weight: 700; }
    .tip {
      position: absolute; pointer-events: none; z-index: 4;
      background: var(--ink); color: var(--paper);
      padding: 6px 8px; font-size: 12px; line-height: 1.35;
      min-width: 140px; display: none;
    }
    .swatch { width: 8px; height: 8px; display: inline-block; margin-right: 4px; vertical-align: middle; }
    .table-wrap { overflow: auto; border: 1px solid var(--ink); }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { padding: 7px 8px; border-bottom: 1px solid var(--rule); white-space: nowrap; text-align: right; }
    th { position: sticky; top: 0; background: var(--ink); color: var(--paper); cursor: pointer; font-weight: 600; text-align: right; }
    th.left, td.left { text-align: left; }
    tbody tr:hover { background: #efe3c4; }
    tbody tr.sel { background: #dfe9d8; outline: 1px solid var(--green); }
    .pos { color: var(--green); }
    .neg { color: var(--red); }
    .m-us { color: var(--us); }
    .m-india { color: var(--india); }
    .chip { display: inline-block; padding: 1px 6px; background: var(--chip); font-size: 11px; }
    .foot { margin-top: 18px; color: var(--mute); font-size: 12px; max-width: 80ch; line-height: 1.45; }
    .err { color: var(--red); }
    @media (max-width: 980px) {
      .stats, .mast { grid-template-columns: 1fr; }
      canvas { height: 320px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="mast">
      <div>
        <div class="kicker">Research ledger v4 · fixed vs compounding · not financial advice</div>
        <h1>Every named strategy, two books.</h1>
        <p class="lede">India Nifty 500 PIT. Windows 5 / 3 / 2 / 1 year. Hold 20 / top 10. Fixed = equal_slot from realized equity. Compounding = reinvested_equal_slot from marked equity. Switch year, cost, and sizing with the tabs. Click a row to pin its path.</p>
      </div>
      <div class="stamp">in-sample<br>research</div>
    </div>
    <div class="stats" id="stats"></div>
    <div class="controls">
      <div class="tabs" id="years"></div>
      <div class="tabs" id="variants"></div>
      <div class="tabs" id="sizing"></div>
      <div class="tabs" id="markets"></div>
      <div class="tabs" id="families"></div>
      <select id="chartMode">
        <option value="sharpe-dd">Scatter: Sharpe vs max DD</option>
        <option value="real-p05">Scatter: realized vs MC p05</option>
      </select>
      <input id="q" type="search" placeholder="filter name">
    </div>
    <div class="charts">
      <section class="panel">
        <div class="panel-head">
          <h2 id="scatterTitle">Sharpe vs max drawdown</h2>
          <span class="hint" id="scatterHint">up and left is better</span>
        </div>
        <div class="chart-box">
          <canvas id="scatter"></canvas>
          <div class="tip" id="scatterTip"></div>
        </div>
        <div class="legend" id="scatterLegend"></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <h2>Equity, start = 100</h2>
          <span class="hint" id="equityHint">top 5 by Sharpe · click a row to pin</span>
        </div>
        <div class="chart-box">
          <canvas id="equity"></canvas>
          <div class="tip" id="equityTip"></div>
        </div>
        <div class="legend" id="equityLegend"></div>
      </section>
    </div>
    <div class="table-wrap">
      <table>
        <thead id="head"></thead>
        <tbody id="body"></tbody>
      </table>
    </div>
    <p class="foot" id="foot"></p>
  </div>
  <script>
    const DATA = __DATA__;
    const COLS = [
      ["market","Mkt"],["strategy","Strategy"],["family","Family"],["sizing","Book"],
      ["sharpe","Sharpe"],["cagr","CAGR"],["total_return","Total"],
      ["excess_return","Excess"],["max_drawdown","Max DD"],
      ["hit_rate","Hit"],["trade_count","Trades"],
      ["mc_median_return","MC med"],["mc_return_p05","MC p05"],
      ["mc_return_p95","MC p95"],["mc_probability_of_profit","P(profit)"],
      ["mc_risk_of_ruin","Ruin"],["mc_worst_drawdown","MC worst DD"]
    ];
    const PCT = new Set(["cagr","total_return","excess_return","max_drawdown","hit_rate","mc_median_return","mc_return_p05","mc_return_p95","mc_probability_of_profit","mc_risk_of_ruin","mc_worst_drawdown"]);
    const VARIANT_LABEL = {baseline:"0-cost", lag60:"lag 60d", lag60_india_slip10:"lag60 + India costs"};
    const SIZING_LABEL = {equal_slot:"fixed", reinvested_equal_slot:"compounding"};
    const state = { market: "india", family: "all", year: "5", variant: "lag60_india_slip10", sizing: "all", q: "", sort: "sharpe", dir: -1, sel: null, mode: "sharpe-dd" };

    const m = (row, key) => (row.metrics || {})[key];
    const num = (v) => (typeof v === "number" && Number.isFinite(v)) ? v : null;
    const fmtPct = (v) => { const n = num(v); return n === null ? "—" : (n*100).toFixed(1) + "%"; };
    const fmtNum = (v) => { const n = num(v); return n === null ? "—" : n.toFixed(2); };
    const cls = (v) => { const n = num(v); if (n === null) return ""; return n >= 0 ? "pos" : "neg"; };

    function rows() {
      const q = state.q.toLowerCase();
      return DATA.runs.filter(r => {
        if (state.market !== "all" && r.market !== state.market) return false;
        if (state.family !== "all" && r.family !== state.family) return false;
        if (state.year !== "all" && String(r.years || 2) !== String(state.year)) return false;
        if (state.variant !== "all" && (r.variant || "baseline") !== state.variant) return false;
        if (state.sizing !== "all" && (r.sizing || "equal_slot") !== state.sizing) return false;
        if (q && !(r.strategy || "").toLowerCase().includes(q)) return false;
        return true;
      }).sort((a,b) => {
        const strKeys = ["market","strategy","family","sizing"];
        const ka = strKeys.includes(state.sort) ? a[state.sort] : m(a, state.sort);
        const kb = strKeys.includes(state.sort) ? b[state.sort] : m(b, state.sort);
        if (ka == null && kb == null) return 0;
        if (ka == null) return 1;
        if (kb == null) return -1;
        if (typeof ka === "string") return ka.localeCompare(kb) * state.dir;
        return (ka - kb) * state.dir;
      });
    }

    function statsHtml(list) {
      const ok = list.filter(r => !r.error);
      const sharpes = ok.map(r => num(m(r,"sharpe"))).filter(v => v !== null);
      const p05 = ok.map(r => num(m(r,"mc_return_p05"))).filter(v => v !== null);
      const ruin = ok.map(r => num(m(r,"mc_risk_of_ruin"))).filter(v => v !== null);
      const best = sharpes.length ? Math.max(...sharpes) : null;
      const worstP05 = p05.length ? Math.min(...p05) : null;
      const maxRuin = ruin.length ? Math.max(...ruin) : null;
      const cells = [
        ["Runs in view", `${ok.length}/${DATA.planned}`],
        ["Failed", String(list.filter(r => r.error).length)],
        ["Best Sharpe", best === null ? "—" : best.toFixed(2)],
        ["Worst MC p05", worstP05 === null ? "—" : fmtPct(worstP05)],
        ["Highest ruin", maxRuin === null ? "—" : fmtPct(maxRuin)],
        ["Window", `${DATA.years}y to ${DATA.end}`],
      ];
      return cells.map(([k,v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join("");
    }

    function bookLabel(row) {
      return row.sizing_label || SIZING_LABEL[row.sizing || "equal_slot"] || (row.sizing || "fixed");
    }

    function nameLabel(row) {
      const book = bookLabel(row);
      return state.sizing === "all" ? `${row.strategy} · ${book}` : row.strategy;
    }

    function cell(row, key) {
      if (key === "market" || key === "strategy" || key === "family" || key === "sizing") {
        const extra = key === "market" ? ` m-${row.market}` : "";
        let text = row[key];
        if (key === "strategy" && row.error) text = row.strategy + " !";
        if (key === "sizing") text = bookLabel(row);
        return `<td class="left${extra}">${text || ""}</td>`;
      }
      const v = m(row, key);
      const text = key === "trade_count" ? (num(v) === null ? "—" : String(Math.round(v))) : (PCT.has(key) ? fmtPct(v) : fmtNum(v));
      return `<td class="${cls(v)}">${text}</td>`;
    }

    function renderTable() {
      const list = rows();
      document.getElementById("stats").innerHTML = statsHtml(list);
      document.getElementById("head").innerHTML = "<tr>" + COLS.map(([k,l]) => {
        const mark = state.sort === k ? (state.dir > 0 ? " ▲" : " ▼") : "";
        const left = (k==="market"||k==="strategy"||k==="family"||k==="sizing") ? " left" : "";
        return `<th class="${left}" data-k="${k}">${l}${mark}</th>`;
      }).join("") + "</tr>";
      document.getElementById("body").innerHTML = list.map(r => {
        const sel = state.sel === r.run_id ? " sel" : "";
        return `<tr class="${sel}" data-id="${r.run_id}">${COLS.map(([k]) => cell(r,k)).join("")}</tr>`;
      }).join("");
      drawScatter(list);
      drawEquity(list);
    }

    const PALETTE = ["#1d6b4f","#1e4d7b","#9a3412","#b45309","#0f766e","#7c2d12"];
    const scatterHits = [];
    const equityHits = { series: [], X: null, y0: 0, y1: 1, L: 0, T: 0, B: 0, w: 0, h: 0 };
    let listenersBound = false;

    function padRange(min, max, frac) {
      if (!(max > min)) { min -= 1; max += 1; }
      const pad = (max - min) * frac;
      return [min - pad, max + pad];
    }

    function ticks(min, max, count) {
      const span = Math.abs(max - min) || 1;
      const raw = span / count;
      const mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const step = [1, 2, 2.5, 5, 10].map(n => n * mag).find(n => n >= raw) || mag * 10;
      const start = Math.ceil(min / step) * step;
      const out = [];
      for (let v = start; v <= max + step * 0.01; v += step) out.push(v);
      return out;
    }

    function fitCanvas(c) {
      const dpr = window.devicePixelRatio || 1;
      const rect = c.getBoundingClientRect();
      const w = Math.max(1, rect.width);
      const h = Math.max(1, rect.height);
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
      const ctx = c.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, w, h };
    }

    function axisBox(ctx, w, h, x0, x1, y0, y1, L, R, T, B, fmtX, fmtY) {
      const X = v => L + (v - x0) / (x1 - x0) * (w - L - R);
      const Y = v => T + (1 - (v - y0) / (y1 - y0)) * (h - T - B);
      ctx.strokeStyle = "#e6dcc4";
      ctx.lineWidth = 1;
      ctx.font = "11px Inconsolata";
      ctx.fillStyle = "#6b5e4e";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const v of ticks(y0, y1, 5)) {
        const y = Y(v);
        ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke();
        ctx.fillText(fmtY(v), L - 6, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const v of ticks(x0, x1, 5)) {
        const x = X(v);
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke();
        ctx.fillText(fmtX(v), x, h - B + 6);
      }
      ctx.strokeStyle = "#1a1410";
      ctx.beginPath();
      ctx.moveTo(L, T);
      ctx.lineTo(L, h - B);
      ctx.lineTo(w - R, h - B);
      ctx.stroke();
      return { X, Y };
    }

    function showTip(el, html, px, py, box) {
      el.innerHTML = html;
      el.style.display = "block";
      const tw = el.offsetWidth;
      const th = el.offsetHeight;
      let left = px + 12;
      let top = py - th - 8;
      if (left + tw > box.clientWidth - 4) left = px - tw - 12;
      if (top < 4) top = py + 12;
      el.style.left = left + "px";
      el.style.top = top + "px";
    }

    function hideTip(el) { el.style.display = "none"; }

    function scatterPoint(r) {
      if (state.mode === "real-p05") {
        return { r, x: num(m(r, "mc_return_p05")), y: num(m(r, "total_return")) };
      }
      const dd = num(m(r, "max_drawdown"));
      return { r, x: dd === null ? null : Math.abs(dd), y: num(m(r, "sharpe")) };
    }

    function drawScatter(list) {
      const c = document.getElementById("scatter");
      const { ctx, w, h } = fitCanvas(c);
      scatterHits.length = 0;
      const pct = state.mode !== "sharpe-dd";
      document.getElementById("scatterTitle").textContent =
        pct ? "Realized return vs Monte Carlo p05" : "Sharpe vs max drawdown";
      document.getElementById("scatterHint").textContent =
        pct ? "above the diagonal beat the 5th percentile" : "up and left is better";
      const pts = list.filter(r => !r.error).map(scatterPoint).filter(p => p.x !== null && p.y !== null);
      document.getElementById("scatterLegend").innerHTML =
        '<span><i class="swatch" style="background:#1e4d7b;border-radius:50%;width:8px;height:8px"></i>US circle</span>' +
        '<span><i class="swatch" style="background:#9a3412;width:8px;height:8px"></i>India square</span>' +
        '<span>labels = top Sharpe in view · hover any point</span>';
      if (!pts.length) return;
      const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
      const [x0, x1] = padRange(Math.min(...xs), Math.max(...xs), 0.14);
      const [y0, y1] = padRange(Math.min(...ys), Math.max(...ys), 0.16);
      const L = 58, R = 200, T = 18, B = 40;
      const fmt = v => pct ? (v * 100).toFixed(0) + "%" : (state.mode === "sharpe-dd" ? (v >= 1 || v <= -1 ? v.toFixed(1) : v.toFixed(2)) : v.toFixed(2));
      const fmtX = v => state.mode === "sharpe-dd" ? (v * 100).toFixed(0) + "%" : fmt(v);
      const { X, Y } = axisBox(ctx, w, h, x0, x1, y0, y1, L, R, T, B, fmtX, fmt);
      ctx.fillStyle = "#6b5e4e";
      ctx.font = "11px Inconsolata";
      ctx.textAlign = "center";
      ctx.fillText(state.mode === "sharpe-dd" ? "max drawdown" : "MC p05 return", (L + w - R) / 2, h - 8);
      ctx.save();
      ctx.translate(12, (T + h - B) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(state.mode === "sharpe-dd" ? "Sharpe" : "realized return", 0, 0);
      ctx.restore();
      if (pct && x0 < 0 && x1 > 0) {
        ctx.strokeStyle = "rgba(26,20,16,.25)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), h - B); ctx.stroke();
        ctx.setLineDash([]);
      }
      if (state.mode === "sharpe-dd" && y0 < 0 && y1 > 0) {
        ctx.strokeStyle = "rgba(26,20,16,.25)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(w - R, Y(0)); ctx.stroke();
        ctx.setLineDash([]);
      }
      pts.forEach(p => {
        const x = X(p.x), y = Y(p.y);
        const selected = p.r.run_id === state.sel;
        ctx.fillStyle = p.r.market === "india" ? "#9a3412" : "#1e4d7b";
        ctx.beginPath();
        if (p.r.market === "india") {
          const s = selected ? 6 : 4;
          ctx.rect(x - s, y - s, s * 2, s * 2);
        } else {
          ctx.arc(x, y, selected ? 6 : 4, 0, Math.PI * 2);
        }
        ctx.fill();
        if (selected) {
          ctx.strokeStyle = "#1a1410";
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.lineWidth = 1;
        }
        scatterHits.push({ x, y, p });
      });
      const labelIds = new Set();
      const byMarket = { us: [], india: [] };
      pts.forEach(p => { (byMarket[p.r.market] || []).push(p); });
      ["us", "india"].forEach(mk => {
        [...(byMarket[mk] || [])].sort((a, b) => (b.y ?? -999) - (a.y ?? -999))
          .slice(0, 5).forEach(p => labelIds.add(p.r.run_id));
      });
      if (state.sel) labelIds.add(state.sel);
      const labeled = pts.filter(p => labelIds.has(p.r.run_id))
        .sort((a, b) => Y(a.y) - Y(b.y));
      const usedY = [];
      labeled.forEach(p => {
        const x = X(p.x), y = Y(p.y);
        let ly = y;
        while (usedY.some(v => Math.abs(v - ly) < 13)) ly += (ly >= y ? 13 : -13);
        usedY.push(ly);
        const lx = w - R + 8;
        ctx.strokeStyle = "rgba(26,20,16,.28)";
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(lx - 4, ly);
        ctx.stroke();
        ctx.fillStyle = p.r.market === "india" ? "#9a3412" : "#1e4d7b";
        ctx.font = (p.r.run_id === state.sel ? "700 " : "") + "12px Inconsolata";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText((p.r.market === "india" ? "IN " : "US ") + nameLabel(p.r), lx, ly);
      });
    }

    function rebase(eq) {
      if (!eq.length) return [];
      const base = eq[0][1];
      if (!base) return [];
      return eq.map(([d, v]) => [d, 100 * v / base]);
    }

    function pickEquity(list) {
      const ok = list.filter(r => !r.error && r.equity && r.equity.length);
      const top = ok.slice(0, 5);
      const ids = new Set(top.map(r => r.run_id));
      if (state.sel && !ids.has(state.sel)) {
        const extra = ok.find(r => r.run_id === state.sel);
        if (extra) top.push(extra);
      }
      return top;
    }

    function drawEquity(list) {
      const c = document.getElementById("equity");
      const { ctx, w, h } = fitCanvas(c);
      const chosen = pickEquity(list);
      const series = chosen.map((r, i) => ({
        r,
        eq: rebase(r.equity),
        color: PALETTE[i % PALETTE.length],
      })).filter(s => s.eq.length);
      document.getElementById("equityHint").textContent = state.sel
        ? "pinned row in bold"
        : "top 5 by current sort";
      document.getElementById("equityLegend").innerHTML = series.map(s => {
        const on = s.r.run_id === state.sel ? " on" : "";
        return `<button class="${on}" data-id="${s.r.run_id}"><i style="background:${s.color}"></i>${nameLabel(s.r)}</button>`;
      }).join("");
      equityHits.series = series;
      if (!series.length) return;
      const ys = series.flatMap(s => s.eq.map(p => p[1]));
      let [y0, y1] = padRange(Math.min(...ys), Math.max(...ys), 0.08);
      if (y0 > 100) y0 = 100;
      if (y1 < 100) y1 = 100;
      const L = 48, R = 168, T = 12, B = 34;
      const n = Math.max(...series.map(s => s.eq.length));
      const X = i => L + i / Math.max(n - 1, 1) * (w - L - R);
      const Y = v => T + (1 - (v - y0) / (y1 - y0)) * (h - T - B);
      ctx.strokeStyle = "#e6dcc4";
      ctx.lineWidth = 1;
      ctx.font = "11px Inconsolata";
      ctx.fillStyle = "#6b5e4e";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const v of ticks(y0, y1, 5)) {
        const y = Y(v);
        ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke();
        ctx.fillText(v.toFixed(0), L - 6, y);
      }
      ctx.strokeStyle = "#1a1410";
      ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, h - B); ctx.lineTo(w - R, h - B); ctx.stroke();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(26,20,16,.35)";
      ctx.beginPath(); ctx.moveTo(L, Y(100)); ctx.lineTo(w - R, Y(100)); ctx.stroke();
      ctx.setLineDash([]);
      const dates = series[0].eq;
      const marks = [0, Math.floor((dates.length - 1) / 2), dates.length - 1];
      ctx.fillStyle = "#6b5e4e";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      marks.forEach(i => {
        const label = (dates[i][0] || "").slice(0, 7);
        ctx.fillText(label, X(i), h - B + 6);
      });
      series.forEach(s => {
        const pinned = s.r.run_id === state.sel;
        ctx.beginPath();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = pinned ? 2.4 : 1.35;
        ctx.globalAlpha = state.sel && !pinned ? 0.35 : 1;
        s.eq.forEach((p, i) => {
          const x = X(i), y = Y(p[1]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
      const endY = [];
      series.forEach(s => {
        const last = s.eq[s.eq.length - 1];
        let ly = Y(last[1]);
        while (endY.some(v => Math.abs(v - ly) < 13)) ly += 13;
        endY.push(ly);
        ctx.fillStyle = s.color;
        ctx.font = (s.r.run_id === state.sel ? "700 " : "") + "12px Inconsolata";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(nameLabel(s.r) + " " + last[1].toFixed(0), w - R + 8, ly);
      });
      equityHits.X = X;
      equityHits.y0 = y0;
      equityHits.y1 = y1;
      equityHits.L = L;
      equityHits.R = R;
      equityHits.T = T;
      equityHits.B = B;
      equityHits.w = w;
      equityHits.h = h;
      equityHits.Y = Y;
    }

    function nearestScatter(mx, my) {
      let best = null, bestD = 14;
      for (const hit of scatterHits) {
        const d = Math.hypot(hit.x - mx, hit.y - my);
        if (d < bestD) { bestD = d; best = hit; }
      }
      return best;
    }

    function tabs(id, items, key) {
      const el = document.getElementById(id);
      el.innerHTML = items.map(([val, label]) =>
        `<button class="tab${state[key]===val?" active":""}" data-k="${key}" data-v="${val}">${label}</button>`
      ).join("");
    }

    function boot() {
      const families = ["all", ...new Set(DATA.runs.map(r => r.family).filter(Boolean))].sort();
      const years = [...new Set(DATA.runs.map(r => r.years).filter(Boolean))].sort((a,b) => b-a);
      const variants = [...new Set(DATA.runs.map(r => r.variant || "baseline"))];
      const vOrder = ["lag60_india_slip10", "lag60", "baseline"];
      variants.sort((a,b) => vOrder.indexOf(a) - vOrder.indexOf(b));
      tabs("years", [["all","All years"], ...years.map(y => [String(y), y+"y"])], "year");
      tabs("variants", variants.map(v => [v, VARIANT_LABEL[v] || v]), "variant");
      tabs("sizing", [["all","Both books"],["equal_slot","Fixed"],["reinvested_equal_slot","Compounding"]], "sizing");
      tabs("markets", [["india","India Nifty500 PIT"],["us","US SP500"],["all","All markets"]], "market");
      tabs("families", families.map(f => [f, f]), "family");
      document.getElementById("foot").innerHTML =
        `Generated ${DATA.generated}. MC ${DATA.mc_iterations} paths, block ${DATA.mc_block}, seed ${DATA.mc_seed}. ` +
        `Skipped callable-only: ${DATA.callable_skipped.join(", ")}. ` +
        `India PIT, FMP prices and fundamentals. Baseline is 0 bps. lag60+India costs is delivery STT plus 10 bps slip. ` +
        `Fixed = equal_slot from realized equity. Compounding = reinvested_equal_slot from marked equity. Research only.`;
      if (!listenersBound) {
        listenersBound = true;
        document.body.addEventListener("click", (ev) => {
          const tab = ev.target.closest("[data-k][data-v]");
          if (tab && tab.classList.contains("tab")) {
            state[tab.dataset.k] = tab.dataset.v;
            boot(); renderTable(); return;
          }
          const th = ev.target.closest("th[data-k]");
          if (th) {
            const k = th.dataset.k;
            if (state.sort === k) state.dir *= -1;
            else { state.sort = k; state.dir = (k==="strategy"||k==="family"||k==="market"||k==="sizing") ? 1 : -1; }
            renderTable(); return;
          }
          const tr = ev.target.closest("tbody tr[data-id]");
          if (tr) { state.sel = tr.dataset.id; renderTable(); return; }
          const leg = ev.target.closest("#equityLegend [data-id]");
          if (leg) { state.sel = leg.dataset.id; renderTable(); }
        });
        document.getElementById("q").addEventListener("input", (e) => { state.q = e.target.value; renderTable(); });
        document.getElementById("chartMode").addEventListener("change", (e) => { state.mode = e.target.value; renderTable(); });
        window.addEventListener("resize", () => renderTable());
        const scatter = document.getElementById("scatter");
        const scatterTip = document.getElementById("scatterTip");
        scatter.addEventListener("mousemove", (ev) => {
          const rect = scatter.getBoundingClientRect();
          const mx = ev.clientX - rect.left;
          const my = ev.clientY - rect.top;
          const hit = nearestScatter(mx, my);
          if (!hit) { hideTip(scatterTip); scatter.style.cursor = "crosshair"; return; }
          scatter.style.cursor = "pointer";
          const p = hit.p;
          const xLabel = state.mode === "sharpe-dd" ? "max DD " + fmtPct(m(p.r, "max_drawdown")) : "MC p05 " + fmtPct(p.x);
          const yLabel = state.mode === "sharpe-dd" ? "Sharpe " + fmtNum(p.y) : "total " + fmtPct(p.y);
          showTip(scatterTip, `<b>${nameLabel(p.r)}</b><br>${p.r.market} · ${p.r.family}<br>${yLabel}<br>${xLabel}`, mx, my, scatter.parentElement);
        });
        scatter.addEventListener("mouseleave", () => hideTip(scatterTip));
        scatter.addEventListener("click", (ev) => {
          const rect = scatter.getBoundingClientRect();
          const hit = nearestScatter(ev.clientX - rect.left, ev.clientY - rect.top);
          if (hit) { state.sel = hit.p.r.run_id; renderTable(); }
        });
        const equity = document.getElementById("equity");
        const equityTip = document.getElementById("equityTip");
        equity.addEventListener("mousemove", (ev) => {
          const { series, X, L, w, R } = equityHits;
          if (!X || !series.length) { hideTip(equityTip); return; }
          const rect = equity.getBoundingClientRect();
          const mx = ev.clientX - rect.left;
          const n = Math.max(...series.map(s => s.eq.length));
          const i = Math.round((mx - L) / Math.max(w - L - (R || 12), 1) * (n - 1));
          const idx = Math.max(0, Math.min(n - 1, i));
          const lines = series.map(s => {
            const pt = s.eq[Math.min(idx, s.eq.length - 1)];
            return `<span style="color:${s.color}">${nameLabel(s.r)}</span> ${pt ? pt[1].toFixed(1) : "—"}`;
          });
          const day = series[0].eq[Math.min(idx, series[0].eq.length - 1)][0];
          showTip(equityTip, `<b>${day}</b><br>${lines.join("<br>")}`, mx, my, equity.parentElement);
        });
        equity.addEventListener("mouseleave", () => hideTip(equityTip));
      }
      renderTable();
      if ((DATA.runs || []).length < (DATA.planned || 0)) {
        setTimeout(() => location.reload(), 20000);
      }
    }
    boot();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
