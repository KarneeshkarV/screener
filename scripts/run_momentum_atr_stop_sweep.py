#!/usr/bin/env python
"""Nifty 500 point-in-time ATR-stop sweep over the momentum strategies.

Crosses every momentum-family strategy against an ATR stop grid over nested
trailing windows, so the question "does a per-ticker stop help, and at what
width" gets one answer per strategy per horizon instead of one anecdote.

    uv run python scripts/run_momentum_atr_stop_sweep.py
    uv run python scripts/run_momentum_atr_stop_sweep.py --smoke
    uv run python scripts/run_momentum_atr_stop_sweep.py --report-only

Grid, per strategy:

* ``no_stop``       - the control. Whatever the strategy's own exits do.
* ``atr_{m}``       - ``--stop-atr m`` for m in 0.5 .. 4.0 step 0.5.
* ``pct_08``/``pct_12`` - flat ``--stop-loss``, so the ATR arm is measured
  against the mechanism it replaces and not only against no stop at all.

The ATR stop fields are book fields (``BOOK_CONFIG_FIELDS``), so the whole
grid for one strategy reuses a single prepared 5-year panel and the nested
windows are slices of it. That is the only reason this is affordable.

Windows are nested and all end on ``END_DATE``, so the 1-year result sits
inside the 5-year one. Read them as "how did this hold up as the window
shortened", not as four independent samples.

This is research, not financial advice.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_nifty500_pit_strategy_sweep import (  # noqa: E402
    METRIC_KEYS,
    compact_metrics,
    configure_fmp_sources,
    make_request,
    slice_prepared,
    write_run,
)

DEFAULT_OUT_DIR = ROOT / "reports" / "momentum_atr_stop_sweep"
END_DATE = date(2026, 9, 12)
YEARS = (5, 3, 2, 1)
HOLD = 20
TOP = 10
ATR_MULTIPLES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
ATR_WINDOW = 14
FLAT_STOPS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)

# The momentum family in ``NAMED_STRATEGIES``. Listed rather than pattern
# matched: "trend" and "breakout" names are momentum in substance, and a
# regex over the registry would silently gain or lose members when someone
# adds a plugin. Anything dropped from the registry is skipped with a note.
MOMENTUM_STRATEGIES = (
    "bb_breakout",
    "breakout",
    "donchian_breakout",
    "ema_trend",
    "ha_momentum",
    "mom_lowvol_combo",
    "momentum_12_1",
    "momentum_12_1_ema10",
    "momentum_12_1_riskadj",
    "momentum_12_1_trend",
    "rs_breakout",
    "rs_momentum_regime",
    "supertrend",
    "supertrend_flip",
    "supertrend_rsi",
)


def window_start(years: int, end: date = END_DATE) -> date:
    """CLI-identical trailing window: ``end - 365 * years`` calendar days."""
    return end - timedelta(days=365 * int(years))


def run_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "runs" / f"{run_id}.json"


def stop_arms() -> list[tuple[str, dict[str, Any]]]:
    """``(arm_name, book update)`` for the control and every stop setting."""
    arms: list[tuple[str, dict[str, Any]]] = [("no_stop", {})]
    for multiple in ATR_MULTIPLES:
        arms.append(
            (
                f"atr_{multiple:g}",
                {
                    "stop_mode": "atr",
                    "stop_atr_multiple": float(multiple),
                    "stop_atr_window": ATR_WINDOW,
                },
            )
        )
    for pct in FLAT_STOPS:
        arms.append((f"pct_{int(pct * 100):02d}", {"stop_loss": float(pct)}))
    return arms


def available_strategies() -> tuple[list[str], list[str]]:
    from screener.strategies.expressions import NAMED_STRATEGIES

    known = set(NAMED_STRATEGIES)
    present = [name for name in MOMENTUM_STRATEGIES if name in known]
    missing = [name for name in MOMENTUM_STRATEGIES if name not in known]
    return present, missing


def result_record(
    *,
    strategy: str,
    years: int,
    start: date,
    arm: str,
    cfg: Any,
    result: Any | None,
    error: str | None,
    elapsed: float,
    universe_note: str | None,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "years": years,
        "start": start.isoformat(),
        "end": END_DATE.isoformat(),
        "arm": arm,
        "hold": cfg.hold,
        "top": cfg.top,
        "stop_mode": cfg.stop_mode,
        "stop_atr_multiple": cfg.stop_atr_multiple,
        "stop_atr_window": cfg.stop_atr_window,
        "stop_loss": cfg.stop_loss,
        "sizing_rule": cfg.sizing_rule,
        "universe": "nifty500_pit",
        "benchmark": cfg.benchmark,
        "point_in_time": True,
        "universe_note": universe_note,
        "metrics": compact_metrics(result.metrics) if result is not None else {},
        "stop_exit_share": stop_exit_share(result),
        "warnings": list(result.warnings)[:8] if result is not None else [],
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }


def stop_exit_share(result: Any | None) -> float | None:
    """Share of trades the stop actually closed.

    A stop wide enough never to fire is indistinguishable from ``no_stop`` in
    the metrics; this is what separates "the stop did nothing" from "the stop
    helped".
    """
    if result is None or not result.trades:
        return None
    stopped = sum(1 for trade in result.trades if trade.exit_reason == "stop")
    return round(stopped / len(result.trades), 4)


def sweep_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    """One strategy: prepare the 5-year panel once, then every window x arm."""
    from screener.backtester.rolling_simulation import (
        prepare_rolling_backtest,
        run_prepared_rolling_backtest,
    )
    from screener.backtester.workflow import resolve_backtest_run

    strategy = str(payload["strategy"])
    out_dir = Path(payload["out_dir"])
    smoke = bool(payload["smoke"])
    years_list = [1] if smoke else list(YEARS)
    arms = stop_arms()[:2] if smoke else stop_arms()

    done = skipped = failed = 0
    t0 = time.time()
    configure_fmp_sources()

    pending = [
        (years, arm)
        for years in years_list
        for arm, _ in arms
        if not run_path(out_dir, f"{strategy}__{years}y__{arm}").exists()
    ]
    if not pending:
        return {
            "strategy": strategy,
            "done": 0,
            "skipped": len(years_list) * len(arms),
            "failed": 0,
            "elapsed": 0.0,
            "error": None,
        }

    try:
        prepare_years = 1 if smoke else max(years_list)
        request = make_request(
            strategy_name=strategy,
            hold=HOLD,
            top=TOP,
            years=prepare_years,
            start_arg=datetime.combine(
                window_start(prepare_years), datetime.min.time()
            ),
            end_arg=datetime.combine(END_DATE, datetime.min.time()),
        )
        run = resolve_backtest_run(request)
        assert run.start_date is not None and run.end_date is not None
        prepared = prepare_rolling_backtest(
            run.config,
            run.price_fetcher,
            start_date=run.start_date,
            end_date=run.end_date,
            fundamental_fetcher=run.fundamental_fetcher,
        )

        for years in years_list:
            start = window_start(years)
            sliced = slice_prepared(prepared, start)
            if sliced is None:
                failed += 1
                continue
            for arm, update in arms:
                run_id = f"{strategy}__{years}y__{arm}"
                path = run_path(out_dir, run_id)
                if path.exists():
                    skipped += 1
                    continue
                cfg = run.config.model_copy(update={"hold": HOLD, "top": TOP, **update})
                t_run = time.time()
                error = None
                result = None
                try:
                    if not sliced.supports(cfg):
                        raise RuntimeError(f"prepared panel rejects arm {arm}")
                    result = run_prepared_rolling_backtest(sliced, cfg)
                except Exception as exc:  # noqa: BLE001 - sweep must continue
                    error = f"{type(exc).__name__}: {exc}"
                    failed += 1
                write_run(
                    path,
                    result_record(
                        strategy=strategy,
                        years=years,
                        start=start,
                        arm=arm,
                        cfg=cfg,
                        result=result,
                        error=error,
                        elapsed=time.time() - t_run,
                        universe_note=run.universe_note,
                    ),
                )
                done += 1
        del prepared
        gc.collect()
        return {
            "strategy": strategy,
            "done": done,
            "skipped": skipped,
            "failed": failed,
            "elapsed": round(time.time() - t0, 1),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "strategy": strategy,
            "done": done,
            "skipped": skipped,
            "failed": failed + 1,
            "elapsed": round(time.time() - t0, 1),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}",
        }


def load_runs(out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((out_dir / "runs").glob("*.json")):
        payload = json.loads(path.read_text())
        row = {k: v for k, v in payload.items() if k not in {"metrics", "warnings"}}
        row.update({f"m_{k}": payload.get("metrics", {}).get(k) for k in METRIC_KEYS})
        row["m_excess_return"] = payload.get("metrics", {}).get("excess_return")
        rows.append(row)
    return pd.DataFrame(rows)


def _arm_order(arms: list[str]) -> list[str]:
    """Control, then ATR arms by width, then the flat controls."""
    ordered = [a for a in ("no_stop",) if a in arms]
    ordered += sorted(
        (a for a in arms if a.startswith("atr_")),
        key=lambda a: float(a.removeprefix("atr_")),
    )
    ordered += sorted(a for a in arms if a.startswith("pct_"))
    return ordered + [a for a in arms if a not in ordered]


def _markdown_table(pivot: pd.DataFrame, digits: int) -> str:
    """Render a pivot as a Markdown table. ``pandas.to_markdown`` needs
    ``tabulate``, which is not a dependency of this repo."""
    header = ["arm"] + [f"{c}y" for c in pivot.columns]
    rows = [header, ["---"] * len(header)]
    for arm in _arm_order(list(pivot.index)):
        cells = [arm]
        for column in pivot.columns:
            value = pivot.loc[arm, column]
            cells.append("-" if pd.isna(value) else f"{float(value):.{digits}f}")
        rows.append(cells)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_report(out_dir: Path, frame: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "runs.csv", index=False)

    ok = frame[frame["error"].isna()].copy()
    if ok.empty:
        (out_dir / "summary.md").write_text("# ATR stop sweep\n\nNo successful runs.\n")
        return

    lines = [
        "# Momentum x ATR stop sweep - Nifty 500 point-in-time",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')}. "
        f"Windows end {END_DATE.isoformat()} and are nested. "
        f"hold={HOLD}, top={TOP}, ATR window {ATR_WINDOW}.",
        "",
        f"{len(ok)} successful runs, {int(frame['error'].notna().sum())} failed, "
        f"across {ok['strategy'].nunique()} strategies.",
        "",
    ]
    panels = (
        ("Median Sharpe", "m_sharpe", 3),
        ("Median CAGR", "m_cagr", 4),
        ("Median max drawdown", "m_max_drawdown", 4),
        ("Median hit rate", "m_hit_rate", 3),
        ("Median trade count", "m_trade_count", 0),
        ("Share of trades closed by the stop (median)", "stop_exit_share", 3),
    )
    for title, column, digits in panels:
        pivot = ok.pivot_table(
            index="arm", columns="years", values=column, aggfunc="median"
        )
        lines.extend(
            [f"## {title} by arm and window", "", _markdown_table(pivot, digits), ""]
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--strategies", default=None, help="Comma-separated override.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        if args.strategies:
            strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
            missing: list[str] = []
        else:
            strategies, missing = available_strategies()
        if missing:
            print(f"not in the registry, skipped: {', '.join(missing)}")
        if args.smoke:
            strategies = strategies[:2]

        arms = stop_arms()[:2] if args.smoke else stop_arms()
        years_list = [1] if args.smoke else list(YEARS)
        total = len(strategies) * len(years_list) * len(arms)
        print(
            f"{len(strategies)} strategies x {len(years_list)} windows x "
            f"{len(arms)} arms = {total} runs, {args.workers} workers"
        )

        payloads = [
            {"strategy": s, "out_dir": str(out_dir), "smoke": args.smoke}
            for s in strategies
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(sweep_strategy, p): p["strategy"] for p in payloads}
            for finished, future in enumerate(as_completed(futures), start=1):
                summary = future.result()
                flag = " ERROR" if summary["error"] else ""
                print(
                    f"[{finished}/{len(futures)}] {summary['strategy']}: "
                    f"done={summary['done']} skipped={summary['skipped']} "
                    f"failed={summary['failed']} {summary['elapsed']}s{flag}",
                    flush=True,
                )
                if summary["error"]:
                    print(f"    {summary['error'].splitlines()[0]}", flush=True)

    frame = load_runs(out_dir)
    if frame.empty:
        print("no runs on disk")
        return 1
    write_report(out_dir, frame)
    print(f"wrote {out_dir / 'runs.csv'} and {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
