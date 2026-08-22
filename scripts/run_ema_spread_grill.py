#!/usr/bin/env python
"""EMA-spread weighting A/B across every strategy in the research set.

One difference from the baseline run: position weight. Under ``ema_spread``
sizing each entry gets a slice of its slot proportional to the strategy's own
fast-minus-slow EMA gap at the signal bar, so a stock deeper into its trend is
weighted more than one hugging its slow EMA. Everything else - entry and exit
criteria, universe, costs, slots, hold - is untouched.

Every strategy is run twice on the *same* prepared bars: ``equal_slot`` (the
baseline weighting) and ``ema_spread``. Sizing is a book field, so the pair
shares one prepare and the delta is attributable to weighting alone.

Output goes to an external directory (default ``~/grill-me``), never the
worktree and never ``tmp``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from run_full_factorial_compare import (
    ALL_STRATEGIES,
    END_DATE,
    FY_START,
    STRATEGY_BY_NAME,
    WINDOWS,
    build_request as _build_base_request,
)
from run_pit_midsmall_study import (
    CAPITAL,
    TOP_SLOTS,
    UNIVERSES,
    Strategy,
    _curve_records,
    _scalar_metrics,
    _trade_records,
    rekey_fmp_cache,
)
from screener.backtester.data import build_price_fetcher
from screener.backtester.rolling_simulation import (
    prepare_rolling_backtest,
    run_prepared_rolling_backtest,
)
from screener.backtester.sizing import ema_spread_windows
from screener.backtester.workflow import resolve_backtest_run

DEFAULT_OUT = Path.home() / "grill-me"
# An arm is a name plus the config overrides that define it, so the same rule
# can appear twice at different settings. Baseline first, so a partial run
# always has something to compare against.
#
# The risk_pct values on inverse_vol / atr_risk are calibrated to land near the
# same mean slot fraction as ema_spread. Without that, comparing arms would
# mostly compare how much leverage each one gave up rather than how it ranked
# stocks against each other.
ARM_SPECS: dict[str, dict[str, Any]] = {
    # Baseline: every entry gets a full, equal slot.
    "equal_slot": {"sizing_rule": "equal_slot"},
    # Trend-strength family: the strategy's own MA gap, normalized and clamped.
    "ema_spread": {"sizing_rule": "ema_spread"},
    "sma_spread": {"sizing_rule": "sma_spread"},
    "ma_extension": {"sizing_rule": "ma_extension"},
    # Same EMA gap, run hot: a tighter cap and a higher floor put the arm back
    # near full exposure, which separates the weighting shape from the cash
    # drag every clamped rule necessarily carries.
    "ema_spread_hi": {
        "sizing_rule": "ema_spread",
        "sizing_ema_spread_cap": 0.08,
        "sizing_ema_spread_floor": 0.50,
    },
    # Classic risk weighting, as controls: is the trend-gap effect anything
    # more than a volatility proxy?
    "inverse_vol": {"sizing_rule": "inverse_vol", "sizing_risk_pct": 0.0022},
    "atr_risk": {"sizing_rule": "atr_risk", "sizing_risk_pct": 0.0065},
}
ARMS: tuple[str, ...] = tuple(ARM_SPECS)
BASELINE = "equal_slot"
WINDOW_ORDER = ("3y", "1y", "5y")
UNIVERSE_ORDER = ("midsmall", "n500", "mid", "small")
# The book is deliberately flat: strategy-default hold, no stop, no trail, no
# regime gate. Weighting is the only lever under test.
REGIME = "off"
SPREAD_CAP = 0.20
SPREAD_FLOOR = 0.25


def cell_path(
    out_dir: Path, universe: str, window: str, strategy: str, arm: str
) -> Path:
    return out_dir / "runs" / universe / window / strategy / f"{arm}.json"


def build_request(
    strategy: Strategy,
    universe_key: str,
    window: str,
    fetcher: Any,
    *,
    arm: str,
) -> Any:
    request = _build_base_request(
        strategy,
        universe_key,
        window,
        fetcher,
        hold=strategy.hold,
        stop_loss=None,
        trailing_stop=None,
        regime=REGIME,
    )
    return replace(
        request,
        sizing_ema_spread_cap=SPREAD_CAP,
        sizing_ema_spread_floor=SPREAD_FLOOR,
        **ARM_SPECS[arm],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str, allow_nan=False))
    tmp.replace(path)


def _base_row(group: dict[str, Any], arm: str) -> dict[str, Any]:
    return {
        "strategy": group["strategy"],
        "family": group["family"],
        "universe": group["universe"],
        "window": group["window"],
        "arm": arm,
        "hold": group["base_hold"],
    }


def _error_payload(
    group: dict[str, Any], arm: str, exc: BaseException
) -> dict[str, Any]:
    return {
        **_base_row(group, arm),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "generated": date.today().isoformat(),
    }


def _success_payload(
    group: dict[str, Any],
    arm: str,
    run: Any,
    result: Any,
    elapsed: float,
    prepare_s: float,
    ema_windows: tuple[int, int],
) -> dict[str, Any]:
    trades = _trade_records(result.trades)
    # ``_trade_records`` drops entry_cost, so read it off the ledger objects.
    entry_costs = [float(t.entry_cost) for t in result.trades]
    slot = CAPITAL / TOP_SLOTS
    return {
        **_base_row(group, arm),
        "fund": group["fund"],
        "market": "india",
        "universe_name": UNIVERSES[group["universe"]],
        "years": WINDOWS[group["window"]]["years"],
        "regime": REGIME,
        "sizing_rule": ARM_SPECS[arm]["sizing_rule"],
        "arm_spec": ARM_SPECS[arm],
        "ema_fast": ema_windows[0],
        "ema_slow": ema_windows[1],
        "spread_cap": SPREAD_CAP,
        "spread_floor": SPREAD_FLOOR,
        "start": run.start_date.isoformat() if run.start_date else None,
        "end": run.end_date.isoformat() if run.end_date else None,
        "top": TOP_SLOTS,
        "initial_capital": CAPITAL,
        "cost_model": "india",
        "slippage_bps": 10.0,
        "price_provider": "fmp",
        "point_in_time": True,
        "universe_note": run.universe_note,
        "elapsed_seconds": round(elapsed, 1),
        "prepare_seconds": round(prepare_s, 1),
        "n_trades": len(result.trades),
        # How hard the weighting actually bit: mean fraction of a full slot
        # each entry was funded to. 1.0 under equal_slot by construction.
        "mean_slot_fraction": (
            round(float(sum(entry_costs) / len(entry_costs) / slot), 4)
            if entry_costs
            else None
        ),
        "metrics": _scalar_metrics(result.metrics),
        "equity_curve": _curve_records(result.equity_curve),
        "trades": trades,
        "warnings": list(result.warnings),
        "generated": date.today().isoformat(),
    }


def _metric_row(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    return {
        "strategy": payload.get("strategy"),
        "family": payload.get("family"),
        "universe": payload.get("universe"),
        "window": payload.get("window"),
        "arm": payload.get("arm"),
        "hold": payload.get("hold"),
        "ema_fast": payload.get("ema_fast"),
        "ema_slow": payload.get("ema_slow"),
        "start": payload.get("start"),
        "end": payload.get("end"),
        "n_trades": payload.get("n_trades"),
        "mean_slot_fraction": payload.get("mean_slot_fraction"),
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get("sharpe"),
        "sortino": metrics.get("sortino"),
        "max_drawdown": metrics.get("max_drawdown"),
        "hit_rate": metrics.get("hit_rate"),
        "total_return": metrics.get("total_return"),
        "exposure": metrics.get("exposure"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "error": payload.get("error") or "",
    }


def run_group(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("SCREENER_PRICE_PROVIDER", "fmp")
    os.environ.setdefault("SCREENER_AGENT", "0")
    out_dir = Path(payload["out_dir"])
    strategy = STRATEGY_BY_NAME[payload["strategy"]]
    group = {
        "strategy": strategy.name,
        "family": strategy.family,
        "fund": strategy.fund,
        "universe": payload["universe"],
        "window": payload["window"],
        "base_hold": strategy.hold,
    }
    pending = [
        arm
        for arm in payload["arms"]
        if not cell_path(
            out_dir, group["universe"], group["window"], strategy.name, arm
        ).exists()
    ]
    if not pending:
        return {"ok": True, "key": payload["key"], "wrote": 0, "sec": 0.0}

    fetcher = build_price_fetcher(provider="fmp")
    started = time.time()
    try:
        request = build_request(
            strategy,
            group["universe"],
            group["window"],
            fetcher,
            arm=pending[0],
        )
        run = resolve_backtest_run(request)
        fund = getattr(run, "fundamental_fetcher", None)
        if fund is not None and hasattr(fund, "cache_ttl"):
            fund.cache_ttl = -1.0
        assert run.start_date is not None and run.end_date is not None
        prepare_t0 = time.time()
        prepared = prepare_rolling_backtest(
            run.config,
            run.price_fetcher,
            start_date=run.start_date,
            end_date=run.end_date,
            fundamental_fetcher=run.fundamental_fetcher,
        )
        prepare_s = time.time() - prepare_t0
    except Exception as exc:  # noqa: BLE001
        for arm in pending:
            _write_json(
                cell_path(
                    out_dir, group["universe"], group["window"], strategy.name, arm
                ),
                _error_payload(group, arm, exc),
            )
        return {
            "ok": False,
            "key": payload["key"],
            "wrote": len(pending),
            "error": f"{type(exc).__name__}: {exc}",
            "sec": round(time.time() - started, 1),
        }

    ema_windows = ema_spread_windows(run.config)
    wrote = 0
    errors = 0
    rows: list[dict[str, Any]] = []
    for arm in pending:
        dest = cell_path(
            out_dir, group["universe"], group["window"], strategy.name, arm
        )
        sim_t0 = time.time()
        try:
            cfg = run.config.model_copy(update=ARM_SPECS[arm])
            if not prepared.supports(cfg):
                raise RuntimeError("prepared state rejected the sizing arm")
            result = run_prepared_rolling_backtest(prepared, cfg)
            payload_out = _success_payload(
                group, arm, run, result, time.time() - sim_t0, prepare_s, ema_windows
            )
        except Exception as exc:  # noqa: BLE001
            payload_out = _error_payload(group, arm, exc)
            errors += 1
        _write_json(dest, payload_out)
        rows.append(_metric_row(payload_out))
        wrote += 1

    shard = out_dir / "metrics" / f"w{os.getpid()}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with shard.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str, allow_nan=False) + "\n")

    return {
        "ok": errors == 0,
        "key": payload["key"],
        "wrote": wrote,
        "errors": errors,
        "prepare_s": round(prepare_s, 1),
        "sec": round(time.time() - started, 1),
    }


def plan(
    out_dir: Path, universes: list[str], windows: list[str], names: list[str] | None
) -> list[dict[str, Any]]:
    wanted = set(names or STRATEGY_BY_NAME)
    rank = {name: i for i, name in enumerate(STRATEGY_BY_NAME)}
    cells: list[dict[str, Any]] = []
    for window in windows:
        for universe in universes:
            for strategy in ALL_STRATEGIES:
                if strategy.name not in wanted:
                    continue
                pending = [
                    arm
                    for arm in ARMS
                    if not cell_path(
                        out_dir, universe, window, strategy.name, arm
                    ).exists()
                ]
                if not pending:
                    continue
                cells.append(
                    {
                        "key": f"{universe}__{strategy.name}__{window}",
                        "out_dir": str(out_dir),
                        "universe": universe,
                        "strategy": strategy.name,
                        "family": strategy.family,
                        "window": window,
                        "arms": pending,
                        "rank": rank.get(strategy.name, 999),
                    }
                )
    cells.sort(
        key=lambda c: (
            WINDOW_ORDER.index(c["window"]),
            UNIVERSE_ORDER.index(c["universe"]),
            c["rank"],
        )
    )
    return cells


def write_manifest(
    out_dir: Path, universes: list[str], windows: list[str], pending: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "question": "Does weighting entries by the strategy's own EMA gap beat equal slots?",
                "end_date": END_DATE.isoformat(),
                "fy_start": FY_START.isoformat(),
                "universes": universes,
                "windows": windows,
                "regime": REGIME,
                "arms": list(ARMS),
                "book": "strategy-default hold, no stop, no trailing stop",
                "sizing_ema_spread_cap": SPREAD_CAP,
                "sizing_ema_spread_floor": SPREAD_FLOOR,
                "ema_pair_rule": "the strategy's own fastest/slowest ema() windows, else 50/200",
                "strategies": [asdict(s) for s in ALL_STRATEGIES],
                "n_strategies": len(ALL_STRATEGIES),
                "capital": CAPITAL,
                "top": TOP_SLOTS,
                "cost_model": "india",
                "slippage_bps": 10.0,
                "point_in_time": True,
                "price_provider": "fmp",
                "pending_groups": pending,
                "notes": [
                    "Both arms share one prepare, so signals are identical across the pair.",
                    "ema_spread can only size DOWN from a slot, so it always holds less gross exposure than equal_slot.",
                    "Read sharpe / max_drawdown / mean_slot_fraction, not total_return alone.",
                ],
            },
            default=str,
            indent=2,
        )
    )


def write_progress(
    out_dir: Path, *, total: int, done: int, fail: int, wrote: int
) -> None:
    (out_dir / "progress.json").write_text(
        json.dumps(
            {
                "total_groups": total,
                "done_groups": done,
                "fail_groups": fail,
                "wrote_cells": wrote,
                "updated": datetime.now().isoformat(timespec="seconds"),
            }
        )
    )


def write_results(out_dir: Path) -> int:
    """Fold the metric shards into one flat table plus arm-vs-baseline deltas."""
    rows: list[dict[str, Any]] = []
    for shard in sorted((out_dir / "metrics").glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return 0
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["strategy", "universe", "window", "arm"], keep="last"
    )
    frame.to_csv(out_dir / "results.csv", index=False)

    keys = ["strategy", "family", "universe", "window", "hold"]
    metrics = ["cagr", "sharpe", "sortino", "max_drawdown", "total_return", "n_trades"]
    present = [m for m in metrics if m in frame.columns]
    base = frame[frame["arm"] == BASELINE].set_index(keys)
    if base.empty:
        return len(frame)

    blocks: list[pd.DataFrame] = []
    for arm in ARMS:
        if arm == BASELINE:
            continue
        test = frame[frame["arm"] == arm].set_index(keys)
        if test.empty:
            continue
        cols = [*present, "mean_slot_fraction", "ema_fast", "ema_slow"]
        cols = [c for c in cols if c in test.columns]
        joined = base[present].join(
            test[cols], lsuffix="_base", rsuffix="_arm", how="inner"
        )
        if joined.empty:
            continue
        joined.insert(0, "arm", arm)
        for metric in present:
            left = f"{metric}_base" if f"{metric}_base" in joined else metric
            right = f"{metric}_arm" if f"{metric}_arm" in joined else metric
            joined[f"d_{metric}"] = joined[right] - joined[left]
        blocks.append(joined.reset_index())
    if blocks:
        compare = pd.concat(blocks, ignore_index=True)
        compare.sort_values(["arm", "d_sharpe"], ascending=[True, False]).to_csv(
            out_dir / "compare.csv", index=False
        )
        # Arm-level scoreboard: the table to read first.
        summary = (
            compare.groupby(["arm", "window"])
            .agg(
                n=("strategy", "size"),
                sharpe_up=("d_sharpe", lambda x: int((x > 0).sum())),
                dd_better=("d_max_drawdown", lambda x: int((x > 0).sum())),
                cagr_up=("d_cagr", lambda x: int((x > 0).sum())),
                d_sharpe=("d_sharpe", "mean"),
                d_sortino=("d_sortino", "mean"),
                d_max_drawdown=("d_max_drawdown", "mean"),
                d_cagr=("d_cagr", "mean"),
                slot_frac=("mean_slot_fraction", "mean"),
            )
            .round(4)
        )
        summary.to_csv(out_dir / "summary_by_arm.csv")
    return len(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-u", "--universe", action="append", choices=sorted(UNIVERSES))
    parser.add_argument("-s", "--strategy", action="append")
    parser.add_argument("-w", "--window", action="append", choices=list(WINDOWS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    if args.report_only:
        print(f"results rows={write_results(out_dir)}")
        return 0

    universes = args.universe or ["midsmall"]
    windows = args.window or ["3y", "1y", "5y"]
    if args.strategy:
        unknown = set(args.strategy) - set(STRATEGY_BY_NAME)
        if unknown:
            print(f"unknown strategy: {sorted(unknown)}", file=sys.stderr)
            return 2

    cells = plan(out_dir, universes, windows, args.strategy)
    if args.limit_groups:
        cells = cells[: args.limit_groups]
    write_manifest(out_dir, universes, windows, len(cells))
    print(
        f"pending_groups={len(cells)} pending_cells={sum(len(c['arms']) for c in cells)} "
        f"workers={args.workers} universes={universes} windows={windows}",
        flush=True,
    )
    if args.plan_only or not cells:
        return 0

    rekey_fmp_cache()
    workers = max(1, args.workers)
    done = fail = wrote = 0
    log_path = out_dir / "run.log"
    started = time.time()

    def _log(line: str) -> None:
        text = f"{datetime.now().isoformat(timespec='seconds')} {line}"
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    pool = None
    if workers == 1:
        iterator = ((i, run_group(cell)) for i, cell in enumerate(cells, start=1))
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(run_group, cell) for cell in cells]
        iterator = (
            (i, fut.result()) for i, fut in enumerate(as_completed(futures), start=1)
        )

    try:
        for index, result in iterator:
            wrote += int(result.get("wrote") or 0)
            if result.get("ok"):
                done += 1
            else:
                fail += 1
            _log(
                f"[{index}/{len(cells)}] {'ok' if result.get('ok') else 'FAIL'} "
                f"{result.get('key')} wrote={result.get('wrote')} "
                f"prep={result.get('prepare_s')}s tot={result.get('sec')}s "
                f"{result.get('error') or ''}"
            )
            if index % 5 == 0 or index == len(cells):
                write_progress(
                    out_dir, total=len(cells), done=done + fail, fail=fail, wrote=wrote
                )
                write_results(out_dir)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)
        write_progress(
            out_dir, total=len(cells), done=done + fail, fail=fail, wrote=wrote
        )
        rows = write_results(out_dir)
        _log(
            f"finished done={done} fail={fail} wrote={wrote} rows={rows} "
            f"sec={round(time.time() - started, 1)}"
        )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
