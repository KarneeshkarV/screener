#!/usr/bin/env python
"""Full-factorial India PIT comparison: main + PR130 + PR126.

Reuse prepared bars across hold / stop / trail. Regime changes signals,
so each regime is a new prepare. Output goes to an external temp dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from run_pit_midsmall_study import (
    CAPITAL,
    EXTRA_FIELD_STRATEGIES,
    FUND_FIELDS,
    STRATEGIES as RESEARCH_STRATEGIES,
    TOP_SLOTS,
    UNIVERSE_CONFIG,
    UNIVERSES,
    Strategy,
    _cli_defaults,
    _curve_records,
    _scalar_metrics,
    _trade_records,
    rekey_fmp_cache,
)
from screener.backtester.data import build_price_fetcher
from screener.backtester.fundamentals import DEFAULT_FUNDAMENTAL_FIELDS
from screener.backtester.rolling_simulation import (
    prepare_rolling_backtest,
    run_prepared_rolling_backtest,
)
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run

END_DATE = date(2026, 8, 17)
FY_START = date(2026, 1, 1)
DEFAULT_OUT = Path.home() / "tmp" / "screener-india-pit-compare-2026-08-17"
SKIP = frozenset({"sloan_low_accruals", "piotroski_value", "fcf_yield_value"})
HOLD_SHORT = {10: 10, 20: 10, 21: 10, 63: 21, 126: 63, 250: 126}
HOLD_LONG = {10: 21, 20: 63, 21: 63, 63: 126, 126: 250, 250: 250}
STOPS: tuple[float | None, ...] = (None, 0.08, 0.15)
TRAILS: tuple[float | None, ...] = (None, 0.12, 0.20)
UNIVERSE_ORDER = ("midsmall", "n500", "mid", "small")
WINDOW_ORDER = ("5y", "3y", "2y", "1y", "fy")
REGIME_ORDER = ("off", "bull", "bp")

MAIN_TA: tuple[Strategy, ...] = (
    Strategy("awesome_oscillator", "main_ta", 20),
    Strategy("bb_breakout", "main_ta", 20),
    Strategy("bb_pattern", "main_ta", 20),
    Strategy("donchian_breakout", "main_ta", 20),
    Strategy("heikin_ashi", "main_ta", 20),
    Strategy("macd_oscillator", "main_ta", 20),
    Strategy("macd_rsi", "main_ta", 20),
    Strategy("ma_cross", "main_ta", 20),
    Strategy("ma_cross_regime", "main_ta", 20),
    Strategy("ma_cross_st_entry", "main_ta", 20),
    Strategy("ma_cross_st_exit", "main_ta", 20),
    Strategy("parabolic_sar", "main_ta", 20),
    Strategy("rsi_ema", "main_ta", 20),
    Strategy("rsi_pattern", "main_ta", 20),
    Strategy("rsi_reversion", "main_ta", 20),
    Strategy("shooting_star", "main_ta", 20),
    Strategy("supertrend", "main_ta", 20),
    Strategy("supertrend_rsi", "main_ta", 20),
)

PR126: tuple[Strategy, ...] = (
    Strategy("dual_momentum_gem", "pr126", 63),
    Strategy("dual_momentum_market", "pr126", 63),
    Strategy("dual_momentum_paa", "pr126", 63),
    Strategy("dual_momentum_daa", "pr126", 63),
    Strategy("faber_sma10", "pr126", 126),
    Strategy("absolute_momentum", "pr126", 126),
    Strategy("industry_trend_breakout", "pr126", 20),
    Strategy("momentum_12_1_volmanaged", "pr126", 63),
    Strategy("momentum_12_1_dynamic", "pr126", 63),
    Strategy("tsmom_12", "pr126", 126),
    Strategy("tsmom_blend", "pr126", 126),
)

ALL_STRATEGIES: tuple[Strategy, ...] = tuple(
    s for s in (*RESEARCH_STRATEGIES, *MAIN_TA, *PR126) if s.name not in SKIP
)
STRATEGY_BY_NAME = {s.name: s for s in ALL_STRATEGIES}

WINDOWS: dict[str, dict[str, Any]] = {
    "5y": {"years": 5, "start": None, "end": END_DATE},
    "3y": {"years": 3, "start": None, "end": END_DATE},
    "2y": {"years": 2, "start": None, "end": END_DATE},
    "1y": {"years": 1, "start": None, "end": END_DATE},
    "fy": {"years": 1, "start": FY_START, "end": END_DATE},
}

REGIMES: dict[str, tuple[str, ...]] = {
    "off": (),
    "bull": ("bull",),
    "bp": ("bull", "pullback"),
}


@dataclass(frozen=True)
class BookCfg:
    hold: int
    sl: float | None
    tr: float | None

    def cfg_id(self) -> str:
        return f"h{self.hold}_{_pct('sl', self.sl)}_{_pct('tr', self.tr)}"


def _pct(prefix: str, value: float | None) -> str:
    if value is None:
        return f"{prefix}none"
    return f"{prefix}{int(round(value * 100)):02d}"


def holds_for(base: int) -> tuple[int, ...]:
    values = (HOLD_SHORT.get(base, max(10, base // 2)), base, HOLD_LONG.get(base, min(250, base * 2)))
    return tuple(dict.fromkeys(values))


def books_for(base_hold: int) -> list[BookCfg]:
    return [BookCfg(hold, sl, tr) for hold in holds_for(base_hold) for sl in STOPS for tr in TRAILS]


def cell_path(out_dir: Path, universe: str, window: str, strategy: str, regime: str, book: BookCfg) -> Path:
    return (
        out_dir
        / "runs"
        / universe
        / window
        / strategy
        / f"{book.cfg_id()}_rg{regime}.json"
    )


def build_request(
    strategy: Strategy,
    universe_key: str,
    window: str,
    fetcher: Any,
    *,
    hold: int,
    stop_loss: float | None,
    trailing_stop: float | None,
    regime: str,
) -> BacktestRequest:
    spec = WINDOWS[window]
    params = _cli_defaults()
    start = spec["start"]
    end = spec["end"]
    params.update(
        market="india",
        years=int(spec["years"]),
        start_arg=datetime(start.year, start.month, start.day) if start else None,
        end_arg=datetime(end.year, end.month, end.day),
        strategy_name=strategy.name,
        hold=int(hold),
        stop_loss=stop_loss,
        take_profit=None,
        trailing_stop=trailing_stop,
        regime_filter_args=REGIMES[regime],
        top=TOP_SLOTS,
        initial_capital=CAPITAL,
        universe=UNIVERSES[universe_key],
        universe_config=UNIVERSE_CONFIG,
        point_in_time=True,
        benchmark="^NSEI",
        cost_model="india",
        slippage_bps=10.0,
        commission_bps=0.0,
    )
    if strategy.fund:
        params["fundamentals_provider"] = "fmp"
        params["fundamental_field_args"] = (
            FUND_FIELDS if strategy.name in EXTRA_FIELD_STRATEGIES else DEFAULT_FUNDAMENTAL_FIELDS
        )
    return BacktestRequest(mode="rolling", context_obj=fetcher, **params)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str, allow_nan=False))
    tmp.replace(path)


def _error_payload(group: dict[str, Any], book: BookCfg, exc: BaseException) -> dict[str, Any]:
    return {
        "strategy": group["strategy"],
        "family": group["family"],
        "universe": group["universe"],
        "window": group["window"],
        "hold": book.hold,
        "stop_loss": book.sl,
        "trailing_stop": book.tr,
        "regime": group["regime"],
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "generated": date.today().isoformat(),
    }


def _success_payload(
    group: dict[str, Any],
    book: BookCfg,
    run: Any,
    result: Any,
    elapsed: float,
    prepare_s: float,
) -> dict[str, Any]:
    return {
        "strategy": group["strategy"],
        "family": group["family"],
        "fund": group["fund"],
        "source": group["family"],
        "market": "india",
        "universe": group["universe"],
        "universe_name": UNIVERSES[group["universe"]],
        "window": group["window"],
        "years": WINDOWS[group["window"]]["years"],
        "hold": book.hold,
        "base_hold": group["base_hold"],
        "stop_loss": book.sl,
        "trailing_stop": book.tr,
        "regime": group["regime"],
        "regime_filter": list(REGIMES[group["regime"]]),
        "start": run.start_date.isoformat() if run.start_date else None,
        "end": run.end_date.isoformat() if run.end_date else None,
        "top": TOP_SLOTS,
        "cost_model": "india",
        "slippage_bps": 10.0,
        "price_provider": "fmp",
        "point_in_time": True,
        "universe_note": run.universe_note,
        "elapsed_seconds": round(elapsed, 1),
        "prepare_seconds": round(prepare_s, 1),
        "n_trades": len(result.trades),
        "metrics": _scalar_metrics(result.metrics),
        "equity_curve": _curve_records(result.equity_curve),
        "trades": _trade_records(result.trades),
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
        "hold": payload.get("hold"),
        "base_hold": payload.get("base_hold"),
        "stop_loss": payload.get("stop_loss"),
        "trailing_stop": payload.get("trailing_stop"),
        "regime": payload.get("regime"),
        "start": payload.get("start"),
        "end": payload.get("end"),
        "n_trades": payload.get("n_trades"),
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
        "regime": payload["regime"],
        "base_hold": strategy.hold,
    }
    books = [BookCfg(int(b["hold"]), b["sl"], b["tr"]) for b in payload["books"]]
    pending = [b for b in books if not cell_path(out_dir, group["universe"], group["window"], strategy.name, group["regime"], b).exists()]
    if not pending:
        return {"ok": True, "key": payload["key"], "wrote": 0, "skipped": len(books), "sec": 0.0}

    fetcher = build_price_fetcher(provider="fmp")
    first = pending[0]
    started = time.time()
    try:
        request = build_request(
            strategy,
            group["universe"],
            group["window"],
            fetcher,
            hold=first.hold,
            stop_loss=first.sl,
            trailing_stop=first.tr,
            regime=group["regime"],
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
        for book in pending:
            _write_json(cell_path(out_dir, group["universe"], group["window"], strategy.name, group["regime"], book), _error_payload(group, book, exc))
        return {
            "ok": False,
            "key": payload["key"],
            "wrote": len(pending),
            "error": f"{type(exc).__name__}: {exc}",
            "sec": round(time.time() - started, 1),
        }

    wrote = 0
    errors = 0
    rows: list[dict[str, Any]] = []
    for book in pending:
        dest = cell_path(out_dir, group["universe"], group["window"], strategy.name, group["regime"], book)
        sim_t0 = time.time()
        try:
            cfg = run.config.model_copy(
                update={"hold": book.hold, "stop_loss": book.sl, "trailing_stop": book.tr}
            )
            if not prepared.supports(cfg):
                request = build_request(
                    strategy,
                    group["universe"],
                    group["window"],
                    fetcher,
                    hold=book.hold,
                    stop_loss=book.sl,
                    trailing_stop=book.tr,
                    regime=group["regime"],
                )
                alt = resolve_backtest_run(request)
                result = run_prepared_rolling_backtest(prepared, alt.config) if prepared.supports(alt.config) else None
                if result is None:
                    raise RuntimeError("prepared state rejected book config")
                run_for_payload = alt
            else:
                result = run_prepared_rolling_backtest(prepared, cfg)
                run_for_payload = run
            payload_out = _success_payload(group, book, run_for_payload, result, time.time() - sim_t0, prepare_s)
        except Exception as exc:  # noqa: BLE001
            payload_out = _error_payload(group, book, exc)
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
        "skipped": len(books) - wrote,
        "errors": errors,
        "prepare_s": round(prepare_s, 1),
        "sec": round(time.time() - started, 1),
    }


def plan(
    out_dir: Path,
    universes: list[str],
    windows: list[str],
    names: list[str] | None,
    regimes: list[str],
    limit_book: int,
) -> list[dict[str, Any]]:
    wanted = set(names or STRATEGY_BY_NAME)
    rank = {name: i for i, name in enumerate(STRATEGY_BY_NAME)}
    cells: list[dict[str, Any]] = []
    for window in windows:
        for universe in universes:
            for strategy in ALL_STRATEGIES:
                if strategy.name not in wanted:
                    continue
                books = books_for(strategy.hold)
                if limit_book:
                    books = books[:limit_book]
                for regime in regimes:
                    pending = [
                        b
                        for b in books
                        if not cell_path(out_dir, universe, window, strategy.name, regime, b).exists()
                    ]
                    if not pending:
                        continue
                    key = f"{universe}__{strategy.name}__{window}__rg{regime}"
                    cells.append(
                        {
                            "key": key,
                            "out_dir": str(out_dir),
                            "universe": universe,
                            "strategy": strategy.name,
                            "family": strategy.family,
                            "window": window,
                            "regime": regime,
                            "books": [{"hold": b.hold, "sl": b.sl, "tr": b.tr} for b in pending],
                            "rank": rank.get(strategy.name, 999),
                        }
                    )
    cells.sort(key=lambda c: (WINDOW_ORDER.index(c["window"]), UNIVERSE_ORDER.index(c["universe"]), c["rank"], REGIME_ORDER.index(c["regime"])))
    return cells


def write_manifest(out_dir: Path, args: argparse.Namespace, pending: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "end_date": END_DATE.isoformat(),
        "fy_start": FY_START.isoformat(),
        "universes": list(args.universe or UNIVERSE_ORDER),
        "windows": list(args.window or WINDOW_ORDER),
        "regimes": list(args.regime or REGIME_ORDER),
        "stops": STOPS,
        "trails": TRAILS,
        "strategies": [asdict(s) if hasattr(s, "__dataclass_fields__") else s.__dict__ for s in ALL_STRATEGIES],
        "n_strategies": len(ALL_STRATEGIES),
        "skipped": sorted(SKIP),
        "capital": CAPITAL,
        "top": TOP_SLOTS,
        "cost_model": "india",
        "slippage_bps": 10.0,
        "point_in_time": True,
        "pending_groups": pending,
        "engine": "screener-pit-mid-small PR130 + copied PR126 plugins",
        "price_provider": "fmp",
        "notes": [
            "Rerun from scratch. End date pinned to 2026-08-17.",
            "Full factorial of hold x stop x trail x regime.",
            "Prepared bars reused across hold/stop/trail. Regime rebuilds signals.",
            "Prices come from FMP.",
        ],
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(payload, default=str, indent=2))


def write_progress(out_dir: Path, *, total: int, done: int, fail: int, wrote: int) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-u", "--universe", action="append", choices=sorted(UNIVERSES))
    parser.add_argument("-s", "--strategy", action="append")
    parser.add_argument("-w", "--window", action="append", choices=list(WINDOWS))
    parser.add_argument("-r", "--regime", action="append", choices=list(REGIMES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--limit-book", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    universes = args.universe or list(UNIVERSE_ORDER)
    windows = args.window or list(WINDOW_ORDER)
    regimes = args.regime or list(REGIME_ORDER)
    if args.strategy:
        unknown = set(args.strategy) - set(STRATEGY_BY_NAME)
        if unknown:
            print(f"unknown strategy: {sorted(unknown)}", file=sys.stderr)
            return 2

    cells = plan(out_dir, universes, windows, args.strategy, regimes, args.limit_book)
    if args.limit_groups:
        cells = cells[: args.limit_groups]
    write_manifest(out_dir, args, len(cells))
    n_books = sum(len(c["books"]) for c in cells)
    print(
        f"pending_groups={len(cells)} pending_cells={n_books} "
        f"workers={args.workers} strategies={len(args.strategy or ALL_STRATEGIES)} "
        f"universes={universes} windows={windows} regimes={regimes}",
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
        stamp = datetime.now().isoformat(timespec="seconds")
        text = f"{stamp} {line}"
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    if workers == 1:
        iterator = ((i, run_group(cell)) for i, cell in enumerate(cells, start=1))
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = {pool.submit(run_group, cell): cell for cell in cells}
        iterator = (
            (i, fut.result())
            for i, fut in enumerate(as_completed(futures), start=1)
        )

    try:
        for index, result in iterator:
            wrote += int(result.get("wrote") or 0)
            if result.get("ok"):
                done += 1
            else:
                fail += 1
            _log(
                f"[{index}/{len(cells)}] "
                f"{'ok' if result.get('ok') else 'FAIL'} "
                f"{result.get('key')} wrote={result.get('wrote')} "
                f"prep={result.get('prepare_s')}s tot={result.get('sec')}s "
                f"{result.get('error') or ''}"
            )
            if index % 5 == 0 or index == len(cells):
                write_progress(out_dir, total=len(cells), done=done + fail, fail=fail, wrote=wrote)
    finally:
        if workers > 1:
            pool.shutdown(wait=True, cancel_futures=False)
        write_progress(out_dir, total=len(cells), done=done + fail, fail=fail, wrote=wrote)
        _log(f"finished done={done} fail={fail} wrote={wrote} sec={round(time.time() - started, 1)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
