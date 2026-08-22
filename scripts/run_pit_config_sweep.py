#!/usr/bin/env python
"""Sweep hold / stop-loss / take-profit / trailing-stop around each PIT base book.

Writes findings/pit_midsmall/configs/*.json and refreshes configs/index.json.
Base books stay in findings/pit_midsmall/runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("SCREENER_PRICE_PROVIDER", "yfinance")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from pit_config_grid import cfg_id, variants_for, write_index  # noqa: E402
from run_pit_midsmall_study import (  # noqa: E402
    DEFAULT_OUT_DIR,
    STRATEGIES,
    STRATEGY_BY_NAME,
    UNIVERSES,
    rekey_fmp_cache,
    run_one,
)

UNIVERSE_ORDER = ("midsmall", "n50", "mid", "small", "n500")
SKIP = {"sloan_low_accruals", "piotroski_value", "fcf_yield_value"}
PRIORITY = (
    "momentum_12_1_trend",
    "momentum_12_1",
    "hurst_trend_quality",
    "tsmom_12_1",
    "nifty_momentum",
    "nifty_momentum_trend",
    "ema150_200_revenue_up_3q",
    "earnings_momentum",
    "pead_drift",
    "quality_lowvol",
    "quality_value",
    "max_avoidance",
    "seasonal_strong_trend",
    "gw52_proximity",
)


def config_name(universe: str, strategy: str, years: int, cfg_key: str) -> str:
    return f"india__{universe}__{strategy}__{years}y__{cfg_key}"


def _run_cell(payload: dict) -> dict:
    from screener.backtester.data import build_price_fetcher

    strategy = STRATEGY_BY_NAME[payload["strategy"]]
    fetcher = build_price_fetcher(provider="yfinance")
    started = time.time()
    try:
        result = run_one(
            strategy,
            payload["universe"],
            payload["years"],
            fetcher,
            hold=payload["hold"],
            stop_loss=payload["sl"],
            take_profit=payload["tp"],
            trailing_stop=payload["tr"],
        )
        result["tag"] = payload["tag"]
        result["cfg_id"] = payload["cfg_id"]
        out = Path(payload["out"])
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, default=str, allow_nan=False))
        tmp.replace(out)
        metrics = result.get("metrics") or {}
        return {
            "ok": True,
            "key": payload["key"],
            "sharpe": metrics.get("sharpe"),
            "cagr": metrics.get("cagr"),
            "n": result.get("n_trades"),
            "sec": round(time.time() - started, 1),
        }
    except Exception as exc:  # noqa: BLE001
        out = Path(payload["out"])
        out.write_text(
            json.dumps(
                {
                    "strategy": payload["strategy"],
                    "universe": payload["universe"],
                    "years": payload["years"],
                    "hold": payload["hold"],
                    "stop_loss": payload["sl"],
                    "take_profit": payload["tp"],
                    "trailing_stop": payload["tr"],
                    "tag": payload["tag"],
                    "cfg_id": payload["cfg_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        return {"ok": False, "key": payload["key"], "error": f"{type(exc).__name__}: {exc}"}


def plan(
    out_dir: Path,
    universes: list[str],
    years: list[int],
    names: list[str] | None,
) -> list[dict]:
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(names or [s.name for s in STRATEGIES]) - SKIP
    rank = {name: i for i, name in enumerate(PRIORITY)}
    cells: list[dict] = []
    strategies = sorted(
        STRATEGIES,
        key=lambda s: (rank.get(s.name, 100), s.name),
    )
    for year in years:
        for univ in universes:
            if univ not in UNIVERSES:
                continue
            for strategy in strategies:
                if strategy.name not in wanted:
                    continue
                for cfg in variants_for(strategy.hold):
                    cid = cfg_id(cfg.hold, cfg.sl, cfg.tp, cfg.tr)
                    key = config_name(univ, strategy.name, year, cid)
                    dest = configs_dir / f"{key}.json"
                    if dest.exists():
                        continue
                    cells.append(
                        {
                            "key": key,
                            "out": str(dest),
                            "universe": univ,
                            "strategy": strategy.name,
                            "years": year,
                            "hold": cfg.hold,
                            "sl": cfg.sl,
                            "tp": cfg.tp,
                            "tr": cfg.tr,
                            "tag": cfg.tag,
                            "cfg_id": cid,
                        }
                    )
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("-u", "--universe", action="append", choices=sorted(UNIVERSES))
    parser.add_argument("-s", "--strategy", action="append")
    parser.add_argument("-y", "--years", action="append", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    if args.index_only:
        path = write_index(out_dir)
        print(f"wrote {path}")
        return 0

    universes = args.universe or list(UNIVERSE_ORDER)
    years = args.years or [5, 3, 2, 1]
    rekey_fmp_cache()
    cells = plan(out_dir, universes, years, args.strategy)
    if args.limit:
        cells = cells[: args.limit]
    total = len(cells)
    print(f"pending {total} configs workers={args.workers} years={years}", flush=True)
    if not cells:
        write_index(out_dir)
        return 0

    ok = fail = 0
    workers = max(1, args.workers)
    if workers == 1:
        results = (_run_cell(cell) for cell in cells)
        iterator = enumerate(results, start=1)
        for index, result in iterator:
            if result["ok"]:
                ok += 1
                print(
                    f"[{index}/{total}] {result['key']} sharpe={result.get('sharpe')} "
                    f"cagr={result.get('cagr')} n={result.get('n')} {result.get('sec')}s",
                    flush=True,
                )
            else:
                fail += 1
                print(f"[{index}/{total}] FAIL {result['key']} {result.get('error')}", flush=True)
            if index % 20 == 0:
                write_index(out_dir)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_cell, cell): cell for cell in cells}
            for index, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                if result["ok"]:
                    ok += 1
                    print(
                        f"[{index}/{total}] {result['key']} sharpe={result.get('sharpe')} "
                        f"cagr={result.get('cagr')} n={result.get('n')} {result.get('sec')}s",
                        flush=True,
                    )
                else:
                    fail += 1
                    print(f"[{index}/{total}] FAIL {result['key']} {result.get('error')}", flush=True)
                if index % 15 == 0:
                    write_index(out_dir)

    dest = write_index(out_dir)
    print(f"done ok={ok} fail={fail} index={dest}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
