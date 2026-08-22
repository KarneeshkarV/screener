"""Hold / stop / take-profit grid and config index for the PIT study."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

UNIVERSES = ("midsmall", "n500", "n50", "mid", "small")
HOLD_SHORT = {10: 10, 20: 10, 21: 10, 63: 21, 126: 63, 250: 126}
HOLD_LONG = {10: 21, 20: 63, 21: 63, 63: 126, 126: 250, 250: 250}


@dataclass(frozen=True)
class Cfg:
    hold: int
    sl: float | None
    tp: float | None
    tr: float | None
    tag: str

    def key(self) -> tuple[int, float | None, float | None, float | None]:
        return (self.hold, self.sl, self.tp, self.tr)


def _pct_tag(prefix: str, value: float | None) -> str:
    if value is None:
        return f"{prefix}none"
    return f"{prefix}{int(round(value * 100)):02d}"


def cfg_id(hold: int, sl: float | None, tp: float | None, tr: float | None) -> str:
    return f"h{hold}_{_pct_tag('sl', sl)}_{_pct_tag('tp', tp)}_{_pct_tag('tr', tr)}"


def is_base(cfg: Cfg, base_hold: int) -> bool:
    return cfg.hold == base_hold and cfg.sl is None and cfg.tp is None and cfg.tr is None


def variants_for(base_hold: int) -> list[Cfg]:
    short = HOLD_SHORT.get(base_hold, max(10, base_hold // 2))
    long = HOLD_LONG.get(base_hold, min(250, base_hold * 2))
    raw = [
        Cfg(short, None, None, None, "h_short"),
        Cfg(long, None, None, None, "h_long"),
        Cfg(base_hold, 0.08, None, None, "sl8"),
        Cfg(base_hold, 0.15, None, None, "sl15"),
        Cfg(base_hold, None, 0.25, None, "tp25"),
        Cfg(base_hold, None, 0.50, None, "tp50"),
        Cfg(base_hold, None, None, 0.12, "trail12"),
        Cfg(base_hold, 0.08, 0.25, None, "sl8_tp25"),
        Cfg(base_hold, 0.15, 0.50, None, "sl15_tp50"),
        Cfg(short, 0.08, None, None, "hshort_sl8"),
        Cfg(long, None, 0.50, None, "hlong_tp50"),
    ]
    seen: set[tuple[int, float | None, float | None, float | None]] = set()
    out: list[Cfg] = []
    for cfg in raw:
        if is_base(cfg, base_hold):
            continue
        if cfg.key() in seen:
            continue
        seen.add(cfg.key())
        out.append(cfg)
    return out


def label(cfg: dict[str, Any]) -> str:
    if cfg.get("tag") == "base" or (
        cfg.get("sl") is None and cfg.get("tp") is None and cfg.get("tr") is None and cfg.get("is_base")
    ):
        return f"base h{cfg.get('hold')}"
    parts = [f"h{cfg.get('hold')}"]
    if cfg.get("sl") is not None:
        parts.append(f"sl{int(round(float(cfg['sl']) * 100))}")
    if cfg.get("tp") is not None:
        parts.append(f"tp{int(round(float(cfg['tp']) * 100))}")
    if cfg.get("tr") is not None:
        parts.append(f"tr{int(round(float(cfg['tr']) * 100))}")
    return " ".join(parts)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _row_from_payload(payload: dict[str, Any], *, tag: str, file: str, is_base: bool) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    hold = int(payload.get("hold") or 0)
    sl = payload.get("stop_loss")
    tp = payload.get("take_profit")
    tr = payload.get("trailing_stop")
    return {
        "tag": tag,
        "id": cfg_id(hold, sl, tp, tr),
        "hold": hold,
        "sl": sl,
        "tp": tp,
        "tr": tr,
        "is_base": is_base,
        "sharpe": _num(metrics.get("sharpe")),
        "cagr": _num(metrics.get("cagr")),
        "dd": _num(metrics.get("max_drawdown")),
        "exp": _num(metrics.get("exposure")),
        "hit": _num(metrics.get("hit_rate")),
        "n_trades": int(payload.get("n_trades") or 0),
        "file": file,
    }


def _min_trades(years: int) -> int:
    return {5: 10, 3: 8, 2: 5, 1: 3}.get(int(years), 8)


def parse_years(name: str) -> int | None:
    match = re.search(r"__(\d+)y(?:__|\.json$)", name)
    return int(match.group(1)) if match else None


def pick_best(rows: list[dict[str, Any]], years: int = 5) -> dict[str, Any] | None:
    if not rows:
        return None
    floor0 = _min_trades(years)
    base = next((r for r in rows if r.get("is_base")), rows[0])
    pool = [r for r in rows if (r.get("n_trades") or 0) >= floor0]
    if not pool:
        pool = rows
    base_n = int(base.get("n_trades") or 0)
    if base_n >= floor0 * 2:
        floor = max(floor0, int(0.3 * base_n))
        robust = [r for r in pool if (r.get("n_trades") or 0) >= floor]
        if robust:
            pool = robust

    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            row.get("sharpe") if row.get("sharpe") is not None else -99.0,
            row.get("cagr") if row.get("cagr") is not None else -99.0,
            row.get("dd") if row.get("dd") is not None else -1.0,
        )

    return max(pool, key=key)


def _empty_win(expected: int = 12) -> dict[str, Any]:
    return {"expected": expected, "base": None, "best": None, "configs": [], "n": 0}


def _finish_win(cell: dict[str, Any], years: int) -> None:
    cell["n"] = len(cell["configs"])
    cell["best"] = pick_best(cell["configs"], years)
    for row in cell["configs"]:
        row["label"] = label(row)
        row["years"] = years
    if cell["best"]:
        cell["best"]["label"] = label(cell["best"])
        cell["best"]["years"] = years
    if cell["base"]:
        cell["base"]["label"] = label(cell["base"])
        cell["base"]["years"] = years


def build_index(root: Path) -> dict[str, Any]:
    runs = root / "runs"
    configs = root / "configs"
    by: dict[str, dict[str, Any]] = {}
    expected = 0
    done = 0

    from run_pit_midsmall_study import STRATEGY_BY_NAME

    def bucket(strat: str, univ: str, years: int, n_var: int) -> dict[str, Any]:
        slot = by.setdefault(strat, {}).setdefault(univ, {"win": {}})
        win = slot["win"].setdefault(str(years), _empty_win(1 + n_var))
        return win

    for path in sorted(runs.glob("india__*__*__*y.json")):
        years = parse_years(path.name)
        if years is None:
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("error"):
            continue
        strat = payload.get("strategy")
        univ = payload.get("universe")
        if not strat or not univ:
            continue
        spec = STRATEGY_BY_NAME.get(strat)
        base_hold = int(payload.get("hold") or (spec.hold if spec else 0))
        n_var = len(variants_for(base_hold))
        expected += 1 + n_var
        cell = bucket(strat, univ, years, n_var)
        cell["expected"] = 1 + n_var
        row = _row_from_payload(
            payload,
            tag="base",
            file=f"runs/{path.name}",
            is_base=True,
        )
        cell["base"] = row
        cell["configs"].append(row)
        done += 1

    if configs.exists():
        for path in sorted(configs.glob("india__*.json")):
            if path.name == "index.json":
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if payload.get("error"):
                continue
            strat = payload.get("strategy")
            univ = payload.get("universe")
            if not strat or not univ:
                continue
            years = int(payload.get("years") or parse_years(path.name) or 5)
            spec = STRATEGY_BY_NAME.get(strat)
            n_var = len(variants_for(spec.hold)) if spec else 11
            cell = bucket(strat, univ, years, n_var)
            tag = payload.get("tag") or "cfg"
            row = _row_from_payload(
                payload,
                tag=tag,
                file=f"configs/{path.name}",
                is_base=False,
            )
            if any(c.get("id") == row["id"] for c in cell["configs"]):
                continue
            cell["configs"].append(row)
            done += 1

    for strat, universes in by.items():
        for univ, slot in universes.items():
            for year_key, cell in slot["win"].items():
                _finish_win(cell, int(year_key))
            five = slot["win"].get("5") or _empty_win()
            slot["base"] = five.get("base")
            slot["best"] = five.get("best")
            slot["configs"] = five.get("configs") or []
            slot["n"] = sum(c.get("n") or 0 for c in slot["win"].values())
            slot["expected"] = sum(c.get("expected") or 0 for c in slot["win"].values())

    return {
        "windows": [5, 3, 2, 1],
        "done": done,
        "expected": expected if expected else done,
        "by": by,
    }


def write_index(root: Path) -> Path:
    payload = build_index(root)
    dest = root / "configs" / "index.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    return dest
