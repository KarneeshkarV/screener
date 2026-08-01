#!/usr/bin/env python
"""Deterministic offline backtest delta harness.

Captures a pinned matrix of backtest engine x cost model x sizing rule x
interval cells as sorted JSON. Later stages that deliberately move numbers
can re-run the matrix with ``--compare`` and see exactly which cells changed
and by how much.

Fully offline: synthetic bars only, no network. Same inputs must produce
byte-identical JSON on repeated runs.

    uv run python scripts/backtest_delta.py --out /tmp/baseline.json
    uv run python scripts/backtest_delta.py --compare /tmp/baseline.json

Bar factories and the price-fetcher stub are defined here on purpose. The
harness is the instrument that measures whether later stages moved the
numbers; importing fixtures from ``tests/`` would let a test refactor silently
move the baseline.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import click
import numpy as np
import pandas as pd
from pydantic import ValidationError

from screener.backtester import run_backtest, run_rolling_backtest
from screener.backtester.models import BacktestConfig, BacktestResult, Trade

# --------------------------------------------------------------------------- #
# Pinned inputs - every free variable is fixed so the baseline is stable for
# the right reasons.
# --------------------------------------------------------------------------- #

SEED = 42
MARKET = "us"
BENCHMARK = "SPY"
TICKERS: tuple[str, ...] = ("AAA", "BBB", "CCC")
ENTRY_EXPR = "close > 0"
HOLD = 5
TOP = 2
INITIAL_CAPITAL = 100_000.0
SLIPPAGE_BPS = 5.0
COMMISSION_BPS = 1.0
SIZING_ATR_WINDOW = 14
# atr_risk budget = equity * risk_pct / (atr_multiple * ATR / close), then
# clamped to the equal-slot ceiling. Intraday ATR/close is tiny, so the daily
# default risk_pct (0.01) blows past the slot and the rule becomes a no-op.
# Pin both knobs, and use a much smaller risk_pct on 15m so the rule binds.
SIZING_ATR_MULTIPLE = 2.0
SIZING_RISK_PCT_1D = 0.01
SIZING_RISK_PCT_15M = 0.0001

# Daily synthetic panel: a few hundred business days so atr_risk clears warmup.
DAILY_START = "2023-01-02"
DAILY_N_BARS = 400
# Historical signal day sits well inside the panel with room for hold exits.
DAILY_AS_OF = date(2024, 6, 3)  # ~ bar 350 of the 2023-01-02 bdate panel
DAILY_ROLL_START = date(2023, 6, 1)
DAILY_ROLL_END = date(2024, 7, 15)

# Intraday: naive-UTC 15m stamps built from 09:30 America/New_York opens so
# DST is handled per session (US 14:30 UTC pre-DST, 13:30 UTC after).
INTRADAY_SESSIONS = 60
INTRADAY_BARS_PER_SESSION = 26  # 09:30..15:45 ET at 15m
INTRADAY_TZ = "America/New_York"
INTRADAY_SESSION0_DATE = date(2024, 3, 4)  # Monday
# Historical as_of is a real bar timestamp near the end of the panel.
INTRADAY_AS_OF_SESSION = 50
INTRADAY_AS_OF_BAR = 10
INTRADAY_ROLL_START = date(2024, 3, 18)
INTRADAY_ROLL_END = date(2024, 5, 17)

ENGINES: tuple[str, ...] = ("rolling", "historical")
COST_MODELS: tuple[str, ...] = ("flat", "india", "us_vested")
SIZING_RULES: tuple[str, ...] = ("equal_slot", "atr_risk")
INTERVALS: tuple[str, ...] = ("1d", "15m")

FLOAT_DECIMALS = 9


# --------------------------------------------------------------------------- #
# Bar factories (self-contained; patterns copied from tests, not imported)
# --------------------------------------------------------------------------- #


def make_daily_bars(
    *,
    start: str = DAILY_START,
    n: int = DAILY_N_BARS,
    open_base: float = 100.0,
    drift: float = 0.02,
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV frame (``np.random.default_rng``)."""
    rng = np.random.default_rng(seed)
    close = open_base + np.cumsum(rng.normal(drift, 0.5, n))
    openp = np.concatenate(([open_base], close[:-1]))
    high = np.maximum(openp, close) + rng.uniform(0.1, 0.5, n)
    low = np.minimum(openp, close) - rng.uniform(0.1, 0.5, n)
    volume = rng.integers(10_000, 50_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _us_15m_index(*, sessions: int) -> pd.DatetimeIndex:
    """Naive-UTC 15m stamps for US regular sessions (09:30 ET open).

    Each session open is constructed in ``America/New_York`` and converted to
    naive UTC, so the spring-forward DST boundary keeps opens at 09:30 ET
    (14:30 UTC before 2024-03-10, 13:30 UTC after).
    """
    stamps: list[pd.Timestamp] = []
    day = INTRADAY_SESSION0_DATE
    made = 0
    while made < sessions:
        if day.weekday() < 5:
            open_et = pd.Timestamp(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=9,
                minute=30,
                tz=INTRADAY_TZ,
            )
            for b in range(INTRADAY_BARS_PER_SESSION):
                ts_et = open_et + pd.Timedelta(minutes=15 * b)
                stamps.append(ts_et.tz_convert("UTC").tz_localize(None))
            made += 1
        day = day + timedelta(days=1)
    return pd.DatetimeIndex(stamps)


def make_intraday_bars(
    *,
    sessions: int = INTRADAY_SESSIONS,
    open_base: float = 100.0,
    drift: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic synthetic 15m OHLCV on a session-aware US index."""
    idx = _us_15m_index(sessions=sessions)
    n = len(idx)
    rng = np.random.default_rng(seed)
    close = open_base + np.cumsum(rng.normal(drift, 0.15, n))
    openp = np.concatenate(([open_base], close[:-1]))
    high = np.maximum(openp, close) + rng.uniform(0.05, 0.2, n)
    low = np.minimum(openp, close) - rng.uniform(0.05, 0.2, n)
    volume = rng.integers(1_000, 8_000, n).astype(float)
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _intraday_as_of() -> datetime:
    """Pinned historical as_of that lands on a real 15m bar."""
    idx = _us_15m_index(sessions=INTRADAY_SESSIONS)
    pos = INTRADAY_AS_OF_SESSION * INTRADAY_BARS_PER_SESSION + INTRADAY_AS_OF_BAR
    ts = idx[pos]
    return datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


# --------------------------------------------------------------------------- #
# Price fetcher stub (structural PriceFetcher)
# --------------------------------------------------------------------------- #


class StubPriceFetcher:
    """In-memory price fetcher for offline runs.

    Same shape as the test stub: ``fetch(tickers, start, end) -> dict``.
    End-of-day bounds are made inclusive for date-only ends so intraday bars
    on the final session are not clipped at midnight.
    """

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = {k: v.copy() for k, v in data.items()}

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if e == e.normalize():
            e = e + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
        for t in tickers:
            frame = self._data.get(t, pd.DataFrame())
            if frame.empty:
                out[t] = frame
                continue
            out[t] = frame.loc[(frame.index >= s) & (frame.index <= e)].copy()
        return out


# --------------------------------------------------------------------------- #
# Matrix construction
# --------------------------------------------------------------------------- #


EngineName = Literal["rolling", "historical"]


def _cell_key(engine: str, cost_model: str, sizing_rule: str, interval: str) -> str:
    return f"{engine}|{cost_model}|{sizing_rule}|{interval}"


def _build_panels() -> dict[str, dict[str, pd.DataFrame]]:
    """Build the two interval panels once; shared across cost/sizing cells."""
    daily: dict[str, pd.DataFrame] = {}
    intraday: dict[str, pd.DataFrame] = {}
    # Distinct seeds per symbol so dollar-volume ranking is stable but not tied.
    bases = {"AAA": 100.0, "BBB": 80.0, "CCC": 120.0, BENCHMARK: 400.0}
    for i, (sym, base) in enumerate(bases.items()):
        daily[sym] = make_daily_bars(open_base=base, seed=SEED + i, drift=0.03)
        intraday[sym] = make_intraday_bars(
            open_base=base, seed=SEED + 100 + i, drift=0.01
        )
    return {"1d": daily, "15m": intraday}


def _make_config(
    *,
    engine: str,
    cost_model: str,
    sizing_rule: str,
    interval: str,
) -> BacktestConfig:
    if interval == "1d":
        as_of: date | datetime = DAILY_AS_OF
    else:
        as_of = _intraday_as_of()
    return BacktestConfig(
        market=MARKET,
        as_of=as_of,
        benchmark=BENCHMARK,
        tickers=TICKERS,
        entry_expr=ENTRY_EXPR,
        exit_expr=None,
        strategy_name=None,
        interval=interval,
        hold=HOLD,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=SLIPPAGE_BPS,
        commission_bps=COMMISSION_BPS,
        cost_model=cost_model,  # type: ignore[arg-type]
        top=TOP,
        initial_capital=INITIAL_CAPITAL,
        sizing_rule=sizing_rule,
        sizing_atr_window=SIZING_ATR_WINDOW,
        sizing_atr_multiple=SIZING_ATR_MULTIPLE,
        sizing_risk_pct=(
            SIZING_RISK_PCT_1D if interval == "1d" else SIZING_RISK_PCT_15M
        ),
        min_price=None,
        min_avg_dollar_volume=None,
    )


def _run_cell(
    *,
    engine: str,
    cost_model: str,
    sizing_rule: str,
    interval: str,
    panels: dict[str, dict[str, pd.DataFrame]],
) -> tuple[BacktestResult | None, str | None]:
    """Run one matrix cell. Returns (result, skip_reason)."""
    try:
        cfg = _make_config(
            engine=engine,
            cost_model=cost_model,
            sizing_rule=sizing_rule,
            interval=interval,
        )
    except (ValidationError, ValueError) as exc:
        return None, f"config rejected: {exc}"

    fetcher = StubPriceFetcher(panels[interval])
    try:
        if engine == "historical":
            result = run_backtest(cfg, fetcher)
        elif engine == "rolling":
            if interval == "1d":
                start, end = DAILY_ROLL_START, DAILY_ROLL_END
            else:
                start, end = INTRADAY_ROLL_START, INTRADAY_ROLL_END
            result = run_rolling_backtest(cfg, fetcher, start_date=start, end_date=end)
        else:
            return None, f"unknown engine {engine!r}"
    except (ValidationError, ValueError) as exc:
        return None, f"engine rejected: {exc}"

    return result, None


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def _round_number(value: float) -> float | str:
    """Round finites; keep NaN/±Inf as distinct canonical strings."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return round(float(value), FLOAT_DECIMALS)


def _jsonable(value: Any) -> Any:
    """Recursively convert a value into sorted-JSON-friendly form."""
    if isinstance(value, float):
        return _round_number(value)
    if isinstance(value, (np.floating,)):
        return _round_number(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        # Timestamps normalize to ISO; plain dates stay YYYY-MM-DD.
        if isinstance(value, datetime) and not isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Timestamp):
            if value.hour or value.minute or value.second or value.microsecond:
                return value.to_pydatetime().isoformat()
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Trade):
        return _jsonable(value.model_dump(mode="python"))
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _trade_records(trades: list[Trade]) -> list[dict[str, Any]]:
    return [_jsonable(t) for t in trades]


def _metrics_record(metrics: dict[str, Any]) -> dict[str, Any]:
    return _jsonable(dict(metrics))


def _meta_block() -> dict[str, Any]:
    return {
        "seed": SEED,
        "market": MARKET,
        "benchmark": BENCHMARK,
        "tickers": list(TICKERS),
        "entry_expr": ENTRY_EXPR,
        "hold": HOLD,
        "top": TOP,
        "initial_capital": INITIAL_CAPITAL,
        "slippage_bps": SLIPPAGE_BPS,
        "commission_bps": COMMISSION_BPS,
        "sizing_atr_window": SIZING_ATR_WINDOW,
        "sizing_atr_multiple": SIZING_ATR_MULTIPLE,
        "sizing_risk_pct_1d": SIZING_RISK_PCT_1D,
        "sizing_risk_pct_15m": SIZING_RISK_PCT_15M,
        "daily_start": DAILY_START,
        "daily_n_bars": DAILY_N_BARS,
        "daily_as_of": DAILY_AS_OF.isoformat(),
        "daily_roll_start": DAILY_ROLL_START.isoformat(),
        "daily_roll_end": DAILY_ROLL_END.isoformat(),
        "intraday_sessions": INTRADAY_SESSIONS,
        "intraday_bars_per_session": INTRADAY_BARS_PER_SESSION,
        "intraday_tz": INTRADAY_TZ,
        "intraday_session0_date": INTRADAY_SESSION0_DATE.isoformat(),
        "intraday_as_of": _intraday_as_of().isoformat(),
        "intraday_roll_start": INTRADAY_ROLL_START.isoformat(),
        "intraday_roll_end": INTRADAY_ROLL_END.isoformat(),
        "float_decimals": FLOAT_DECIMALS,
        "engines": list(ENGINES),
        "cost_models": list(COST_MODELS),
        "sizing_rules": list(SIZING_RULES),
        "intervals": list(INTERVALS),
    }


def run_matrix() -> dict[str, Any]:
    """Execute every valid matrix cell and return the baseline document."""
    panels = _build_panels()
    cells: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []

    for engine in ENGINES:
        for cost_model in COST_MODELS:
            for sizing_rule in SIZING_RULES:
                for interval in INTERVALS:
                    key = _cell_key(engine, cost_model, sizing_rule, interval)
                    result, reason = _run_cell(
                        engine=engine,
                        cost_model=cost_model,
                        sizing_rule=sizing_rule,
                        interval=interval,
                        panels=panels,
                    )
                    if reason is not None:
                        skipped.append({"key": key, "reason": reason})
                        continue
                    assert result is not None
                    trades = list(result.trades)
                    if not trades:
                        raise RuntimeError(
                            f"cell {key!r} produced an empty trade ledger; "
                            "extend the synthetic panel or relax the entry "
                            "expression so atr_risk / rolling paths actually "
                            "trade (an empty ledger is a broken cell, not a "
                            "stable baseline)"
                        )
                    cells[key] = {
                        "key": {
                            "engine": engine,
                            "cost_model": cost_model,
                            "sizing_rule": sizing_rule,
                            "interval": interval,
                        },
                        "metrics": _metrics_record(result.metrics),
                        "trades": _trade_records(trades),
                        "trade_count": len(trades),
                    }

    atr_share_diffs = _assert_atr_risk_binds(cells)

    doc = {
        "meta": _meta_block(),
        "cells": {k: cells[k] for k in sorted(cells)},
        "skipped": skipped,
        "atr_share_diffs": {
            f"{engine}|{interval}": count
            for (engine, interval), count in sorted(atr_share_diffs.items())
        },
    }
    return _jsonable(doc)


def _assert_atr_risk_binds(
    cells: dict[str, Any],
) -> dict[tuple[str, str], int]:
    """Fail if atr_risk is a silent no-op vs equal_slot for any engine/interval.

    Compares the ``flat`` cost-model pair (cost model must not mask sizing).
    Returns the per-(engine, interval) count of trades whose ``shares`` differ.
    """
    counts: dict[tuple[str, str], int] = {}
    for engine in ENGINES:
        for interval in INTERVALS:
            eq_key = _cell_key(engine, "flat", "equal_slot", interval)
            atr_key = _cell_key(engine, "flat", "atr_risk", interval)
            if eq_key not in cells or atr_key not in cells:
                raise RuntimeError(
                    f"cannot verify atr_risk binding for {engine}/{interval}: "
                    f"missing cell(s) {eq_key!r} / {atr_key!r}"
                )
            eq_trades = cells[eq_key]["trades"]
            atr_trades = cells[atr_key]["trades"]
            n = min(len(eq_trades), len(atr_trades))
            differ = 0
            for i in range(n):
                if eq_trades[i].get("shares") != atr_trades[i].get("shares"):
                    differ += 1
            # Also count length mismatch as evidence the paths diverged.
            differ += abs(len(eq_trades) - len(atr_trades))
            counts[(engine, interval)] = differ
            if differ == 0:
                raise RuntimeError(
                    f"atr_risk is inert for engine={engine!r} interval={interval!r}: "
                    f"all {n} trade shares match equal_slot (budget is clamping to the "
                    f"slot ceiling). Lower sizing_risk_pct for this interval so the "
                    f"ATR-derived budget lands below the slot budget."
                )
    return counts


def dumps_baseline(doc: dict[str, Any]) -> str:
    """Canonical JSON: sorted keys, stable separators, trailing newline."""
    return json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n"


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #


def _walk_diff(left: Any, right: Any, path: str = "") -> Iterator[tuple[str, Any, Any]]:
    """Yield (path, left_value, right_value) for every leaf mismatch."""
    if type(left) is not type(right) and not (
        isinstance(left, (int, float)) and isinstance(right, (int, float))
    ):
        yield path or "$", left, right
        return
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right), key=str)
        for k in keys:
            p = f"{path}.{k}" if path else str(k)
            if k not in left:
                yield p, None, right[k]
            elif k not in right:
                yield p, left[k], None
            else:
                yield from _walk_diff(left[k], right[k], p)
        return
    if isinstance(left, list):
        n = max(len(left), len(right))
        if len(left) != len(right):
            yield f"{path}.len", len(left), len(right)
        for i in range(n):
            p = f"{path}[{i}]"
            if i >= len(left):
                yield p, None, right[i]
            elif i >= len(right):
                yield p, left[i], None
            else:
                yield from _walk_diff(left[i], right[i], p)
        return
    if left != right:
        yield path or "$", left, right


def _format_delta(old: Any, new: Any) -> str:
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if old is None or new is None:
            return f"{old!r} -> {new!r}"
        delta = float(new) - float(old)
        return f"{old!r} -> {new!r} (delta={delta:+.9g})"
    return f"{old!r} -> {new!r}"


def compare_baselines(baseline: dict[str, Any], current: dict[str, Any]) -> int:
    """Print a readable per-cell diff. Return process exit code (0 = identical)."""
    base_cells = baseline.get("cells", {})
    cur_cells = current.get("cells", {})
    base_skipped = {
        row["key"]: row.get("reason", "")
        for row in baseline.get("skipped", [])
        if isinstance(row, dict) and "key" in row
    }
    cur_skipped = {
        row["key"]: row.get("reason", "")
        for row in current.get("skipped", [])
        if isinstance(row, dict) and "key" in row
    }

    all_keys = sorted(
        set(base_cells) | set(cur_cells) | set(base_skipped) | set(cur_skipped)
    )
    n_diff_cells = 0
    n_diffs = 0

    meta_diffs = list(
        _walk_diff(baseline.get("meta", {}), current.get("meta", {}), "meta")
    )
    if meta_diffs:
        click.echo("meta:")
        for path, old, new in meta_diffs:
            click.echo(f"  {path}: {_format_delta(old, new)}")
            n_diffs += 1

    for key in all_keys:
        in_base = key in base_cells
        in_cur = key in cur_cells
        if key in base_skipped or key in cur_skipped:
            b_reason = base_skipped.get(key)
            c_reason = cur_skipped.get(key)
            if b_reason != c_reason or in_base != in_cur:
                n_diff_cells += 1
                click.echo(f"cell {key}:")
                click.echo(f"  skipped: baseline={b_reason!r} current={c_reason!r}")
                n_diffs += 1
            continue
        if in_base and not in_cur:
            n_diff_cells += 1
            n_diffs += 1
            click.echo(f"cell {key}: missing from current run")
            continue
        if in_cur and not in_base:
            n_diff_cells += 1
            n_diffs += 1
            click.echo(f"cell {key}: new cell not in baseline")
            continue

        cell_diffs = list(_walk_diff(base_cells[key], cur_cells[key], key))
        if not cell_diffs:
            continue
        n_diff_cells += 1
        click.echo(f"cell {key}:")
        # Prefer a short metrics/trade summary before the full leaf dump.
        b_tc = base_cells[key].get("trade_count")
        c_tc = cur_cells[key].get("trade_count")
        if b_tc != c_tc:
            click.echo(f"  trade_count: {_format_delta(b_tc, c_tc)}")
        shown = 0
        for path, old, new in cell_diffs:
            click.echo(f"  {path}: {_format_delta(old, new)}")
            n_diffs += 1
            shown += 1
            if shown >= 40:
                remaining = len(cell_diffs) - shown
                if remaining > 0:
                    click.echo(f"  ... ({remaining} more diffs)")
                    n_diffs += remaining
                break

    if n_diffs == 0:
        click.echo(f"identical: {len(cur_cells)} cells, {len(cur_skipped)} skipped")
        return 0

    click.echo(f"differ: {n_diff_cells} cell(s) with changes, {n_diffs} leaf diff(s)")
    return 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write the baseline JSON to PATH.",
)
@click.option(
    "--compare",
    "compare_path",
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    default=None,
    help="Rerun the matrix and diff against a previously written baseline.",
)
def main(out_path: Path | None, compare_path: Path | None) -> None:
    """Pinned offline backtest matrix for delta detection across refactors."""
    if (out_path is None) == (compare_path is None):
        raise click.UsageError("provide exactly one of --out or --compare")

    click.echo("running backtest delta matrix (offline, deterministic)...", err=True)
    doc = run_matrix()
    payload = dumps_baseline(doc)

    n_cells = len(doc["cells"])
    n_skipped = len(doc["skipped"])
    for key in sorted(doc["cells"]):
        tc = doc["cells"][key]["trade_count"]
        click.echo(f"  {key}: {tc} trades", err=True)
    for row in doc["skipped"]:
        click.echo(f"  SKIP {row['key']}: {row['reason']}", err=True)
    for pair, count in sorted(doc.get("atr_share_diffs", {}).items()):
        click.echo(
            f"  atr_risk share diffs [{pair}]: {count}",
            err=True,
        )
    click.echo(f"matrix: {n_cells} cells, {n_skipped} skipped", err=True)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        click.echo(f"wrote {out_path}", err=True)
        return

    assert compare_path is not None
    baseline = json.loads(compare_path.read_text(encoding="utf-8"))
    current = json.loads(payload)
    code = compare_baselines(baseline, current)
    sys.exit(code)


if __name__ == "__main__":
    main()
