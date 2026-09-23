#!/usr/bin/env python
"""Bounded India momentum_12_1 hold/rank-exit comparison on nifty500_pit.

Descriptive fixed-parameter comparison only. Not parameter selection, not
walk-forward / research-report optimization, and not a trading recommendation.

Live (acquires once via FMP caches, then freezes):

    uv run python scripts/run_india_momentum_comparison.py

Offline replay from a frozen snapshot into a different out-dir (no FMP key,
no live universe resolution):

    uv run python scripts/run_india_momentum_comparison.py \\
      --replay-from reports/next_actions_2026-09-07/momentum \\
      --out-dir reports/next_actions_2026-09-07/momentum_verified

Owns this script plus ``reports/next_actions_2026-09-07/momentum/`` and
``reports/next_actions_2026-09-07/momentum_verified/``.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
import os
import pstats
import resource
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "reports" / "next_actions_2026-09-07" / "momentum"
UNIVERSE_CONFIG = ROOT / "universes.yaml"
STRATEGY = "momentum_12_1"
MARKET = "india"
UNIVERSE = "nifty500_pit"
BENCHMARK = "^NSEI"
TOP = 10
INITIAL_CAPITAL = 100_000.0
SLIPPAGE_BPS = 5.0
COMMISSION_BPS = 0.0
COST_MODEL = "india"
HOLDS = (21, 63, 126)
RANK_UNIVERSE_SIZES = (10, 20, 30)
RANK_EXIT_EVERY = 21
MAIN_START = date(2022, 9, 1)
MAIN_END = date(2026, 9, 1)
ANNUAL_WINDOWS: tuple[tuple[str, date, date], ...] = (
    ("y2022", date(2022, 9, 1), date(2023, 8, 31)),
    ("y2023", date(2023, 9, 1), date(2024, 8, 31)),
    ("y2024", date(2024, 9, 1), date(2025, 8, 31)),
    ("y2025", date(2025, 9, 1), date(2026, 9, 1)),
)
TURNOVER_DEFINITION = (
    "both-sided traded notional = "
    "sum(shares*entry_price + shares*exit_price) / "
    "(initial_capital * calendar_years); cash fees excluded"
)


@dataclass(frozen=True)
class VariantSpec:
    """One predeclared hold / rank-exit cell."""

    hold: int
    rank_exit_every: int | None
    rank_universe_size: int | None

    @property
    def name(self) -> str:
        if self.rank_exit_every is None:
            return f"hold{self.hold}__rank_off"
        return f"hold{self.hold}__rank{self.rank_exit_every}_u{self.rank_universe_size}"


def variant_grid() -> list[VariantSpec]:
    """Return the 12 fixed variants: 3 holds x (rank off + 3 retention bands)."""
    out: list[VariantSpec] = []
    for hold in HOLDS:
        out.append(
            VariantSpec(hold=hold, rank_exit_every=None, rank_universe_size=None)
        )
        for universe_size in RANK_UNIVERSE_SIZES:
            out.append(
                VariantSpec(
                    hold=hold,
                    rank_exit_every=RANK_EXIT_EVERY,
                    rank_universe_size=universe_size,
                )
            )
    return out


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def peak_rss_mb() -> float:
    """Linux ``ru_maxrss`` is kilobytes; return megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def calendar_years(start: date, end: date) -> float:
    return max((end - start).days / 365.25, 1e-12)


def both_sided_turnover(
    trades: pd.DataFrame, *, initial_capital: float, years: float
) -> float:
    """Both-sided traded notional / (starting capital * years).

    Uses ``shares * entry_price`` and ``shares * exit_price`` only.
    Does not use ``entry_cost`` / ``exit_value`` (those include cash fees).
    This study has no partial exits, so one entry and one exit per trade.
    """
    if trades.empty or initial_capital <= 0 or years <= 0:
        return 0.0
    needed = {"shares", "entry_price", "exit_price"}
    if not needed.issubset(trades.columns):
        return 0.0
    shares = trades["shares"].fillna(0.0).astype(float)
    buy = float((shares * trades["entry_price"].fillna(0.0).astype(float)).sum())
    sell = float((shares * trades["exit_price"].fillna(0.0).astype(float)).sum())
    return (buy + sell) / (initial_capital * years)


def exit_reason_counts(trades: pd.DataFrame) -> dict[str, int]:
    if trades.empty or "exit_reason" not in trades.columns:
        return {}
    return {
        str(k): int(v) for k, v in Counter(trades["exit_reason"].astype(str)).items()
    }


def _hash_path_tree(root: Path, pattern: str) -> dict[str, Any]:
    """Aggregate SHA256 over relative paths + file digests under ``root``."""
    if not root.exists():
        return {"aggregate_sha256": "MISSING", "n_files": 0}
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        file_hash = sha256_file(path)
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\0")
    return {
        "aggregate_sha256": digest.hexdigest(),
        "n_files": len(files),
        "root": str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
        "pattern": pattern,
    }


def local_portfolio_edit_record() -> dict[str, Any]:
    """Record the existing uncommitted ``portfolio.py`` working-tree edit."""
    rel = "screener/backtester/portfolio.py"
    path = ROOT / rel
    record: dict[str, Any] = {
        "path": rel,
        "working_tree_sha256": sha256_file(path) if path.exists() else "MISSING",
        "note": (
            "Existing local edit; not introduced by this comparison script. "
            "Recorded so the trial footprint identifies the code that ran."
        ),
    }
    try:
        head = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        record["head_sha256"] = sha256_bytes(head)
    except (subprocess.CalledProcessError, FileNotFoundError):
        record["head_sha256"] = None
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--", rel],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        record["diff_sha256"] = sha256_bytes(diff.encode()) if diff else None
        record["diff_stat"] = subprocess.check_output(
            ["git", "diff", "--stat", "--", rel],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        record["modified"] = bool(diff.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        record["modified"] = None
    return record


def code_fingerprints() -> dict[str, Any]:
    """Trial footprint: recursive package tree, lockfiles, universe, script."""
    return {
        "screener_py_tree": _hash_path_tree(ROOT / "screener", "**/*.py"),
        "files": {
            "pyproject.toml": sha256_file(ROOT / "pyproject.toml"),
            "uv.lock": sha256_file(ROOT / "uv.lock"),
            "universes.yaml": sha256_file(UNIVERSE_CONFIG),
            "scripts/run_india_momentum_comparison.py": sha256_file(Path(__file__)),
        },
        "local_edits": {
            "screener/backtester/portfolio.py": local_portfolio_edit_record(),
        },
    }


def restore_slippage_model(raw: Any) -> Any:
    """Rebuild a SlippageModel from ``BacktestConfig.model_dump(mode='json')``."""
    from screener.backtester.slippage import (
        CompositeSlippage,
        EstimatedHalfSpreadSlippage,
        FixedBpsSlippage,
        HalfSpreadSlippage,
        VolumeImpactSlippage,
    )

    if raw is None or hasattr(raw, "adverse_fraction"):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(
            f"restore_slippage_model unsupported payload type: {type(raw).__name__}"
        )
    keys = set(raw)
    if keys == {"bps"}:
        return FixedBpsSlippage(bps=float(raw["bps"]))
    if keys == {"bps", "half_spread_bps"} or keys == {"half_spread_bps"}:
        return HalfSpreadSlippage(
            bps=float(raw.get("bps", 0.0)),
            half_spread_bps=float(raw.get("half_spread_bps", 0.0)),
        )
    if keys == set():
        return EstimatedHalfSpreadSlippage()
    if "k" in keys or "vol_impact_k" in keys:
        return VolumeImpactSlippage.model_validate(raw)
    if "models" in keys:
        models = tuple(restore_slippage_model(m) for m in raw["models"])
        return CompositeSlippage(models=models)
    # Fixed-bps dump from this study is ``{"bps": 5.0}``; reject unknowns loudly.
    raise ValueError(f"restore_slippage_model cannot classify keys={sorted(keys)}")


def restore_backtest_config(full_config: dict[str, Any]) -> Any:
    """Validate a saved ``full_config`` into ``BacktestConfig`` offline."""
    from screener.backtester.models import BacktestConfig

    payload = dict(full_config)
    payload["slippage_model"] = restore_slippage_model(payload.get("slippage_model"))
    return BacktestConfig.model_validate(payload)


def resolved_summary_from_config(cfg: Any, *, universe_note: str) -> dict[str, Any]:
    resolved = cfg.model_dump(mode="json")
    return {
        "eligibility_gates": {
            "min_price": cfg.min_price,
            "min_avg_dollar_volume": cfg.min_avg_dollar_volume,
            "avg_dollar_volume_window": cfg.avg_dollar_volume_window,
            "min_score": cfg.min_score,
            "regime_filter": list(cfg.regime_filter or ()),
            "sector_neutral": cfg.sector_neutral,
            "earnings_blackout_days": cfg.earnings_blackout_days,
            "entry_expr": cfg.entry_expr,
            "exit_expr": cfg.exit_expr,
        },
        "execution": {
            "top": cfg.top,
            "hold": cfg.hold,
            "sizing_rule": cfg.sizing_rule,
            "initial_capital": cfg.initial_capital,
            "entry_order_type": cfg.entry_order_type,
            "cost_model": cfg.cost_model,
            "slippage_bps": cfg.slippage_bps,
            "commission_bps": cfg.commission_bps,
            "slippage_model": resolved.get("slippage_model"),
            "stop_loss": cfg.stop_loss,
            "take_profit": cfg.take_profit,
            "trailing_stop": cfg.trailing_stop,
        },
        "universe": {
            "strategy_name": cfg.strategy_name,
            "market": cfg.market,
            "benchmark": cfg.benchmark,
            "n_tickers": len(cfg.tickers or ()),
            "n_membership_windows": len(cfg.membership_windows or ()),
            "fundamentals_provider": cfg.fundamentals_provider,
            "universe_note": universe_note,
            "start": MAIN_START.isoformat(),
            "end": MAIN_END.isoformat(),
        },
        "full_config": resolved,
    }


class RecordingPriceFetcher:
    """Thin recorder over an existing ``PriceFetcher`` acquisition path."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.frames: dict[str, pd.DataFrame] = {}
        self.fetch_calls: list[dict[str, Any]] = []
        self.missing_provider_bars: list[dict[str, Any]] = []
        self.seen_symbols: set[str] = set()

    def fetch(self, tickers: Any, start: date, end: date) -> dict[str, pd.DataFrame]:
        symbols = [str(t) for t in tickers]
        self.seen_symbols.update(symbols)
        raw = self.inner.fetch(symbols, start, end)
        call_missing: list[str] = []
        for symbol in symbols:
            frame = raw.get(symbol)
            if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
                call_missing.append(symbol)
                self.missing_provider_bars.append(
                    {
                        "symbol": symbol,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "reason": "provider_returned_empty_or_absent",
                    }
                )
                empty = pd.DataFrame()
                raw[symbol] = empty
                self._store(symbol, empty)
                continue
            stored = frame.copy()
            if not isinstance(stored.index, pd.DatetimeIndex):
                stored.index = pd.to_datetime(stored.index)
            stored = stored.sort_index()
            self._store(symbol, stored)
            raw[symbol] = stored
        self.fetch_calls.append(
            {
                "n_symbols": len(symbols),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "n_missing": len(call_missing),
                "missing_sample": call_missing[:20],
            }
        )
        return raw

    def _store(self, symbol: str, frame: pd.DataFrame) -> None:
        prior = self.frames.get(symbol)
        if prior is None or prior.empty:
            self.frames[symbol] = frame.copy()
            return
        if frame.empty:
            return
        merged = pd.concat([prior, frame])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.frames[symbol] = merged


class ReplayPriceFetcher:
    """Serve only frozen raw bars; raise on symbols absent from the snapshot."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        recorded_symbols: set[str],
    ) -> None:
        self.frames = frames
        self.recorded_symbols = set(recorded_symbols)
        self.fetch_calls: list[dict[str, Any]] = []
        self.missing_provider_bars: list[dict[str, Any]] = []
        self.fetch_wall_s = 0.0

    def fetch(self, tickers: Any, start: date, end: date) -> dict[str, pd.DataFrame]:
        t0 = time.perf_counter()
        try:
            symbols = [str(t) for t in tickers]
            start_ts = pd.Timestamp(start).normalize()
            end_ts = pd.Timestamp(end).normalize()
            out: dict[str, pd.DataFrame] = {}
            unexpected = [s for s in symbols if s not in self.recorded_symbols]
            if unexpected:
                raise KeyError(
                    "ReplayPriceFetcher unexpected symbols not in frozen snapshot: "
                    + ", ".join(sorted(unexpected)[:20])
                )
            missing: list[str] = []
            for symbol in symbols:
                frame = self.frames.get(symbol)
                if frame is None or frame.empty:
                    missing.append(symbol)
                    self.missing_provider_bars.append(
                        {
                            "symbol": symbol,
                            "start": start.isoformat(),
                            "end": end.isoformat(),
                            "reason": "frozen_snapshot_empty",
                        }
                    )
                    out[symbol] = pd.DataFrame()
                    continue
                idx = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
                work = frame.copy()
                work.index = idx
                sliced = work.loc[(work.index >= start_ts) & (work.index <= end_ts)]
                out[symbol] = sliced.copy()
            self.fetch_calls.append(
                {
                    "n_symbols": len(symbols),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "n_missing": len(missing),
                }
            )
            return out
        finally:
            self.fetch_wall_s += time.perf_counter() - t0


def frames_to_parquet(frames: dict[str, pd.DataFrame], path: Path) -> dict[str, Any]:
    """Persist raw OHLCV bars as one replayable parquet with a symbol column."""
    rows: list[pd.DataFrame] = []
    coverage: dict[str, Any] = {}
    for symbol, frame in sorted(frames.items()):
        if frame is None or frame.empty:
            coverage[symbol] = {
                "n_bars": 0,
                "first": None,
                "last": None,
                "empty": True,
            }
            continue
        piece = frame.copy()
        if not isinstance(piece.index, pd.DatetimeIndex):
            piece.index = pd.to_datetime(piece.index)
        piece = piece.sort_index()
        piece.index = pd.DatetimeIndex(piece.index).tz_localize(None)
        coverage[symbol] = {
            "n_bars": int(len(piece)),
            "first": str(piece.index.min().date()),
            "last": str(piece.index.max().date()),
            "empty": False,
            "columns": list(piece.columns),
        }
        out = piece.reset_index()
        date_col = out.columns[0]
        out = out.rename(columns={date_col: "date"})
        out.insert(0, "symbol", symbol)
        rows.append(out)
    if rows:
        table = pd.concat(rows, ignore_index=True)
    else:
        table = pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "volume"]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return coverage


def load_frames_from_parquet(path: Path) -> dict[str, pd.DataFrame]:
    table = pd.read_parquet(path)
    frames: dict[str, pd.DataFrame] = {}
    if table.empty:
        return frames
    for symbol, group in table.groupby("symbol", sort=True):
        frame = group.drop(columns=["symbol"]).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        frames[str(symbol)] = frame
    return frames


def load_symbol_manifest(snap_dir: Path) -> list[str]:
    """Full captured symbol list including empty provider symbols."""
    manifest_path = snap_dir / "symbol_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        symbols = payload.get("symbols") or payload
        return [str(s) for s in symbols]
    coverage = json.loads((snap_dir / "coverage.json").read_text())
    return sorted(coverage.keys())


def configure_fmp_no_refresh() -> None:
    """Load ``.env`` and force FMP prices without cache refresh."""
    from screener.config import load_env_file

    load_env_file()
    os.environ["SCREENER_PRICE_PROVIDER"] = "fmp"
    if not os.environ.get("FMP_API_KEY"):
        raise RuntimeError("FMP_API_KEY is not set after load_env_file()")


def build_request(
    *,
    start: date,
    end: date,
    hold: int,
    rank_exit_every: int | None,
    rank_universe_size: int | None,
) -> Any:
    from scripts.run_nifty500_pit_strategy_sweep import make_request

    overrides: dict[str, Any] = {
        "strategy_name": STRATEGY,
        "market": MARKET,
        "universe": UNIVERSE,
        "universe_config": UNIVERSE_CONFIG,
        "point_in_time": True,
        "point_in_time_was_explicit": True,
        "benchmark": BENCHMARK,
        "top": TOP,
        "hold": hold,
        "initial_capital": INITIAL_CAPITAL,
        "sizing_rule": "equal_slot",
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "entry_order": "moo",
        "cost_model": COST_MODEL,
        "slippage_bps": SLIPPAGE_BPS,
        "slippage_model": "fixed",
        "commission_bps": COMMISSION_BPS,
        "fundamentals_provider": None,
        "refresh": False,
        "start_arg": datetime.combine(start, datetime.min.time()),
        "end_arg": datetime.combine(end, datetime.min.time()),
        "years": max(1, int(round(calendar_years(start, end)))),
    }
    if rank_exit_every is None:
        overrides["rank_exit"] = None
        overrides["rank_universe_size"] = 50
    else:
        overrides["rank_exit"] = str(rank_exit_every)
        overrides["rank_universe_size"] = int(rank_universe_size or TOP)
    return make_request(**overrides)


def apply_variant(cfg: Any, variant: VariantSpec) -> Any:
    update = {"hold": int(variant.hold)}
    if variant.rank_exit_every is None:
        update["rank_exit_every"] = None
        update["rank_universe_size"] = 50
    else:
        update["rank_exit_every"] = int(variant.rank_exit_every)
        update["rank_universe_size"] = int(variant.rank_universe_size or TOP)
    return cfg.model_copy(update=update)


def window_cfg_from_frozen(base_cfg: Any, *, end: date) -> Any:
    """Annual/main window config: update as_of only; keep frozen membership."""
    return base_cfg.model_copy(update={"as_of": end})


def trades_frame(result: Any) -> pd.DataFrame:
    from screener.backtester.display import trades_dataframe

    return trades_dataframe(result)


def equity_frame(result: Any) -> pd.DataFrame:
    equity = result.equity_curve.copy()
    if equity.empty:
        return pd.DataFrame(columns=["date", "equity"])
    out = equity.rename("equity").to_frame().reset_index()
    out.columns = ["date", "equity"]
    return out


def normalize_trades_csv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("signal_date", "entry_date", "exit_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    for col in out.columns:
        if col in {"signal_date", "entry_date", "exit_date", "ticker", "exit_reason"}:
            out[col] = out[col].astype(str)
        elif col == "rank":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def normalize_equity_csv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    if "equity" in out.columns:
        out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    return out.reset_index(drop=True)


def ledger_fingerprint(trades: pd.DataFrame, equity: pd.DataFrame) -> str:
    payload = {
        "trades": normalize_trades_csv(trades)
        .fillna("")
        .astype(str)
        .to_dict(orient="list"),
        "equity": normalize_equity_csv(equity)
        .fillna("")
        .astype(str)
        .to_dict(orient="list"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(blob)


def metrics_row(
    *,
    window: str,
    start: date,
    end: date,
    variant: VariantSpec,
    result: Any,
) -> dict[str, Any]:
    metrics = result.metrics or {}
    trades = trades_frame(result)
    years = calendar_years(start, end)
    total_return = metrics.get("total_return")
    benchmark_return = metrics.get("benchmark_return")
    excess = None
    if isinstance(total_return, (int, float)) and isinstance(
        benchmark_return, (int, float)
    ):
        if math.isfinite(float(total_return)) and math.isfinite(
            float(benchmark_return)
        ):
            excess = float(total_return) - float(benchmark_return)
    return {
        "window": window,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "hold": variant.hold,
        "rank_exit_every": variant.rank_exit_every,
        "rank_universe_size": (
            None if variant.rank_exit_every is None else variant.rank_universe_size
        ),
        "variant": variant.name,
        "sharpe": metrics.get("sharpe"),
        "cagr": metrics.get("cagr"),
        "total_return": total_return,
        "max_drawdown": metrics.get("max_drawdown"),
        "benchmark_return": benchmark_return,
        "excess_return": excess,
        "trade_count": metrics.get("trade_count"),
        "exposure": metrics.get("exposure"),
        "total_fees": metrics.get("total_fees"),
        "both_sided_turnover": both_sided_turnover(
            trades, initial_capital=INITIAL_CAPITAL, years=years
        ),
        "turnover_definition": TURNOVER_DEFINITION,
        "calendar_years": years,
        "exit_reasons": exit_reason_counts(trades),
        "n_warnings": len(result.warnings or []),
    }


def save_run_artifacts(
    run_dir: Path,
    *,
    result: Any,
    row: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    trades = trades_frame(result)
    equity = equity_frame(result)
    trades.to_csv(run_dir / "trades.csv", index=False)
    equity.to_csv(run_dir / "equity.csv", index=False)
    write_json(run_dir / "metrics.json", {**row, "metrics": result.metrics})
    write_json(run_dir / "exit_reasons.json", row["exit_reasons"])
    write_json(run_dir / "warnings.json", list(result.warnings or []))


def membership_day_counts(cfg: Any, dates: list[date]) -> dict[str, int]:
    windows = list(cfg.membership_windows or ())
    out: dict[str, int] = {}
    for day in dates:
        n = 0
        for _symbol, start, end in windows:
            if start <= day and (end is None or end >= day):
                n += 1
        out[day.isoformat()] = n
    return out


def progress(msg: str) -> None:
    print(msg, flush=True)


def run_prepared_rolling_backtest_safe(prepared: Any, cfg: Any) -> Any:
    from screener.backtester import run_prepared_rolling_backtest

    return run_prepared_rolling_backtest(prepared, cfg)


def run_window(
    *,
    window: str,
    start: date,
    end: date,
    base_cfg: Any,
    fetcher: Any,
    out_dir: Path,
    variants: list[VariantSpec],
    rows: list[dict[str, Any]],
    all_warnings: list[dict[str, Any]],
) -> Any:
    from screener.backtester import (
        prepare_rolling_backtest,
        run_prepared_rolling_backtest,
    )

    progress(f"window={window} prepare {start.isoformat()}..{end.isoformat()}")
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    prepared = prepare_rolling_backtest(
        base_cfg,
        fetcher,
        start_date=start,
        end_date=end,
        fundamental_fetcher=None,
    )
    prep_wall = time.perf_counter() - t0
    prep_cpu = time.process_time() - cpu0
    write_json(
        out_dir / "runs" / window / "prepare_timings.json",
        {
            "window": window,
            "prepare_wall_s": prep_wall,
            "prepare_cpu_s": prep_cpu,
            "n_master_dates": len(prepared.master_dates),
            "n_bars_by_tv": len(prepared.bars_by_tv),
            "peak_rss_mb": peak_rss_mb(),
            "warnings": list(prepared.warnings),
            "replay_adapter_fetch_wall_s": getattr(fetcher, "fetch_wall_s", None),
        },
    )
    for warning in prepared.warnings:
        all_warnings.append({"window": window, "phase": "prepare", "warning": warning})

    for variant in variants:
        cfg = apply_variant(base_cfg, variant)
        if not prepared.supports(cfg):
            raise RuntimeError(
                f"prepared data does not support variant {variant.name} in {window}"
            )
        result = run_prepared_rolling_backtest(prepared, cfg)
        row = metrics_row(
            window=window, start=start, end=end, variant=variant, result=result
        )
        rows.append(row)
        save_run_artifacts(
            out_dir / "runs" / window / variant.name,
            result=result,
            row=row,
        )
        for warning in result.warnings or []:
            all_warnings.append(
                {
                    "window": window,
                    "variant": variant.name,
                    "phase": "simulate",
                    "warning": warning,
                }
            )
    progress(
        f"window={window} done variants={len(variants)} "
        f"prep_wall_s={prep_wall:.1f} peak_rss_mb={peak_rss_mb():.0f}"
    )
    return prepared


def speed_benchmark(
    *,
    base_cfg: Any,
    replay: ReplayPriceFetcher,
    out_dir: Path,
    start: date,
    end: date,
) -> dict[str, Any]:
    from screener.backtester import (
        prepare_rolling_backtest,
        run_prepared_rolling_backtest,
    )

    baseline = VariantSpec(hold=21, rank_exit_every=None, rank_universe_size=None)
    cfg = apply_variant(base_cfg, baseline)
    speed_dir = out_dir / "speed"
    speed_dir.mkdir(parents=True, exist_ok=True)

    fresh_times: list[float] = []
    fresh_cpu: list[float] = []
    fresh_fps: list[str] = []
    reused_times: list[float] = []
    reused_cpu: list[float] = []
    reused_fps: list[str] = []
    replay_fetch_times: list[float] = []

    reference_prepared = None
    for i in range(3):
        replay.fetch_wall_s = 0.0
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        prepared = prepare_rolling_backtest(
            cfg,
            replay,
            start_date=start,
            end_date=end,
            fundamental_fetcher=None,
        )
        result = run_prepared_rolling_backtest(prepared, cfg)
        fresh_times.append(time.perf_counter() - wall0)
        fresh_cpu.append(time.process_time() - cpu0)
        replay_fetch_times.append(replay.fetch_wall_s)
        fresh_fps.append(ledger_fingerprint(trades_frame(result), equity_frame(result)))
        if i == 0:
            reference_prepared = prepared
    if reference_prepared is None:
        raise RuntimeError("speed benchmark failed to prepare a reference panel")

    for _ in range(3):
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        result = run_prepared_rolling_backtest(reference_prepared, cfg)
        reused_times.append(time.perf_counter() - wall0)
        reused_cpu.append(time.process_time() - cpu0)
        reused_fps.append(
            ledger_fingerprint(trades_frame(result), equity_frame(result))
        )

    identical = len(set(fresh_fps + reused_fps)) == 1
    identity = {
        "identical_full_trade_ledger_and_equity": identical,
        "fresh_fingerprints": fresh_fps,
        "reused_fingerprints": reused_fps,
        "baseline_variant": baseline.name,
    }
    write_json(speed_dir / "identity_assert.json", identity)
    if not identical:
        raise AssertionError(
            "fresh prepare+simulate and reused simulate ledgers/equity differ"
        )

    profile_path = speed_dir / "baseline_cprofile.prof"
    top_path = speed_dir / "baseline_cprofile_top.txt"
    profiler = cProfile.Profile()
    replay.fetch_wall_s = 0.0
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    profiler.enable()
    prepared_p = prepare_rolling_backtest(
        cfg,
        replay,
        start_date=start,
        end_date=end,
        fundamental_fetcher=None,
    )
    result_p = run_prepared_rolling_backtest(prepared_p, cfg)
    profiler.disable()
    profiled_wall = time.perf_counter() - wall0
    profiled_cpu = time.process_time() - cpu0
    profiled_replay_fetch = replay.fetch_wall_s
    profiler.dump_stats(str(profile_path))
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(40)
    top_path.write_text(stream.getvalue())
    _ = result_p

    unprofiled_mean = sum(fresh_times) / len(fresh_times)
    payload = {
        "universe_ever_members": len(base_cfg.tickers or ()),
        "membership_counts_on_selected_dates": membership_day_counts(
            base_cfg,
            [
                MAIN_START,
                date(2023, 9, 1),
                date(2024, 9, 1),
                date(2025, 9, 1),
                MAIN_END,
            ],
        ),
        "fresh_prepare_plus_simulate_wall_s": fresh_times,
        "fresh_prepare_plus_simulate_cpu_s": fresh_cpu,
        "reused_simulate_wall_s": reused_times,
        "reused_simulate_cpu_s": reused_cpu,
        "fresh_wall_mean_s": unprofiled_mean,
        "reused_wall_mean_s": sum(reused_times) / len(reused_times),
        "reuse_note": (
            "Reused-simulate vs fresh prepare+simulate demonstrates the existing "
            "PreparedRollingBacktest.supports reuse path. It is not a new engine "
            "speedup."
        ),
        "replay_adapter_fetch_wall_s_fresh_runs": replay_fetch_times,
        "replay_adapter_fetch_wall_s_mean": sum(replay_fetch_times)
        / len(replay_fetch_times),
        "replay_adapter_fetch_wall_s_cprofile_run": profiled_replay_fetch,
        "cprofile_prepare_plus_simulate_wall_s": profiled_wall,
        "cprofile_prepare_plus_simulate_cpu_s": profiled_cpu,
        "cprofile_overhead_wall_s": profiled_wall - unprofiled_mean,
        "cprofile_overhead_note": (
            "cProfile wall minus mean of three unprofiled fresh prepare+simulate "
            "runs on the same frozen snapshot; positive means profiler overhead. "
            "Replay adapter fetch time is reported separately and is included in "
            "prepare wall time."
        ),
        "peak_rss_mb": peak_rss_mb(),
        "identity": identity,
        "profile_path": str(profile_path.relative_to(out_dir)),
        "profile_top_path": str(top_path.relative_to(out_dir)),
    }
    write_json(speed_dir / "timings.json", payload)
    return payload


def compare_main_ledgers(
    *,
    original_dir: Path,
    replay_dir: Path,
    variants: list[VariantSpec],
) -> dict[str, Any]:
    """Exact pandas equality of main-window trades/equity after CSV parse."""
    details: list[dict[str, Any]] = []
    all_equal = True
    for variant in variants:
        rel = Path("runs") / "main" / variant.name
        o_trades = original_dir / rel / "trades.csv"
        r_trades = replay_dir / rel / "trades.csv"
        o_equity = original_dir / rel / "equity.csv"
        r_equity = replay_dir / rel / "equity.csv"
        ot = normalize_trades_csv(pd.read_csv(o_trades))
        rt = normalize_trades_csv(pd.read_csv(r_trades))
        oe = normalize_equity_csv(pd.read_csv(o_equity))
        re = normalize_equity_csv(pd.read_csv(r_equity))
        trades_eq = ot.equals(rt)
        equity_eq = oe.equals(re)
        ok = bool(trades_eq and equity_eq)
        all_equal = all_equal and ok
        details.append(
            {
                "variant": variant.name,
                "trades_equal": trades_eq,
                "equity_equal": equity_eq,
                "n_trades_original": int(len(ot)),
                "n_trades_replay": int(len(rt)),
                "n_equity_original": int(len(oe)),
                "n_equity_replay": int(len(re)),
            }
        )
    return {
        "all_main_ledgers_and_equity_equal": all_equal,
        "n_variants_compared": len(details),
        "details": details,
        "original_dir": str(original_dir),
        "replay_dir": str(replay_dir),
        "method": "pandas.DataFrame.equals after CSV parse + date/numeric normalize",
    }


def write_findings(
    out_dir: Path,
    *,
    rows: list[dict[str, Any]],
    initial_timings: dict[str, Any],
    speed: dict[str, Any],
    warnings: list[dict[str, Any]],
    universe_note: str,
    status: str,
    mode: str,
    command: str,
    equality: dict[str, Any] | None = None,
    source_fingerprints: dict[str, Any] | None = None,
) -> None:
    main_rows = [r for r in rows if r["window"] == "main"]
    lines = [
        "# India momentum_12_1 comparison findings",
        "",
        "Research comparison only. Not a trading recommendation.",
        "Descriptive fixed variants. Not independent out-of-sample proof.",
        "",
        f"- Status: `{status}`",
        f"- Mode: `{mode}`",
        f"- Command: `{command}`",
        f"- Output: `{out_dir}`",
        f"- Strategy: `{STRATEGY}` market `{MARKET}` universe `{UNIVERSE}` PIT explicit",
        f"- Dates main: `{MAIN_START}` .. `{MAIN_END}` inclusive via rolling end date",
        (
            "- Provider: offline replay of frozen FMP snapshot"
            if mode == "replay"
            else "- Provider: FMP via existing workflow caches, `refresh=False`"
        ),
        f"- Costs: `cost_model=india`, slippage fixed `{SLIPPAGE_BPS}` bps/side, "
        f"`commission_bps={COMMISSION_BPS}`",
        f"- Sizing: `equal_slot` top `{TOP}`, capital `{INITIAL_CAPITAL:.0f}`, "
        "no stops/targets/trails, entry `moo`",
        f"- Universe note: {universe_note}",
        f"- Ever members: {speed.get('universe_ever_members')}",
        f"- Membership counts: {speed.get('membership_counts_on_selected_dates')}",
        "",
        "## Semantics",
        "",
        "- `hold` is maximum age in bars.",
        "- Rank exit every 21 bars uses prior completed bar top "
        "`rank_universe_size` as a retention band.",
        "- Entry fills still occur whenever slots are free, not only on monthly "
        "rank-review bars.",
        "- Annual windows are descriptive slices on the same frozen prices and "
        "frozen membership (as_of updated per window end only).",
        f"- Turnover: `{TURNOVER_DEFINITION}`.",
        "",
        "## Initial prepare",
        "",
        f"- Wall s: {initial_timings.get('provider_prepare_wall_s')}",
        f"- CPU s: {initial_timings.get('provider_prepare_cpu_s')}",
        f"- Peak RSS MB after main prepare: {initial_timings.get('peak_rss_mb')}",
        f"- Replay adapter fetch wall s: "
        f"{initial_timings.get('replay_adapter_fetch_wall_s')}",
        "",
        "## Frozen-data speed (baseline hold21 / rank off)",
        "",
        "- Fresh prepare+simulate vs reused simulate shows the existing "
        "`PreparedRollingBacktest` reuse capability, not a new engine speedup.",
        f"- Fresh prepare+simulate wall s (3): {speed.get('fresh_prepare_plus_simulate_wall_s')}",
        f"- Reused simulate wall s (3): {speed.get('reused_simulate_wall_s')}",
        f"- Replay adapter fetch wall s mean: "
        f"{speed.get('replay_adapter_fetch_wall_s_mean')}",
        f"- cProfile wall s: {speed.get('cprofile_prepare_plus_simulate_wall_s')}",
        f"- Profiler overhead wall s: {speed.get('cprofile_overhead_wall_s')}",
        f"- Ledger/equity identity across fresh+reused: "
        f"{speed.get('identity', {}).get('identical_full_trade_ledger_and_equity')}",
        f"- Peak RSS MB: {speed.get('peak_rss_mb')}",
        "",
        "## Main window headline metrics",
        "",
        "| variant | Sharpe | CAGR | total_return | max_dd | excess | trades | "
        "both_sided_turnover | total_fees |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            "| {variant} | {sharpe} | {cagr} | {total_return} | {max_drawdown} | "
            "{excess_return} | {trade_count} | {both_sided_turnover} | {total_fees} |".format(
                variant=row["variant"],
                sharpe=_fmt(row.get("sharpe")),
                cagr=_fmt(row.get("cagr")),
                total_return=_fmt(row.get("total_return")),
                max_drawdown=_fmt(row.get("max_drawdown")),
                excess_return=_fmt(row.get("excess_return")),
                trade_count=row.get("trade_count"),
                both_sided_turnover=_fmt(row.get("both_sided_turnover")),
                total_fees=_fmt(row.get("total_fees")),
            )
        )
    if equality is not None:
        lines.extend(
            [
                "",
                "## Replay equality vs original main ledgers",
                "",
                f"- all_main_ledgers_and_equity_equal: "
                f"`{equality.get('all_main_ledgers_and_equity_equal')}`",
                f"- method: {equality.get('method')}",
                f"- compared variants: {equality.get('n_variants_compared')}",
                "",
            ]
        )
    if source_fingerprints is not None:
        lines.extend(
            [
                "## Source snapshot fingerprints (logged)",
                "",
                "See `fingerprints.json` and `price_snapshot/meta.json` for full "
                "tree hashes, lockfiles, and local portfolio.py edit record.",
                "",
            ]
        )
    lines.extend(
        [
            "## Warnings / limitations",
            "",
            f"- Warning events recorded: {len(warnings)} "
            f"(see `warnings.json` and per-run files).",
            "- Frozen FMP snapshot feeds every window; missing provider bars are "
            "listed under `price_snapshot/`.",
            "- Annual windows are not a walk-forward test and were not used to "
            "pick parameters.",
            "- Nifty 500 vs `^NSEI` can embed size/sector effects.",
            "- India delivery fee stack + 5 bps/side slippage; no extra flat commission.",
            "- No production code was changed.",
            "",
        ]
    )
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.6g}"
    return str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Report directory (default: reports/next_actions_2026-09-07/momentum)",
    )
    parser.add_argument(
        "--replay-from",
        type=Path,
        default=None,
        help=(
            "Offline mode: path to an existing run directory with "
            "price_snapshot/bars.parquet and resolved_config.json. "
            "Does not call FMP or live universe resolution."
        ),
    )
    return parser.parse_args(argv)


def _semantics(variants: list[VariantSpec]) -> dict[str, Any]:
    return {
        "strategy": STRATEGY,
        "market": MARKET,
        "universe": UNIVERSE,
        "point_in_time": True,
        "point_in_time_was_explicit": True,
        "benchmark": BENCHMARK,
        "top": TOP,
        "initial_capital": INITIAL_CAPITAL,
        "sizing_rule": "equal_slot",
        "compounding": False,
        "stops_targets_trails": None,
        "entry_order": "moo",
        "cost_model": COST_MODEL,
        "slippage_model": "fixed",
        "slippage_bps_per_side": SLIPPAGE_BPS,
        "commission_bps": COMMISSION_BPS,
        "commission_note": "0 because India fee stack handles statutory/broker fees",
        "hold_semantics": "maximum holding age in bars",
        "rank_exit_semantics": (
            "When enabled, every rank_exit_every bars flag holdings absent from "
            "the prior completed bar's top rank_universe_size. Hold remains the "
            "maximum age. Entry fills still occur when slots free, not only on "
            "rank-review bars."
        ),
        "variants": [v.name for v in variants],
        "windows": {
            "main": [MAIN_START.isoformat(), MAIN_END.isoformat()],
            **{name: [s.isoformat(), e.isoformat()] for name, s, e in ANNUAL_WINDOWS},
        },
        "selection_policy": (
            "predeclared descriptive comparison; no best-parameter selection; "
            "no research-report / walk-forward optimizer"
        ),
        "turnover_definition": TURNOVER_DEFINITION,
        "membership_policy": (
            "After the initial main resolve (live) or restore (replay), annual "
            "windows update as_of to the window end and keep the frozen "
            "membership_windows / tickers unchanged."
        ),
    }


def _run_all_windows(
    *,
    base_cfg: Any,
    replay: ReplayPriceFetcher,
    out_dir: Path,
    variants: list[VariantSpec],
    rows: list[dict[str, Any]],
    all_warnings: list[dict[str, Any]],
    main_prepared: Any | None = None,
    main_prepare_timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate main + annual windows on one frozen replay fetcher."""
    from screener.backtester import prepare_rolling_backtest

    initial_timings = dict(main_prepare_timings or {})
    if main_prepared is None:
        progress(
            f"window=main prepare (frozen) ever_members={len(base_cfg.tickers or ())}"
        )
        replay.fetch_wall_s = 0.0
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        main_cfg = window_cfg_from_frozen(base_cfg, end=MAIN_END)
        main_prepared = prepare_rolling_backtest(
            main_cfg,
            replay,
            start_date=MAIN_START,
            end_date=MAIN_END,
            fundamental_fetcher=None,
        )
        initial_timings = {
            "provider_prepare_wall_s": time.perf_counter() - wall0,
            "provider_prepare_cpu_s": time.process_time() - cpu0,
            "peak_rss_mb": peak_rss_mb(),
            "replay_adapter_fetch_wall_s": replay.fetch_wall_s,
            "n_master_dates": len(main_prepared.master_dates),
        }
        for warning in main_prepared.warnings:
            all_warnings.append(
                {"window": "main", "phase": "prepare", "warning": warning}
            )
    else:
        main_cfg = window_cfg_from_frozen(base_cfg, end=MAIN_END)

    progress("window=main simulate 12 variants on prepared panel")
    for variant in variants:
        cfg = apply_variant(main_cfg, variant)
        if not main_prepared.supports(cfg):
            raise RuntimeError(f"main prepared data rejects {variant.name}")
        result = run_prepared_rolling_backtest_safe(main_prepared, cfg)
        row = metrics_row(
            window="main",
            start=MAIN_START,
            end=MAIN_END,
            variant=variant,
            result=result,
        )
        rows.append(row)
        save_run_artifacts(
            out_dir / "runs" / "main" / variant.name, result=result, row=row
        )
        for warning in result.warnings or []:
            all_warnings.append(
                {
                    "window": "main",
                    "variant": variant.name,
                    "phase": "simulate",
                    "warning": warning,
                }
            )
    progress(
        f"window=main done variants=12 "
        f"prep_wall_s={float(initial_timings.get('provider_prepare_wall_s', 0.0)):.1f} "
        f"peak_rss_mb={peak_rss_mb():.0f}"
    )

    for name, start, end in ANNUAL_WINDOWS:
        window_cfg = window_cfg_from_frozen(base_cfg, end=end)
        run_window(
            window=name,
            start=start,
            end=end,
            base_cfg=window_cfg,
            fetcher=replay,
            out_dir=out_dir,
            variants=variants,
            rows=rows,
            all_warnings=all_warnings,
        )
    return initial_timings


def run_live(out_dir: Path, variants: list[VariantSpec]) -> int:
    from screener.backtester import prepare_rolling_backtest
    from screener.backtester.workflow import resolve_backtest_run

    configure_fmp_no_refresh()
    rows: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []

    baseline_req = build_request(
        start=MAIN_START,
        end=MAIN_END,
        hold=21,
        rank_exit_every=None,
        rank_universe_size=None,
    )
    run = resolve_backtest_run(baseline_req)
    base_cfg = run.config
    universe_note = run.universe_note or ""

    write_json(
        out_dir / "resolved_config.json",
        resolved_summary_from_config(base_cfg, universe_note=universe_note),
    )
    fingerprints = {
        "code": code_fingerprints(),
        "universe_note": universe_note,
        "membership_windows_sha256": sha256_bytes(
            json.dumps(
                base_cfg.model_dump(mode="json").get("membership_windows"),
                sort_keys=True,
            ).encode()
        ),
        "tickers_sha256": sha256_bytes(
            json.dumps(list(base_cfg.tickers or ()), sort_keys=True).encode()
        ),
    }
    write_json(out_dir / "fingerprints.json", fingerprints)

    recorder = RecordingPriceFetcher(run.price_fetcher)
    progress(f"window=main acquire+prepare ever_members={len(base_cfg.tickers or ())}")
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    main_prepared = prepare_rolling_backtest(
        base_cfg,
        recorder,
        start_date=MAIN_START,
        end_date=MAIN_END,
        fundamental_fetcher=None,
    )
    live_timings = {
        "provider_prepare_wall_s": time.perf_counter() - wall0,
        "provider_prepare_cpu_s": time.process_time() - cpu0,
        "peak_rss_mb": peak_rss_mb(),
        "n_fetch_calls": len(recorder.fetch_calls),
        "n_recorded_symbols": len(recorder.seen_symbols),
        "n_missing_provider_bars": len(recorder.missing_provider_bars),
        "fetch_calls": recorder.fetch_calls,
        "replay_adapter_fetch_wall_s": None,
    }
    write_json(out_dir / "initial_provider_prepare_timings.json", live_timings)

    snap_dir = out_dir / "price_snapshot"
    parquet_path = snap_dir / "bars.parquet"
    coverage = frames_to_parquet(recorder.frames, parquet_path)
    digest = sha256_file(parquet_path)
    (snap_dir / "bars.sha256").write_text(digest + "\n")
    write_json(snap_dir / "coverage.json", coverage)
    write_json(snap_dir / "symbol_manifest.json", {"symbols": sorted(coverage.keys())})
    write_json(snap_dir / "missing_provider_bars.json", recorder.missing_provider_bars)
    write_json(
        snap_dir / "meta.json",
        {
            "sha256": digest,
            "n_symbols_recorded": len(recorder.seen_symbols),
            "n_symbols_with_bars": sum(
                1 for c in coverage.values() if not c.get("empty")
            ),
            "n_symbols_empty": sum(1 for c in coverage.values() if c.get("empty")),
            "fetch_calls": recorder.fetch_calls,
            "fingerprints": fingerprints,
            "main_prepare_warnings": list(main_prepared.warnings),
            "replayable": True,
            "provider": "fmp",
            "refresh": False,
        },
    )
    for warning in main_prepared.warnings:
        all_warnings.append(
            {"window": "main", "phase": "initial_prepare", "warning": warning}
        )

    frames = load_frames_from_parquet(parquet_path)
    for symbol, info in coverage.items():
        if info.get("empty") and symbol not in frames:
            frames[symbol] = pd.DataFrame()
    replay = ReplayPriceFetcher(frames, recorded_symbols=set(coverage.keys()))

    initial_timings = _run_all_windows(
        base_cfg=base_cfg,
        replay=replay,
        out_dir=out_dir,
        variants=variants,
        rows=rows,
        all_warnings=all_warnings,
        main_prepared=main_prepared,
        main_prepare_timings=live_timings,
    )
    speed = speed_benchmark(
        base_cfg=base_cfg,
        replay=replay,
        out_dir=out_dir,
        start=MAIN_START,
        end=MAIN_END,
    )
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    write_json(out_dir / "warnings.json", all_warnings)
    command = "uv run python scripts/run_india_momentum_comparison.py"
    write_findings(
        out_dir,
        rows=rows,
        initial_timings=initial_timings,
        speed=speed,
        warnings=all_warnings,
        universe_note=universe_note,
        status="completed",
        mode="live",
        command=command,
    )
    _print_key_results(rows, speed)
    write_json(
        out_dir / "status.json",
        {
            "status": "completed",
            "mode": "live",
            "n_rows": len(rows),
            "results_csv": str(out_dir / "results.csv"),
            "peak_rss_mb": peak_rss_mb(),
            "finished_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    return 0


def run_replay(out_dir: Path, replay_from: Path, variants: list[VariantSpec]) -> int:
    rows: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []

    snap_dir = replay_from / "price_snapshot"
    parquet_path = snap_dir / "bars.parquet"
    sha_path = snap_dir / "bars.sha256"
    resolved_path = replay_from / "resolved_config.json"
    if not parquet_path.exists():
        raise FileNotFoundError(f"missing frozen bars parquet: {parquet_path}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"missing resolved_config.json: {resolved_path}")

    expected = sha_path.read_text().strip() if sha_path.exists() else None
    actual = sha256_file(parquet_path)
    if expected and actual != expected:
        raise RuntimeError(
            f"snapshot SHA256 mismatch: file={actual} expected={expected}"
        )
    meta_path = snap_dir / "meta.json"
    source_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if source_meta.get("sha256") and source_meta["sha256"] != actual:
        raise RuntimeError(
            f"snapshot meta sha256 mismatch: file={actual} meta={source_meta['sha256']}"
        )

    resolved_payload = json.loads(resolved_path.read_text())
    full_config = resolved_payload.get("full_config")
    if not isinstance(full_config, dict):
        raise RuntimeError("resolved_config.json missing full_config object")
    base_cfg = restore_backtest_config(full_config)
    universe_note = (resolved_payload.get("universe") or {}).get(
        "universe_note"
    ) or "restored from frozen resolved_config.json"

    source_fingerprints = source_meta.get("fingerprints") or {}
    if (replay_from / "fingerprints.json").exists():
        source_fingerprints = json.loads(
            (replay_from / "fingerprints.json").read_text()
        )
    fingerprints = {
        "code": code_fingerprints(),
        "source_run_fingerprints": source_fingerprints,
        "source_snapshot_sha256": actual,
        "source_snapshot_path": str(parquet_path),
        "universe_note": universe_note,
        "membership_windows_sha256": sha256_bytes(
            json.dumps(
                base_cfg.model_dump(mode="json").get("membership_windows"),
                sort_keys=True,
            ).encode()
        ),
        "tickers_sha256": sha256_bytes(
            json.dumps(list(base_cfg.tickers or ()), sort_keys=True).encode()
        ),
    }
    write_json(out_dir / "fingerprints.json", fingerprints)
    write_json(
        out_dir / "resolved_config.json",
        resolved_summary_from_config(base_cfg, universe_note=universe_note),
    )

    manifest = load_symbol_manifest(snap_dir)
    frames = load_frames_from_parquet(parquet_path)
    for symbol in manifest:
        if symbol not in frames:
            frames[symbol] = pd.DataFrame()
    # Copy snapshot metadata into out-dir without rewriting the source parquet.
    out_snap = out_dir / "price_snapshot"
    out_snap.mkdir(parents=True, exist_ok=True)
    (out_snap / "bars.sha256").write_text(actual + "\n")
    write_json(out_snap / "symbol_manifest.json", {"symbols": manifest})
    if (snap_dir / "coverage.json").exists():
        (out_snap / "coverage.json").write_text(
            (snap_dir / "coverage.json").read_text()
        )
    if (snap_dir / "missing_provider_bars.json").exists():
        (out_snap / "missing_provider_bars.json").write_text(
            (snap_dir / "missing_provider_bars.json").read_text()
        )
    write_json(
        out_snap / "meta.json",
        {
            "sha256": actual,
            "source_run": str(replay_from),
            "n_symbols_manifest": len(manifest),
            "n_symbols_with_bars": sum(
                1 for s in manifest if s in frames and not frames[s].empty
            ),
            "n_symbols_empty": sum(
                1 for s in manifest if s not in frames or frames[s].empty
            ),
            "fingerprints": fingerprints,
            "replayable": True,
            "provider": "frozen_snapshot",
            "refresh": False,
            "bars_parquet_source": str(parquet_path),
        },
    )
    # Point replay at the original parquet bytes (verified SHA256).
    write_json(
        out_snap / "bars_source.json",
        {"path": str(parquet_path), "sha256": actual},
    )

    replay = ReplayPriceFetcher(frames, recorded_symbols=set(manifest))
    initial_timings = _run_all_windows(
        base_cfg=base_cfg,
        replay=replay,
        out_dir=out_dir,
        variants=variants,
        rows=rows,
        all_warnings=all_warnings,
    )
    write_json(out_dir / "initial_provider_prepare_timings.json", initial_timings)

    speed = speed_benchmark(
        base_cfg=base_cfg,
        replay=replay,
        out_dir=out_dir,
        start=MAIN_START,
        end=MAIN_END,
    )
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    write_json(out_dir / "warnings.json", all_warnings)

    equality = compare_main_ledgers(
        original_dir=replay_from, replay_dir=out_dir, variants=variants
    )
    write_json(out_dir / "verification_equality.json", equality)

    command = (
        "uv run python scripts/run_india_momentum_comparison.py "
        f"--replay-from {replay_from} --out-dir {out_dir}"
    )
    write_findings(
        out_dir,
        rows=rows,
        initial_timings=initial_timings,
        speed=speed,
        warnings=all_warnings,
        universe_note=universe_note,
        status="completed",
        mode="replay",
        command=command,
        equality=equality,
        source_fingerprints=source_fingerprints,
    )
    _print_key_results(rows, speed)
    progress(
        f"EQUALITY main ledgers/equity: {equality['all_main_ledgers_and_equity_equal']}"
    )
    write_json(
        out_dir / "status.json",
        {
            "status": "completed",
            "mode": "replay",
            "n_rows": len(rows),
            "results_csv": str(out_dir / "results.csv"),
            "peak_rss_mb": peak_rss_mb(),
            "snapshot_sha256": actual,
            "main_ledger_equity_equal": equality["all_main_ledgers_and_equity_equal"],
            "finished_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    return 0 if equality["all_main_ledgers_and_equity_equal"] else 2


def _print_key_results(rows: list[dict[str, Any]], speed: dict[str, Any]) -> None:
    progress("KEY main-window results:")
    for row in rows:
        if row["window"] != "main":
            continue
        progress(
            f"  {row['variant']}: sharpe={_fmt(row.get('sharpe'))} "
            f"cagr={_fmt(row.get('cagr'))} "
            f"tot={_fmt(row.get('total_return'))} "
            f"mdd={_fmt(row.get('max_drawdown'))} "
            f"excess={_fmt(row.get('excess_return'))} "
            f"trades={row.get('trade_count')} "
            f"turn={_fmt(row.get('both_sided_turnover'))}"
        )
    progress(
        "SPEED fresh_mean_s="
        f"{_fmt(speed.get('fresh_wall_mean_s'))} reused_mean_s="
        f"{_fmt(speed.get('reused_wall_mean_s'))} "
        f"(existing reuse path, not a new speedup) "
        f"replay_adapter_fetch_mean_s="
        f"{_fmt(speed.get('replay_adapter_fetch_wall_s_mean'))} "
        f"cprofile_overhead_s={_fmt(speed.get('cprofile_overhead_wall_s'))} "
        f"peak_rss_mb={_fmt(speed.get('peak_rss_mb'))}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    write_json(
        status_path,
        {
            "status": "running",
            "mode": "replay" if args.replay_from else "live",
            "started_at": datetime.utcnow().isoformat() + "Z",
        },
    )

    variants = variant_grid()
    assert len(variants) == 12, len(variants)
    write_json(out_dir / "semantics.json", _semantics(variants))

    try:
        if args.replay_from is not None:
            return run_replay(out_dir, args.replay_from.resolve(), variants)
        return run_live(out_dir, variants)
    except Exception as exc:
        err = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finished_at": datetime.utcnow().isoformat() + "Z",
        }
        write_json(status_path, err)
        write_json(out_dir / "failure.json", err)
        (out_dir / "findings.md").write_text(
            "\n".join(
                [
                    "# India momentum_12_1 comparison findings",
                    "",
                    "Status: failed before completion.",
                    f"Error: `{type(exc).__name__}: {exc}`",
                    "See `failure.json` and `status.json`.",
                    "",
                ]
            )
            + "\n"
        )
        progress(f"FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
