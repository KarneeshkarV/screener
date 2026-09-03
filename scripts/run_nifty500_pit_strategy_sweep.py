#!/usr/bin/env python
"""Nifty 500 point-in-time rolling sweep across every expression strategy.

Runs every registered backtest-rolling strategy on ``nifty500_pit`` with
point-in-time membership, for trailing 1/2/3/4/5-year windows, plus the
config-flag variants that change fills, ranking, or candidate gates.

    uv run python scripts/run_nifty500_pit_strategy_sweep.py
    uv run python scripts/run_nifty500_pit_strategy_sweep.py --smoke
    uv run python scripts/run_nifty500_pit_strategy_sweep.py --report-only

Callable-only plugins (``bb_pattern``, ``heikin_ashi``, ``rsi_pattern``,
``shooting_star``) have no entry/exit expressions and are skipped.

This is research, not financial advice.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "reports" / "nifty500_pit_sweep_fmp"
UNIVERSE_CONFIG = ROOT / "universes.yaml"
PRICE_PROVIDER = "fmp"
FUNDAMENTALS_PROVIDER = "fmp"
END_DATE = date(2026, 9, 1)
YEARS = (5, 4, 3, 2, 1)
HOLDS = (10, 20, 63, 126)
TOPS = (5, 10, 20)
DEFAULT_HOLD = 20
DEFAULT_TOP = 10
CALLABLE_ONLY = ("bb_pattern", "heikin_ashi", "rsi_pattern", "shooting_star")
FUNDAMENTAL_STRATEGIES = (
    "ema150_200_revenue_up_3q",
    "minervini_growth_in",
    "minervini_growth_us",
    "minervini_pro_in",
    "minervini_pro_us",
    "mq_in1",
    "mq_in2",
    "mq_in3",
    "mq_us1",
    "mq_us2",
)

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
)


def configure_fmp_sources() -> None:
    """Force FMP for prices and load ``FMP_API_KEY`` from the project ``.env``.

    ``SCREENER_PRICE_PROVIDER=fmp`` selects :class:`FMPPriceFetcher` with no
    yfinance primary. Fundamentals are selected per request via
    ``fundamentals_provider='fmp'``.
    """
    import os

    from screener.config import load_env_file

    load_env_file()
    os.environ["SCREENER_PRICE_PROVIDER"] = PRICE_PROVIDER


def window_start(years: int, end: date = END_DATE) -> date:
    """CLI-identical trailing window: ``end - 365 * years`` calendar days."""
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


def run_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / "runs" / f"{run_id}.json"


def make_request(**overrides: Any) -> Any:
    from screener.backtester.workflow import BacktestRequest

    payload: dict[str, Any] = {
        "mode": "rolling",
        "context_obj": None,
        "market": "india",
        "hold": DEFAULT_HOLD,
        "top": DEFAULT_TOP,
        "entry_expr": None,
        "exit_expr": None,
        "strategy_name": "ema_trend",
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "cost_model": "flat",
        "initial_capital": 100_000.0,
        "benchmark": None,
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
        "sizing_risk_pct": 0.01,
        "sizing_position_pct": 0.10,
        "sizing_atr_window": 14,
        "sizing_atr_multiple": 2.0,
        "sizing_vol_window": 20,
        "intraday_only": False,
        "start_arg": datetime.combine(window_start(5), datetime.min.time()),
        "end_arg": datetime.combine(END_DATE, datetime.min.time()),
        "years": 5,
        "universe": "nifty500_pit",
        "universe_config": UNIVERSE_CONFIG,
        "point_in_time": True,
        "point_in_time_was_explicit": True,
        "compare_reinvestment": False,
        "fundamentals_provider": FUNDAMENTALS_PROVIDER,
    }
    payload.update(overrides)
    return BacktestRequest(**payload)


def slice_prepared(prepared: Any, start: date) -> Any:
    start_ts = pd.Timestamp(start).normalize()
    dates = tuple(d for d in prepared.master_dates if d >= start_ts)
    if not dates:
        return None
    return replace(prepared, start_ts=start_ts, master_dates=dates)


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


def write_run(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))


def expression_strategy_names() -> list[str]:
    from screener.strategies.expressions import NAMED_STRATEGIES

    return sorted(NAMED_STRATEGIES)


def signal_variants() -> list[dict[str, Any]]:
    """Overlays that change the prepared panel. Each one rebuilds signals."""
    return [
        {"variant": "gates_default", "request": {}},
        {"variant": "sector_neutral", "request": {"sector_neutral": True}},
        {"variant": "regime_bull", "request": {"regime_filter_args": ("bull",)}},
        {
            "variant": "regime_pullback",
            "request": {"regime_filter_args": ("pullback",)},
        },
        {"variant": "regime_bear", "request": {"regime_filter_args": ("bear",)}},
        {
            "variant": "regime_nonbear",
            "request": {"regime_filter_args": ("bull", "pullback")},
        },
        {"variant": "min_score_30", "request": {"min_score": 30.0}},
        {"variant": "min_score_50", "request": {"min_score": 50.0}},
        {"variant": "min_score_70", "request": {"min_score": 70.0}},
        {"variant": "min_price_0", "request": {"min_price": 0.0}},
        {"variant": "min_price_50", "request": {"min_price": 50.0}},
        {
            "variant": "min_adv_0",
            "request": {"min_avg_dollar_volume": 0.0},
        },
        {
            "variant": "min_adv_1e6",
            "request": {"min_avg_dollar_volume": 1_000_000.0},
        },
        {
            "variant": "adv_window_60",
            "request": {"adv_window": 60, "adv_window_was_explicit": True},
        },
        {
            "variant": "earnings_blackout_5",
            "request": {"earnings_blackout_days": 5},
        },
        {
            "variant": "earnings_blackout_21",
            "request": {"earnings_blackout_days": 21},
        },
        {
            "variant": "adj_splits_only",
            "request": {"price_adjustment": "splits_only"},
        },
        {"variant": "adj_none", "request": {"price_adjustment": "none"}},
        {
            "variant": "fund_lag_60",
            "request": {"fundamental_lag_days": 60},
        },
        {"variant": "max_univ_50", "request": {"max_universe": 50}},
        {"variant": "max_univ_100", "request": {"max_universe": 100}},
        {"variant": "max_univ_200", "request": {"max_universe": 200}},
    ]


def book_overlays() -> list[tuple[str, dict[str, Any]]]:
    """One-at-a-time book overlays at hold 20, top 10."""
    return [
        (
            "sizing_reinvested_equal_slot",
            {"sizing_rule": "reinvested_equal_slot"},
        ),
        ("sizing_inverse_vol", {"sizing_rule": "inverse_vol"}),
        ("sizing_atr_risk", {"sizing_rule": "atr_risk"}),
        ("sizing_fixed_fraction", {"sizing_rule": "fixed_fraction"}),
        (
            "sizing_fixed_risk_sl08",
            {"sizing_rule": "fixed_risk", "stop_loss": 0.08},
        ),
        ("stop_0.08", {"stop_loss": 0.08}),
        ("tp_0.20", {"take_profit": 0.20}),
        ("trail_0.15", {"trailing_stop": 0.15}),
        (
            "costs_india_5bps",
            {
                "slippage_bps": 5.0,
                "commission_bps": 5.0,
                "cost_model": "india",
            },
        ),
        (
            "rank_exit_monthly_univ20",
            {"rank_exit_every": 21, "rank_universe_size": 20},
        ),
        (
            "rank_exit_monthly_univ100",
            {"rank_exit_every": 21, "rank_universe_size": 100},
        ),
        ("rank_exit_1", {"rank_exit_every": 1}),
        ("rank_exit_63", {"rank_exit_every": 63}),
        (
            "slip_vol_impact_k_0.05",
            {"slippage_model_name": "vol-impact", "vol_impact_k": 0.05},
        ),
        (
            "slip_vol_impact_k_0.25",
            {"slippage_model_name": "vol-impact", "vol_impact_k": 0.25},
        ),
        (
            "slip_half_spread_10",
            {
                "slippage_model_name": "half-spread",
                "half_spread_bps": 10.0,
            },
        ),
        (
            "entry_limit_10bps",
            {"entry_order_type": "limit", "entry_limit_bps": 10.0},
        ),
        (
            "entry_limit_50bps",
            {"entry_order_type": "limit", "entry_limit_bps": 50.0},
        ),
        ("partial_exit_20_25", {"partial_exits": ((0.20, 0.25),)}),
        (
            "sizing_inv_vol_pct_0.005",
            {"sizing_rule": "inverse_vol", "sizing_risk_pct": 0.005},
        ),
        (
            "sizing_inv_vol_pct_0.02",
            {"sizing_rule": "inverse_vol", "sizing_risk_pct": 0.02},
        ),
        ("stop_0.05", {"stop_loss": 0.05}),
        ("stop_0.12", {"stop_loss": 0.12}),
        ("tp_0.10", {"take_profit": 0.10}),
        ("tp_0.30", {"take_profit": 0.30}),
        ("trail_0.08", {"trailing_stop": 0.08}),
        ("trail_0.25", {"trailing_stop": 0.25}),
        ("slippage_5bps", {"slippage_bps": 5.0}),
        ("slippage_10bps", {"slippage_bps": 10.0}),
        ("commission_5bps", {"commission_bps": 5.0}),
        ("commission_10bps", {"commission_bps": 10.0}),
        ("cost_model_india", {"cost_model": "india"}),
        (
            "slip_half_spread_5",
            {
                "slippage_model_name": "half-spread",
                "half_spread_bps": 5.0,
            },
        ),
        (
            "slip_vol_impact",
            {"slippage_model_name": "vol-impact", "vol_impact_k": 0.1},
        ),
        (
            "slip_composite_5_5",
            {
                "slippage_model_name": "composite",
                "slippage_bps": 5.0,
                "half_spread_bps": 5.0,
                "vol_impact_k": 0.1,
            },
        ),
        ("spread_proxy", {"spread_proxy": True}),
        ("no_gap_fills", {"gap_fills": False}),
        ("entry_moc", {"entry_order_type": "moc"}),
        (
            "entry_limit_20bps",
            {"entry_order_type": "limit", "entry_limit_bps": 20.0},
        ),
        ("partial_exit_10_50", {"partial_exits": ((0.10, 0.50),)}),
        ("no_reinvest", {"reinvest": False}),
        (
            "allow_reentry_3",
            {"allow_reentry": True, "max_reentries": 3},
        ),
        ("reserve_1", {"reserve_multiple": 1}),
        ("reserve_5", {"reserve_multiple": 5}),
        ("capital_500k", {"initial_capital": 500_000.0}),
        (
            "sizing_atr_risk_pct_0.005",
            {"sizing_rule": "atr_risk", "sizing_risk_pct": 0.005},
        ),
        (
            "sizing_atr_risk_pct_0.02",
            {"sizing_rule": "atr_risk", "sizing_risk_pct": 0.02},
        ),
        (
            "sizing_atr_window_7",
            {"sizing_rule": "atr_risk", "sizing_atr_window": 7},
        ),
        (
            "sizing_atr_window_21",
            {"sizing_rule": "atr_risk", "sizing_atr_window": 21},
        ),
        (
            "sizing_atr_mult_1",
            {"sizing_rule": "atr_risk", "sizing_atr_multiple": 1.0},
        ),
        (
            "sizing_atr_mult_3",
            {"sizing_rule": "atr_risk", "sizing_atr_multiple": 3.0},
        ),
        (
            "sizing_frac_0.05",
            {"sizing_rule": "fixed_fraction", "sizing_position_pct": 0.05},
        ),
        (
            "sizing_frac_0.20",
            {"sizing_rule": "fixed_fraction", "sizing_position_pct": 0.20},
        ),
        (
            "sizing_vol_window_10",
            {"sizing_rule": "inverse_vol", "sizing_vol_window": 10},
        ),
        (
            "sizing_vol_window_40",
            {"sizing_rule": "inverse_vol", "sizing_vol_window": 40},
        ),
    ]


def book_jobs(signal_variant: str) -> list[dict[str, Any]]:
    """Book-config jobs that reuse one prepared panel.

    Default-gate prepares sweep hold x top plus one-at-a-time overlays at
    the CLI default of hold=20, top=10. Overlay prepares keep hold/top fixed
    so the gate change is the only moving piece.
    """
    jobs: list[dict[str, Any]] = []
    if signal_variant == "gates_default":
        for hold in HOLDS:
            for top in TOPS:
                jobs.append(
                    {
                        "variant": f"h{hold}_t{top}",
                        "update": {"hold": hold, "top": top},
                    }
                )
                jobs.append(
                    {
                        "variant": f"h{hold}_t{top}__rank_exit_weekly",
                        "update": {
                            "hold": hold,
                            "top": top,
                            "rank_exit_every": 5,
                        },
                    }
                )
                jobs.append(
                    {
                        "variant": f"h{hold}_t{top}__rank_exit_monthly",
                        "update": {
                            "hold": hold,
                            "top": top,
                            "rank_exit_every": 21,
                        },
                    }
                )
        overlays = book_overlays()
        for name, update in overlays:
            jobs.append(
                {
                    "variant": f"h{DEFAULT_HOLD}_t{DEFAULT_TOP}__{name}",
                    "update": {
                        "hold": DEFAULT_HOLD,
                        "top": DEFAULT_TOP,
                        **update,
                    },
                }
            )
    else:
        jobs.append(
            {
                "variant": f"h{DEFAULT_HOLD}_t{DEFAULT_TOP}__{signal_variant}",
                "update": {"hold": DEFAULT_HOLD, "top": DEFAULT_TOP},
            }
        )
    return jobs


_CLI_BY_FIELD = {
    "hold": "--hold",
    "top": "--top",
    "stop_loss": "--stop-loss",
    "take_profit": "--take-profit",
    "trailing_stop": "--trailing-stop",
    "slippage_bps": "--slippage-bps",
    "commission_bps": "--commission-bps",
    "cost_model": "--cost-model",
    "half_spread_bps": "--half-spread-bps",
    "vol_impact_k": "--vol-impact-k",
    "slippage_model_name": "--slippage-model",
    "entry_order_type": "--entry-order",
    "entry_limit_bps": "--entry-limit-bps",
    "initial_capital": "--initial-capital",
    "sizing_rule": "--sizing",
    "sizing_risk_pct": "--sizing-risk-pct",
    "sizing_position_pct": "--sizing-position-pct",
    "sizing_atr_window": "--sizing-atr-window",
    "sizing_atr_multiple": "--sizing-atr-multiple",
    "sizing_vol_window": "--sizing-vol-window",
    "rank_universe_size": "--rank-universe-size",
    "min_score": "--min-score",
    "min_price": "--min-price",
    "min_avg_dollar_volume": "--min-avg-dollar-volume",
    "adv_window": "--adv-window",
    "earnings_blackout_days": "--earnings-blackout",
    "price_adjustment": "--price-adjustment",
    "fundamental_lag_days": "--fundamental-lag-days",
    "max_universe": "--max-universe",
}


def fields_to_cli(fields: dict[str, Any]) -> list[str]:
    """Turn an overlay dict into `backtest-rolling` flags."""
    flags: list[str] = []
    for key, value in fields.items():
        if key == "adv_window_was_explicit":
            continue
        if key == "regime_filter_args":
            for item in value:
                flags.append(f"--regime-filter {item}")
            continue
        if key == "rank_exit_every":
            if value == 5:
                flags.append("--rank-exit weekly")
            elif value == 21:
                flags.append("--rank-exit monthly")
            else:
                flags.append(f"--rank-exit {value}")
            continue
        if key == "partial_exits":
            for profit, shares in value:
                flags.append(f"--partial-exit {profit}:{shares}")
            continue
        if key == "spread_proxy" and value:
            flags.append("--spread-proxy")
            continue
        if key == "sector_neutral" and value:
            flags.append("--sector-neutral")
            continue
        if key == "gap_fills" and value is False:
            flags.append("--no-gap-fills")
            continue
        if key == "reinvest" and value is False:
            flags.append("# historical --no-reinvest (not on backtest-rolling)")
            continue
        if key == "allow_reentry" and value:
            flags.append("# historical --allow-reentry")
            continue
        if key == "max_reentries":
            flags.append(f"# historical --max-reentries {value}")
            continue
        if key == "reserve_multiple":
            flags.append(f"# historical --reserve-multiple {value}")
            continue
        flag = _CLI_BY_FIELD.get(key)
        if flag is None:
            flags.append(f"# {key}={jsonable(value)}")
            continue
        flags.append(f"{flag} {value}")
    return flags


BASELINE_CLI = (
    "-m india",
    "--universe nifty500_pit",
    "--universe-config universes.yaml",
    "--point-in-time",
    "--interval 1d",
    "--benchmark ^NSEI",
    "--fundamentals-provider fmp",
    f"--hold {DEFAULT_HOLD}",
    f"--top {DEFAULT_TOP}",
    "--sizing equal_slot",
    "--slippage-bps 0",
    "--commission-bps 0",
    "--cost-model flat",
    "--entry-order moo",
    "--price-adjustment full",
    "--max-universe 0",
    "--initial-capital 100000",
    "--adv-window 20",
    "--end 2026-09-01",
)


def build_config_catalog() -> dict[str, Any]:
    """Overlay name -> CLI flags, for the HTML config tab."""
    overlays: list[dict[str, Any]] = [
        {
            "name": "rank_exit_weekly",
            "kind": "book",
            "cli": ["--rank-exit weekly"],
            "fields": {"rank_exit_every": 5},
            "note": "Crossed with every hold x top cell.",
        },
        {
            "name": "rank_exit_monthly",
            "kind": "book",
            "cli": ["--rank-exit monthly"],
            "fields": {"rank_exit_every": 21},
            "note": "Crossed with every hold x top cell.",
        },
    ]
    for hold in HOLDS:
        for top in TOPS:
            overlays.append(
                {
                    "name": f"h{hold}_t{top}",
                    "kind": "grid",
                    "cli": [f"--hold {hold}", f"--top {top}"],
                    "fields": {"hold": hold, "top": top},
                    "note": "Core grid cell. No extra overlay.",
                }
            )
    for variant in signal_variants():
        name = str(variant["variant"])
        if name == "gates_default":
            continue
        request = dict(variant["request"])
        overlays.append(
            {
                "name": name,
                "kind": "signal",
                "cli": fields_to_cli(request),
                "fields": jsonable(request),
                "note": "Rebuilds the candidate panel. Hold 20, top 10.",
            }
        )
    for name, update in book_overlays():
        overlays.append(
            {
                "name": name,
                "kind": "book",
                "cli": fields_to_cli(update),
                "fields": jsonable(update),
                "note": "Book overlay on hold 20, top 10.",
            }
        )
    return {
        "baseline": {
            "name": "h20_t10 / gates_default",
            "kind": "baseline",
            "cli": list(BASELINE_CLI),
            "fields": {
                "market": "india",
                "universe": "nifty500_pit",
                "point_in_time": True,
                "interval": "1d",
                "benchmark": "^NSEI",
                "fundamentals_provider": FUNDAMENTALS_PROVIDER,
                "price_provider": PRICE_PROVIDER,
                "hold": DEFAULT_HOLD,
                "top": DEFAULT_TOP,
                "sizing_rule": "equal_slot",
                "slippage_bps": 0.0,
                "commission_bps": 0.0,
                "cost_model": "flat",
                "entry_order": "moo",
                "price_adjustment": "full",
                "max_universe": 0,
                "initial_capital": 100000.0,
                "adv_window": 20,
                "end": END_DATE.isoformat(),
            },
            "note": (
                "India min-price and min-ADV floors still apply unless an overlay "
                "sets them. Named strategies keep their own entry/exit expressions."
            ),
        },
        "overlays": overlays,
        "command_prefix": "uv run screener backtest-rolling",
    }


def apply_book_update(cfg: Any, update: dict[str, Any]) -> Any:
    from screener.backtester.cli_common import build_slippage_model

    patched = dict(update)
    model_name = patched.pop("slippage_model_name", None)
    half_spread_bps = patched.pop("half_spread_bps", 0.0)
    vol_impact_k = patched.pop("vol_impact_k", 0.1)
    rebuild_slip = (
        model_name is not None
        or "slippage_bps" in patched
        or "spread_proxy" in patched
        or "cost_model" in patched
        or "half_spread_bps" in update
        or "vol_impact_k" in update
    )
    if rebuild_slip:
        patched["slippage_model"] = build_slippage_model(
            model_name or "fixed",
            float(patched.get("slippage_bps", cfg.slippage_bps)),
            float(half_spread_bps),
            float(vol_impact_k),
            spread_proxy=bool(patched.get("spread_proxy", cfg.spread_proxy)),
        )
    return cfg.model_copy(update=patched)


def result_record(
    *,
    strategy: str,
    years: int,
    start: date,
    variant: str,
    signal_variant: str,
    cfg: Any,
    result: Any | None,
    error: str | None,
    elapsed: float,
    universe_note: str | None,
) -> dict[str, Any]:
    metrics = compact_metrics(result.metrics) if result is not None else {}
    warnings = list(result.warnings) if result is not None else []
    return {
        "strategy": strategy,
        "years": years,
        "start": start.isoformat(),
        "end": END_DATE.isoformat(),
        "variant": variant,
        "signal_variant": signal_variant,
        "hold": cfg.hold,
        "top": cfg.top,
        "sizing_rule": cfg.sizing_rule,
        "stop_loss": cfg.stop_loss,
        "take_profit": cfg.take_profit,
        "trailing_stop": cfg.trailing_stop,
        "sector_neutral": cfg.sector_neutral,
        "regime_filter": list(cfg.regime_filter),
        "rank_exit_every": cfg.rank_exit_every,
        "cost_model": cfg.cost_model,
        "slippage_bps": cfg.slippage_bps,
        "commission_bps": cfg.commission_bps,
        "min_score": cfg.min_score,
        "earnings_blackout_days": cfg.earnings_blackout_days,
        "universe": "nifty500_pit",
        "benchmark": cfg.benchmark,
        "price_provider": PRICE_PROVIDER,
        "fundamentals_provider": FUNDAMENTALS_PROVIDER,
        "point_in_time": True,
        "universe_note": universe_note,
        "metrics": metrics,
        "warnings": warnings[:12],
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }


def sweep_signal_unit(payload: dict[str, Any]) -> dict[str, Any]:
    """One strategy + one gate variant: prepare once, then every year/book job."""
    from screener.backtester.rolling_simulation import (
        prepare_rolling_backtest,
        run_prepared_rolling_backtest,
    )
    from screener.backtester.workflow import resolve_backtest_run

    strategy = str(payload["strategy"])
    signal_variant = str(payload["signal_variant"])
    out_dir = Path(payload["out_dir"])
    smoke = bool(payload["smoke"])
    request_overlay = dict(payload["request"])
    years_list = [1] if smoke else list(YEARS)
    jobs = book_jobs(signal_variant)
    if smoke:
        jobs = [jobs[0]]

    done = 0
    skipped = 0
    failed = 0
    t0 = time.time()
    configure_fmp_sources()
    pending = []
    for years in years_list:
        for job in jobs:
            run_id = f"{strategy}__{years}y__{job['variant']}"
            if not run_path(out_dir, run_id).exists():
                pending.append((years, job, run_id))
    if not pending:
        return {
            "strategy": strategy,
            "signal_variant": signal_variant,
            "done": 0,
            "skipped": len(years_list) * len(jobs),
            "failed": 0,
            "elapsed": 0.0,
            "error": None,
        }
    try:
        years_for_prepare = 1 if smoke else 5
        request = make_request(
            strategy_name=strategy,
            years=years_for_prepare,
            start_arg=datetime.combine(
                window_start(years_for_prepare), datetime.min.time()
            ),
            **request_overlay,
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
            for job in jobs:
                run_id = f"{strategy}__{years}y__{job['variant']}"
                path = run_path(out_dir, run_id)
                if path.exists():
                    skipped += 1
                    continue
                cfg = apply_book_update(run.config, job["update"])
                t_run = time.time()
                error = None
                result = None
                try:
                    if not sliced.supports(cfg):
                        raise RuntimeError(
                            f"prepared panel does not support book update {job['variant']}"
                        )
                    result = run_prepared_rolling_backtest(sliced, cfg)
                except Exception as exc:  # noqa: BLE001 - sweep must continue
                    error = f"{type(exc).__name__}: {exc}"
                    failed += 1
                record = result_record(
                    strategy=strategy,
                    years=years,
                    start=start,
                    variant=job["variant"],
                    signal_variant=signal_variant,
                    cfg=cfg,
                    result=result,
                    error=error,
                    elapsed=time.time() - t_run,
                    universe_note=run.universe_note,
                )
                write_run(path, record)
                done += 1
        del prepared
        gc.collect()
        return {
            "strategy": strategy,
            "signal_variant": signal_variant,
            "done": done,
            "skipped": skipped,
            "failed": failed,
            "elapsed": round(time.time() - t0, 1),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "strategy": strategy,
            "signal_variant": signal_variant,
            "done": done,
            "skipped": skipped,
            "failed": failed + 1,
            "elapsed": round(time.time() - t0, 1),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}",
        }


def planned_units(strategies: list[str], smoke: bool) -> list[dict[str, Any]]:
    variants = signal_variants()
    if smoke:
        variants = variants[:1]
        strategies = strategies[:1]
    units = []
    for strategy in strategies:
        for variant in variants:
            units.append(
                {
                    "strategy": strategy,
                    "signal_variant": variant["variant"],
                    "request": variant["request"],
                    "smoke": smoke,
                }
            )
    return units


def warm_price_cache() -> str:
    """Fetch the 5-year Nifty 500 PIT panel once so workers hit parquet."""
    configure_fmp_sources()
    from screener.backtester.rolling_simulation import prepare_rolling_backtest
    from screener.backtester.workflow import resolve_backtest_run

    request = make_request(strategy_name="ema_trend")
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
    return f"{n_tickers} tickers, {n_days} session days. {note}"


def load_runs(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out_dir / "runs").glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def _fmt_pct(value: Any) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _metric(row: dict[str, Any], key: str) -> Any:
    return (row.get("metrics") or {}).get(key)


def _is_default_core(row: dict[str, Any]) -> bool:
    return (
        row.get("signal_variant") == "gates_default"
        and row.get("variant") == f"h{DEFAULT_HOLD}_t{DEFAULT_TOP}"
        and not row.get("error")
    )


def build_markdown(rows: list[dict[str, Any]], universe_note: str) -> str:
    ok_all = [r for r in rows if not r.get("error")]
    ok = list(ok_all)
    failed = [r for r in rows if r.get("error")]
    default_5y = [r for r in ok if _is_default_core(r) and r.get("years") == 5]
    default_5y.sort(
        key=lambda r: (_metric(r, "sharpe") is None, -(_metric(r, "sharpe") or -999))
    )
    lines = [
        "# Nifty 500 PIT strategy sweep",
        "",
        "Research, not financial advice.",
        "",
        f"Generated {datetime.now().isoformat(timespec='minutes')}.",
        "",
        "## Decision",
        "",
        "Prices and dated fundamentals come from Financial Modeling Prep (`SCREENER_PRICE_PROVIDER=fmp`, `--fundamentals-provider fmp`). Membership is still `nifty500_pit`.",
        "",
        "Read the 5-year baseline table for the ranking. Zero-cost Sharpe is not a live number; the `costs_india_5bps` overlay is the costed view.",
        "",
        "## Setup",
        "",
        "- Price source: FMP daily `historical-price-full` (cache `~/.screener/fmp_prices`).",
        "- Fundamentals source: FMP (cache `~/.screener/cache/backtester_fmp_fundamentals`), lag 1 calendar day.",
        "- Universe: `nifty500_pit` (13 membership snapshots, 850 names ever in the index).",
        "- Point-in-time: on. A name is eligible only inside its snapshot window.",
        "- Market: India. Benchmark: `^NSEI`.",
        "- Windows: trailing 5 / 4 / 3 / 2 / 1 calendar years ending 2026-09-01.",
        "- CLI-identical year math: `end - 365 * years`.",
        "- Baseline book: hold 20, top 10, equal_slot, no stops, 0 bps costs.",
        "- Core grid: hold in {10, 20, 63, 126} x top in {5, 10, 20}, every year window.",
        "- Rank-exit: weekly and monthly crossed with every hold x top cell. Extra: `--rank-exit 1` and `63`, `--rank-universe-size` 20 and 100.",
        "- Book overlays (hold 20, top 10): every sizing rule and sizing knob; stops 5/8/12%; take-profit 10/20/30%; trail 8/15/25%; slippage/commission 5 and 10 bps; cost-model india; half-spread 5 and 10; vol-impact k 0.05/0.1/0.25; composite; spread-proxy; no-gap-fills; entry moc/limit 10/20/50 bps; partial-exit 10:50 and 20:25; no-reinvest; reentry 3; reserve 1/5; capital 500k.",
        "- Gate overlays (hold 20, top 10): sector-neutral; regime bull/pullback/bear/nonbear; min_score 30/50/70; min-price 0 and 50; min-ADV 0 and 1e6; adv-window 60; earnings-blackout 5 and 21; max-universe 50/100/200.",
        "- Price-adjustment overlays: splits_only and none. Fundamental lag 60 days.",
        "- Callable-only strategies skipped (no entry/exit expressions): "
        + ", ".join(CALLABLE_ONLY)
        + ".",
        "- Fundamental strategies use FMP, not openscreener: "
        + ", ".join(FUNDAMENTAL_STRATEGIES)
        + ".",
        "",
        f"Universe note: {universe_note or 'see per-run JSON'}",
        "",
        "PIT crawl gap: 2020-07-25 to 2022-05-04 is one frozen membership, not resolved history.",
        "",
        f"Runs written: {len(rows)}. Error-free: {len(ok_all)}. Failed: {len(failed)}.",
        "",
        "## 5-year baseline ranking (hold 20, top 10, equal_slot, no extra flags)",
        "",
        "| Rank | Strategy | Sharpe | CAGR | Total | Excess vs Nifty | Max DD | Hit rate | Trades |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(default_5y, start=1):
        lines.append(
            "| {rank} | `{name}` | {sharpe} | {cagr} | {total} | {excess} | {dd} | {hit} | {trades} |".format(
                rank=i,
                name=row["strategy"],
                sharpe=_fmt_num(_metric(row, "sharpe")),
                cagr=_fmt_pct(_metric(row, "cagr")),
                total=_fmt_pct(_metric(row, "total_return")),
                excess=_fmt_pct(_metric(row, "excess_return")),
                dd=_fmt_pct(_metric(row, "max_drawdown")),
                hit=_fmt_pct(_metric(row, "hit_rate")),
                trades=_metric(row, "trade_count") or 0,
            )
        )
    lines += [
        "",
        "## Beat-Nifty count on the baseline (hold 20, top 10)",
        "",
        "Count of year windows where total return beats `^NSEI` (Nifty 50). The candidate pool is Nifty 500 PIT.",
        "",
        "| Strategy | Windows beaten | 5y | 4y | 3y | 2y | 1y |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_name: dict[str, dict[int, dict[str, Any]]] = {}
    for row in ok:
        if not _is_default_core(row):
            continue
        by_name.setdefault(row["strategy"], {})[int(row["years"])] = row
    beat_rows = []
    for name, by_year in by_name.items():
        marks = []
        beats = 0
        for years in YEARS:
            row = by_year.get(years)
            excess = _metric(row, "excess_return") if row else None
            if isinstance(excess, (int, float)) and excess > 0:
                beats += 1
                marks.append("yes")
            elif row is None:
                marks.append("-")
            else:
                marks.append("no")
        beat_rows.append((beats, name, marks))
    beat_rows.sort(key=lambda item: (-item[0], item[1]))
    for beats, name, marks in beat_rows:
        lines.append(f"| `{name}` | {beats}/5 | " + " | ".join(marks) + " |")

    # Hold lever on 5y, top=10
    lines += [
        "",
        "## Hold-period lever (5y, top 10, gates default)",
        "",
        "Sharpe at each hold. Baseline is hold 20.",
        "",
        "| Strategy | h10 | h20 | h63 | h126 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    hold_map: dict[str, dict[int, float | None]] = {}
    for row in ok:
        if row.get("years") != 5 or row.get("top") != 10:
            continue
        if row.get("signal_variant") != "gates_default":
            continue
        if row.get("variant") not in {f"h{h}_t10" for h in HOLDS}:
            continue
        hold_map.setdefault(row["strategy"], {})[int(row["hold"])] = _metric(
            row, "sharpe"
        )
    for name in sorted(hold_map):
        cells = " | ".join(_fmt_num(hold_map[name].get(h)) for h in HOLDS)
        lines.append(f"| `{name}` | {cells} |")

    lines += [
        "",
        "## Overlay deltas vs baseline (5y, hold 20, top 10)",
        "",
        "Median Sharpe change across strategies for each overlay. Positive means the overlay helped.",
        "",
        "| Overlay | Median Sharpe delta | Strategies improved | n |",
        "| --- | ---: | ---: | ---: |",
    ]
    baseline = {
        r["strategy"]: _metric(r, "sharpe")
        for r in default_5y
        if _metric(r, "sharpe") is not None
    }
    overlay_groups: dict[str, list[float]] = {}
    for row in ok:
        if row.get("years") != 5:
            continue
        if int(row.get("hold") or 0) != DEFAULT_HOLD:
            continue
        if int(row.get("top") or 0) != DEFAULT_TOP:
            continue
        variant = str(row.get("variant") or "")
        if "__" not in variant:
            continue
        overlay = variant.split("__", 1)[1]
        base = baseline.get(row["strategy"])
        sharpe = _metric(row, "sharpe")
        if base is None or sharpe is None:
            continue
        overlay_groups.setdefault(overlay, []).append(float(sharpe) - float(base))
    for overlay, deltas in sorted(overlay_groups.items()):
        improved = sum(1 for d in deltas if d > 0)
        deltas_sorted = sorted(deltas)
        mid = deltas_sorted[len(deltas_sorted) // 2]
        lines.append(
            f"| `{overlay}` | {mid:+.2f} | {improved}/{len(deltas)} | {len(deltas)} |"
        )

    lines += [
        "",
        "## Rank-exit rebalance (5y, top 10)",
        "",
        "Sharpe with `--rank-exit monthly` (every 21 bars) vs the same hold without rebalance.",
        "",
        "| Strategy | h10 | h10 monthly | h20 | h20 monthly | h63 | h63 monthly | h126 | h126 monthly |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rebal: dict[str, dict[tuple[int, str], float | None]] = {}
    for row in ok:
        if row.get("years") != 5 or int(row.get("top") or 0) != 10:
            continue
        if row.get("signal_variant") != "gates_default":
            continue
        variant = str(row.get("variant") or "")
        hold = int(row.get("hold") or 0)
        if variant == f"h{hold}_t10":
            kind = "plain"
        elif variant == f"h{hold}_t10__rank_exit_monthly":
            kind = "monthly"
        else:
            continue
        rebal.setdefault(row["strategy"], {})[(hold, kind)] = _metric(row, "sharpe")
    for name in sorted(rebal):
        cells = []
        for hold in HOLDS:
            cells.append(_fmt_num(rebal[name].get((hold, "plain"))))
            cells.append(_fmt_num(rebal[name].get((hold, "monthly"))))
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    if failed:
        lines += [
            "",
            "## Failed runs",
            "",
        ]
        for row in failed[:40]:
            lines.append(
                f"- `{row.get('strategy')}` {row.get('variant')} {row.get('years')}y: {row.get('error')}"
            )
        if len(failed) > 40:
            lines.append(f"- ... {len(failed) - 40} more in `runs/`")

    lines += [
        "",
        "## How to read this",
        "",
        "Default CLI flags are the baseline. Core grid answers hold and concentration. Overlays answer one flag at a time. Gate overlays rebuild the candidate panel, so they are not comparable as a free lunch without looking at trade count.",
        "",
        "FMP may lack prices or fundamentals for removed or delisted names. Missing history is reported, never filled. Names without `revenue_up_3q` / `pe_ttm` fail the fundamental entry and are skipped, not ranked last.",
        "",
        "Full per-run JSON is under `reports/nifty500_pit_sweep_fmp/runs/`. The compact CSV is `results.csv`.",
        "",
    ]
    return "\n".join(lines)


def build_csv(rows: list[dict[str, Any]]) -> str:
    fields = [
        "strategy",
        "years",
        "variant",
        "signal_variant",
        "hold",
        "top",
        "sizing_rule",
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "sector_neutral",
        "regime_filter",
        "rank_exit_every",
        "cost_model",
        "slippage_bps",
        "commission_bps",
        "min_score",
        "earnings_blackout_days",
        "sharpe",
        "sortino",
        "calmar",
        "cagr",
        "total_return",
        "excess_return",
        "max_drawdown",
        "hit_rate",
        "alpha_annual",
        "beta",
        "exposure",
        "benchmark_return",
        "trade_count",
        "unique_tickers",
        "profit_factor",
        "error",
        "elapsed_seconds",
    ]
    lines = [",".join(fields)]
    for row in rows:
        metrics = row.get("metrics") or {}
        values = []
        for field in fields:
            if field in metrics or field in {
                "sharpe",
                "sortino",
                "calmar",
                "cagr",
                "total_return",
                "excess_return",
                "max_drawdown",
                "hit_rate",
                "alpha_annual",
                "beta",
                "exposure",
                "benchmark_return",
                "trade_count",
                "unique_tickers",
                "profit_factor",
            }:
                raw = metrics.get(field, row.get(field))
            else:
                raw = row.get(field)
            if isinstance(raw, (list, tuple)):
                raw = "|".join(str(x) for x in raw)
            if raw is None:
                values.append("")
            else:
                text = str(raw).replace('"', "'")
                if "," in text:
                    values.append(f'"{text}"')
                else:
                    values.append(text)
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def build_html(rows: list[dict[str, Any]], universe_note: str) -> str:
    """Copy the static FMP sweep viewer. Rows live in results.csv next to it."""
    _ = (rows, universe_note)
    viewer = Path(__file__).resolve().parent / "nifty500_pit_sweep_viewer.html"
    return viewer.read_text()


def write_report(out_dir: Path, universe_note: str) -> None:
    rows = load_runs(out_dir)
    (out_dir / "report.md").write_text(build_markdown(rows, universe_note))
    (out_dir / "results.csv").write_text(build_csv(rows))
    (out_dir / "index.html").write_text(build_html(rows, universe_note))
    (out_dir / "configs.json").write_text(
        json.dumps(build_config_catalog(), indent=2) + "\n"
    )
    (out_dir / "skipped.json").write_text(
        json.dumps(
            {
                "callable_only": list(CALLABLE_ONLY),
                "callable_reason": (
                    "These plugins register @strategy callables for the pine "
                    "runner and have no entry/exit expressions, so "
                    "backtest-rolling cannot run them."
                ),
                "fundamental_strategies": list(FUNDAMENTAL_STRATEGIES),
                "fundamental_reason": (
                    "These expressions read FMP dated fields "
                    "(revenue_up_3q, pe_ttm, eps_growth_yoy, revenue_growth_yoy)."
                ),
                "earnings_blackout": "Ran as earnings_blackout_5 and earnings_blackout_21.",
            },
            indent=2,
        )
    )
    print(f"Wrote report for {len(rows)} runs -> {out_dir / 'report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-warm", action="store_true")
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma-separated subset of strategy names.",
    )
    return parser.parse_args()


def main() -> int:
    configure_fmp_sources()
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    names = expression_strategy_names()
    if args.strategies:
        wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]
        missing = [s for s in wanted if s not in names]
        if missing:
            print(f"unknown strategies: {missing}", file=sys.stderr)
            print(f"known: {names}", file=sys.stderr)
            return 2
        names = wanted

    universe_note = ""
    note_path = out_dir / "universe_note.txt"
    if note_path.exists():
        universe_note = note_path.read_text().strip()
    if args.report_only:
        write_report(out_dir, universe_note)
        return 0

    if not args.skip_warm:
        print("warming Nifty 500 PIT price cache (5y window)...", flush=True)
        universe_note = warm_price_cache()
        print(universe_note, flush=True)
        (out_dir / "universe_note.txt").write_text(universe_note + "\n")
    elif (out_dir / "universe_note.txt").exists():
        universe_note = (out_dir / "universe_note.txt").read_text().strip()

    units = planned_units(names, args.smoke)
    for unit in units:
        unit["out_dir"] = str(out_dir)
    print(
        f"work units: {len(units)} "
        f"({len(names)} strategies x {len(signal_variants() if not args.smoke else signal_variants()[:1])} gate variants)",
        flush=True,
    )

    workers = max(1, int(args.workers))
    if args.smoke:
        workers = 1
    results = []
    if workers == 1:
        for i, unit in enumerate(units, start=1):
            print(
                f"[{i}/{len(units)}] {unit['strategy']} {unit['signal_variant']}",
                flush=True,
            )
            result = sweep_signal_unit(unit)
            results.append(result)
            print(result, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(sweep_signal_unit, unit): unit for unit in units}
            done_n = 0
            for fut in as_completed(futs):
                done_n += 1
                unit = futs[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "strategy": unit["strategy"],
                        "signal_variant": unit["signal_variant"],
                        "error": str(exc),
                    }
                results.append(result)
                print(f"[{done_n}/{len(units)}] {result}", flush=True)

    (out_dir / "sweep_status.json").write_text(json.dumps(jsonable(results), indent=2))
    write_report(out_dir, universe_note)
    failed_units = [r for r in results if r.get("error")]
    print(
        f"units with prepare/run errors: {len(failed_units)} / {len(results)}",
        flush=True,
    )
    return 1 if failed_units and args.smoke else 0


if __name__ == "__main__":
    raise SystemExit(main())
