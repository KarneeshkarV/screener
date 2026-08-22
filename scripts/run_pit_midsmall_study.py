#!/usr/bin/env python
"""India mid / small / mid+small point-in-time rolling backtest matrix.

Uses PR 130 strategy names plus the local expression backtest set.
Windows: 5y, 3y, 2y, 1y. Universes: midcap150, smallcap250, midsmall400.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import click
import pandas as pd

from screener.backtester.data import build_price_fetcher
from screener.backtester.fundamentals import DEFAULT_FUNDAMENTAL_FIELDS
from screener.backtester.rolling import backtest_rolling
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run

DEFAULT_OUT_DIR = Path("findings/pit_midsmall")
UNIVERSE_CONFIG = Path("data/universes/india_pit.toml")
PERIODS = (5, 3, 2, 1)
TOP_SLOTS = 10
CAPITAL = 100_000.0

FUND_FIELDS = (
    "pe_ttm",
    "pb_ttm",
    "roe_ttm",
    "debt_to_equity",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "revenue_up_3q",
    "market_cap",
    "total_assets",
    "gross_profit_to_assets",
    "asset_growth",
    "accruals",
    "piotroski_fscore",
    "z_score",
    "current_ratio",
    "operating_cash_flow",
    "free_cash_flow",
    "fcf_yield",
    "roa_ttm",
    "asset_turnover",
    "interest_coverage",
    "dividend_yield_ttm",
    "gross_margin_ttm",
    "net_margin_ttm",
)


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    hold: int
    fund: bool = False


# Holds match the paper-factor / new-strategy matrices when those exist.
# Local expression strategies without a prior hold use 20 (CLI default).
STRATEGIES: tuple[Strategy, ...] = (
    # Paper-factor 20
    Strategy("sloan_low_accruals", "paper_factor", 126, True),
    Strategy("piotroski_value", "paper_factor", 126, True),
    Strategy("gross_profitability", "paper_factor", 126, True),
    Strategy("conservative_investment", "paper_factor", 126, True),
    Strategy("low_idio_vol", "paper_factor", 126),
    Strategy("betting_against_beta", "paper_factor", 126),
    Strategy("downside_risk", "paper_factor", 126),
    Strategy("max_avoidance", "paper_factor", 126, True),
    Strategy("pead_drift", "paper_factor", 63, True),
    Strategy("earnings_momentum", "paper_factor", 126, True),
    Strategy("fcf_yield_value", "paper_factor", 126, True),
    Strategy("qmj_quality", "paper_factor", 126, True),
    Strategy("lt_reversal_path", "paper_factor", 126),
    Strategy("str_reversal_trend", "paper_factor", 21),
    Strategy("gw52_proximity", "paper_factor", 126),
    Strategy("hs_same_month", "paper_factor", 21),
    Strategy("tsmom_12_1", "paper_factor", 126),
    Strategy("kama_trend", "paper_factor", 63),
    Strategy("hurst_trend_quality", "paper_factor", 126),
    Strategy("ma_timing_200", "paper_factor", 250),
    # PR 130 earlier families (skip pe40/pe55/f6)
    Strategy("nifty_momentum", "pr130", 126),
    Strategy("nifty_momentum_trend", "pr130", 126),
    Strategy("momentum_quality", "pr130", 63, True),
    Strategy("momentum_quality_pe", "pr130", 63, True),
    Strategy("momentum_quality_pb", "pr130", 63, True),
    Strategy("momentum_quality_pe60", "pr130", 63, True),
    Strategy("quality_mom_lowvol", "pr130", 63, True),
    Strategy("rs_trend", "pr130", 63),
    Strategy("vwap_trend", "pr130", 63),
    Strategy("vwap_reversion", "pr130", 20),
    Strategy("chandelier_breakout", "pr130", 250),
    Strategy("turtle_breakout", "pr130", 250),
    Strategy("supertrend_expr", "pr130", 250),
    Strategy("value_rank", "pr130", 126, True),
    Strategy("garp", "pr130", 126, True),
    Strategy("deep_value", "pr130", 126, True),
    Strategy("value_momentum_harness", "pr130", 126, True),
    Strategy("quality_lowvol", "pr130", 126, True),
    Strategy("quality_lowbeta", "pr130", 126, True),
    Strategy("quality_stability", "pr130", 126, True),
    Strategy("quality_value", "pr130", 126, True),
    Strategy("volume_surge", "pr130", 21),
    Strategy("obv_flow_trend", "pr130", 63),
    Strategy("cmf_flow_factor", "pr130", 63),
    Strategy("delivery_accumulation", "pr130", 63),
    Strategy("seasonal_strong_trend", "pr130", 63),
    Strategy("tom_window_trend", "pr130", 21),
    Strategy("pre_holiday_trend", "pr130", 10),
    Strategy("nov_apr_trend", "pr130", 63),
    Strategy("vcp_breakout", "pr130", 250),
    Strategy("vol_expansion_breakout", "pr130", 250),
    Strategy("vol_target_lowvol", "pr130", 63),
    Strategy("keltner_squeeze_breakout", "pr130", 250),
    # Local expression set
    Strategy("breakout", "local", 20),
    Strategy("ema150_200_revenue_up_3q", "local", 20, True),
    Strategy("ema_trend", "local", 20),
    Strategy("low_volatility", "local", 63),
    Strategy("mark_minervini", "local", 20),
    Strategy("minervini_growth_in", "local", 20, True),
    Strategy("minervini_pro_in", "local", 20, True),
    Strategy("mom_lowvol_combo", "local", 63),
    Strategy("momentum_12_1", "local", 63),
    Strategy("momentum_12_1_trend", "local", 63),
    Strategy("rs_breakout", "local", 20),
    Strategy("rs_momentum_regime", "local", 20),
    Strategy("vivek_equity_tool", "local", 20),
)

STRATEGY_BY_NAME = {s.name: s for s in STRATEGIES}

# These entry expressions need FMP opt-in columns. Live FMP is 401 here;
# rekey_fmp_cache copies the raw income/balance cache under this field key.
EXTRA_FIELD_STRATEGIES = frozenset(
    {
        "sloan_low_accruals",
        "piotroski_value",
        "gross_profitability",
        "conservative_investment",
        "fcf_yield_value",
        "qmj_quality",
    }
)

UNIVERSES: dict[str, str] = {
    "mid": "nifty_midcap150_pit",
    "small": "nifty_smallcap250_pit",
    "midsmall": "nifty_midsmall400_pit",
    "n50": "nifty50_pit",
    "n500": "nifty500_pit",
}


def _cli_defaults() -> dict[str, Any]:
    context = click.Context(backtest_rolling)
    defaults: dict[str, Any] = {}
    for param in backtest_rolling.params:
        if param.name is None:
            continue
        value = param.get_default(context, call=True)
        if type(value).__name__ == "Sentinel":
            value = () if param.multiple else None
        defaults[param.name] = value
    return defaults


def build_request(
    strategy: Strategy,
    universe_key: str,
    years: int,
    fetcher: Any,
    *,
    hold: int | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
) -> BacktestRequest:
    params = _cli_defaults()
    params.update(
        market="india",
        years=years,
        strategy_name=strategy.name,
        hold=strategy.hold if hold is None else int(hold),
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
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
        # Default-field strategies hit the existing on-disk FMP cache.
        # Extra-field names use FUND_FIELDS; rekey_fmp_cache() copies raw
        # payloads under that key so a dead FMP key still serves stale data.
        params["fundamental_field_args"] = (
            FUND_FIELDS if strategy.name in EXTRA_FIELD_STRATEGIES else DEFAULT_FUNDAMENTAL_FIELDS
        )
    return BacktestRequest(mode="rolling", context_obj=fetcher, **params)


def _scalar_metrics(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(raw).items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(value, float) and (pd.isna(value) or value == float("inf") or value == float("-inf")):
            value = None
        out[key] = None if value is None else value
    return out


def run_key(universe_key: str, strategy: Strategy, years: int) -> str:
    return f"india__{universe_key}__{strategy.name}__{years}y"


def rekey_fmp_cache() -> int:
    """Copy existing FMP payloads under the extra-field cache key.

    Cached files store raw FMP sections, not the computed columns. A copy
    under the FUND_FIELDS key lets the normal fetcher recompute whatever
    those sections support without calling the dead API.
    """
    from screener.cache import CACHE_ROOT, cache_path, stable_key

    src_dir = CACHE_ROOT / "backtester_fmp_fundamentals"
    if not src_dir.exists():
        return 0
    copied = 0
    for path in src_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        income = payload.get("income") or []
        if not income or not isinstance(income[0], dict):
            continue
        symbol = str(income[0].get("symbol") or "").strip().upper()
        if not symbol.endswith(".NS"):
            continue
        dest = cache_path(
            "backtester_fmp_fundamentals",
            stable_key(("india", symbol, 120, FUND_FIELDS)),
            "json",
        )
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        copied += 1
    return copied


def run_one(
    strategy: Strategy,
    universe_key: str,
    years: int,
    fetcher: Any,
    *,
    hold: int | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
) -> dict[str, Any]:
    request = build_request(
        strategy,
        universe_key,
        years,
        fetcher,
        hold=hold,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
    )
    run = resolve_backtest_run(request)
    # Prefer the on-disk FMP cache forever. Live FMP is 401 on this machine.
    fund = getattr(run, "fundamental_fetcher", None)
    if fund is not None and hasattr(fund, "cache_ttl"):
        fund.cache_ttl = -1.0
    assert run.start_date is not None and run.end_date is not None
    started = time.time()
    result = run_rolling_backtest(
        run.config,
        run.price_fetcher,
        start_date=run.start_date,
        end_date=run.end_date,
        fundamental_fetcher=run.fundamental_fetcher,
    )
    metrics = _scalar_metrics(result.metrics)
    return {
        "strategy": strategy.name,
        "family": strategy.family,
        "hold": strategy.hold if hold is None else int(hold),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "trailing_stop": trailing_stop,
        "fund": strategy.fund,
        "market": "india",
        "universe": universe_key,
        "universe_name": UNIVERSES[universe_key],
        "years": years,
        "start": run.start_date.isoformat(),
        "end": run.end_date.isoformat(),
        "top": TOP_SLOTS,
        "cost_model": "india",
        "slippage_bps": 10.0,
        "universe_note": run.universe_note,
        "elapsed_seconds": round(time.time() - started, 1),
        "n_trades": len(result.trades),
        "metrics": metrics,
        "equity_curve": _curve_records(result.equity_curve),
        "benchmark_curve": _curve_records(result.benchmark_curve),
        "trades": _trade_records(result.trades),
        "warnings": list(result.warnings),
        "generated": date.today().isoformat(),
    }


def _curve_records(curve: Any) -> list[dict[str, Any]]:
    if curve is None or len(curve) == 0:
        return []
    series = pd.Series(curve).dropna()
    stamps = pd.DatetimeIndex(series.index)
    return [
        {"d": stamp.date().isoformat(), "v": round(float(value), 4)}
        for stamp, value in zip(stamps, series.to_numpy(), strict=True)
    ]


def _trade_records(trades: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in trades:
        rows.append(
            {
                "ticker": str(getattr(trade, "ticker", "")),
                "entry_date": str(getattr(trade, "entry_date", "")),
                "entry_price": float(trade.entry_price),
                "exit_date": str(getattr(trade, "exit_date", "")),
                "exit_price": float(trade.exit_price),
                "exit_reason": str(getattr(trade, "exit_reason", "")),
                "shares": float(getattr(trade, "shares", 0)),
                "pnl": float(trade.pnl),
                "return_pct": float(trade.return_pct),
            }
        )
    rows.sort(key=lambda row: (row["entry_date"], row["ticker"]))
    return rows


def write_summary(out_dir: Path) -> Path:
    rows = []
    for path in sorted((out_dir / "runs").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("error"):
            rows.append(
                {
                    "key": path.stem,
                    "universe": payload.get("universe"),
                    "strategy": payload.get("strategy"),
                    "years": payload.get("years"),
                    "error": payload["error"],
                }
            )
            continue
        metrics = payload.get("metrics") or {}
        rows.append(
            {
                "key": path.stem,
                "universe": payload.get("universe"),
                "strategy": payload.get("strategy"),
                "family": payload.get("family"),
                "years": payload.get("years"),
                "hold": payload.get("hold"),
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
                "error": "",
            }
        )
    frame = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("-u", "--universe", action="append", choices=sorted(UNIVERSES))
    parser.add_argument("-y", "--years", action="append", type=int)
    parser.add_argument("-s", "--strategy", action="append")
    parser.add_argument("--price-only", action="store_true")
    parser.add_argument("--fund-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="re-run a cell only when the JSON has no trade ledger",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        path = write_summary(args.out_dir)
        print(f"wrote {path}")
        return 0

    universes = args.universe or ["mid", "small", "midsmall"]
    periods = tuple(args.years or PERIODS)
    strategies = [
        s
        for s in STRATEGIES
        if (not args.strategy or s.name in args.strategy)
        and (not args.price_only or not s.fund)
        and (not args.fund_only or s.fund)
    ]
    missing = set(args.strategy or ()) - {s.name for s in strategies}
    if missing:
        print(f"unknown strategy: {sorted(missing)}", file=sys.stderr)
        return 2

    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if any(s.fund for s in strategies):
        copied = rekey_fmp_cache()
        print(f"rekeyed {copied} FMP fundamental cache files")
    # yfinance-only prices. FMP price fallback 401s share the fmp circuit and
    # would block fundamental fetches for the paper-factor strategies.
    fetcher = build_price_fetcher(provider="yfinance")

    cells = [
        (universe_key, years, strategy)
        for universe_key in universes
        for years in sorted(periods, reverse=True)
        for strategy in strategies
    ]
    total = len(cells)
    failures: list[str] = []
    for index, (universe_key, years, strategy) in enumerate(cells, start=1):
        key = run_key(universe_key, strategy, years)
        out_path = runs_dir / f"{key}.json"
        if out_path.exists() and not args.force:
            if args.enrich:
                try:
                    prev = json.loads(out_path.read_text())
                except json.JSONDecodeError:
                    prev = {}
                if prev.get("trades") is not None and prev.get("equity_curve"):
                    print(f"[{index}/{total}] skip {key}")
                    continue
            else:
                print(f"[{index}/{total}] skip {key}")
                continue
        print(f"[{index}/{total}] run {key}", flush=True)
        try:
            payload = run_one(strategy, universe_key, years, fetcher)
            out_path.write_text(json.dumps(payload, default=str, allow_nan=False))
            metrics = payload["metrics"]
            print(
                f"  sharpe={metrics.get('sharpe')} "
                f"cagr={metrics.get('cagr')} "
                f"trades={payload['n_trades']} "
                f"{payload['elapsed_seconds']}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - research runner must continue
            failures.append(key)
            out_path.write_text(
                json.dumps(
                    {
                        "strategy": strategy.name,
                        "universe": universe_key,
                        "years": years,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            )
            print(f"  FAIL {type(exc).__name__}: {exc}", flush=True)

    summary = write_summary(args.out_dir)
    print(f"summary {summary}; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
