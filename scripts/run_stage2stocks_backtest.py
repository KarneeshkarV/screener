#!/usr/bin/env python
"""Run multi-horizon PIT backtests of the Stage2Stocks strategies.

Uses the repo's rolling backtester engine (``screener.backtester.workflow`` and
``screener.backtester.rolling_simulation``) across point-in-time India 500 and
750 universes over trailing 10Y, 5Y, 3Y, and 1Y windows::

    uv run python -m scripts.run_stage2stocks_backtest
    uv run python -m scripts.run_stage2stocks_backtest \
        --universes india_turnover500_pit,india_turnover750_pit \
        --horizons 10,5,3,1 --bhavcopy-prices

Writes ``summary.csv``, ``summary.json``, and ``equity_10y.json`` under
``--out-dir`` (default ``reports/stage2stocks``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screener.strategies.plugins.stage2stocks import VARIANTS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UNIVERSES = (
    "india_turnover500_pit",
    "india_turnover750_pit",
    "nifty500_pit_entity",
)
DEFAULT_HORIZONS = (10, 5, 3, 1)
#: Official Wayback Nifty 500 snapshot history starts in Oct 2018.
NIFTY500_PIT_FIRST_SNAPSHOT = date(2018, 10, 4)


def _subtract_years(end: date, years: int) -> date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:
        return end.replace(year=end.year - years, day=28)


class InMemoryPriceFetcher:
    """In-memory slice wrapper so multi-horizon runs fetch each ticker once."""

    def __init__(
        self,
        base_fetcher: Any,
        cache: dict[str, pd.DataFrame],
        *,
        prefetch_start: date,
        prefetch_end: date,
    ) -> None:
        self.base_fetcher = base_fetcher
        self.cache = cache
        self.prefetch_start = prefetch_start
        self.prefetch_end = prefetch_end

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        tickers_list = list(tickers)
        missing = [ticker for ticker in tickers_list if ticker not in self.cache]
        if missing:
            fetched = self.base_fetcher.fetch(
                missing, self.prefetch_start, self.prefetch_end
            )
            for ticker in missing:
                self.cache[ticker] = fetched.get(ticker, pd.DataFrame())
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers_list:
            frame = self.cache.get(ticker, pd.DataFrame())
            out[ticker] = (
                pd.DataFrame()
                if frame is None or frame.empty
                else frame.loc[start_ts:end_ts]
            )
        return out


def make_request(
    *,
    strategy_name: str,
    universe: str,
    universe_config: Path,
    start: date,
    end: date,
    top: int = 20,
    initial_capital: float = 1_000_000.0,
    slippage_bps: float = 10.0,
    risk_free_rate: float = 0.06,
):
    """Build a ``BacktestRequest`` wired to the codebase's rolling engine."""
    from screener.backtester.workflow import BacktestRequest

    spec = VARIANTS[strategy_name]
    return BacktestRequest(
        mode="rolling",
        context_obj=None,
        market="india",
        hold=5000,
        top=top,
        entry_expr=None,
        exit_expr=None,
        strategy_name=spec.name,
        stop_loss=None,
        take_profit=None,
        trailing_stop=spec.trailing_stop,
        slippage_bps=slippage_bps,
        commission_bps=0.0,
        cost_model="india",
        initial_capital=initial_capital,
        benchmark="^NSEI",
        tickers=None,
        universe_file=None,
        max_universe=0,
        min_price=None,
        min_avg_dollar_volume=None,
        adv_window=20,
        slippage_model="fixed",
        half_spread_bps=0.0,
        vol_impact_k=0.1,
        no_gap_fills=False,
        entry_order="moo",
        entry_limit_bps=None,
        partial_exit_args=(),
        price_adjustment="full",
        interval="1d",
        output_csv=False,
        report_path=None,
        open_report=False,
        sizing_rule="reinvested_equal_slot",
        compounding=True,
        sizing_risk_pct=0.01,
        sizing_position_pct=0.10,
        sizing_atr_window=14,
        sizing_atr_multiple=2.0,
        sizing_vol_window=20,
        intraday_only=False,
        risk_free_rate=risk_free_rate,
        start_arg=datetime.combine(start, datetime.min.time()),
        end_arg=datetime.combine(end, datetime.min.time()),
        universe=universe,
        universe_config=universe_config,
        point_in_time=True,
        point_in_time_was_explicit=True,
        rank_exit=spec.rank_exit,
        rank_universe_size=spec.rank_universe_size,
    )


def benchmark_stats(bench: pd.Series, *, risk_free_rate: float) -> dict[str, float]:
    """Compute CAGR, max drawdown, and Sharpe on a benchmark close series."""
    series = bench.dropna()
    if len(series) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe_rf": 0.0,
            "sharpe_0": 0.0,
        }
    years = max((series.index[-1] - series.index[0]).days / 365.25, 1e-6)
    total_return = float(series.iloc[-1] / series.iloc[0] - 1.0)
    cagr = float((series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1.0)
    max_drawdown = float((series / series.cummax() - 1.0).min())
    rets = series.pct_change().dropna()
    std = float(rets.std())
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    sharpe_0 = float(rets.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0
    sharpe_rf = (
        float((rets.mean() - rf_daily) / std * np.sqrt(252.0)) if std > 0 else 0.0
    )
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe_rf": sharpe_rf,
        "sharpe_0": sharpe_0,
    }


def _to_yf_symbol(symbol: str) -> str:
    bare = symbol.split(":", 1)[-1]
    return bare if bare.endswith(".NS") else f"{bare}.NS"


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    from screener.backtester.data import build_price_fetcher
    from screener.backtester.rolling_simulation import (
        prepare_rolling_backtest,
        run_prepared_rolling_backtest,
    )
    from screener.backtester.workflow import resolve_backtest_run
    from screener.strategies.spec import discover_plugins
    from screener.universes import load_universe_selection

    discover_plugins()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    universes = [u.strip() for u in args.universes.split(",") if u.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = sorted(set(strategies) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")

    longest_start = _subtract_years(args.end, max(horizons))
    prefetch_start = longest_start - timedelta(days=500)

    all_yf: set[str] = {"^NSEI"}
    selections = {}
    for uname in universes:
        sel_start = (
            max(longest_start, NIFTY500_PIT_FIRST_SNAPSHOT)
            if uname == "nifty500_pit_entity"
            else longest_start
        )
        sel = load_universe_selection(
            uname,
            market="india",
            as_of=args.end,
            config_path=args.universe_config,
            point_in_time=True,
            start=sel_start,
        )
        selections[uname] = sel
        all_yf.update(_to_yf_symbol(sym) for sym in sel.symbols)

    base_fetcher: Any = build_price_fetcher()
    if args.bhavcopy_prices:
        from scripts.india_pit_bhavcopy import BhavcopyPriceFetcher

        base_fetcher = BhavcopyPriceFetcher(base_fetcher)

    print(
        f"Pre-fetching {len(all_yf)} tickers ({prefetch_start} -> {args.end})...",
        flush=True,
    )
    mem_cache: dict[str, pd.DataFrame] = dict(
        base_fetcher.fetch(sorted(all_yf), prefetch_start, args.end)
    )
    mem_fetcher = InMemoryPriceFetcher(
        base_fetcher,
        mem_cache,
        prefetch_start=prefetch_start,
        prefetch_end=args.end,
    )

    coverage: dict[str, dict[str, Any]] = {}
    for uname, sel in selections.items():
        yfs = [_to_yf_symbol(sym) for sym in sel.symbols]
        have = sum(1 for yf in yfs if yf in mem_cache and not mem_cache[yf].empty)
        coverage[uname] = {
            "total_pit_symbols": len(yfs),
            "symbols_with_bars": have,
            "coverage_pct": round(100.0 * have / max(len(yfs), 1), 1),
        }

    benchmarks: dict[str, dict[str, float]] = {}
    equity_10y: dict[str, list[list[Any]]] = {}
    bench_full = mem_cache.get("^NSEI", pd.DataFrame())
    for years in horizons:
        start_d = _subtract_years(args.end, years)
        hname = f"{years}Y"
        if not bench_full.empty:
            sliced = bench_full["close"].loc[
                pd.Timestamp(start_d) : pd.Timestamp(args.end)
            ]
            benchmarks[hname] = benchmark_stats(
                sliced, risk_free_rate=args.risk_free_rate
            )
            if years == 10 and not sliced.empty:
                weekly = sliced.resample("W-FRI").last().dropna()
                equity_10y["BENCH_^NSEI_10Y"] = [
                    [str(day.date()), round(float(val / weekly.iloc[0] * 1_000_000), 2)]
                    for day, val in weekly.items()
                ]
        else:
            benchmarks[hname] = benchmark_stats(
                pd.Series(dtype=float), risk_free_rate=args.risk_free_rate
            )

    rows: list[dict[str, Any]] = []
    for uname in universes:
        for years in horizons:
            start_d = _subtract_years(args.end, years)
            if uname == "nifty500_pit_entity" and start_d < NIFTY500_PIT_FIRST_SNAPSHOT:
                continue
            hname = f"{years}Y"
            for strat in strategies:
                t_run = time.time()
                req = make_request(
                    strategy_name=strat,
                    universe=uname,
                    universe_config=args.universe_config,
                    start=start_d,
                    end=args.end,
                    top=args.top,
                    slippage_bps=args.slippage_bps,
                    risk_free_rate=args.risk_free_rate,
                )
                run = resolve_backtest_run(req)
                run = dataclasses.replace(run, price_fetcher=mem_fetcher)
                prepared = prepare_rolling_backtest(
                    run.config,
                    run.price_fetcher,
                    start_date=run.start_date,
                    end_date=run.end_date,
                    fundamental_fetcher=run.fundamental_fetcher,
                )
                result = run_prepared_rolling_backtest(prepared, run.config)
                m = dict(result.metrics)
                eq = result.equity_curve.dropna()
                rets = eq.pct_change().dropna()
                sharpe_0 = (
                    float(rets.mean() / rets.std() * np.sqrt(252.0))
                    if len(rets) > 1 and float(rets.std()) > 0
                    else 0.0
                )
                cagr = float(m.get("calendar_cagr", m.get("cagr", 0.0)))
                row = {
                    "universe": uname,
                    "horizon": hname,
                    "start": start_d.isoformat(),
                    "end": args.end.isoformat(),
                    "strategy": strat,
                    "strategy_label": VARIANTS[strat].label,
                    "total_return": float(m.get("total_return", 0.0)),
                    "cagr": cagr,
                    "bench_cagr": benchmarks[hname]["cagr"],
                    "excess_cagr": cagr - benchmarks[hname]["cagr"],
                    "sharpe_rf": float(m.get("sharpe", 0.0)),
                    "sharpe_0": sharpe_0,
                    "sortino_rf": float(m.get("sortino", 0.0)),
                    "max_drawdown": float(m.get("max_drawdown", 0.0)),
                    "calmar": float(m.get("calmar", 0.0)),
                    "hit_rate": float(m.get("hit_rate", 0.0)),
                    "profit_factor": float(m.get("profit_factor", 0.0)),
                    "trade_count": int(m.get("trade_count", 0)),
                    "avg_capital_exposure": float(m.get("avg_capital_exposure", 0.0)),
                    "warnings": list(result.warnings),
                    "elapsed_s": round(time.time() - t_run, 1),
                }
                rows.append(row)
                if years == 10 and not eq.empty:
                    weekly = eq.resample("W-FRI").last().dropna()
                    equity_10y[f"{uname}__{strat}__10Y"] = [
                        [str(day.date()), round(float(val), 2)]
                        for day, val in weekly.items()
                    ]
                print(
                    f"[{uname:22s} | {hname:3s} | {strat:26s}] "
                    f"CAGR {row['cagr']:6.1%} (vs Nifty {row['bench_cagr']:5.1%}) | "
                    f"Sh {row['sharpe_rf']:5.2f} / {row['sharpe_0']:5.2f} | "
                    f"MaxDD {row['max_drawdown']:6.1%} | Hit {row['hit_rate']:4.0%} | "
                    f"Trades {row['trade_count']:4d} ({row['elapsed_s']}s)",
                    flush=True,
                )

    summary = {
        "end_date": args.end.isoformat(),
        "risk_free_rate": args.risk_free_rate,
        "top": args.top,
        "slippage_bps": args.slippage_bps,
        "bhavcopy_prices": args.bhavcopy_prices,
        "coverage": coverage,
        "benchmarks": benchmarks,
        "results": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "equity_10y.json").write_text(json.dumps(equity_10y))
    pd.DataFrame(rows).drop(columns=["warnings"], errors="ignore").to_csv(
        args.out_dir / "summary.csv", index=False
    )
    print(f"Done in {time.time() - t0:.1f}s -> {args.out_dir}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--universes",
        default=",".join(DEFAULT_UNIVERSES),
        help="Comma-separated PIT universe names from --universe-config.",
    )
    parser.add_argument(
        "--universe-config",
        type=Path,
        default=ROOT / "universes.yaml",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated trailing year windows, e.g. 10,5,3,1.",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(VARIANTS),
        help=f"Comma-separated Stage2Stocks variants (default: {', '.join(VARIANTS)}).",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date.today(),
        help="End date YYYY-MM-DD (default: today).",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.06)
    parser.add_argument(
        "--bhavcopy-prices",
        action="store_true",
        help="Use scripts.india_pit_bhavcopy adjusted price frames for .NS tickers.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "stage2stocks",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_grid(args)


if __name__ == "__main__":
    main()
