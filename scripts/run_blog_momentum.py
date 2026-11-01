#!/usr/bin/env python
"""Run fixed daily blog momentum adaptations through the rolling backtester.

Usage: uv run python -m scripts.run_blog_momentum
Each cell writes its config, metrics, warnings, trade ledger, and equity curve.
Missing data is an error or a recorded limitation, never a synthetic return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.blog_momentum_rules import register_blog_momentum_rules
from screener.backtester import BacktestConfig, run_rolling_backtest
from screener.backtester.data import PriceFetcher, build_price_fetcher
from screener.backtester.models import EquityLedgerTrade
from screener.strategies.spec import ExpressionStrategySpec, resolve_strategy_spec
from screener.universes import UniverseSelection, load_universe_selection

ROOT = Path(__file__).resolve().parent.parent
ETF_SYMBOLS = {
    "us": ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "USO"),
    "india": (
        "NSE:NIFTYBEES",
        "NSE:JUNIORBEES",
        "NSE:BANKBEES",
        "NSE:GOLDBEES",
    ),
}
CHAN_STRATEGIES = (
    "chan_return_12m",
    "chan_weighted_13612",
    "chan_log_ma_7_10",
    "chan_log_ma_50_200",
    "momentum_12_1",
)


def write_study_json(path: Path, payload: object) -> None:
    """Save research metadata with dates converted to ISO strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


class FrozenStudyFetcher:
    """Save the exact fetched bars and reuse them without another provider call."""

    def __init__(self, provider: PriceFetcher, directory: Path) -> None:
        self.provider = provider
        self.directory = directory

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        symbols = tuple(tickers)
        self.directory.mkdir(parents=True, exist_ok=True)
        frames = {}
        missing = []
        for symbol in symbols:
            path = self.directory / f"{symbol.replace('/', '_')}.parquet"
            if path.exists():
                frames[symbol] = pd.read_parquet(path)
            else:
                missing.append(symbol)
        if missing:
            acquired = self.provider.fetch(missing, start, end)
            for symbol in missing:
                frame = acquired.get(symbol, pd.DataFrame())
                frame.to_parquet(self.directory / f"{symbol.replace('/', '_')}.parquet")
                frames[symbol] = frame
        result = {}
        for symbol, frame in frames.items():
            if frame.empty:
                result[symbol] = frame
                continue
            valid = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
            valid &= (
                frame[["open", "high", "low", "close"]]
                .apply(
                    lambda column: column.map(
                        lambda value: pd.notna(value) and float(value) < float("inf")
                    )
                )
                .all(axis=1)
            )
            if not symbol.startswith("^"):
                valid &= frame["volume"].gt(0)
            result[symbol] = frame.loc[valid].loc[str(start) : str(end)]
        return result


def run_blog_study(args: argparse.Namespace) -> None:
    """Run each requested cell independently and record acquisition failures."""
    register_blog_momentum_rules()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text())
        if previous["start"] != str(args.start) or previous["end"] != str(args.end):
            raise ValueError(
                "Blog momentum frozen window mismatch: use a new output directory"
            )
    write_study_json(
        out / "manifest.json",
        {
            "start": args.start,
            "end": args.end,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rules_sha256": hashlib.sha256(
                (ROOT / "scripts/blog_momentum_rules.py").read_bytes()
            ).hexdigest(),
            "monthly_rules_sha256": hashlib.sha256(
                (ROOT / "scripts/blog_monthly_rules.py").read_bytes()
            ).hexdigest(),
            "scope": "daily bars, long-only adaptations; not original portfolio replicas",
            "parameters_selected_on_results": False,
            "slippage_bps_per_fill": 5,
            "us_commission_bps_per_fill": 1,
            "india_cost_model": "india statutory delivery fees",
            "india_etf_cost_limitation": "Equity delivery schedule is an approximation, not ETF-specific taxation",
            "us_expression_exit_timing": "Previous completed signal, next close",
            "india_expression_exit_timing": "Previous completed signal, next open",
        },
    )
    rows = []
    fetcher = build_price_fetcher()
    for market in args.markets:
        for asset in args.assets:
            try:
                universe = None
                if asset == "stocks":
                    universe_path = out / f"{market}_stocks_universe.json"
                    if universe_path.exists():
                        saved = json.loads(universe_path.read_text())
                        universe = UniverseSelection(
                            name=saved["name"],
                            market=market,
                            benchmark=saved["benchmark"],
                            symbols=tuple(saved["symbols"]),
                            source=saved["source"],
                            warnings=tuple(saved["warnings"]),
                            membership_windows=tuple(
                                (
                                    symbol,
                                    date.fromisoformat(first),
                                    date.fromisoformat(last) if last else None,
                                )
                                for symbol, first, last in saved["membership_windows"]
                            ),
                        )
                    else:
                        universe = load_universe_selection(
                            "nifty500_pit" if market == "india" else "sp500",
                            market=market,
                            as_of=args.end,
                            start=args.start,
                            config_path=ROOT / "universes.yaml",
                            point_in_time=True,
                        )
                        if not universe.membership_windows:
                            raise ValueError(
                                "Blog momentum stock universe lacks historical membership"
                            )
                        write_study_json(
                            universe_path,
                            {
                                "name": universe.name,
                                "benchmark": universe.benchmark,
                                "symbols": universe.symbols,
                                "source": universe.source,
                                "warnings": universe.warnings,
                                "membership_windows": universe.membership_windows,
                            },
                        )
                    symbols = universe.symbols
                else:
                    symbols = ETF_SYMBOLS[market]
            except Exception as exc:
                rows.append(
                    {
                        "market": market,
                        "asset": asset,
                        "status": "blocked",
                        "error": repr(exc),
                    }
                )
                write_study_json(out / "summary.json", rows)
                continue
            for strategy in args.strategies:
                cell = f"{market}_{asset}_{strategy}"
                cell_dir = out / cell
                print(f"START {cell} symbols={len(symbols)}", flush=True)
                try:
                    spec = resolve_strategy_spec(strategy)
                    if not isinstance(spec, ExpressionStrategySpec):
                        raise ValueError(
                            f"Blog momentum strategy not registered: {strategy}"
                        )
                    cfg = BacktestConfig(
                        market=market,
                        as_of=args.start,
                        benchmark="SPY" if market == "us" else "^NSEI",
                        tickers=tuple(symbols),
                        max_universe=0,
                        membership_windows=universe.membership_windows
                        if universe
                        else (),
                        strategy_name=strategy,
                        entry_expr=spec.entry,
                        exit_expr=spec.exit,
                        hold=(
                            62
                            if strategy == "momentum_12_1"
                            else 20
                            if strategy in ("chan_return_12m", "chan_weighted_13612")
                            else 10000
                        ),
                        top=10 if asset == "stocks" else 1,
                        stop_loss=None,
                        take_profit=None,
                        trailing_stop=None,
                        slippage_bps=5,
                        commission_bps=1 if market == "us" else 0,
                        cost_model="flat" if market == "us" else "india",
                        initial_capital=1000000 if market == "india" else 100000,
                        sizing_rule="reinvested_equal_slot",
                    )
                    for stale in (
                        "metrics.json",
                        "warnings.json",
                        "trades.csv",
                        "equity.csv",
                        "error.json",
                        "early_history_exits.json",
                    ):
                        (cell_dir / stale).unlink(missing_ok=True)
                    write_study_json(
                        cell_dir / "config.json", cfg.model_dump(mode="json")
                    )
                    write_study_json(
                        cell_dir / "universe.json",
                        {
                            "symbols": symbols,
                            "source": universe.source
                            if universe
                            else "fixed research ETF basket",
                            "warnings": universe.warnings
                            if universe
                            else [
                                "Static surviving ETF basket, not point-in-time market coverage",
                                "India basket is not an equivalent of the US multi-asset basket",
                            ],
                        },
                    )
                    result = run_rolling_backtest(
                        cfg,
                        FrozenStudyFetcher(fetcher, out / f"{market}_{asset}_bars"),
                        start_date=args.start,
                        end_date=args.end,
                    )
                    early_eod = [
                        trade
                        for trade in result.trades
                        if trade.exit_reason == "eod" and trade.exit_date < args.end
                    ]
                    if early_eod:
                        result.warnings.append(
                            f"PROVISIONAL: {len(early_eod)} early history-end exits use last available quotes, not verified sales"
                        )
                        write_study_json(
                            cell_dir / "early_history_exits.json",
                            [trade.model_dump(mode="json") for trade in early_eod],
                        )
                    row = {
                        "cell": cell,
                        "market": market,
                        "asset": asset,
                        "strategy": strategy,
                        "status": "provisional" if early_eod else "completed",
                        "early_history_exits": len(early_eod),
                        "start": str(args.start),
                        "end": str(args.end),
                        **result.metrics,
                    }
                    write_study_json(cell_dir / "metrics.json", result.metrics)
                    write_study_json(cell_dir / "warnings.json", result.warnings)
                    pd.DataFrame(
                        [trade.model_dump() for trade in result.trades],
                        columns=list(EquityLedgerTrade.model_fields),
                    ).to_csv(cell_dir / "trades.csv", index=False)
                    if (cell_dir / "error.json").exists():
                        (cell_dir / "error.json").unlink()
                    result.equity_curve.to_csv(cell_dir / "equity.csv")
                    rows.append(row)
                    print(
                        f"DONE {cell} {json.dumps(result.metrics, default=str)}",
                        flush=True,
                    )
                except Exception as exc:
                    rows.append({"cell": cell, "status": "failed", "error": repr(exc)})
                    write_study_json(cell_dir / "error.json", {"error": repr(exc)})
                    print(f"FAILED {cell}: {exc!r}", flush=True)
                write_study_json(out / "summary.json", rows)
                pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

    failed = [row for row in rows if row["status"] in ("blocked", "failed")]
    if failed:
        raise SystemExit(
            f"Blog momentum study incomplete: {len(failed)} blocked or failed cells"
        )


def main() -> None:
    """Read a fixed research window and explicit market and asset lists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--out-dir", default="reports/blog_momentum/chan")
    parser.add_argument(
        "--markets", nargs="+", choices=("us", "india"), default=["us", "india"]
    )
    parser.add_argument(
        "--assets", nargs="+", choices=("etfs", "stocks"), default=["etfs", "stocks"]
    )
    parser.add_argument("--strategies", nargs="+", default=list(CHAN_STRATEGIES))
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error("start must precede end")
    run_blog_study(args)


if __name__ == "__main__":
    main()
