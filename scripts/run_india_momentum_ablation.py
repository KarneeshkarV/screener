#!/usr/bin/env python
"""Filter-by-filter ablation of a 12-month Indian momentum model, plus allocations.

The model (from a retail momentum video): stage-2 trend (close above the 30,
50 and 100-day SMA), within 20% of the 52-week high, rank by 12-month rate of
change divided by 12-month volatility, skip names stuck in price-band locks,
blend in distance to the 52-week high, top 20 equal weight, monthly rank exit
with a top-50 hold buffer. Each stage adds one rule to the one before it.
The stages are strategies in ``screener.strategies.plugins.india_momentum``;
this script runs them with the rank exit, hold buffer and costs they need.

    uv run python -m scripts.india_pit_bhavcopy ...   # build data first
    uv run python -m scripts.run_india_momentum_ablation stages \
        --universe india_turnover750_pit --bhavcopy-prices
    uv run python -m scripts.run_india_momentum_ablation allocate s4_volpen

``stages`` writes ``<out>/<stage>.json`` (metrics and equity curve) and
``<stage>_trades.csv``. ``allocate`` blends one stage's equity curve with gold,
REITs and a flat 6% debt proxy.

``--stop-loss 0.4`` sells a name when its low touches 40% under the entry fill
(the open if it gaps through). ``--lower-circuit-exit`` sells at the next open
after a close locked at the lower band, and blocks a new buy of a name that
had a lower lock in the last month. Each option adds a suffix to the output
name, for example ``s1_roc__sl40_lc``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from screener.strategies.plugins.india_momentum import TOP, STAGES, strategy_name

ROOT = Path(__file__).resolve().parent.parent
DRAWDOWN_TRIGGERS = (-0.10, -0.20, -0.30)
ALLOCATION_ASSETS = ("GOLDBEES.NS", "EMBASSY.NS", "MINDSPACE.NS", "^NSEI")
DEBT_YIELD = 0.06


def output_name(stage: str, args: argparse.Namespace) -> str:
    """Stage name plus a suffix for each exit option, ``s1_roc__sl40_lc``."""
    parts = []
    if args.stop_loss:
        parts.append(f"sl{round(args.stop_loss * 100)}")
    if args.lower_circuit_exit:
        parts.append("lc")
    return f"{stage}__{'_'.join(parts)}" if parts else stage


def make_request(stage: str, args: argparse.Namespace):
    from screener.backtester.workflow import BacktestRequest

    spec = STAGES[stage]
    name = strategy_name(stage, lower_circuit_exit=args.lower_circuit_exit)
    return BacktestRequest(
        mode="rolling",
        context_obj=None,
        market="india",
        hold=5000,
        top=TOP,
        entry_expr=None,
        exit_expr=None,
        strategy_name=name,
        stop_loss=args.stop_loss,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=10.0,
        commission_bps=0.0,
        cost_model="india",
        initial_capital=1_000_000.0,
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
        start_arg=datetime.combine(args.start, datetime.min.time()),
        end_arg=datetime.combine(args.end, datetime.min.time()),
        universe=args.universe,
        universe_config=args.universe_config,
        point_in_time=True,
        point_in_time_was_explicit=True,
        rank_exit="monthly",
        rank_universe_size=spec.hold_buffer,
    )


def run_stage(stage: str, args: argparse.Namespace) -> None:
    from screener.backtester.rolling_simulation import (
        prepare_rolling_backtest,
        run_prepared_rolling_backtest,
    )
    from screener.backtester.workflow import resolve_backtest_run
    from screener.strategies.spec import discover_plugins

    discover_plugins()
    run = resolve_backtest_run(make_request(stage, args))
    if args.bhavcopy_prices:
        from scripts.india_pit_bhavcopy import BhavcopyPriceFetcher

        run = dataclasses.replace(
            run, price_fetcher=BhavcopyPriceFetcher(run.price_fetcher)
        )
    prepared = prepare_rolling_backtest(
        run.config,
        run.price_fetcher,
        start_date=run.start_date,
        end_date=run.end_date,
        fundamental_fetcher=run.fundamental_fetcher,
    )
    result = run_prepared_rolling_backtest(prepared, run.config)
    metrics = {
        key: float(value) if isinstance(value, int | float | np.floating) else value
        for key, value in dict(result.metrics).items()
        if isinstance(value, int | float | np.floating | str)
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = output_name(stage, args)
    pd.DataFrame([vars(trade) for trade in result.trades]).to_csv(
        args.out_dir / f"{out}_trades.csv", index=False
    )
    (args.out_dir / f"{out}.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "description": STAGES[stage].description,
                "stop_loss": args.stop_loss,
                "lower_circuit_exit": args.lower_circuit_exit,
                "universe": args.universe,
                "bhavcopy_prices": args.bhavcopy_prices,
                "start": str(run.start_date),
                "end": str(run.end_date),
                "metrics": metrics,
                "equity": [
                    [str(day.date()), float(value)]
                    for day, value in result.equity_curve.items()
                ],
                "warnings": list(result.warnings),
            }
        )
    )
    print(
        f"{out:15s} CAGR {metrics['cagr']:6.1%}  Sharpe {metrics['sharpe']:5.2f}  "
        f"MaxDD {metrics['max_drawdown']:6.1%}  trades {int(metrics['trade_count'])}",
        flush=True,
    )


# -- allocations ---------------------------------------------------------------


def curve_stats(equity: pd.Series) -> dict[str, float]:
    equity = equity.dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    returns = equity.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = (equity / equity.cummax() - 1).min()
    return {"cagr": cagr, "sharpe": sharpe, "max_drawdown": max_drawdown}


def fixed_mix(returns: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Rebalance to ``weights`` on each month's first session; drift between."""
    target = pd.Series(weights, dtype=float)
    holding = target.copy()
    values = []
    previous_month = None
    for day, row in returns[list(weights)].iterrows():
        if previous_month is not None and day.month != previous_month:
            holding = target * holding.sum()
        holding = holding * (1 + row.fillna(0.0))
        values.append(holding.sum())
        previous_month = day.month
    return pd.Series(values, index=returns.index)


def dry_powder(
    returns: pd.DataFrame,
    momentum_equity: pd.Series,
    sleeve: Mapping[str, float],
    core: float,
    triggers: tuple[float, ...] = DRAWDOWN_TRIGGERS,
) -> pd.Series:
    """Move the sleeve into momentum in steps as momentum's drawdown deepens.

    ``core`` starts in momentum (column ``mom``) and the rest in ``sleeve``.
    Each trigger the drawdown crosses moves one equal step of the sleeve in
    at that close. A new momentum high resets to the start split. Monthly
    rebalance to the current target.
    """
    drawdown = momentum_equity / momentum_equity.cummax() - 1
    columns = ["mom", *sleeve]

    def target(steps: int) -> pd.Series:
        moved = steps / len(triggers)
        weights = {"mom": core + (1 - core) * moved}
        weights.update(
            {asset: (1 - core) * (1 - moved) * share for asset, share in sleeve.items()}
        )
        return pd.Series(weights, dtype=float).reindex(columns)

    deployed = 0
    holding = target(0)
    values = []
    previous_month = None
    for day, row in returns[columns].iterrows():
        holding = holding * (1 + row.fillna(0.0))
        total = holding.sum()
        depth = drawdown.loc[day]
        steps = (
            0 if depth >= -1e-12 else max(deployed, sum(depth <= t for t in triggers))
        )
        if steps != deployed or (
            previous_month is not None and day.month != previous_month
        ):
            deployed = steps
            holding = target(deployed) * total
        values.append(total)
        previous_month = day.month
    return pd.Series(values, index=returns.index)


def allocate(stage: str, args: argparse.Namespace) -> None:
    from screener.backtester.data import build_price_fetcher

    payload = json.loads((args.out_dir / f"{stage}.json").read_text())
    momentum = pd.Series(
        {pd.Timestamp(day): value for day, value in payload["equity"]}
    ).sort_index()
    fetched = build_price_fetcher().fetch(
        list(ALLOCATION_ASSETS), momentum.index[0].date(), momentum.index[-1].date()
    )
    prices = pd.DataFrame(
        {
            ticker: frame["close"].astype(float)
            for ticker, frame in fetched.items()
            if not frame.empty
        }
    )
    prices = prices.reindex(momentum.index).ffill()
    returns = prices.pct_change()
    # GOLDBEES has a one-day bad print (2019-12-19, -99% then back). No real
    # one-day move above 50% exists in these ETFs and REITs.
    returns = returns.mask(returns.abs() > 0.5, 0.0)
    # LIQUIDBEES pays its yield as units, so its yfinance price misses it.
    returns["debt"] = (1 + DEBT_YIELD) ** (1 / 252) - 1
    returns["mom"] = momentum.pct_change()
    returns["reit"] = returns[["EMBASSY.NS", "MINDSPACE.NS"]].mean(axis=1, skipna=True)
    returns = returns.iloc[1:]
    variants = {
        "100% momentum": momentum,
        "70 mom / 30 gold": fixed_mix(returns, {"mom": 0.7, "GOLDBEES.NS": 0.3}),
        "80 mom / 10 gold / 10 REIT": fixed_mix(
            returns, {"mom": 0.8, "GOLDBEES.NS": 0.1, "reit": 0.1}
        ),
        "80 / 20 gold+REIT dry powder": dry_powder(
            returns, momentum.iloc[1:], {"GOLDBEES.NS": 0.5, "reit": 0.5}, 0.8
        ),
        "70 / 30 gold dry powder": dry_powder(
            returns, momentum.iloc[1:], {"GOLDBEES.NS": 1.0}, 0.7
        ),
        f"70 mom / 30 debt ({DEBT_YIELD:.0%})": fixed_mix(
            returns, {"mom": 0.7, "debt": 0.3}
        ),
        "Nifty 50 (price)": prices["^NSEI"],
        "Gold (GOLDBEES)": (1 + returns["GOLDBEES.NS"]).cumprod(),
    }
    print(f"{stage} {payload['universe']} {payload['start']} to {payload['end']}")
    print(f"{'variant':32s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s}")
    for name, equity in variants.items():
        result = curve_stats(equity)
        print(
            f"{name:32s} {result['cagr']:7.1%} {result['sharpe']:7.2f} "
            f"{result['max_drawdown']:7.1%}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "reports" / "india_momentum"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    stages = commands.add_parser("stages", help="run ablation stages")
    stages.add_argument("names", nargs="*", help=f"default: all of {', '.join(STAGES)}")
    stages.add_argument("--universe", default="india_turnover750_pit")
    stages.add_argument("--universe-config", type=Path, default=ROOT / "universes.yaml")
    stages.add_argument("--start", type=date.fromisoformat, default=date(2019, 4, 1))
    stages.add_argument("--end", type=date.fromisoformat, default=date.today())
    stages.add_argument(
        "--bhavcopy-prices",
        action="store_true",
        help="read .NS bars from scripts.india_pit_bhavcopy output (includes delisted names)",
    )
    stages.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="sell when the low is this fraction under the entry fill, e.g. 0.4",
    )
    stages.add_argument(
        "--lower-circuit-exit",
        action="store_true",
        help="sell at the next open after a lower-band lock; block a re-buy for a month",
    )
    blend = commands.add_parser(
        "allocate", help="blend a stage's equity with gold/REIT/debt"
    )
    blend.add_argument("names", nargs="+", choices=list(STAGES))
    args = parser.parse_args(argv)
    args.names = args.names or list(STAGES)
    unknown = sorted(set(args.names) - set(STAGES))
    if unknown:
        parser.error(f"unknown stages: {', '.join(unknown)}")
    for stage in args.names:
        if args.command == "stages":
            run_stage(stage, args)
        else:
            allocate(stage, args)


if __name__ == "__main__":
    main()
