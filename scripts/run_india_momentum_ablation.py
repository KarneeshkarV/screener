#!/usr/bin/env python
"""Filter-by-filter ablation of a 12-month Indian momentum model, plus allocations.

The model (from a retail momentum video): stage-2 trend (close above the 30,
50 and 100-day SMA), within 20% of the 52-week high, rank by 12-month rate of
change divided by 12-month volatility, skip names stuck in price-band locks,
blend in distance to the 52-week high, top 20 equal weight, monthly rank exit
with a top-50 hold buffer. Each stage adds one rule to the one before it.

    uv run python -m scripts.india_pit_bhavcopy ...   # build data first
    uv run python -m scripts.run_india_momentum_ablation stages \
        --universe india_turnover750_pit --bhavcopy-prices
    uv run python -m scripts.run_india_momentum_ablation allocate s4_volpen

``stages`` writes ``<out>/<stage>.json`` (metrics and equity curve) and
``<stage>_trades.csv``. ``allocate`` blends one stage's equity curve with gold,
REITs and a flat 6% debt proxy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOP = 20
LOOKBACK = 252
#: Sessions 12-1 momentum skips, as in ``screener.factors.momentum_12_1``.
SKIP = 21
CIRCUIT_WINDOW = 126
CIRCUIT_MAX_DAYS = 3
NEAR_HIGH = 0.80
LOCK_MOVE = 0.049
DRAWDOWN_TRIGGERS = (-0.10, -0.20, -0.30)
ALLOCATION_ASSETS = ("GOLDBEES.NS", "EMBASSY.NS", "MINDSPACE.NS", "^NSEI")
DEBT_YIELD = 0.06


def circuit_hits(bars: pd.DataFrame, window: int = CIRCUIT_WINDOW) -> pd.Series:
    """Trailing count of sessions that look like a price-band lock.

    A lock closes at least 4.9% away from the prior close and on the day's
    high (upper band) or low (lower band). A 2% band lock is not caught, which
    errs toward keeping names.
    """
    close = bars["close"].astype(float)
    move = close.pct_change()
    tolerance = 0.001 * close
    upper = (move >= LOCK_MOVE) & ((bars["high"] - close).abs() <= tolerance)
    lower = (move <= -LOCK_MOVE) & ((close - bars["low"]).abs() <= tolerance)
    return (upper | lower).astype(float).rolling(window, min_periods=window).sum()


def rank_score(
    kind: str, momentum: pd.DataFrame, vol: pd.DataFrame, near_high: pd.DataFrame
) -> pd.DataFrame:
    """Cross-sectional score per date (higher ranks first)."""
    if kind == "near_hi":
        return near_high
    if kind == "momentum":
        return momentum
    risk_adjusted = momentum / vol.replace(0, np.nan)
    if kind == "momentum_vol":
        return risk_adjusted
    if kind == "composite":
        return 0.5 * risk_adjusted.rank(axis=1, pct=True) + 0.5 * near_high.rank(
            axis=1, pct=True
        )
    raise ValueError(f"unknown rank kind: {kind}")


def prepare_bars(kind: str, momentum: str = "roc_12m") -> Callable:
    """``prepare_bars`` hook: add the factor columns and ``rank_score``.

    ``momentum`` is the column the rank reads: ``roc_12m`` (the last 12
    months) or ``mom_12_1`` (12 months back to 1 month back).
    """
    from screener.tradeable import tradeable_span

    def prepare(ctx):
        factors: dict[str, dict[str, pd.Series]] = {
            "momentum": {},
            "vol": {},
            "near_hi": {},
        }
        out: dict[str, pd.DataFrame] = {}
        for ticker, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[ticker] = bars
                continue
            frame = bars.copy()
            close = frame["close"].astype(float)
            live = tradeable_span(close, frame.get("volume"), window=LOOKBACK)
            high = (
                frame["high"]
                .astype(float)
                .rolling(LOOKBACK, min_periods=LOOKBACK)
                .max()
            )
            frame["roc_12m"] = (close / close.shift(LOOKBACK) - 1).where(live)
            frame["mom_12_1"] = (close.shift(SKIP) / close.shift(LOOKBACK) - 1).where(
                live
            )
            frame["vol_12m"] = (
                close.pct_change()
                .rolling(LOOKBACK, min_periods=LOOKBACK)
                .std()
                .where(live)
            )
            frame["near_hi"] = (close / high).where(live)
            frame["circuit_hits"] = circuit_hits(frame)
            for name, column in (
                ("momentum", momentum),
                ("vol", "vol_12m"),
                ("near_hi", "near_hi"),
            ):
                factors[name][ticker] = frame[column]
            out[ticker] = frame
        if not factors["momentum"]:
            return out
        momentum_frame, vol, near_high = (
            ctx.mask_rank_reference(pd.DataFrame(factors[name]))
            for name in ("momentum", "vol", "near_hi")
        )
        score = rank_score(kind, momentum_frame, vol, near_high)
        for ticker, frame in out.items():
            if frame is not None and not frame.empty and ticker in score.columns:
                frame["rank_score"] = score[ticker].reindex(frame.index)
        return out

    return prepare


@dataclass(frozen=True)
class Stage:
    rank: str
    entry: str
    hold_buffer: int
    description: str
    momentum: str = "roc_12m"


BASE = "roc_12m > -10 and vol_12m > 0"
TREND = f"{BASE} and close > sma(close, 30) and close > sma(close, 50) and close > sma(close, 100)"
NEAR = f"{TREND} and near_hi >= {NEAR_HIGH}"
NO_LOCKS = f"{NEAR} and circuit_hits <= {CIRCUIT_MAX_DAYS}"
STAGES: dict[str, Stage] = {
    "s0_52wh": Stage("near_hi", BASE, TOP, "Rank by closeness to the 52-week high"),
    "s1_roc": Stage("momentum", BASE, TOP, "Rank by 12-month rate of change"),
    "s1b_roc_buf50": Stage("momentum", BASE, 50, "s1 + hold while in the top 50"),
    "s2_stage2": Stage("momentum", TREND, TOP, "+ close above the 30/50/100-day SMA"),
    "s3_near52": Stage("momentum", NEAR, TOP, "+ within 20% of the 52-week high"),
    "s4_volpen": Stage("momentum_vol", NEAR, TOP, "+ rank by ROC / volatility"),
    "s5_circuit": Stage(
        "momentum_vol", NO_LOCKS, TOP, "+ at most 3 band-lock days in 126"
    ),
    "s6_buffer50": Stage("momentum_vol", NO_LOCKS, 50, "+ hold while in the top 50"),
    "s7_composite": Stage(
        "composite", NO_LOCKS, 50, "+ 52-week-high distance in the rank"
    ),
    "s8_dma200": Stage(
        "composite",
        f"{NO_LOCKS} and close > sma(close, 200)",
        50,
        "+ close above the 200-day SMA",
    ),
    # 12-1 variants: the same stages ranked on 12-1 momentum, which skips the
    # last month's short-term reversal. ``mom_12_1 > 0`` is the floor the
    # repo's momentum_12_1 recipe uses.
    "m1_mom121": Stage(
        "momentum",
        f"{BASE} and mom_12_1 > 0",
        TOP,
        "s1 ranked by 12-1 momentum",
        "mom_12_1",
    ),
    "m4_mom121_volpen": Stage(
        "momentum_vol",
        f"{NEAR} and mom_12_1 > 0",
        TOP,
        "s4 ranked by 12-1 momentum / volatility",
        "mom_12_1",
    ),
    "m6_mom121_buffer50": Stage(
        "momentum_vol",
        f"{NO_LOCKS} and mom_12_1 > 0",
        50,
        "s6 ranked by 12-1 momentum / volatility",
        "mom_12_1",
    ),
    "m7_mom121_composite": Stage(
        "composite",
        f"{NO_LOCKS} and mom_12_1 > 0",
        50,
        "s7 with 12-1 momentum in the composite",
        "mom_12_1",
    ),
}


def make_request(stage: str, args: argparse.Namespace):
    from screener.backtester.workflow import BacktestRequest
    from screener.strategies.spec import (
        DEFAULT_STRATEGY_PROFILE,
        register_expression_strategy,
    )

    spec = STAGES[stage]
    name = f"india_mom_{stage}"
    register_expression_strategy(
        name,
        entry=spec.entry,
        exit=None,
        prepare_bars=prepare_bars(spec.rank, spec.momentum),
        required_lookback=lambda: LOOKBACK + 8,
        profile=DEFAULT_STRATEGY_PROFILE,
    )
    return BacktestRequest(
        mode="rolling",
        context_obj=None,
        market="india",
        hold=5000,
        top=TOP,
        entry_expr=None,
        exit_expr=None,
        strategy_name=name,
        stop_loss=None,
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
    pd.DataFrame([vars(trade) for trade in result.trades]).to_csv(
        args.out_dir / f"{stage}_trades.csv", index=False
    )
    (args.out_dir / f"{stage}.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "description": STAGES[stage].description,
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
        f"{stage:15s} CAGR {metrics['cagr']:6.1%}  Sharpe {metrics['sharpe']:5.2f}  "
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
