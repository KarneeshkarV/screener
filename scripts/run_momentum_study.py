#!/usr/bin/env python
"""Run the momentum-literature backtest matrix and dump one JSON per run.

Sixteen strategies drawn from four momentum literatures, over five trailing
windows, in two markets. Every run goes through the same rolling engine the
``backtest-rolling`` CLI uses - this script only supplies parameters and
serializes the result, so nothing here can produce a number the CLI would not.

    uv run python scripts/run_momentum_study.py                  # whole matrix
    uv run python scripts/run_momentum_study.py -m india -y 5    # one slice
    uv run python scripts/run_momentum_study.py --regime-sweep   # + overlays
    uv run python scripts/run_momentum_study.py --force          # ignore cache

Runs are resumable: a run whose JSON already exists is skipped unless
``--force`` is passed, so an interrupted matrix picks up where it stopped.

Holding periods follow each paper rather than a single house default, because
holding period is the parameter these strategies are most sensitive to:

* cross-sectional momentum holds three months, Jegadeesh & Titman's most
  profitable formation/holding cell;
* ``momentum_6_6`` holds six months, their headline J=6/K=6 specification;
* dual momentum and time-series momentum hold one month, matching Antonacci's
  monthly decisions and Moskowitz-Ooi-Pedersen's one-month holding period.

Costs are charged in every run. India uses the STT/stamp/exchange stack in
``screener.backtester.costs``; the US uses a flat five basis points of
commission. Both add slippage, wider for India where the small and mid caps in
the universe trade thinner.

This is research, not financial advice.
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
from screener.backtester.rolling import backtest_rolling
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run

DEFAULT_OUT_DIR = Path("reports/momentum_study")
INDIA_UNIVERSE_CONFIG = Path("data/universes/india_pit.toml")

PERIODS = (1, 2, 3, 5, 10)
TOP_SLOTS = 20

CROSS_SECTIONAL_HOLD = 63
JT_HOLD = 126
MONTHLY_HOLD = 21

# Holding periods swept by ``--hold-sweep``: one month, one quarter, six months.
# Holding period is the single parameter these strategies are most sensitive to,
# and it interacts with a regime overlay - a filter that blocks entries changes
# how often slots turn over, so the best hold under a filter need not be the
# best hold without one. That interaction is why the two sweeps cross rather
# than run separately.
HOLD_GRID = (MONTHLY_HOLD, CROSS_SECTIONAL_HOLD, JT_HOLD)


@dataclass(frozen=True)
class Strategy:
    name: str
    family: str
    label: str
    hold: int
    paper: str
    note: str


STRATEGIES: tuple[Strategy, ...] = (
    Strategy(
        "momentum_12_1",
        "A",
        "12-1 cross-sectional",
        CROSS_SECTIONAL_HOLD,
        "Jegadeesh & Titman (1993)",
        "Rank on the 12-month return skipping the last month; hold the winners.",
    ),
    Strategy(
        "momentum_6_6",
        "A",
        "6-6 (JT headline)",
        JT_HOLD,
        "Jegadeesh & Titman (1993)",
        "The paper's J=6/K=6 cell: six-month formation, no skip, six-month hold.",
    ),
    Strategy(
        "momentum_12_1_trend",
        "A",
        "12-1 + 200-day trend",
        CROSS_SECTIONAL_HOLD,
        "Jegadeesh & Titman (1993); Antonacci (2012)",
        "Winners that have also held their long-term trend.",
    ),
    Strategy(
        "momentum_12_1_riskadj",
        "A",
        "12-1 risk-adjusted rank",
        CROSS_SECTIONAL_HOLD,
        "Barroso & Santa-Clara (2015)",
        "Ranks by momentum per unit of volatility instead of raw momentum.",
    ),
    Strategy(
        "momentum_12_1_volmanaged",
        "A",
        "12-1 volatility-managed",
        CROSS_SECTIONAL_HOLD,
        "Barroso & Santa-Clara (2015)",
        "Stops entering when the momentum portfolio's own volatility is in its top quintile.",
    ),
    Strategy(
        "momentum_12_1_dynamic",
        "A",
        "12-1 dynamic (crash state)",
        CROSS_SECTIONAL_HOLD,
        "Daniel & Moskowitz (2016)",
        "Stops entering in a bear market with elevated variance - their crash state.",
    ),
    Strategy(
        "momentum_12_1_defensive",
        "A",
        "12-1 + risk-on regime",
        CROSS_SECTIONAL_HOLD,
        "Daniel & Moskowitz (2016)",
        "Enters only while the benchmark is above rising 50/200-day averages.",
    ),
    Strategy(
        "dual_momentum_gem",
        "B",
        "Dual momentum (per-name gate)",
        MONTHLY_HOLD,
        "Antonacci (2012/2017)",
        "A winner must have beaten Treasury bills over the same twelve months.",
    ),
    Strategy(
        "dual_momentum_market",
        "B",
        "Dual momentum (market gate)",
        MONTHLY_HOLD,
        "Antonacci (2012/2017)",
        "Holds stocks only while the benchmark itself beat bills - GEM's risk-off switch.",
    ),
    Strategy(
        "dual_momentum_paa",
        "B",
        "Protective asset allocation",
        MONTHLY_HOLD,
        "Keller & Keuning (2016)",
        "Breadth crash protection: flat once half the universe is below its 12-month average.",
    ),
    Strategy(
        "dual_momentum_daa",
        "B",
        "Defensive asset allocation",
        MONTHLY_HOLD,
        "Keller & Keuning (2018)",
        "Canary gate on the benchmark's 13612W momentum.",
    ),
    Strategy(
        "tsmom_12",
        "C",
        "Time-series momentum 12m",
        MONTHLY_HOLD,
        "Moskowitz, Ooi & Pedersen (2012)",
        "Long each name while its own 12-month return is positive.",
    ),
    Strategy(
        "tsmom_blend",
        "C",
        "Trend blend 1/3/12m",
        MONTHLY_HOLD,
        "Hurst, Ooi & Pedersen (2013, 2017)",
        "Long when at least two of the 1-, 3- and 12-month trends agree.",
    ),
    Strategy(
        "faber_sma10",
        "D",
        "10-month moving average",
        MONTHLY_HOLD,
        "Faber (2018)",
        "Hold each name while it is above its own ten-month average.",
    ),
    Strategy(
        "absolute_momentum",
        "D",
        "Absolute momentum",
        MONTHLY_HOLD,
        "Antonacci (2013)",
        "Hold while the full 12-month return beat bills; no skip month.",
    ),
    Strategy(
        "industry_trend_breakout",
        "D",
        "Channel breakout + trailing stop",
        MONTHLY_HOLD,
        "Zarattini & Antonacci (2024)",
        "Buy the 20-day breakout, sell the 40-day breakdown.",
    ),
)

FAMILY_TITLES = {
    "A": "Cross-sectional momentum",
    "B": "Dual momentum",
    "C": "Time-series momentum / trend following",
    "D": "Long-only trend rules",
}


@dataclass(frozen=True)
class RegimeFilter:
    """A benchmark-state overlay applied on top of a strategy's own rules.

    The engine suppresses entries on days whose benchmark trend or breadth
    regime is outside the listed labels. It is an entry gate only, so a filtered
    run holds the same positions it would otherwise have held, it just opens
    fewer of them - which makes this a clean test of whether market state adds
    anything to a signal that already has its own risk rules.
    """

    key: str
    label: str
    trend: tuple[str, ...] = ()
    breadth: tuple[str, ...] = ()


REGIME_FILTERS: tuple[RegimeFilter, ...] = (
    RegimeFilter("", "No overlay"),
    RegimeFilter("bull", "Benchmark bull only", trend=("bull",)),
    RegimeFilter("nonbear", "Anything but a bear", trend=("bull", "pullback")),
    RegimeFilter(
        "breadth",
        "Bullish breadth",
        breadth=("strong_bull", "bullish", "recovery_attempt"),
    ),
)
REGIME_BY_KEY = {regime.key: regime for regime in REGIME_FILTERS}


@dataclass(frozen=True)
class Lever:
    """A single construction change tested against the baseline portfolio.

    These are swept one at a time rather than crossed with each other. Crossing
    five levers with the hold and regime grids would be tens of thousands of
    runs and would make any single good cell indistinguishable from the best of
    a very large search - the deflated Sharpe in each run's metrics exists
    precisely because that distinction matters.
    """

    key: str
    label: str
    why: str
    overrides: dict[str, Any]


LEVERS: tuple[Lever, ...] = (
    Lever(
        "invvol",
        "Inverse-volatility sizing",
        "Moskowitz-Ooi-Pedersen, Hurst and Zarattini all size positions inversely "
        "to recent volatility; the baseline's equal slots do not.",
        # The engine clamps every sizing rule to the equal-slot budget, because
        # it cannot lever. inverse_vol asks for equity * risk_pct / daily_vol,
        # so at the 1% default and 20 slots (a 5% slot) the clamp binds unless a
        # name moves 20% a day - the lever measured nothing at all on its first
        # pass. Calibrating risk_pct so the median name (~1.7% daily) lands at
        # its slot cap puts the tilt back: quiet names fill the slot, volatile
        # ones get cut below it, which is the whole point of the rule.
        {"sizing_rule": "inverse_vol", "sizing_risk_pct": 0.00085},
    ),
    Lever(
        "top10",
        "10 positions",
        "Piras finds concentrated long-only winner portfolios trade momentum "
        "exposure against idiosyncratic risk, and get unstable below ten names.",
        {"top": 10},
    ),
    Lever(
        "top50",
        "50 positions",
        "The diffuse end of the same trade-off: more names, less stock-specific "
        "noise, weaker exposure to the strongest signals.",
        {"top": 50},
    ),
    Lever(
        "sectorneutral",
        "Sector-neutral ranking",
        "Momentum baskets concentrate in whatever sector last led. Z-scoring the "
        "rank score within sector strips that bet out of the signal.",
        {"sector_neutral": True},
    ),
    Lever(
        "trail25",
        "25% trailing stop",
        "Zarattini's exit is a trailing channel stop. A wide trailing stop is the "
        "closest equivalent here; narrow stops are already known to lose.",
        {"trailing_stop": 0.25},
    ),
)
LEVER_BY_KEY = {lever.key: lever for lever in LEVERS}


@dataclass(frozen=True)
class MarketSpec:
    market: str
    universe: str
    universe_config: Path | None
    benchmark: str
    cost_model: str
    slippage_bps: float
    commission_bps: float
    min_price: float
    label: str


MARKETS: dict[str, MarketSpec] = {
    "india": MarketSpec(
        market="india",
        universe="nifty500_extended_pit",
        universe_config=INDIA_UNIVERSE_CONFIG,
        benchmark="^NSEI",
        cost_model="india",
        slippage_bps=10.0,
        commission_bps=0.0,
        min_price=10.0,
        label="India - Nifty 500 (point-in-time)",
    ),
    "us": MarketSpec(
        market="us",
        universe="sp500",
        universe_config=None,
        benchmark="SPY",
        cost_model="flat",
        slippage_bps=5.0,
        commission_bps=5.0,
        min_price=1.0,
        label="US - S&P 500 (point-in-time)",
    ),
}


def _cli_defaults() -> dict[str, Any]:
    """Every ``backtest-rolling`` option at its declared default.

    Reading the defaults off the Click command rather than restating them keeps
    this script from drifting away from the CLI when an option changes.
    """
    context = click.Context(backtest_rolling)
    defaults: dict[str, Any] = {}
    for param in backtest_rolling.params:
        if param.name is None:
            continue
        value = param.get_default(context, call=True)
        # Click leaves a repeatable option with no declared default as UNSET;
        # the command itself only ever sees the empty tuple the parser builds.
        if type(value).__name__ == "Sentinel":
            value = () if param.multiple else None
        defaults[param.name] = value
    return defaults


def build_request(
    strategy: Strategy,
    spec: MarketSpec,
    years: int,
    fetcher: Any,
    regime: RegimeFilter = REGIME_FILTERS[0],
    hold: int | None = None,
    lever: Lever | None = None,
) -> BacktestRequest:
    params = _cli_defaults()
    params.update(
        market=spec.market,
        years=years,
        strategy_name=strategy.name,
        hold=hold or strategy.hold,
        top=TOP_SLOTS,
        universe=spec.universe,
        universe_config=spec.universe_config,
        point_in_time=True,
        benchmark=spec.benchmark,
        cost_model=spec.cost_model,
        slippage_bps=spec.slippage_bps,
        commission_bps=spec.commission_bps,
        min_price=spec.min_price,
        regime_filter_args=regime.trend,
        breadth_filter_args=regime.breadth,
        # The study reports CAGR and an equity curve over up to ten years, so
        # it has to reinvest. With the frozen slot budget a run that quadruples
        # ends up three quarters in idle cash, and its measured volatility and
        # drawdown decay for reasons that have nothing to do with the signal -
        # which makes Sharpe incomparable between the 1y and 10y windows.
        compounding=True,
    )
    if lever is not None:
        params.update(lever.overrides)
    # The fetcher is injected as the context object so all runs share one warm
    # on-disk price cache instead of rebuilding it per run.
    return BacktestRequest(mode="rolling", context_obj=fetcher, **params)


def _curve_records(curve: pd.Series) -> list[dict[str, Any]]:
    if curve is None or len(curve) == 0:
        return []
    series = curve.dropna()
    stamps = pd.DatetimeIndex(series.index)
    return [
        {"date": stamp.date().isoformat(), "value": float(value)}
        for stamp, value in zip(stamps, series.to_numpy(), strict=True)
    ]


def _trade_records(trades: list[Any]) -> list[dict[str, Any]]:
    records = []
    for trade in trades:
        records.append(
            {
                "ticker": trade.ticker,
                "entry_date": str(getattr(trade, "entry_date", "")),
                "exit_date": str(getattr(trade, "exit_date", "")),
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "shares": float(trade.shares),
                "pnl": float(trade.pnl),
                "return_pct": float(trade.return_pct),
                "exit_reason": str(trade.exit_reason),
            }
        )
    return records


def run_one(
    strategy: Strategy,
    spec: MarketSpec,
    years: int,
    fetcher: Any,
    regime: RegimeFilter = REGIME_FILTERS[0],
    hold: int | None = None,
    lever: Lever | None = None,
) -> dict[str, Any]:
    request = build_request(strategy, spec, years, fetcher, regime, hold, lever)
    run = resolve_backtest_run(request)
    assert run.start_date is not None and run.end_date is not None
    started = time.time()
    result = run_rolling_backtest(
        run.config,
        run.price_fetcher,
        start_date=run.start_date,
        end_date=run.end_date,
        fundamental_fetcher=run.fundamental_fetcher,
    )
    metrics = {
        key: (None if value is None else value)
        for key, value in dict(result.metrics).items()
        if not isinstance(value, (pd.Series, pd.DataFrame))
    }
    return {
        "strategy": strategy.name,
        "family": strategy.family,
        "label": strategy.label,
        "paper": strategy.paper,
        "note": strategy.note,
        "market": spec.market,
        "market_label": spec.label,
        "universe": spec.universe,
        "benchmark": spec.benchmark,
        "years": years,
        "start": run.start_date.isoformat(),
        "end": run.end_date.isoformat(),
        "top": request.top,
        "hold": hold or strategy.hold,
        "default_hold": strategy.hold,
        "lever": lever.key if lever else "",
        "lever_label": lever.label if lever else "Baseline",
        "cost_model": spec.cost_model,
        "slippage_bps": spec.slippage_bps,
        "commission_bps": spec.commission_bps,
        "universe_note": run.universe_note,
        "regime": regime.key,
        "regime_label": regime.label,
        "elapsed_seconds": round(time.time() - started, 1),
        "metrics": metrics,
        "equity_curve": _curve_records(result.equity_curve),
        "benchmark_curve": _curve_records(result.benchmark_curve),
        "trades": _trade_records(result.trades),
        "warnings": list(result.warnings),
        "generated": date.today().isoformat(),
    }


def run_key(
    strategy: Strategy,
    spec: MarketSpec,
    years: int,
    regime: RegimeFilter = REGIME_FILTERS[0],
    hold: int | None = None,
    lever: Lever | None = None,
) -> str:
    # A run at the strategy's own defaults carries no suffix, so keys written
    # before the sweep dimensions existed still resolve.
    suffix = f"__{regime.key}" if regime.key else ""
    if hold and hold != strategy.hold:
        suffix += f"__h{hold}"
    if lever is not None:
        suffix += f"__{lever.key}"
    return f"{spec.market}__{strategy.name}__{years}y{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("-m", "--market", action="append", choices=sorted(MARKETS))
    parser.add_argument("-y", "--years", action="append", type=int)
    parser.add_argument("-s", "--strategy", action="append")
    parser.add_argument(
        "-r",
        "--regime",
        action="append",
        choices=sorted(REGIME_BY_KEY),
        help=(
            "Benchmark-state overlay to apply (repeatable). Defaults to the "
            "unfiltered run only; pass --regime-sweep for all of them."
        ),
    )
    parser.add_argument(
        "--regime-sweep",
        action="store_true",
        help="run every regime overlay, not just the unfiltered one",
    )
    parser.add_argument(
        "--hold",
        action="append",
        type=int,
        help=(
            "Holding period in trading days to test (repeatable). Defaults to "
            "each strategy's own paper-derived hold; --hold-sweep uses the grid."
        ),
    )
    parser.add_argument(
        "--hold-sweep",
        action="store_true",
        help=f"test every holding period in {HOLD_GRID}",
    )
    parser.add_argument(
        "-l",
        "--lever",
        action="append",
        choices=sorted(LEVER_BY_KEY),
        help="Construction lever to test against the baseline (repeatable).",
    )
    parser.add_argument(
        "--lever-sweep",
        action="store_true",
        help="test every construction lever, one at a time, against the baseline",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-run even when the JSON exists"
    )
    args = parser.parse_args()

    markets = [MARKETS[name] for name in (args.market or sorted(MARKETS))]
    periods = tuple(args.years or PERIODS)
    strategies = [s for s in STRATEGIES if not args.strategy or s.name in args.strategy]
    if args.regime_sweep:
        regimes = list(REGIME_FILTERS)
    elif args.regime:
        regimes = [REGIME_BY_KEY[key] for key in args.regime]
    else:
        regimes = [REGIME_FILTERS[0]]
    if args.hold_sweep:
        holds: list[int | None] = list(HOLD_GRID)
    elif args.hold:
        holds = list(args.hold)
    else:
        holds = [None]
    # Levers are tested one at a time against the baseline, so the baseline is
    # always in the list and never crossed with another lever.
    if args.lever_sweep:
        levers: list[Lever | None] = [None, *LEVERS]
    elif args.lever:
        levers = [None, *(LEVER_BY_KEY[key] for key in args.lever)]
    else:
        levers = [None]

    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    fetcher = build_price_fetcher()

    cells = [
        (spec, years, strategy, regime, hold, lever)
        # Longest window first: it warms the price cache that every shorter
        # window then reads, so the expensive fetch happens once per market.
        for spec in markets
        for years in sorted(periods, reverse=True)
        for strategy in strategies
        for regime in regimes
        for hold in holds
        for lever in levers
    ]
    # A hold that equals the strategy's own default is the same run as the
    # unswept one, so the sweep does not duplicate it under a second key.
    seen: set[str] = set()
    total = len(cells)
    done = 0
    failures: list[str] = []
    for spec, years, strategy, regime, hold, lever in cells:
        done += 1
        key = run_key(strategy, spec, years, regime, hold, lever)
        if key in seen:
            print(f"[{done}/{total}] skip {key} (duplicate cell)", file=sys.stderr)
            continue
        seen.add(key)
        path = runs_dir / f"{key}.json"
        if path.exists() and not args.force:
            print(f"[{done}/{total}] skip {key} (cached)", file=sys.stderr)
            continue
        print(f"[{done}/{total}] run  {key}", file=sys.stderr, flush=True)
        try:
            payload = run_one(strategy, spec, years, fetcher, regime, hold, lever)
        except Exception:  # noqa: BLE001 - one bad run must not stop the matrix
            failures.append(key)
            traceback.print_exc()
            continue
        path.write_text(json.dumps(payload), encoding="utf-8")
        metrics = payload["metrics"]
        print(
            f"    sharpe={metrics.get('sharpe', float('nan')):.2f}  "
            f"cagr={metrics.get('cagr', float('nan')):.2%}  "
            f"maxdd={metrics.get('max_drawdown', float('nan')):.2%}  "
            f"trades={metrics.get('trade_count', 0)}  "
            f"{payload['elapsed_seconds']}s",
            file=sys.stderr,
            flush=True,
        )

    if failures:
        print(
            f"\n{len(failures)} run(s) failed: {', '.join(failures)}", file=sys.stderr
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
