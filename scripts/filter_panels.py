#!/usr/bin/env python
"""Build the point-in-time panels the filter study evaluates against.

One expensive step, cached to disk, so the evaluation itself is cheap to rerun.

What comes out:

- ``base``       date x ticker boolean. True where the default screen criterion
                 (`ema`: EMA5 > EMA20 > EMA100 > EMA200) holds *and* the name
                 was a point-in-time member of the universe on that date.
- ``forward``    date x ticker forward returns at 1W / 1M / 3M / 6M.
- ``features``   one date x ticker frame per feature per parameter setting.
- ``regime``     per-date market regime and volatility labels.

Bars come from the rolling backtester's own preparation path, so universe
membership, warmup and price adjustment are identical to a real backtest rather
than a second implementation that could drift from it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_pit_midsmall_study import (
    CAPITAL,
    TOP_SLOTS,
    UNIVERSES,
    UNIVERSE_CONFIG,
    _cli_defaults,
)
from screener.backtester.data import build_price_fetcher
from screener.backtester.rolling_simulation import prepare_rolling_backtest
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run
from screener.research.features import FeatureCtx, registry
from screener.research.filter_study import PanelSet, setting_key

# The default screen criterion, as a backtestable expression. `screener screen`
# with no -c flag runs `ema`, whose TradingView form is
# EMA5 > EMA20 > EMA100 > EMA200 with EMA200 positive; this is the same stack
# evaluated on bars.
BASE_ENTRY = (
    "ema(close, 5) > ema(close, 20) "
    "and ema(close, 20) > ema(close, 100) "
    "and ema(close, 100) > ema(close, 200) "
    "and ema(close, 200) > 0"
)
BASE_NAME = "ema_stack"

FORWARD_HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}
DEFAULT_OUT = Path.home() / "grill-me-filters"
END_DATE = date(2026, 8, 17)


def _request(universe_key: str, years: int, fetcher: Any) -> BacktestRequest:
    params = _cli_defaults()
    params.update(
        market="india",
        years=years,
        end_arg=datetime(END_DATE.year, END_DATE.month, END_DATE.day),
        entry_expr=BASE_ENTRY,
        exit_expr=None,
        strategy_name=None,
        hold=21,
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
    return BacktestRequest(mode="rolling", context_obj=fetcher, **params)


def _forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from today's close to the close ``horizon`` bars ahead.

    This is the one place a forward-looking shift is correct: it is the label
    being predicted, never an input. Everything on the feature side is causal,
    so the two can never be confused.
    """
    return close.shift(-horizon) / close - 1.0


def _regime_labels(benchmark: pd.Series) -> pd.DataFrame:
    """Causal market-state labels for conditional reporting.

    Trend: benchmark above/below its own 200-day SMA, with a sideways band when
    the SMA slope is flat. Volatility: benchmark realized vol against its own
    trailing median. Both use trailing windows only, so a date's label is knowable
    on that date.
    """
    close = benchmark.astype(float)
    sma = close.rolling(200, min_periods=200).mean()
    slope = sma - sma.shift(21)
    above = close > sma
    flat = slope.abs() < (close.rolling(200, min_periods=200).std(ddof=1) * 0.02)
    trend = pd.Series("sideways", index=close.index, dtype=object)
    trend[above & ~flat] = "bull"
    trend[~above & ~flat] = "bear"
    trend[sma.isna()] = "unknown"

    daily = np.log(close.where(close > 0.0)).diff()
    vol = daily.rolling(60, min_periods=60).std(ddof=1)
    median = vol.rolling(500, min_periods=250).median()
    vol_state = pd.Series("unknown", index=close.index, dtype=object)
    known = vol.notna() & median.notna()
    vol_state[known & (vol > median)] = "high_vol"
    vol_state[known & (vol <= median)] = "low_vol"
    return pd.DataFrame({"trend": trend, "vol": vol_state})


def build_panels(
    universe_key: str,
    years: int,
    *,
    categories: tuple[str, ...],
    max_tickers: int | None,
) -> PanelSet:
    fetcher = build_price_fetcher(provider="fmp")
    run = resolve_backtest_run(_request(universe_key, years, fetcher))
    assert run.start_date is not None and run.end_date is not None
    prepared = prepare_rolling_backtest(
        run.config,
        run.price_fetcher,
        start_date=run.start_date,
        end_date=run.end_date,
        fundamental_fetcher=run.fundamental_fetcher,
    )
    bars_by_tv = {
        tv: bars
        for tv, bars in prepared.bars_by_tv.items()
        if bars is not None and not bars.empty
    }
    tickers = sorted(bars_by_tv)
    if max_tickers is not None:
        # Deterministic subsample for the expensive experimental features.
        tickers = tickers[:max_tickers]
    dates = pd.DatetimeIndex(prepared.master_dates)
    # Regime labels need their own warmup, so they are computed on the full
    # benchmark series (which includes the pre-window warmup bars) and only
    # then cut to the evaluation window. Computing them on the window alone
    # would leave the first 200 dates unlabelled for no reason.
    full_benchmark = prepared.benchmark.astype(float)
    regime = _regime_labels(full_benchmark).reindex(dates)
    benchmark = full_benchmark.reindex(dates)

    close = pd.DataFrame(
        {tv: bars_by_tv[tv]["close"].astype(float).reindex(dates) for tv in tickers}
    )

    specs = [s for s in registry.values() if s.category in categories]
    columns: dict[str, dict[str, pd.Series]] = {}
    settings: dict[str, dict[str, Any]] = {}
    base_cols: dict[str, pd.Series] = {}

    from screener.backtester.pine import evaluate, parse

    base_ast = parse(BASE_ENTRY)
    started = time.time()
    for position, tv in enumerate(tickers, start=1):
        bars = bars_by_tv[tv]
        ctx = FeatureCtx(
            bars=bars,
            benchmark=prepared.benchmark.astype(float).reindex(bars.index),
            sector=None,
        )
        try:
            signal = evaluate(base_ast, bars).astype(bool)
        except Exception:  # noqa: BLE001 - a malformed ticker must not kill the panel
            signal = pd.Series(False, index=bars.index)
        base_cols[tv] = signal.reindex(dates).fillna(False)

        for spec in specs:
            if spec.needs_benchmark and ctx.benchmark is None:
                continue
            if spec.needs_sector:
                continue
            for params in spec.settings():
                key = setting_key(spec.name, params)
                settings.setdefault(key, {"feature": spec.name, **params})
                try:
                    values = spec.compute(ctx, **params)
                except Exception:  # noqa: BLE001
                    values = pd.Series(np.nan, index=bars.index)
                columns.setdefault(key, {})[tv] = values.reindex(dates)
        if position % 25 == 0:
            print(
                f"  {position}/{len(tickers)} tickers ({time.time() - started:.0f}s)",
                flush=True,
            )

    # float32 halves the artefact: at full universe scale the feature panels are
    # the dominant term, and no downstream statistic needs float64 precision on
    # what is ultimately a cross-sectional rank.
    features = {
        key: pd.DataFrame(cols).astype("float32") for key, cols in columns.items()
    }
    base = pd.DataFrame(base_cols).fillna(False).astype(bool)
    forward = {
        label: _forward_returns(close, h) for label, h in FORWARD_HORIZONS.items()
    }

    return PanelSet(
        base=base,
        close=close,
        forward=forward,
        features=features,
        settings=settings,
        regime=regime,
        benchmark=benchmark,
        meta={
            "universe": universe_key,
            "universe_name": UNIVERSES[universe_key],
            "years": years,
            "start": run.start_date.isoformat(),
            "end": run.end_date.isoformat(),
            "base_name": BASE_NAME,
            "base_entry": BASE_ENTRY,
            "n_tickers": len(tickers),
            "n_dates": len(dates),
            "categories": list(categories),
            "n_feature_settings": len(features),
            "generated": datetime.now().isoformat(timespec="seconds"),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "-u", "--universe", default="midsmall", choices=sorted(UNIVERSES)
    )
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument(
        "--categories",
        default="trend,quality,volatility,acceleration,relative,liquidity",
        help="Comma-separated feature categories to compute.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Subsample the universe (used for the expensive experimental features).",
    )
    parser.add_argument("--tag", default="core", help="Artefact name under --out-dir.")
    args = parser.parse_args()

    categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
    unknown = set(categories) - {s.category for s in registry.values()}
    if unknown:
        print(f"unknown categories: {sorted(unknown)}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(
        f"building panels universe={args.universe} years={args.years} "
        f"categories={categories}",
        flush=True,
    )
    panels = build_panels(
        args.universe,
        args.years,
        categories=categories,
        max_tickers=args.max_tickers,
    )
    dest = args.out_dir / f"panels_{args.tag}.pkl"
    with dest.open("wb") as handle:
        pickle.dump(panels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (args.out_dir / f"panels_{args.tag}.json").write_text(
        json.dumps(panels.meta, indent=2, default=str)
    )
    print(
        f"wrote {dest} "
        f"({panels.meta['n_tickers']} tickers x {panels.meta['n_dates']} dates, "
        f"{panels.meta['n_feature_settings']} feature settings, "
        f"{time.time() - started:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
