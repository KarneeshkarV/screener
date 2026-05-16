"""Walk-forward optimization for strategy parameters."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Literal, Optional

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from screener.backtester.cli_common import (
    DEFAULT_BENCHMARK,
    build_slippage_model,
    parse_partial_exits,
    resolve_min_filters,
    resolve_strategy_exprs,
)
from screener.backtester.data import PriceFetcher, YFinancePriceFetcher
from screener.backtester.metrics import compute_metrics
from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.backtester.portfolio import build_equity_curve
from screener.backtester.rolling import run_rolling_backtest


def _generate_param_combinations(
    param_grid: dict[str, list],
) -> list[dict[str, object]]:
    """Cartesian product of param grid values."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos: list[dict[str, object]] = []
    for combo in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, combo)})
    return combos


@dataclass
class WindowResult:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    best_params: dict[str, object]
    best_metric: float
    oos_result: BacktestResult


@dataclass
class WalkForwardResult:
    window_results: list[WindowResult]
    combined_trades: list[Trade]
    combined_equity: pd.Series
    combined_benchmark: pd.Series
    combined_metrics: dict
    param_stability: dict[str, float] = field(default_factory=dict)


class WalkForwardOptimizer:
    """Split history into train/test windows, optimize params in-sample,
    evaluate out-of-sample, and aggregate OOS results."""

    def __init__(
        self,
        base_cfg: BacktestConfig,
        fetcher: PriceFetcher,
        train_months: int = 6,
        test_months: int = 2,
        param_grid: Optional[dict[str, list]] = None,
        optimization_metric: Literal["sharpe", "total_return", "cagr"] = "sharpe",
        window_type: Literal["rolling", "expanding"] = "rolling",
    ) -> None:
        self.base_cfg = base_cfg
        self.fetcher = fetcher
        self.train_months = train_months
        self.test_months = test_months
        self.param_grid = param_grid or {}
        self.optimization_metric = optimization_metric
        self.window_type = window_type

    def _make_cfg(self, overrides: dict[str, object]) -> BacktestConfig:
        """Clone base config with param overrides."""
        # BacktestConfig is frozen; build a new one from scratch via dict.
        d = {
            "market": self.base_cfg.market,
            "as_of": self.base_cfg.as_of,
            "hold": overrides.get("hold", self.base_cfg.hold),
            "top": self.base_cfg.top,
            "entry_expr": self.base_cfg.entry_expr,
            "exit_expr": self.base_cfg.exit_expr,
            "stop_loss": overrides.get("stop_loss", self.base_cfg.stop_loss),
            "take_profit": overrides.get("take_profit", self.base_cfg.take_profit),
            "trailing_stop": overrides.get("trailing_stop", self.base_cfg.trailing_stop),
            "slippage_bps": self.base_cfg.slippage_bps,
            "commission_bps": self.base_cfg.commission_bps,
            "initial_capital": self.base_cfg.initial_capital,
            "benchmark": self.base_cfg.benchmark,
            "strategy_name": self.base_cfg.strategy_name,
            "tickers": self.base_cfg.tickers,
            "universe_file": self.base_cfg.universe_file,
            "max_universe": self.base_cfg.max_universe,
            "min_price": self.base_cfg.min_price,
            "min_avg_dollar_volume": self.base_cfg.min_avg_dollar_volume,
            "avg_dollar_volume_window": self.base_cfg.avg_dollar_volume_window,
            "reserve_multiple": self.base_cfg.reserve_multiple,
            "reinvest": self.base_cfg.reinvest,
            "slippage_model": self.base_cfg.slippage_model,
            "gap_fills": self.base_cfg.gap_fills,
            "entry_order_type": self.base_cfg.entry_order_type,
            "entry_limit_bps": self.base_cfg.entry_limit_bps,
            "allow_reentry": self.base_cfg.allow_reentry,
            "max_reentries": self.base_cfg.max_reentries,
            "partial_exits": self.base_cfg.partial_exits,
            "price_adjustment": self.base_cfg.price_adjustment,
        }
        return BacktestConfig(**d)

    def _optimize_window(
        self,
        train_start: date,
        train_end: date,
    ) -> tuple[dict[str, object], float]:
        """Grid-search params over the train window."""
        combos = _generate_param_combinations(self.param_grid)
        if not combos:
            combos = [{}]

        best_params: dict[str, object] = {}
        best_metric = float("-inf")

        for combo in combos:
            cfg = self._make_cfg(combo)
            try:
                result = run_rolling_backtest(
                    cfg,
                    self.fetcher,
                    start_date=train_start,
                    end_date=train_end,
                )
            except Exception as exc:
                # noisy data / empty windows are skipped gracefully
                continue
            metric = result.metrics.get(self.optimization_metric, float("-inf"))
            if not np.isfinite(metric):
                continue
            if metric > best_metric:
                best_metric = metric
                best_params = dict(combo)

        return best_params, best_metric

    def _build_windows(
        self,
        start_date: date,
        end_date: date,
    ) -> list[tuple[date, date, date, date]]:
        """Return list of (train_start, train_end, test_start, test_end)."""
        windows: list[tuple[date, date, date, date]] = []
        train_start = start_date
        test_delta = pd.DateOffset(months=self.test_months)
        train_delta = pd.DateOffset(months=self.train_months)

        while True:
            train_end = (pd.Timestamp(train_start) + train_delta).date()
            test_start = train_end
            test_end = (pd.Timestamp(test_start) + test_delta).date()
            if test_end > end_date:
                break
            windows.append((train_start, train_end, test_start, test_end))
            if self.window_type == "rolling":
                train_start = test_end
            else:
                train_start = start_date
        return windows

    def run(
        self,
        start_date: date,
        end_date: date,
    ) -> WalkForwardResult:
        """Run walk-forward optimization and return aggregated OOS results."""
        windows = self._build_windows(start_date, end_date)
        if not windows:
            empty_equity = pd.Series(
                self.base_cfg.initial_capital,
                index=pd.bdate_range(start_date, end_date),
                dtype=float,
            )
            return WalkForwardResult(
                window_results=[],
                combined_trades=[],
                combined_equity=empty_equity,
                combined_benchmark=empty_equity,
                combined_metrics=compute_metrics(
                    empty_equity, empty_equity, [], max(self.base_cfg.top, 1)
                ),
            )

        window_results: list[WindowResult] = []
        all_oos_trades: list[Trade] = []

        for train_start, train_end, test_start, test_end in windows:
            best_params, best_metric = self._optimize_window(train_start, train_end)
            # if no combo worked, fall back to base config
            if not best_params:
                best_params = {}
            oos_cfg = self._make_cfg(best_params)
            try:
                oos_result = run_rolling_backtest(
                    oos_cfg,
                    self.fetcher,
                    start_date=test_start,
                    end_date=test_end,
                )
            except Exception:
                # Build a dummy empty result for this window
                calendar = pd.bdate_range(test_start, test_end)
                empty_equity = pd.Series(
                    self.base_cfg.initial_capital, index=calendar, dtype=float
                )
                oos_result = BacktestResult(
                    config=oos_cfg,
                    trades=[],
                    equity_curve=empty_equity,
                    benchmark_curve=empty_equity,
                    metrics=compute_metrics(
                        empty_equity, empty_equity, [], max(self.base_cfg.top, 1)
                    ),
                )

            window_results.append(
                WindowResult(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    best_params=best_params,
                    best_metric=best_metric,
                    oos_result=oos_result,
                )
            )
            all_oos_trades.extend(oos_result.trades)

        # Aggregate OOS equity curve: stitch together window equity curves
        # Weighted by the capital they started with.  Simpler: just rebuild
        # from combined trades over the full date range.
        full_start = window_results[0].test_start
        full_end = window_results[-1].test_end
        full_calendar = pd.bdate_range(full_start, full_end)

        # We need price panels for build_equity_curve.  Fetch once for the
        # full range with the base config's tickers/universe.
        from screener.backtester.core import _resolve_universe
        from screener.backtester.data import tv_to_yf, fetch_benchmark

        tv_symbols, _ = _resolve_universe(self.base_cfg)
        yf_by_tv = {tv: tv_to_yf(tv, self.base_cfg.market) for tv in tv_symbols}
        yf_symbols = list(dict.fromkeys(list(yf_by_tv.values()) + [self.base_cfg.benchmark]))

        warmup = max(
            (pd.Timestamp(full_end) - pd.Timestamp(full_start)).days + 30,
            365,
        )
        fetch_start = (pd.Timestamp(full_start) - pd.Timedelta(days=warmup)).date()
        price_panel = self.fetcher.fetch(yf_symbols, fetch_start, full_end)
        bars_by_tv = {tv: price_panel.get(yf_by_tv[tv], pd.DataFrame()) for tv in tv_symbols}

        combined_equity = build_equity_curve(
            full_calendar, all_oos_trades, bars_by_tv, self.base_cfg.initial_capital
        )
        benchmark = fetch_benchmark(
            self.base_cfg.benchmark, fetch_start, full_end, self.fetcher
        )
        combined_benchmark = benchmark.reindex(full_calendar, method="ffill").dropna()
        combined_metrics = compute_metrics(
            combined_equity, combined_benchmark, all_oos_trades, max(self.base_cfg.top, 1)
        )

        # param stability = fraction of windows where the most-common value appears
        param_stability: dict[str, float] = {}
        for key in self.param_grid:
            values = [wr.best_params.get(key) for wr in window_results]
            if not values:
                continue
            # Count non-None values
            non_none = [v for v in values if v is not None]
            if not non_none:
                continue
            mode_val = max(set(non_none), key=non_none.count)
            param_stability[key] = sum(1 for v in values if v == mode_val) / len(values)

        return WalkForwardResult(
            window_results=window_results,
            combined_trades=all_oos_trades,
            combined_equity=combined_equity,
            combined_benchmark=combined_benchmark,
            combined_metrics=combined_metrics,
            param_stability=param_stability,
        )


def print_walk_forward(result: WalkForwardResult) -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]Walk-Forward Optimization[/bold]  "
            f"windows={len(result.window_results)}  "
            f"combined_sharpe=[cyan]{result.combined_metrics.get('sharpe', 0.0):.3f}[/cyan]"
        )
    )

    window_table = Table(title="Window Results", show_header=True, header_style="bold")
    window_table.add_column("Window")
    window_table.add_column("Train")
    window_table.add_column("Test")
    window_table.add_column("Best Params")
    window_table.add_column("Best IS Metric")
    window_table.add_column("OOS Trades")
    window_table.add_column("OOS Sharpe")

    for i, wr in enumerate(result.window_results, 1):
        params_str = ", ".join(
            f"{k}={v}" for k, v in wr.best_params.items()
        )
        oos_sharpe = wr.oos_result.metrics.get("sharpe", 0.0)
        window_table.add_row(
            str(i),
            f"{wr.train_start} → {wr.train_end}",
            f"{wr.test_start} → {wr.test_end}",
            params_str or "default",
            f"{wr.best_metric:.3f}",
            str(len(wr.oos_result.trades)),
            f"{oos_sharpe:.3f}",
        )
    console.print(window_table)

    if result.param_stability:
        stab_table = Table(title="Param Stability", show_header=True, header_style="bold")
        stab_table.add_column("Param")
        stab_table.add_column("Stability", justify="right")
        for k, v in result.param_stability.items():
            stab_table.add_row(k, f"{v:.1%}")
        console.print(stab_table)

    metrics_table = Table(title="Combined OOS Metrics", show_header=True, header_style="bold")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")
    for key, label in {
        "total_return": "Total Return",
        "cagr": "CAGR",
        "sharpe": "Sharpe",
        "max_drawdown": "Max Drawdown",
        "hit_rate": "Hit Rate",
        "trade_count": "Trades",
    }.items():
        if key in result.combined_metrics:
            val = result.combined_metrics[key]
            if isinstance(val, float):
                fmt = f"{val * 100:.2f}%" if key != "sharpe" else f"{val:.3f}"
            else:
                fmt = str(val)
            metrics_table.add_row(label, fmt)
    console.print(metrics_table)


@click.command(name="walk-forward")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    help="Market to backtest.",
)
@click.option(
    "--start",
    "start_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    "end_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="End date (YYYY-MM-DD).",
)
@click.option("--train-months", type=int, default=6, show_default=True)
@click.option("--test-months", type=int, default=2, show_default=True)
@click.option("--hold", type=int, default=20)
@click.option("--top", type=int, default=10)
@click.option("--entry", "entry_expr", default=None)
@click.option("--exit", "exit_expr", default=None)
@click.option("--strategy", "strategy_name", default=None)
@click.option("--stop-loss", type=float, default=None)
@click.option("--take-profit", type=float, default=None)
@click.option("--trailing-stop", type=float, default=None)
@click.option("--slippage-bps", type=float, default=0.0)
@click.option("--commission-bps", type=float, default=0.0)
@click.option("--initial-capital", type=float, default=100_000.0)
@click.option("--benchmark", default=None)
@click.option("--tickers", default=None)
@click.option("--universe-file", default=None)
@click.option("--max-universe", type=int, default=0)
@click.option("--min-price", type=float, default=None)
@click.option("--min-avg-dollar-volume", type=float, default=None)
@click.option("--adv-window", type=int, default=20)
@click.option("--slippage-model", type=click.Choice(["fixed", "half-spread", "vol-impact", "composite"]), default="fixed")
@click.option("--half-spread-bps", type=float, default=0.0)
@click.option("--vol-impact-k", type=float, default=0.1)
@click.option("--no-gap-fills", is_flag=True, default=False)
@click.option("--entry-order", type=click.Choice(["moo", "moc", "limit"]), default="moo")
@click.option("--entry-limit-bps", type=float, default=None)
@click.option("--price-adjustment", type=click.Choice(["full", "splits_only", "none"]), default="full")
@click.option(
    "--param-grid",
    default=None,
    help='JSON string of param grid, e.g. \'{"stop_loss":[0.05,0.10],"take_profit":[0.10,0.20]}\'',
)
@click.option(
    "--optimization-metric",
    type=click.Choice(["sharpe", "total_return", "cagr"]),
    default="sharpe",
)
@click.option(
    "--window-type",
    type=click.Choice(["rolling", "expanding"]),
    default="rolling",
)
@click.option("--json-out", default=None, help="Write result JSON to path.")
def walk_forward(
    market,
    start_arg,
    end_arg,
    train_months,
    test_months,
    hold,
    top,
    entry_expr,
    exit_expr,
    strategy_name,
    stop_loss,
    take_profit,
    trailing_stop,
    slippage_bps,
    commission_bps,
    initial_capital,
    benchmark,
    tickers,
    universe_file,
    max_universe,
    min_price,
    min_avg_dollar_volume,
    adv_window,
    slippage_model,
    half_spread_bps,
    vol_impact_k,
    no_gap_fills,
    entry_order,
    entry_limit_bps,
    price_adjustment,
    param_grid,
    optimization_metric,
    window_type,
    json_out,
):
    """Walk-forward optimization: train on in-sample, test out-of-sample."""
    entry_expr, exit_expr = resolve_strategy_exprs(strategy_name, entry_expr, exit_expr)
    slip_model = build_slippage_model(
        slippage_model, slippage_bps, half_spread_bps, vol_impact_k
    )
    resolved_min_price, resolved_min_adv = resolve_min_filters(
        market, min_price, min_avg_dollar_volume
    )
    bench = benchmark or DEFAULT_BENCHMARK.get(market, "SPY")
    start_date: date = start_arg.date() if isinstance(start_arg, datetime) else start_arg
    end_date: date = end_arg.date() if isinstance(end_arg, datetime) else end_arg

    ticker_tuple = None
    if tickers:
        ticker_tuple = tuple(t.strip() for t in tickers.split(",") if t.strip())

    grid: dict[str, list] = {}
    if param_grid:
        grid = json.loads(param_grid)

    cfg = BacktestConfig(
        market=market,
        as_of=end_date,
        hold=int(hold),
        top=int(top),
        strategy_name=strategy_name,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        slippage_bps=float(slippage_bps),
        commission_bps=float(commission_bps),
        initial_capital=float(initial_capital),
        benchmark=bench,
        tickers=ticker_tuple,
        universe_file=universe_file,
        max_universe=int(max_universe),
        min_price=resolved_min_price,
        min_avg_dollar_volume=resolved_min_adv,
        avg_dollar_volume_window=int(adv_window),
        slippage_model=slip_model,
        gap_fills=not no_gap_fills,
        entry_order_type=entry_order,
        entry_limit_bps=entry_limit_bps,
        price_adjustment=price_adjustment,
    )

    fetcher = click.get_current_context().obj or YFinancePriceFetcher(
        auto_adjust=price_adjustment == "full"
    )
    optimizer = WalkForwardOptimizer(
        base_cfg=cfg,
        fetcher=fetcher,
        train_months=int(train_months),
        test_months=int(test_months),
        param_grid=grid,
        optimization_metric=optimization_metric,
        window_type=window_type,
    )
    result = optimizer.run(start_date, end_date)
    print_walk_forward(result)

    if json_out:
        out = {
            "combined_metrics": result.combined_metrics,
            "param_stability": result.param_stability,
            "windows": [
                {
                    "train_start": wr.train_start.isoformat(),
                    "train_end": wr.train_end.isoformat(),
                    "test_start": wr.test_start.isoformat(),
                    "test_end": wr.test_end.isoformat(),
                    "best_params": wr.best_params,
                    "best_metric": wr.best_metric,
                    "oos_sharpe": wr.oos_result.metrics.get("sharpe"),
                    "oos_trades": len(wr.oos_result.trades),
                }
                for wr in result.window_results
            ],
        }
        import pathlib

        pathlib.Path(json_out).write_text(json.dumps(out, indent=2, default=str))
