"""Parameter sensitivity analysis via grid search and heatmaps."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional

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
from screener.backtester.models import BacktestConfig, BacktestResult
from screener.backtester.rolling import run_rolling_backtest


try:
    import matplotlib
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


@dataclass
class SensitivityResult:
    param_names: tuple[str, ...]
    values: list[tuple[tuple, dict]]  # ((p1_val, p2_val, ...), metrics)
    base_config: BacktestConfig
    metrics_map: dict[str, list] = field(default_factory=dict)


class SensitivityAnalyzer:
    """Grid-search 1-2 parameters and report how metrics vary."""

    def __init__(
        self,
        base_cfg: BacktestConfig,
        fetcher: PriceFetcher,
        param_grid: dict[str, list],
        start_date: date,
        end_date: date,
    ) -> None:
        self.base_cfg = base_cfg
        self.fetcher = fetcher
        self.param_grid = param_grid
        self.start_date = start_date
        self.end_date = end_date

    def _make_cfg(self, overrides: dict[str, object]) -> BacktestConfig:
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

    def run(self) -> SensitivityResult:
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        results: list[tuple[tuple, dict]] = []

        for combo in itertools.product(*values):
            overrides = {k: v for k, v in zip(keys, combo)}
            cfg = self._make_cfg(overrides)
            try:
                result = run_rolling_backtest(
                    cfg,
                    self.fetcher,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )
            except Exception:
                # empty / invalid window
                result = BacktestResult(
                    config=cfg,
                    trades=[],
                    equity_curve=pd.Series(dtype=float),
                    benchmark_curve=pd.Series(dtype=float),
                    metrics={},
                )
            results.append((combo, dict(result.metrics)))

        # transpose metrics_map for easy plotting
        metrics_map: dict[str, list] = {}
        all_keys = set()
        for _combo, metrics in results:
            all_keys.update(metrics.keys())
        for mkey in all_keys:
            metrics_map[mkey] = [metrics.get(mkey, np.nan) for _combo, metrics in results]

        return SensitivityResult(
            param_names=tuple(keys),
            values=results,
            base_config=self.base_cfg,
            metrics_map=metrics_map,
        )


def print_sensitivity(result: SensitivityResult) -> None:
    console = Console()
    console.print(
        Panel.fit(
            f"[bold]Sensitivity Analysis[/bold]  params={list(result.param_names)}"
        )
    )

    table = Table(title="Grid Results", show_header=True, header_style="bold")
    for p in result.param_names:
        table.add_column(p)
    table.add_column("Sharpe", justify="right")
    table.add_column("Total Return", justify="right")
    table.add_column("Max Drawdown", justify="right")
    table.add_column("Trades", justify="right")

    for combo, metrics in result.values:
        sharpe = metrics.get("sharpe", np.nan)
        total_ret = metrics.get("total_return", np.nan)
        max_dd = metrics.get("max_drawdown", np.nan)
        trades = metrics.get("trade_count", np.nan)
        table.add_row(
            *(str(v) for v in combo),
            f"{sharpe:.3f}" if np.isfinite(sharpe) else "—",
            f"{total_ret * 100:.2f}%" if np.isfinite(total_ret) else "—",
            f"{max_dd * 100:.2f}%" if np.isfinite(max_dd) else "—",
            str(trades) if np.isfinite(trades) else "—",
        )
    console.print(table)


def save_heatmap(
    result: SensitivityResult,
    metric: str = "sharpe",
    out_path: str = "sensitivity_heatmap.png",
) -> None:
    if not _HAS_MATPLOTLIB:
        Console().print("[yellow]matplotlib not available; skipping heatmap.[/yellow]")
        return

    if len(result.param_names) != 2:
        Console().print("[yellow]Heatmap requires exactly 2 parameters.[/yellow]")
        return

    p1_name, p2_name = result.param_names
    p1_values = sorted({combo[0] for combo, _ in result.values})
    p2_values = sorted({combo[1] for combo, _ in result.values})

    grid = np.full((len(p1_values), len(p2_values)), np.nan)
    for combo, metrics in result.values:
        i = p1_values.index(combo[0])
        j = p2_values.index(combo[1])
        grid[i, j] = metrics.get(metric, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(p2_values)))
    ax.set_yticks(np.arange(len(p1_values)))
    ax.set_xticklabels([str(v) for v in p2_values])
    ax.set_yticklabels([str(v) for v in p1_values])
    ax.set_xlabel(p2_name)
    ax.set_ylabel(p1_name)
    ax.set_title(f"Sensitivity: {metric}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    Console().print(f"[dim]Saved heatmap to {out_path}[/dim]")


@click.command(name="sensitivity")
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
)
@click.option(
    "--end",
    "end_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
)
@click.option("--hold", type=int, default=20)
@click.option("--top", type=int, default=10)
@click.option("--entry", "entry_expr", default=None)
@click.option("--exit", "exit_expr", default=None)
@click.option("--strategy", "strategy_name", default=None)
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
    required=True,
    help='JSON string of param grid, e.g. \'{"stop_loss":[0.05,0.10],"take_profit":[0.10,0.20]}\'',
)
@click.option("--heatmap-metric", default="sharpe")
@click.option("--heatmap-out", default="sensitivity_heatmap.png")
@click.option("--json-out", default=None)
def sensitivity(
    market,
    start_arg,
    end_arg,
    hold,
    top,
    entry_expr,
    exit_expr,
    strategy_name,
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
    heatmap_metric,
    heatmap_out,
    json_out,
):
    """Grid-search 1-2 parameters and show how metrics vary."""
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

    grid = json.loads(param_grid)

    cfg = BacktestConfig(
        market=market,
        as_of=end_date,
        hold=int(hold),
        top=int(top),
        strategy_name=strategy_name,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
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
    analyzer = SensitivityAnalyzer(
        base_cfg=cfg,
        fetcher=fetcher,
        param_grid=grid,
        start_date=start_date,
        end_date=end_date,
    )
    result = analyzer.run()
    print_sensitivity(result)
    if len(grid) == 2:
        save_heatmap(result, metric=heatmap_metric, out_path=heatmap_out)

    if json_out:
        out = {
            "param_names": list(result.param_names),
            "grid": [
                {"params": {k: v for k, v in zip(result.param_names, combo)}, "metrics": metrics}
                for combo, metrics in result.values
            ],
        }
        import pathlib

        pathlib.Path(json_out).write_text(json.dumps(out, indent=2, default=str))
