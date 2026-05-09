"""Click commands for backtester parameter optimization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

import click

from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.optimization.grid import grid_search
from screener.backtester.optimization.monte_carlo import simulate_monte_carlo
from screener.backtester.optimization.reporting import (
    print_grid_table,
    print_walk_forward_table,
    write_html_report,
    write_json_report,
)
from screener.backtester.optimization.walk_forward import walk_forward_optimize


DEFAULT_BENCHMARK = {"us": "SPY", "india": "^NSEI"}
DEFAULT_MIN_PRICE = {"us": 1.0, "india": 10.0}
DEFAULT_MIN_ADV = {"us": 1_000.0, "india": 100_000.0}
P = ParamSpec("P")
R = TypeVar("R")


def _parse_values(
    raw: str | None, cast: type = float, *, allow_none: bool = True
) -> list[Any]:
    if raw is None:
        return [None] if allow_none else []
    values: list[Any] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if allow_none and item.lower() in {"none", "null", "off"}:
            values.append(None)
        elif ":" in item:
            pieces = item.split(":")
            if len(pieces) not in {2, 3}:
                raise click.UsageError(f"Invalid range {item!r}; use start:end[:step].")
            start = float(pieces[0])
            end = float(pieces[1])
            step = float(pieces[2]) if len(pieces) == 3 else 1.0
            if step <= 0:
                raise click.UsageError("Range step must be positive.")
            current = start
            while current <= end + step / 1_000_000:
                values.append(cast(current))
                current += step
        else:
            values.append(cast(item))
    return values


def _parameter_grid(
    stop_loss, take_profit, trailing_stop, hold
) -> dict[str, list[Any]]:
    grid = {
        "stop_loss": _parse_values(stop_loss),
        "take_profit": _parse_values(take_profit),
        "trailing_stop": _parse_values(trailing_stop),
        "hold": _parse_values(hold, int, allow_none=False),
    }
    return {key: value for key, value in grid.items() if value}


def _base_config(
    *,
    market,
    end_date,
    hold,
    top,
    entry_expr,
    exit_expr,
    strategy_name,
    tickers,
    universe_file,
    max_universe,
    stop_loss,
    take_profit,
    trailing_stop,
    slippage_bps,
    commission_bps,
    initial_capital,
    benchmark,
    min_price,
    min_avg_dollar_volume,
    adv_window,
):
    from screener.backtester.strategies import resolve_strategy

    if strategy_name:
        strategy = resolve_strategy(strategy_name)
        entry_expr = entry_expr or strategy.entry
        exit_expr = exit_expr or strategy.exit
    if not entry_expr:
        raise click.UsageError("--entry or --strategy is required.")
    ticker_tuple = (
        tuple(t.strip() for t in tickers.split(",") if t.strip()) if tickers else None
    )
    resolved_min_price = (
        DEFAULT_MIN_PRICE.get(market) if min_price is None else min_price
    )
    resolved_min_adv = (
        DEFAULT_MIN_ADV.get(market)
        if min_avg_dollar_volume is None
        else min_avg_dollar_volume
    )
    if resolved_min_price == 0:
        resolved_min_price = None
    if resolved_min_adv == 0:
        resolved_min_adv = None
    if not ticker_tuple and not universe_file:
        raise click.UsageError(
            "Pass --tickers or --universe-file for optimization runs."
        )
    return BacktestConfig(
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
        benchmark=benchmark or DEFAULT_BENCHMARK.get(market, "SPY"),
        tickers=ticker_tuple,
        universe_file=universe_file,
        max_universe=int(max_universe),
        min_price=resolved_min_price,
        min_avg_dollar_volume=resolved_min_adv,
        avg_dollar_volume_window=int(adv_window),
        reinvest=True,
    )


def _resolve_dates(start_arg, end_arg, years) -> tuple[date, date]:
    end_date = (
        end_arg.date() if isinstance(end_arg, datetime) else (end_arg or date.today())
    )
    start_date = (
        start_arg.date()
        if isinstance(start_arg, datetime)
        else (start_arg or (end_date - timedelta(days=365 * int(years))))
    )
    return start_date, end_date


def _fetcher():
    from screener.backtester.data import build_price_fetcher

    return click.get_current_context().obj or build_price_fetcher()


@click.group(name="optimize")
def optimize() -> None:
    """Optimize and validate backtest parameters."""


def _common_options(fn: Callable[P, R]) -> Callable[P, R]:
    options = [
        click.option(
            "-m", "--market", type=click.Choice(["us", "india"]), default="us"
        ),
        click.option(
            "--start",
            "start_arg",
            type=click.DateTime(formats=["%Y-%m-%d"]),
            default=None,
        ),
        click.option(
            "--end", "end_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
        ),
        click.option("--years", type=int, default=1, show_default=True),
        click.option("--top", type=int, default=10, show_default=True),
        click.option("--entry", "entry_expr", default=None),
        click.option("--exit", "exit_expr", default=None),
        click.option("--strategy", "strategy_name", default=None),
        click.option("--tickers", default=None),
        click.option("--universe-file", default=None),
        click.option("--max-universe", type=int, default=200, show_default=True),
        click.option("--stop-loss", default="none,0.05,0.08,0.10"),
        click.option("--take-profit", default="none,0.10,0.15,0.20"),
        click.option("--trailing-stop", default="none,0.08,0.10"),
        click.option("--hold", default="10,20,30"),
        click.option("--slippage-bps", type=float, default=0.0),
        click.option("--commission-bps", type=float, default=0.0),
        click.option("--initial-capital", type=float, default=100_000.0),
        click.option("--benchmark", default=None),
        click.option("--min-price", type=float, default=None),
        click.option("--min-avg-dollar-volume", type=float, default=None),
        click.option("--adv-window", type=int, default=20),
        click.option(
            "--metric",
            type=click.Choice(
                [
                    "sharpe",
                    "profit_factor",
                    "risk_adjusted_return",
                    "calmar",
                    "total_return",
                ]
            ),
            default="sharpe",
        ),
        click.option("--min-trades", type=int, default=1, show_default=True),
        click.option("--top-n", type=int, default=10, show_default=True),
        click.option("--workers", type=int, default=1, show_default=True),
        click.option(
            "--cache", "cache_path", type=click.Path(path_type=Path), default=None
        ),
        click.option(
            "--json", "json_path", type=click.Path(path_type=Path), default=None
        ),
        click.option(
            "--html", "html_path", type=click.Path(path_type=Path), default=None
        ),
    ]
    for option in reversed(options):
        fn = option(fn)
    return fn


@optimize.command(name="grid")
@_common_options
def optimize_grid(**kwargs) -> None:
    """Run exhaustive grid search over parameter ranges."""
    start_date, end_date = _resolve_dates(
        kwargs.pop("start_arg"), kwargs.pop("end_arg"), kwargs.pop("years")
    )
    parameter_grid = _parameter_grid(
        kwargs["stop_loss"],
        kwargs["take_profit"],
        kwargs["trailing_stop"],
        kwargs["hold"],
    )
    cfg = _base_config(
        market=kwargs["market"],
        end_date=end_date,
        hold=_parse_values(kwargs["hold"], int, allow_none=False)[0],
        top=kwargs["top"],
        entry_expr=kwargs["entry_expr"],
        exit_expr=kwargs["exit_expr"],
        strategy_name=kwargs["strategy_name"],
        tickers=kwargs["tickers"],
        universe_file=kwargs["universe_file"],
        max_universe=kwargs["max_universe"],
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=kwargs["slippage_bps"],
        commission_bps=kwargs["commission_bps"],
        initial_capital=kwargs["initial_capital"],
        benchmark=kwargs["benchmark"],
        min_price=kwargs["min_price"],
        min_avg_dollar_volume=kwargs["min_avg_dollar_volume"],
        adv_window=kwargs["adv_window"],
    )
    results = grid_search(
        cfg,
        _fetcher(),
        parameter_grid,
        metric=kwargs["metric"],
        top_n=kwargs["top_n"],
        min_trades=kwargs["min_trades"],
        max_workers=kwargs["workers"],
        cache_path=kwargs["cache_path"],
        runner="rolling",
        start_date=start_date,
        end_date=end_date,
    )
    print_grid_table(results)
    payload = [asdict(result) for result in results]
    if kwargs["json_path"]:
        write_json_report(payload, kwargs["json_path"])
    if kwargs["html_path"]:
        write_html_report(payload, kwargs["html_path"], "Grid Search Report")


@optimize.command(name="walk-forward")
@_common_options
@click.option("--train-days", type=int, default=252, show_default=True)
@click.option("--test-days", type=int, default=63, show_default=True)
@click.option("--step-days", type=int, default=None)
def optimize_walk_forward(train_days, test_days, step_days, **kwargs) -> None:
    """Run rolling train/test walk-forward optimization."""
    start_date, end_date = _resolve_dates(
        kwargs.pop("start_arg"), kwargs.pop("end_arg"), kwargs.pop("years")
    )
    parameter_grid = _parameter_grid(
        kwargs["stop_loss"],
        kwargs["take_profit"],
        kwargs["trailing_stop"],
        kwargs["hold"],
    )
    cfg = _base_config(
        market=kwargs["market"],
        end_date=end_date,
        hold=_parse_values(kwargs["hold"], int, allow_none=False)[0],
        top=kwargs["top"],
        entry_expr=kwargs["entry_expr"],
        exit_expr=kwargs["exit_expr"],
        strategy_name=kwargs["strategy_name"],
        tickers=kwargs["tickers"],
        universe_file=kwargs["universe_file"],
        max_universe=kwargs["max_universe"],
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=kwargs["slippage_bps"],
        commission_bps=kwargs["commission_bps"],
        initial_capital=kwargs["initial_capital"],
        benchmark=kwargs["benchmark"],
        min_price=kwargs["min_price"],
        min_avg_dollar_volume=kwargs["min_avg_dollar_volume"],
        adv_window=kwargs["adv_window"],
    )
    summary = walk_forward_optimize(
        cfg,
        _fetcher(),
        parameter_grid,
        start_date=start_date,
        end_date=end_date,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        metric=kwargs["metric"],
        min_trades=kwargs["min_trades"],
        max_workers=kwargs["workers"],
        cache_path=kwargs["cache_path"],
    )
    print_walk_forward_table(summary)
    payload = asdict(summary)
    if kwargs["json_path"]:
        write_json_report(payload, kwargs["json_path"])
    if kwargs["html_path"]:
        write_html_report(payload, kwargs["html_path"], "Walk-Forward Report")


def _load_trades(path: Path) -> list[Trade]:
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        rows = data.get("trades", data) if isinstance(data, dict) else data
    else:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    trades: list[Trade] = []
    for idx, row in enumerate(rows, start=1):
        trades.append(
            Trade(
                ticker=str(row.get("ticker", "")),
                rank=int(row.get("rank") or idx),
                signal_date=date.fromisoformat(
                    str(row.get("signal_date") or row.get("entry_date"))
                ),
                entry_date=date.fromisoformat(str(row.get("entry_date"))),
                entry_price=float(row.get("entry_price") or 0.0),
                exit_date=date.fromisoformat(str(row.get("exit_date"))),
                exit_price=float(row.get("exit_price") or 0.0),
                exit_reason=str(row.get("exit_reason") or "time"),
                shares=float(row.get("shares") or 0.0),
                entry_cost=float(row.get("entry_cost") or 0.0),
                exit_value=float(row.get("exit_value") or 0.0),
                pnl=float(row.get("pnl") or 0.0),
                return_pct=float(row.get("return_pct") or 0.0),
            )
        )
    return trades


@optimize.command(name="validate")
@click.option(
    "--trades",
    "trades_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option("--iterations", type=int, default=5000, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--initial-capital", type=float, default=100_000.0, show_default=True)
@click.option("--ruin-threshold", type=float, default=0.5, show_default=True)
@click.option("--json", "json_path", type=click.Path(path_type=Path), default=None)
def optimize_validate(
    trades_path, iterations, seed, initial_capital, ruin_threshold, json_path
) -> None:
    """Run Monte Carlo stress testing on an existing trade ledger."""
    from rich.console import Console
    from rich.table import Table

    result = simulate_monte_carlo(
        _load_trades(trades_path),
        iterations=iterations,
        seed=seed,
        initial_capital=initial_capital,
        ruin_threshold=ruin_threshold,
    )
    table = Table(title="Monte Carlo Validation", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in asdict(result).items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    Console().print(table)
    if json_path:
        write_json_report(asdict(result), json_path)
