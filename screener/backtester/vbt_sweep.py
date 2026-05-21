"""Fast vectorbt parameter sweeps for strategy exploration.

This module is for **fast exploration**, not validation. Results may diverge
from ``backtest-rolling`` because the following are **not** modeled here:

- slot allocation / position sizing rules
- partial exits
- dividends
- custom slippage models (``slippage=0.0`` hard-coded)
- commissions / fees (``fees=0.0`` hard-coded)
- trailing stops, stop-loss, take-profit (custom engine supports all three)

Entries/exits are shifted by one bar and filled at the next bar's **open** to
match the custom engine's MOO (market-on-open) semantics — so there is no
same-bar look-ahead, but residual differences vs MOC-style exits still apply.

Always validate promising parameter combinations with ``backtest-rolling``
before drawing conclusions.

Strategy DSL note
-----------------
The ``sma_cross`` strategy here is a hand-coded vectorbt signal generator
(see ``sma_crossover_signals`` below); it is **not** the same as the
``ma_cross`` plugin in ``screener/strategies/plugins/ma_cross.py`` (which
uses EMA10/EMA20 via the Pine DSL). The two are intentionally decoupled so
this module can stay vectorbt-native for speed; do not assume parameter
results transfer between them.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any, Literal, cast

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.cli_common import DEFAULT_BENCHMARK
from screener.backtester.data import PriceFetcher, build_price_fetcher, tv_to_yf
from screener.universes import load_current_universe

DISCLAIMER = (
    "[yellow]Exploration only — approximations; validate with backtest-rolling.\n"
    "Not modeled: slot allocation, partial exits, dividends, custom slippage "
    "(slippage=0), fees (fees=0), trailing stops, stop-loss, take-profit.\n"
    "Fills: next-bar open (MOO match); residual MOC differences may apply.[/yellow]"
)

MetricName = Literal["sharpe", "total_return", "calmar"]
StrategyName = Literal["sma_cross"]

StrategyBuilder = Callable[
    [pd.DataFrame, int, int, int, Any],
    tuple[pd.DataFrame, pd.DataFrame],
]

INITIAL_CAPITAL_DEFAULT = 100_000.0


def _require_vectorbt() -> Any:
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise click.ClickException(
            "vectorbt is not installed. Install optional deps with: "
            "uv sync --extra vectorbt"
        ) from exc
    return vbt


def parse_int_list(raw: str, *, name: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise click.UsageError(f"--{name} requires at least one integer.")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise click.UsageError(f"--{name} expects comma-separated integers.") from exc


def _sma(close: pd.DataFrame, window: int, vbt: Any) -> pd.DataFrame:
    ma_out = vbt.MA.run(close, window=window).ma
    if isinstance(ma_out.columns, pd.MultiIndex):
        return ma_out.xs(window, axis=1, level="ma_window")
    return ma_out


def _fixed_hold_exits(entries: pd.DataFrame, hold: int) -> pd.DataFrame:
    if hold <= 0:
        return pd.DataFrame(False, index=entries.index, columns=entries.columns)
    arr = entries.to_numpy(dtype=bool)
    out = np.zeros_like(arr, dtype=bool)
    for col in range(arr.shape[1]):
        for entry_i in np.flatnonzero(arr[:, col]):
            exit_i = entry_i + hold
            if exit_i < arr.shape[0]:
                out[exit_i, col] = True
    return pd.DataFrame(out, index=entries.index, columns=entries.columns)


def sma_crossover_signals(
    close: pd.DataFrame,
    fast: int,
    slow: int,
    hold: int,
    vbt: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SMA crossover: enter when close crosses above slow SMA while close > fast SMA."""
    sma_fast = _sma(close, fast, vbt)
    sma_slow = _sma(close, slow, vbt)
    entries = close.vbt.crossed_above(sma_slow) & (close > sma_fast)
    cross_exits = close.vbt.crossed_below(sma_slow)
    if hold > 0:
        exits = cross_exits | _fixed_hold_exits(entries, hold)
    else:
        exits = cross_exits
    return entries.fillna(False), exits.fillna(False)


STRATEGY_BUILDERS: dict[StrategyName, StrategyBuilder] = {
    "sma_cross": sma_crossover_signals,
}


def iter_param_combos(
    fast_values: list[int],
    slow_values: list[int],
    hold_values: list[int],
) -> list[tuple[int, int, int]]:
    combos: list[tuple[int, int, int]] = []
    for fast, slow, hold in itertools.product(fast_values, slow_values, hold_values):
        if slow <= fast:
            continue
        combos.append((fast, slow, hold))
    return combos


def _build_column_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for sym in yf_symbols:
        frame = price_panel.get(sym)
        if frame is None or frame.empty or column not in frame.columns:
            continue
        col = frame[column].astype(float)
        col.index = pd.to_datetime(col.index).tz_localize(None).normalize()
        trimmed = col.loc[(col.index >= start) & (col.index <= end)]
        if trimmed.empty:
            continue
        series[sym] = trimmed
    if not series:
        raise ValueError(f"No usable {column} prices for the requested window.")
    panel = pd.DataFrame(series).sort_index()
    panel = panel.ffill()
    panel = panel.dropna(axis=1, how="any")
    return panel.dropna(how="all")


def build_close_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return _build_column_panel(
        price_panel, yf_symbols, column="close", start=start, end=end
    )


def build_open_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return _build_column_panel(
        price_panel, yf_symbols, column="open", start=start, end=end
    )


def _scalar_metric(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (pd.Series, pd.DataFrame)):
        flat = value.to_numpy().ravel()
        if flat.size == 0:
            return float("nan")
        return float(flat[0])
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.ravel()[0])


def run_combo_backtest(
    close: pd.DataFrame,
    fast: int,
    slow: int,
    hold: int,
    *,
    vbt: Any,
    open_: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
) -> dict[str, float | int]:
    entries, exits = STRATEGY_BUILDERS["sma_cross"](close, fast, slow, hold, vbt)
    # Match custom engine MOO semantics: signals on bar t fill at bar t+1 open.
    # Shift entries/exits by 1 bar and price the fill at the next bar's open.
    entries_shifted = entries.astype(bool).shift(1, fill_value=False).astype(bool)
    exits_shifted = exits.astype(bool).shift(1, fill_value=False).astype(bool)
    fill_price = open_ if open_ is not None else close
    pf = vbt.Portfolio.from_signals(
        close,
        entries_shifted,
        exits_shifted,
        price=fill_price,
        init_cash=float(initial_capital),
        fees=0.0,
        slippage=0.0,
        group_by=True,
        cash_sharing=True,
        freq="1D",
    )
    win_rate = _scalar_metric(pf.trades.win_rate())
    trade_count = pf.trades.count()
    if isinstance(trade_count, (pd.Series, pd.DataFrame)):
        trade_count = int(trade_count.to_numpy().sum())
    else:
        trade_count = int(trade_count)
    return {
        "fast": fast,
        "slow": slow,
        "hold": hold,
        "sharpe": _scalar_metric(pf.sharpe_ratio()),
        "total_return": _scalar_metric(pf.total_return()),
        "calmar": _scalar_metric(pf.calmar_ratio()),
        "max_drawdown": _scalar_metric(pf.max_drawdown()),
        "win_rate": win_rate,
        "trades": trade_count,
    }


def run_parameter_sweep(
    close: pd.DataFrame,
    *,
    fast_values: list[int],
    slow_values: list[int],
    hold_values: list[int],
    open_: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
) -> pd.DataFrame:
    vbt = _require_vectorbt()
    rows: list[dict[str, float | int]] = []
    for fast, slow, hold in iter_param_combos(fast_values, slow_values, hold_values):
        rows.append(
            run_combo_backtest(
                close,
                fast,
                slow,
                hold,
                vbt=vbt,
                open_=open_,
                initial_capital=initial_capital,
            )
        )
    if not rows:
        raise ValueError("No valid parameter combinations (require slow > fast).")
    return pd.DataFrame(rows)


def rank_results(df: pd.DataFrame, metric: MetricName) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Unknown metric: {metric}")
    sort_key = df[metric].replace([np.inf, -np.inf], np.nan)
    return (
        df.assign(_sort_key=sort_key)
        .sort_values("_sort_key", ascending=False, kind="stable", na_position="last")
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )


def print_results_table(
    df: pd.DataFrame,
    *,
    top_n: int,
    metric: MetricName,
    console: Console | None = None,
) -> None:
    out = console or Console()
    out.print(DISCLAIMER)
    table = Table(title=f"Top {top_n} by {metric}")
    for col in [
        "fast",
        "slow",
        "hold",
        "sharpe",
        "total_return",
        "calmar",
        "max_drawdown",
        "win_rate",
        "trades",
    ]:
        table.add_column(col)
    for _, row in df.head(top_n).iterrows():
        table.add_row(
            str(int(row["fast"])),
            str(int(row["slow"])),
            str(int(row["hold"])),
            f"{row['sharpe']:.3f}" if np.isfinite(row["sharpe"]) else "n/a",
            f"{row['total_return'] * 100:+.2f}%",
            f"{row['calmar']:.3f}" if np.isfinite(row["calmar"]) else "n/a",
            f"{row['max_drawdown'] * 100:+.2f}%",
            f"{row['win_rate'] * 100:.1f}%" if np.isfinite(row["win_rate"]) else "n/a",
            str(int(row["trades"])),
        )
    out.print(table)


@click.command(name="vbt-sweep")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    help="Market to backtest.",
)
@click.option(
    "--start", "start_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--end", "end_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--years",
    type=int,
    default=2,
    show_default=True,
    help="Trailing calendar years when --start is omitted.",
)
@click.option(
    "--universe",
    type=click.Choice(["sp500", "nifty50"]),
    default=None,
    help="Current index universe. Defaults to sp500 for US and nifty50 for India.",
)
@click.option(
    "--no-universe-cache",
    is_flag=True,
    default=False,
    help="Force live constituent refresh instead of today's cache.",
)
@click.option("--tickers", default=None, help="Comma-separated ticker list.")
@click.option(
    "--universe-file", default=None, help="Path to newline-separated ticker file."
)
@click.option(
    "--fast",
    default="10,20,50",
    show_default=True,
    help="Comma-separated fast SMA windows.",
)
@click.option(
    "--slow",
    default="50,100,200",
    show_default=True,
    help="Comma-separated slow SMA windows (must be > fast).",
)
@click.option(
    "--hold",
    default="0",
    show_default=True,
    help="Comma-separated fixed hold lengths in bars; 0 disables fixed hold.",
)
@click.option(
    "--top", type=int, default=10, show_default=True, help="Print top N rows."
)
@click.option(
    "--csv", "output_csv", is_flag=True, help="Emit full results as CSV on stdout."
)
@click.option(
    "--metric",
    type=click.Choice(["sharpe", "total_return", "calmar"]),
    default="sharpe",
    show_default=True,
    help="Metric used to rank combinations.",
)
def vbt_sweep(
    market: str,
    start_arg: datetime | None,
    end_arg: datetime | None,
    years: int,
    universe: str | None,
    no_universe_cache: bool,
    tickers: str | None,
    universe_file: str | None,
    fast: str,
    slow: str,
    hold: str,
    top: int,
    output_csv: bool,
    metric: str,
) -> None:
    """Fast vectorbt grid search for exploration (not validation).

    Approximate results — slot allocation, partial exits, dividends, and custom
    slippage are not modeled. Validate winners with backtest-rolling.
    """
    fast_values = parse_int_list(fast, name="fast")
    slow_values = parse_int_list(slow, name="slow")
    hold_values = parse_int_list(hold, name="hold")

    end_date = (
        end_arg.date() if isinstance(end_arg, datetime) else (end_arg or date.today())
    )
    start_date = (
        start_arg.date()
        if isinstance(start_arg, datetime)
        else (start_arg or (end_date - timedelta(days=365 * int(years))))
    )
    if end_date < start_date:
        raise click.UsageError("--end must be on or after --start.")

    bench = DEFAULT_BENCHMARK.get(market, "SPY")
    console = Console()

    tv_symbols: list[str]
    universe_note: str | None = None
    if tickers:
        tv_symbols = [t.strip() for t in tickers.split(",") if t.strip()]
    elif universe_file:
        from pathlib import Path

        content = Path(universe_file).read_text()
        tv_symbols = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        resolved_universe = universe or ("nifty50" if market == "india" else "sp500")
        loaded = load_current_universe(
            resolved_universe,
            as_of=end_date,
            use_cache=not no_universe_cache,
        )
        tv_symbols = list(loaded.symbols)
        universe_note = (
            f"{loaded.name}: {len(loaded.symbols)} symbols from {loaded.source}; "
            f"cache={loaded.cached_path}"
        )

    if not tv_symbols:
        raise click.UsageError("No tickers resolved for the sweep.")

    yf_by_tv = {tv: tv_to_yf(tv, market) for tv in tv_symbols}
    yf_symbols = list(dict.fromkeys(list(yf_by_tv.values()) + [bench]))
    max_slow = max(slow_values)
    warmup_days = max(max_slow * 3 + 30, 90)
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)).date()
    fetch_end = end_date

    fetcher: PriceFetcher = click.get_current_context().obj or build_price_fetcher()
    price_panel = fetcher.fetch(yf_symbols, fetch_start, fetch_end)
    equity_symbols = [yf_by_tv[tv] for tv in tv_symbols if yf_by_tv[tv] in price_panel]
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    close = build_close_panel(
        price_panel,
        equity_symbols,
        start=start_ts,
        end=end_ts,
    )
    try:
        open_panel = build_open_panel(
            price_panel,
            equity_symbols,
            start=start_ts,
            end=end_ts,
        )
        # Align open panel to the same columns/rows as close (handles drop-na divergence).
        open_panel = open_panel.reindex(index=close.index, columns=close.columns)
    except ValueError:
        open_panel = None

    results = run_parameter_sweep(
        close,
        fast_values=fast_values,
        slow_values=slow_values,
        hold_values=hold_values,
        open_=open_panel,
        initial_capital=INITIAL_CAPITAL_DEFAULT,
    )
    ranked = rank_results(results, cast(MetricName, metric))

    if output_csv:
        click.echo(ranked.to_csv(index=False))
        return

    console.print(
        f"[dim]Window: {start_date.isoformat()} to {end_date.isoformat()}  "
        f"symbols={close.shape[1]}  combos={len(results)}  "
        f"capital={INITIAL_CAPITAL_DEFAULT:,.0f}  slippage=0[/dim]"
    )
    if universe_note:
        console.print(f"[dim]Universe: {universe_note}[/dim]")
    print_results_table(ranked, top_n=int(top), metric=cast(MetricName, metric))
