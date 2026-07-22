from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any, cast

import click
import pandas as pd
from rich.console import Console

from screener.backtester.data import PriceFetcher, build_price_fetcher, tv_to_yf
from screener.backtester.optimization.walk_forward import generate_walk_forward_windows
from screener.backtester.vbt.config import (
    COMMISSION_BPS_DEFAULT,
    COST_MODEL_DEFAULT,
    DEFAULT_BBANDS_STD,
    DEFAULT_BBANDS_WINDOWS,
    DEFAULT_BREAKOUT_WINDOWS,
    DEFAULT_KELTNER_MULT,
    DEFAULT_KELTNER_WINDOWS,
    DEFAULT_MACD_SIGNAL,
    DEFAULT_MACD_SLOW,
    DEFAULT_OBV_EMA_WINDOWS,
    DEFAULT_RSI_PERIOD,
    DEFAULT_RSI_THRESHOLDS,
    DEFAULT_SUPERTREND_MULT,
    DEFAULT_SUPERTREND_PERIODS,
    DEFAULT_VOL_MA_WINDOW,
    HL_INDICATORS,
    INITIAL_CAPITAL_DEFAULT,
    MetricName,
    VALID_INDICATORS,
    VOLUME_INDICATORS,
    WALK_FORWARD_DAYS_PER_MONTH,
)
from screener.backtester.vbt.panels import (
    build_close_panel,
    build_high_panel,
    build_low_panel,
    build_open_panel,
    build_volume_panel,
)
from screener.backtester.vbt.render import (
    print_results_table,
    print_walk_forward_sweep_table,
)
from screener.backtester.vbt.sweep import (
    rank_results,
    run_parameter_sweep as _run_parameter_sweep,
    _require_vectorbt,
    _portfolio_chunk_metrics,
)
from screener.backtester.vbt.walk_forward import (
    parse_walk_forward,
    run_walk_forward_sweep as _run_walk_forward_sweep,
)
from screener.markets import get_market, get_price_fetcher, market_option
from screener.universes import available_universes, load_current_universe


def parse_int_list(raw: str, *, name: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise click.UsageError(f"--{name} requires at least one integer.")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise click.UsageError(f"--{name} expects comma-separated integers.") from exc


def parse_indicator_list(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise click.UsageError("--indicator requires at least one value.")
    if parts == ["all"]:
        return list(VALID_INDICATORS)
    bad = [p for p in parts if p not in VALID_INDICATORS]
    if bad:
        raise click.UsageError(
            f"--indicator values must be in {VALID_INDICATORS} (or 'all'); got {bad!r}"
        )
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def run_parameter_sweep(
    close: pd.DataFrame,
    *,
    fast_values: list[int],
    slow_values: list[int],
    hold_values: list[int],
    indicators: list[str] | None = None,
    breakout_windows: list[int] | None = None,
    bbands_windows: list[int] | None = None,
    supertrend_periods: list[int] | None = None,
    keltner_windows: list[int] | None = None,
    rsi_thresholds: list[int] | None = None,
    obv_ema_windows: list[int] | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    open_: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
    chunk_size: int | None = None,
    cost_model: str = COST_MODEL_DEFAULT,
    commission_bps: float = COMMISSION_BPS_DEFAULT,
) -> pd.DataFrame:
    """Compatibility wrapper for the extracted sweep implementation."""
    return _run_parameter_sweep(
        close,
        fast_values=fast_values,
        slow_values=slow_values,
        hold_values=hold_values,
        indicators=indicators,
        breakout_windows=breakout_windows,
        bbands_windows=bbands_windows,
        supertrend_periods=supertrend_periods,
        keltner_windows=keltner_windows,
        rsi_thresholds=rsi_thresholds,
        obv_ema_windows=obv_ema_windows,
        high=high,
        low=low,
        volume=volume,
        open_=open_,
        initial_capital=initial_capital,
        chunk_size=chunk_size,
        cost_model=cost_model,
        commission_bps=commission_bps,
        require_vectorbt_fn=_require_vectorbt,
        portfolio_chunk_metrics_fn=_portfolio_chunk_metrics,
    )


def run_walk_forward_sweep(
    close: pd.DataFrame,
    *,
    windows: list,
    metric: MetricName,
    grid: dict[str, Any],
    open_: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
    sweep_fn: Callable[..., pd.DataFrame] | None = None,
):
    """Compatibility wrapper that honors monkeypatches on this module."""
    import screener.backtester.vbt_sweep as vs

    return _run_walk_forward_sweep(
        close,
        windows=windows,
        metric=metric,
        grid=grid,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        initial_capital=initial_capital,
        sweep_fn=sweep_fn if sweep_fn is not None else vs.run_parameter_sweep,
    )


def _max_indicator_lookback(
    *,
    indicators: list[str],
    slow_values: list[int],
    breakout_windows: list[int],
    bbands_windows: list[int],
    supertrend_periods: list[int],
    keltner_windows: list[int],
    obv_ema_windows: list[int],
) -> int:
    candidate_lookbacks: list[int] = []
    if any(ind in ("sma", "ema", "sma_rsi") for ind in indicators):
        candidate_lookbacks.append(max(slow_values))
    if any(ind in ("breakout", "vol_breakout", "breakout_rsi") for ind in indicators):
        candidate_lookbacks.append(max(breakout_windows))
    if "bbands" in indicators:
        candidate_lookbacks.append(max(bbands_windows))
    if "supertrend" in indicators:
        candidate_lookbacks.append(max(supertrend_periods))
    if "keltner" in indicators:
        candidate_lookbacks.append(max(keltner_windows))
    if "obv_trend" in indicators:
        candidate_lookbacks.append(max(obv_ema_windows))
    if "macd" in indicators:
        candidate_lookbacks.append(DEFAULT_MACD_SLOW + DEFAULT_MACD_SIGNAL)
    if "rsi" in indicators:
        candidate_lookbacks.append(DEFAULT_RSI_PERIOD * 3)
    if "vol_breakout" in indicators:
        candidate_lookbacks.append(DEFAULT_VOL_MA_WINDOW)
    return max(candidate_lookbacks) if candidate_lookbacks else max(slow_values)


def _optional_aligned_panel(
    builder: Callable[..., pd.DataFrame],
    price_panel: dict[str, pd.DataFrame],
    equity_symbols: list[str],
    close: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame | None:
    try:
        panel = builder(price_panel, equity_symbols, start=start_ts, end=end_ts)
        return panel.reindex(index=close.index, columns=close.columns)
    except ValueError:
        return None


def _run_cli_walk_forward(
    *,
    close: pd.DataFrame,
    start_date: date,
    end_date: date,
    train_months: int,
    test_months: int,
    indicators: list[str],
    metric: str,
    output_csv: bool,
    universe_note: str | None,
    console: Console,
    sweep_grid: dict[str, Any],
    open_panel: pd.DataFrame | None,
    high_panel: pd.DataFrame | None,
    low_panel: pd.DataFrame | None,
    volume_panel: pd.DataFrame | None,
) -> None:
    wf_windows = generate_walk_forward_windows(
        start_date,
        end_date,
        train_days=train_months * WALK_FORWARD_DAYS_PER_MONTH,
        test_days=test_months * WALK_FORWARD_DAYS_PER_MONTH,
    )
    if not wf_windows:
        raise click.UsageError(
            "--walk-forward windows do not fit in the date range; use "
            "smaller TRAIN/TEST months or a wider --start/--end."
        )
    import screener.backtester.vbt_sweep as vs

    summary = vs.run_walk_forward_sweep(
        close,
        windows=wf_windows,
        metric=cast(MetricName, metric),
        grid=sweep_grid,
        open_=open_panel,
        high=high_panel,
        low=low_panel,
        volume=volume_panel,
        initial_capital=INITIAL_CAPITAL_DEFAULT,
    )
    if output_csv:
        click.echo(summary.windows.to_csv(index=False))
        return
    console.print(
        f"[dim]Walk-forward: {start_date.isoformat()} to "
        f"{end_date.isoformat()}  train={train_months}mo "
        f"test={test_months}mo windows={len(wf_windows)}  "
        f"indicators={','.join(indicators)}  symbols={close.shape[1]}  "
        f"capital={INITIAL_CAPITAL_DEFAULT:,.0f}  slippage=0[/dim]"
    )
    if universe_note:
        console.print(f"[dim]Universe: {universe_note}[/dim]")
    print_walk_forward_sweep_table(summary, console=console)


@click.command(name="vbt-sweep")
@market_option(
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
    type=click.Choice(list(available_universes())),
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
    "--indicator",
    "indicator_arg",
    default="sma",
    show_default=True,
    help=(
        "Comma-separated subset of "
        f"{','.join(VALID_INDICATORS)} (or 'all'). EMA/SMA/MACD/sma_rsi use "
        "--fast/--slow/--hold; window-based indicators use their own --*-windows."
    ),
)
@click.option(
    "--fast",
    default="10,20,50",
    show_default=True,
    help="Comma-separated fast SMA/EMA windows.",
)
@click.option(
    "--slow",
    default="50,100,200",
    show_default=True,
    help="Comma-separated slow SMA/EMA windows (must be > fast).",
)
@click.option(
    "--hold",
    default="0",
    show_default=True,
    help="Comma-separated fixed hold lengths in bars; 0 disables fixed hold.",
)
@click.option(
    "--breakout-windows",
    "breakout_windows_arg",
    default=",".join(str(w) for w in DEFAULT_BREAKOUT_WINDOWS),
    show_default=True,
    help="Donchian breakout windows (used by breakout, vol_breakout, breakout_rsi).",
)
@click.option(
    "--bbands-windows",
    "bbands_windows_arg",
    default=",".join(str(w) for w in DEFAULT_BBANDS_WINDOWS),
    show_default=True,
    help=f"EMA Bollinger band windows (std fixed at {DEFAULT_BBANDS_STD}).",
)
@click.option(
    "--supertrend-periods",
    "supertrend_periods_arg",
    default=",".join(str(p) for p in DEFAULT_SUPERTREND_PERIODS),
    show_default=True,
    help=f"Supertrend ATR periods (multiplier fixed at {DEFAULT_SUPERTREND_MULT}).",
)
@click.option(
    "--keltner-windows",
    "keltner_windows_arg",
    default=",".join(str(w) for w in DEFAULT_KELTNER_WINDOWS),
    show_default=True,
    help=f"Keltner channel windows (multiplier fixed at {DEFAULT_KELTNER_MULT}).",
)
@click.option(
    "--rsi-thresholds",
    "rsi_thresholds_arg",
    default=",".join(str(t) for t in DEFAULT_RSI_THRESHOLDS),
    show_default=True,
    help=f"RSI entry thresholds (period fixed at {DEFAULT_RSI_PERIOD}).",
)
@click.option(
    "--obv-ema-windows",
    "obv_ema_windows_arg",
    default=",".join(str(w) for w in DEFAULT_OBV_EMA_WINDOWS),
    show_default=True,
    help="OBV smoothing EMA windows for obv_trend.",
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
@click.option(
    "--cost-model",
    type=click.Choice(["flat", "india", "us_vested"]),
    default=COST_MODEL_DEFAULT,
    show_default=True,
    help=(
        "Statutory fee model used to approximate vectorbt's per-side fees. "
        "'flat' applies --commission-bps on every fill (default 0 → fees=0)."
    ),
)
@click.option(
    "--commission-bps",
    type=float,
    default=COMMISSION_BPS_DEFAULT,
    show_default=True,
    help="Commission per fill (bps) when --cost-model=flat.",
)
@click.option(
    "--walk-forward",
    "walk_forward_arg",
    default=None,
    help=(
        "Walk-forward validation as TRAIN_MONTHS:TEST_MONTHS (e.g. 12:3): run "
        "the sweep grid in each training window, pick the in-sample winner by "
        "--metric, and evaluate it out-of-sample on the test window. Months "
        "are approximated as 30 calendar days; windows step by the test "
        "length. With --csv, emits per-window rows."
    ),
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
    indicator_arg: str,
    fast: str,
    slow: str,
    hold: str,
    breakout_windows_arg: str,
    bbands_windows_arg: str,
    supertrend_periods_arg: str,
    keltner_windows_arg: str,
    rsi_thresholds_arg: str,
    obv_ema_windows_arg: str,
    top: int,
    output_csv: bool,
    metric: str,
    cost_model: str,
    commission_bps: float,
    walk_forward_arg: str | None,
) -> None:
    """Fast vectorbt grid search for exploration (not validation).

    Always validate promising parameter combinations with backtest-rolling
    before drawing conclusions.

    This module is for **fast exploration**, not validation. Results may diverge
    from ``backtest-rolling`` because the following are **not** modeled here:
    """
    walk_forward_spec = (
        parse_walk_forward(walk_forward_arg) if walk_forward_arg else None
    )
    indicators = parse_indicator_list(indicator_arg)
    fast_values = parse_int_list(fast, name="fast")
    slow_values = parse_int_list(slow, name="slow")
    hold_values = parse_int_list(hold, name="hold")
    breakout_windows = parse_int_list(breakout_windows_arg, name="breakout-windows")
    bbands_windows = parse_int_list(bbands_windows_arg, name="bbands-windows")
    supertrend_periods = parse_int_list(
        supertrend_periods_arg, name="supertrend-periods"
    )
    keltner_windows = parse_int_list(keltner_windows_arg, name="keltner-windows")
    rsi_thresholds = parse_int_list(rsi_thresholds_arg, name="rsi-thresholds")
    obv_ema_windows = parse_int_list(obv_ema_windows_arg, name="obv-ema-windows")

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

    market_meta = get_market(market)
    bench = market_meta.benchmark
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
        resolved_universe = universe or market_meta.default_universe
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
    max_lookback = _max_indicator_lookback(
        indicators=indicators,
        slow_values=slow_values,
        breakout_windows=breakout_windows,
        bbands_windows=bbands_windows,
        supertrend_periods=supertrend_periods,
        keltner_windows=keltner_windows,
        obv_ema_windows=obv_ema_windows,
    )
    warmup_days = max(max_lookback * 3 + 30, 90)
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)).date()
    fetch_end = end_date

    fetcher: PriceFetcher = get_price_fetcher(
        click.get_current_context().obj, builder=build_price_fetcher
    )
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
    open_panel = _optional_aligned_panel(
        build_open_panel, price_panel, equity_symbols, close, start_ts, end_ts
    )

    high_panel: pd.DataFrame | None = None
    low_panel: pd.DataFrame | None = None
    volume_panel: pd.DataFrame | None = None
    if any(ind in HL_INDICATORS for ind in indicators):
        high_panel = _optional_aligned_panel(
            build_high_panel, price_panel, equity_symbols, close, start_ts, end_ts
        )
        low_panel = _optional_aligned_panel(
            build_low_panel, price_panel, equity_symbols, close, start_ts, end_ts
        )
    if any(ind in VOLUME_INDICATORS for ind in indicators):
        volume_panel = _optional_aligned_panel(
            build_volume_panel, price_panel, equity_symbols, close, start_ts, end_ts
        )

    sweep_grid: dict[str, Any] = {
        "fast_values": fast_values,
        "slow_values": slow_values,
        "hold_values": hold_values,
        "indicators": indicators,
        "breakout_windows": breakout_windows,
        "bbands_windows": bbands_windows,
        "supertrend_periods": supertrend_periods,
        "keltner_windows": keltner_windows,
        "rsi_thresholds": rsi_thresholds,
        "obv_ema_windows": obv_ema_windows,
        "cost_model": cost_model,
        "commission_bps": float(commission_bps),
    }

    if walk_forward_spec is not None:
        _run_cli_walk_forward(
            close=close,
            start_date=start_date,
            end_date=end_date,
            train_months=walk_forward_spec[0],
            test_months=walk_forward_spec[1],
            indicators=indicators,
            metric=metric,
            output_csv=output_csv,
            universe_note=universe_note,
            console=console,
            sweep_grid=sweep_grid,
            open_panel=open_panel,
            high_panel=high_panel,
            low_panel=low_panel,
            volume_panel=volume_panel,
        )
        return

    import screener.backtester.vbt_sweep as vs

    results = vs.run_parameter_sweep(
        close,
        high=high_panel,
        low=low_panel,
        volume=volume_panel,
        open_=open_panel,
        initial_capital=INITIAL_CAPITAL_DEFAULT,
        **sweep_grid,
    )
    ranked = rank_results(results, cast(MetricName, metric))

    if output_csv:
        click.echo(ranked.to_csv(index=False))
        return

    console.print(
        f"[dim]Window: {start_date.isoformat()} to {end_date.isoformat()}  "
        f"indicators={','.join(indicators)}  symbols={close.shape[1]}  "
        f"combos={len(results)}  capital={INITIAL_CAPITAL_DEFAULT:,.0f}  "
        f"slippage=0[/dim]"
    )
    if universe_note:
        console.print(f"[dim]Universe: {universe_note}[/dim]")
    print_results_table(ranked, top_n=int(top), metric=cast(MetricName, metric))
