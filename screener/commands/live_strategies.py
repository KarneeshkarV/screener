"""Live screens for the winning sweep strategies.

These commands evaluate the parameter-sweep winners on the latest available
trading day and list the tickers that are signaling entries (or currently
holding open positions):

  - ``vol-breakout-live``: Donchian N-day breakout with volume confirmation.
    Sweep winner on US SP500 was window=100, hold=15.
  - ``obv-trend-live``: OBV crosses its own EMA. Sweep winner on India
    Nifty50 was ema_window=20, hold=0 (exit on cross-down).

The signal logic exactly matches the vectorized panel logic in vbt_sweep
(Donchian uses the rolling max shifted by one; OBV uses cumulative signed
volume), so a "fresh entry today" here corresponds to the same bar that the
backtester would have entered on.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.data import (
    PriceFetcher,
    build_price_fetcher,
    tv_to_yf,
)
from screener.backtester.vbt_sweep import (
    DEFAULT_VOL_MA_WINDOW,
    DEFAULT_VOL_MULTIPLIER,
    _obv,
    build_close_panel,
    build_volume_panel,
)
from screener.universes import load_current_universe


def _market_to_universe(market: str) -> str:
    return "sp500" if market == "us" else "nifty50"


def _crossed_above_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NaN-aware element-wise ``a`` crosses above ``b``.

    Mirrors ``vectorbt.generic.nb.crossed_above_nb`` so live signals agree with
    the sweep bit-for-bit.
    """
    prev_a = np.empty_like(a)
    prev_b = np.empty_like(b)
    prev_a[0] = np.nan
    prev_b[0] = np.nan
    prev_a[1:] = a[:-1]
    prev_b[1:] = b[:-1]
    return (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(prev_a)
        & np.isfinite(prev_b)
        & (prev_a <= prev_b)
        & (a > b)
    )


def _load_panels(
    market: str,
    as_of: date,
    *,
    lookback_days: int,
    fetcher: PriceFetcher,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    universe = load_current_universe(_market_to_universe(market), as_of=as_of)
    yf_by_tv = {tv: tv_to_yf(tv, market=market) for tv in universe.symbols}
    yf_symbols = list(dict.fromkeys(yf_by_tv.values()))
    fetch_start = as_of - timedelta(days=lookback_days * 2 + 30)
    panel = fetcher.fetch(yf_symbols, fetch_start, as_of)
    equity_symbols = [yf_by_tv[tv] for tv in universe.symbols if yf_by_tv[tv] in panel]
    start_ts = pd.Timestamp(fetch_start).normalize()
    end_ts = pd.Timestamp(as_of).normalize()
    close = build_close_panel(panel, equity_symbols, start=start_ts, end=end_ts)
    volume = build_volume_panel(
        panel, equity_symbols, start=start_ts, end=end_ts
    ).reindex(index=close.index, columns=close.columns)
    yf_to_tv = {yf: tv for tv, yf in yf_by_tv.items()}
    return close, volume, yf_to_tv


def _render_today_table(
    console: Console,
    *,
    title: str,
    rows: list[tuple[str, float]],
    limit: int,
) -> None:
    table = Table(title=title)
    table.add_column("Ticker")
    table.add_column("Close", justify="right")
    if not rows:
        table.add_row("(none)", "—")
    else:
        for sym, px in rows[:limit]:
            table.add_row(sym, f"{px:.2f}")
    console.print(table)


def _render_active_table(
    console: Console,
    *,
    title: str,
    rows: list[tuple[str, date, float, float, float, int]],
    limit: int,
) -> None:
    table = Table(title=title)
    table.add_column("Ticker")
    table.add_column("Entry")
    table.add_column("Entry Px", justify="right")
    table.add_column("Last Px", justify="right")
    table.add_column("PnL %", justify="right")
    table.add_column("Days Held", justify="right")
    if not rows:
        table.add_row("(none)", "—", "—", "—", "—", "—")
    else:
        for sym, ed, ep, lp, pnl, dh in rows[:limit]:
            table.add_row(
                sym, str(ed), f"{ep:.2f}", f"{lp:.2f}", f"{pnl:+.2f}", str(dh)
            )
    console.print(table)


@click.command(name="vol-breakout-live")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    show_default=True,
    help="Market to scan. Sweep winner is on US SP500.",
)
@click.option(
    "--as-of",
    "as_of_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Trading date to evaluate (default: today).",
)
@click.option(
    "--window",
    type=int,
    default=100,
    show_default=True,
    help="Donchian breakout lookback in trading days.",
)
@click.option(
    "--hold",
    type=int,
    default=15,
    show_default=True,
    help="Hold period for active-position window.",
)
@click.option(
    "--vol-ma",
    type=int,
    default=DEFAULT_VOL_MA_WINDOW,
    show_default=True,
    help="Volume moving-average window.",
)
@click.option(
    "--vol-mult",
    type=float,
    default=DEFAULT_VOL_MULTIPLIER,
    show_default=True,
    help="Volume must exceed this multiple of the volume MA.",
)
@click.option("-n", "--limit", type=int, default=30, show_default=True)
def vol_breakout_live(
    market: str,
    as_of_arg: datetime | None,
    window: int,
    hold: int,
    vol_ma: int,
    vol_mult: float,
    limit: int,
) -> None:
    """Donchian N-day high breakout confirmed by above-average volume.

    Sweep winner: ``--market us --window 100 --hold 15``.
    """
    as_of = as_of_arg.date() if isinstance(as_of_arg, datetime) else date.today()
    run_vol_breakout_live(
        market=market,
        as_of=as_of,
        window=window,
        hold=hold,
        vol_ma=vol_ma,
        vol_mult=vol_mult,
        limit=limit,
        fetcher=click.get_current_context().obj,
    )


def run_vol_breakout_live(
    *,
    market: str,
    as_of: date,
    window: int = 100,
    hold: int = 15,
    vol_ma: int = DEFAULT_VOL_MA_WINDOW,
    vol_mult: float = DEFAULT_VOL_MULTIPLIER,
    limit: int = 30,
    fetcher: PriceFetcher | None = None,
) -> None:
    """Run the vol-breakout live screen (no Click context required)."""
    console = Console()
    lookback = max(window, vol_ma) * 3 + 30
    fetcher = fetcher or build_price_fetcher()
    close, volume, yf_to_tv = _load_panels(
        market, as_of, lookback_days=lookback, fetcher=fetcher
    )

    rmax = close.rolling(window).max().shift(1)
    breakout = (close > rmax) & rmax.notna()
    vol_ma_panel = volume.rolling(vol_ma).mean()
    vol_ok = volume > (vol_mult * vol_ma_panel)
    entries = (breakout & vol_ok).to_numpy()

    today_idx = -1
    today_date = close.index[today_idx].date()
    today_mask = entries[today_idx]
    today_list = [
        (yf_to_tv.get(c, c), float(close.iloc[today_idx][c]))
        for j, c in enumerate(close.columns)
        if today_mask[j]
    ]
    today_list.sort(key=lambda r: r[1], reverse=True)

    n_rows = entries.shape[0]
    window_start = max(0, n_rows - hold)
    recent = entries[window_start:n_rows]
    active: list[tuple[str, date, float, float, float, int]] = []
    for j, c in enumerate(close.columns):
        local = np.where(recent[:, j])[0]
        if local.size == 0:
            continue
        last_local = local[-1]
        entry_row = window_start + last_local
        entry_dt = close.index[entry_row].date()
        entry_px = float(close.iloc[entry_row][c])
        last_px = float(close.iloc[today_idx][c])
        if not (np.isfinite(entry_px) and entry_px > 0 and np.isfinite(last_px)):
            continue
        pnl = (last_px / entry_px - 1.0) * 100.0
        days_held = (n_rows - 1) - entry_row
        active.append((yf_to_tv.get(c, c), entry_dt, entry_px, last_px, pnl, days_held))
    active.sort(key=lambda r: r[4], reverse=True)

    console.print(
        f"\n[bold]vol_breakout live — {market.upper()} — as of {today_date}[/bold]"
    )
    console.print(
        f"[dim]Universe: {len(close.columns)} symbols | window={window} "
        f"| hold={hold} | vol_ma={vol_ma} | vol_mult={vol_mult}[/dim]\n"
    )
    _render_today_table(
        console,
        title=f"Fresh entries on {today_date}",
        rows=today_list,
        limit=limit,
    )
    _render_active_table(
        console,
        title=f"Active positions within last {hold} trading days",
        rows=active,
        limit=limit,
    )


@click.command(name="obv-trend-live")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="india",
    show_default=True,
    help="Market to scan. Sweep winner is on India Nifty50.",
)
@click.option(
    "--as-of",
    "as_of_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Trading date to evaluate (default: today).",
)
@click.option(
    "--ema-window",
    type=int,
    default=20,
    show_default=True,
    help="EMA span applied to OBV for the trend filter.",
)
@click.option("-n", "--limit", type=int, default=30, show_default=True)
def obv_trend_live(
    market: str,
    as_of_arg: datetime | None,
    ema_window: int,
    limit: int,
) -> None:
    """OBV crosses above/below its EMA — flow-leads-price trend follower.

    Sweep winner: ``--market india --ema-window 20``.
    """
    as_of = as_of_arg.date() if isinstance(as_of_arg, datetime) else date.today()
    run_obv_trend_live(
        market=market,
        as_of=as_of,
        ema_window=ema_window,
        limit=limit,
        fetcher=click.get_current_context().obj,
    )


def run_obv_trend_live(
    *,
    market: str,
    as_of: date,
    ema_window: int = 20,
    limit: int = 30,
    fetcher: PriceFetcher | None = None,
) -> None:
    """Run the obv-trend live screen (no Click context required)."""
    console = Console()
    lookback = ema_window * 5 + 30
    fetcher = fetcher or build_price_fetcher()
    close, volume, yf_to_tv = _load_panels(
        market, as_of, lookback_days=lookback, fetcher=fetcher
    )

    obv_arr = _obv(close, volume)
    obv_df = pd.DataFrame(obv_arr, index=close.index, columns=close.columns)
    obv_ema_arr = obv_df.ewm(span=ema_window, adjust=False).mean().to_numpy(dtype=float)
    entries = _crossed_above_np(
        np.ascontiguousarray(obv_arr), np.ascontiguousarray(obv_ema_arr)
    )
    exits = _crossed_above_np(
        np.ascontiguousarray(obv_ema_arr), np.ascontiguousarray(obv_arr)
    )

    today_idx = -1
    today_date = close.index[today_idx].date()
    today_mask = entries[today_idx]
    today_list = [
        (yf_to_tv.get(c, c), float(close.iloc[today_idx][c]))
        for j, c in enumerate(close.columns)
        if today_mask[j]
    ]
    today_list.sort(key=lambda r: r[1], reverse=True)

    long_rows: list[tuple[str, date, float, float, float, int]] = []
    n_rows = entries.shape[0]
    for j, c in enumerate(close.columns):
        e_idx = np.where(entries[:, j])[0]
        x_idx = np.where(exits[:, j])[0]
        if e_idx.size == 0:
            continue
        last_e = int(e_idx[-1])
        last_x = int(x_idx[-1]) if x_idx.size else -1
        if last_e <= last_x:
            continue
        entry_dt = close.index[last_e].date()
        entry_px = float(close.iloc[last_e][c])
        last_px = float(close.iloc[today_idx][c])
        if not (np.isfinite(entry_px) and entry_px > 0 and np.isfinite(last_px)):
            continue
        pnl = (last_px / entry_px - 1.0) * 100.0
        days_held = (n_rows - 1) - last_e
        long_rows.append(
            (yf_to_tv.get(c, c), entry_dt, entry_px, last_px, pnl, days_held)
        )
    long_rows.sort(key=lambda r: r[4], reverse=True)

    console.print(
        f"\n[bold]obv_trend live — {market.upper()} — as of {today_date}[/bold]"
    )
    console.print(
        f"[dim]Universe: {len(close.columns)} symbols | ema_window={ema_window}[/dim]\n"
    )
    _render_today_table(
        console,
        title=f"Fresh entries on {today_date}",
        rows=today_list,
        limit=limit,
    )
    _render_active_table(
        console,
        title="Currently long (OBV above EMA, no cross-down yet)",
        rows=long_rows,
        limit=limit,
    )
