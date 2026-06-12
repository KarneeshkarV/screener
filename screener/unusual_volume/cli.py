"""Click sub-command for unusual-volume detection.

The command is registered on the main ``cli`` group in ``main.py`` via:

    from screener.unusual_volume.cli import unusual_volume
    cli.add_command(unusual_volume)
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import requests
from rich.console import Console

from screener.backtester.data import build_price_fetcher, tv_to_yf
from .buildup import (
    BuildupScore,
    DEFAULT_MIN_SCORE as DEFAULT_BUILDUP_MIN,
    DEFAULT_WINDOW as DEFAULT_BUILDUP_WINDOW,
)
from .detector import (
    DEFAULT_MIN_RVOL,
    DEFAULT_MIN_Z,
    Event,
)
from .output import render_rich, sort_events, write_json, write_markdown
from .service import (
    UnusualVolumeRequest,
    run_unusual_volume_scan,
)


_DEFAULT_MIN_AVG_VOLUME = 100_000.0
_DEFAULT_MIN_MCAP = {"us": 300_000_000.0, "india": 5_000_000_000.0}
_STRENGTH_RANK = {"MODERATE": 1, "HIGH": 2, "EXTREME": 3}


def _resolve_universe(
    market: str,
    tickers: Optional[str],
    universe_file: Optional[str],
) -> list[str]:
    if tickers:
        return [t.strip() for t in tickers.split(",") if t.strip()]
    if universe_file:
        path = Path(universe_file)
        if not path.exists():
            raise click.UsageError(f"--universe-file not found: {universe_file}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    # Fallback to the project's default universe loader.
    from screener.backtester.pine_runner import load_universe  # lazy import; pulls TV

    return load_universe(market)


def _fetch_bars(
    tickers: list[str], market: str, as_of: date, console: Console
) -> dict[str, pd.DataFrame]:
    fetcher = build_price_fetcher()
    start = as_of - timedelta(days=400)
    end = as_of + timedelta(days=1)

    yf_map = {t: tv_to_yf(t, market) for t in tickers}
    out: dict[str, pd.DataFrame] = {}

    def _fetch_one(tv_sym: str) -> tuple[str, Optional[pd.DataFrame]]:
        yf_sym = yf_map[tv_sym]
        try:
            frames = fetcher.fetch([yf_sym], start, end)
        except (
            requests.RequestException,
            ConnectionError,
            TimeoutError,
            KeyError,
            ValueError,
        ):
            return tv_sym, None
        df = frames.get(yf_sym)
        if df is None or df.empty:
            return tv_sym, None
        return tv_sym, df

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_one, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            tv_sym, df = fut.result()
            if df is not None and not df.empty:
                out[tv_sym] = df
            if i % 100 == 0:
                console.print(
                    f"  [{market}] fetched {i}/{len(tickers)} ({len(out)} usable)",
                    style="dim",
                )
    return out


def _india_symbol(tv_sym: str) -> str:
    """`NSE:RELIANCE` or `RELIANCE` → `RELIANCE` (matches NSE bhavcopy SYMBOL)."""
    if ":" in tv_sym:
        _exch, rest = tv_sym.split(":", 1)
        return rest.upper()
    return tv_sym.upper()


def _bars_on_or_before_as_of(bars: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            return pd.DataFrame()
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"]).values))
    df = df.sort_index()
    return df[df.index <= pd.Timestamp(as_of).normalize()]


def _standalone_buildup_event(
    score: BuildupScore, bars: pd.DataFrame, as_of: date
) -> Optional[Event]:
    df_s = _bars_on_or_before_as_of(bars, as_of)
    if df_s.empty:
        return None
    last = df_s.iloc[-1]
    prev_close = (
        float(df_s["close"].iloc[-2]) if len(df_s) >= 2 else float(last["close"])
    )
    close_v = float(last["close"])
    pct_change = (close_v - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0
    return Event(
        symbol=score.symbol,
        date=as_of,
        close=close_v,
        pct_change=round(pct_change, 4),
        volume=float(last["volume"]),
        avg_volume_20d=0.0,
        rvol=float("nan"),
        rvol_5d=float("nan"),
        rvol_50d=float("nan"),
        rvol_90d=float("nan"),
        z_score=float("nan"),
        pct_rank_252d=float("nan"),
        direction="BUILDUP",
        strength="MODERATE",
        buildup_score=score.composite,
        buildup_flags=list(score.flags),
        notes=(
            "multi-week build-up: " + ", ".join(score.flags)
            if score.flags
            else "multi-week build-up"
        ),
    )


@click.command(name="unusual-volume")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    help="Market to scan.",
)
@click.option(
    "--as-of",
    "as_of_arg",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Trading date to evaluate (default: today).",
)
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated ticker list. Falls back to load_universe() when omitted.",
)
@click.option(
    "--universe-file",
    default=None,
    help="Newline-separated ticker file (alternative to --tickers).",
)
@click.option(
    "--min-rvol",
    type=float,
    default=DEFAULT_MIN_RVOL,
    help=f"RVOL floor for the moderate tier (default {DEFAULT_MIN_RVOL}).",
)
@click.option(
    "--min-z",
    type=float,
    default=DEFAULT_MIN_Z,
    help=f"Volume Z-score floor for the moderate tier (default {DEFAULT_MIN_Z}).",
)
@click.option(
    "--strength",
    "strength_floor",
    type=click.Choice(["moderate", "high", "extreme"]),
    default="moderate",
    help="Drop events below this strength tier.",
)
@click.option(
    "--min-avg-volume",
    type=float,
    default=_DEFAULT_MIN_AVG_VOLUME,
    help="Minimum 20-day average daily volume (shares).",
)
@click.option(
    "--min-market-cap",
    type=float,
    default=None,
    help="Minimum market cap. Defaults to $300M (US) / ₹500 cr (India).",
)
@click.option(
    "--include-fno-ban",
    is_flag=True,
    default=False,
    help="(India) include tickers in the F&O ban list. Default: drop them.",
)
@click.option(
    "--deep-india",
    is_flag=True,
    default=False,
    help="(India) enrich flagged events with promoter holding via openscreener.",
)
@click.option(
    "--json",
    "json_path",
    default=None,
    help="JSON output path. Default: unusual_volume_<market>_<as_of>.json",
)
@click.option(
    "--md",
    "md_path",
    default=None,
    help="Markdown output path. Default: unusual_volume_<market>_<as_of>.md",
)
@click.option(
    "--no-output-files",
    is_flag=True,
    default=False,
    help="Skip JSON/MD writes (rich-table only).",
)
@click.option(
    "--option-chain",
    is_flag=True,
    default=False,
    help="(India) attach live NSE option-chain PCR / call-put OI ratio and "
    "accumulate a daily snapshot panel.",
)
@click.option(
    "--fii-dii",
    is_flag=True,
    default=False,
    help="(India) attach market-wide FII/DII 5d net + trend and accumulate a "
    "daily snapshot panel.",
)
@click.option(
    "--pledge",
    is_flag=True,
    default=False,
    help="(India) attach promoter pledge %% (NSE filings, openscreener fallback).",
)
@click.option(
    "--refresh", is_flag=True, help="Bypass cached yfinance and enrichment data."
)
@click.option(
    "-n",
    "--limit",
    type=int,
    default=50,
    help="Cap rich-table rows (sorted by strength then RVOL).",
)
@click.option(
    "--buildup/--no-buildup",
    "buildup_enabled",
    default=False,
    help="Score every ticker for multi-week build-up patterns and emit a "
    "BUILDUP bucket. Adds buildup_score+flags onto detected events too.",
)
@click.option(
    "--buildup-window",
    type=int,
    default=DEFAULT_BUILDUP_WINDOW,
    show_default=True,
    help="Bars of lookback for build-up scoring.",
)
@click.option(
    "--buildup-min-score",
    type=float,
    default=DEFAULT_BUILDUP_MIN,
    show_default=True,
    help="Composite score floor for the BUILDUP bucket.",
)
def unusual_volume(
    market: str,
    as_of_arg,
    tickers: Optional[str],
    universe_file: Optional[str],
    min_rvol: float,
    min_z: float,
    strength_floor: str,
    min_avg_volume: float,
    min_market_cap: Optional[float],
    include_fno_ban: bool,
    deep_india: bool,
    option_chain: bool,
    fii_dii: bool,
    pledge: bool,
    json_path: Optional[str],
    md_path: Optional[str],
    no_output_files: bool,
    refresh: bool,
    limit: int,
    buildup_enabled: bool,
    buildup_window: int,
    buildup_min_score: float,
) -> None:
    """Detect abnormal trading volume across a market on a given day."""
    as_of: date = (
        as_of_arg.date()
        if isinstance(as_of_arg, datetime)
        else (as_of_arg or date.today())
    )
    run_unusual_volume(
        market=market,
        as_of=as_of,
        tickers=tickers,
        universe_file=universe_file,
        min_rvol=min_rvol,
        min_z=min_z,
        strength_floor=strength_floor,
        min_avg_volume=min_avg_volume,
        min_market_cap=min_market_cap,
        include_fno_ban=include_fno_ban,
        deep_india=deep_india,
        option_chain=option_chain,
        fii_dii=fii_dii,
        pledge=pledge,
        json_path=json_path,
        md_path=md_path,
        no_output_files=no_output_files,
        refresh=refresh,
        limit=limit,
        buildup_enabled=buildup_enabled,
        buildup_window=buildup_window,
        buildup_min_score=buildup_min_score,
    )


def run_unusual_volume(
    *,
    market: str,
    as_of: date,
    tickers: Optional[str] = None,
    universe_file: Optional[str] = None,
    min_rvol: float = DEFAULT_MIN_RVOL,
    min_z: float = DEFAULT_MIN_Z,
    strength_floor: str = "moderate",
    min_avg_volume: float = _DEFAULT_MIN_AVG_VOLUME,
    min_market_cap: Optional[float] = None,
    include_fno_ban: bool = False,
    deep_india: bool = False,
    option_chain: bool = False,
    fii_dii: bool = False,
    pledge: bool = False,
    json_path: Optional[str] = None,
    md_path: Optional[str] = None,
    no_output_files: bool = False,
    refresh: bool = False,
    limit: int = 50,
    buildup_enabled: bool = False,
    buildup_window: int = DEFAULT_BUILDUP_WINDOW,
    buildup_min_score: float = DEFAULT_BUILDUP_MIN,
) -> None:
    """Run the unusual-volume scan and render it (no Click context required)."""
    console = Console()
    universe = _resolve_universe(market, tickers, universe_file)
    if not universe:
        raise click.UsageError("Empty universe — pass --tickers or --universe-file.")
    request = UnusualVolumeRequest(
        market=market,
        as_of=as_of,
        universe=universe,
        min_rvol=min_rvol,
        min_z=min_z,
        strength_floor=strength_floor,
        min_avg_volume=min_avg_volume,
        min_market_cap=min_market_cap,
        include_fno_ban=include_fno_ban,
        deep_india=deep_india,
        buildup_enabled=buildup_enabled,
        buildup_window=buildup_window,
        buildup_min_score=buildup_min_score,
        option_chain=option_chain,
        fii_dii=fii_dii,
        pledge=pledge,
        refresh=refresh,
    )
    result = run_unusual_volume_scan(request, console)
    if not result.events and result.fetched_count == 0:
        console.print("[red]No OHLCV data fetched. Aborting.[/red]")
        sys.exit(1)
    if not result.events and result.liquid_count == 0:
        console.print("[yellow]No tickers passed the volume floor.[/yellow]")
        return
    if not result.events:
        console.print(
            f"[yellow]No unusual-volume events on {as_of} for {market.upper()}.[/yellow]"
        )
        return

    events = result.events
    sorted_events = sort_events(events)
    render_rich(sorted_events[:limit], market, as_of, console)

    if not no_output_files:
        json_default = f"unusual_volume_{market}_{as_of.isoformat()}.json"
        md_default = f"unusual_volume_{market}_{as_of.isoformat()}.md"
        write_json(events, Path(json_path or json_default))
        write_markdown(events, Path(md_path or md_default), market, as_of)
        console.print(
            f"\n[dim]Wrote {json_path or json_default} + {md_path or md_default}[/dim]"
        )


def _human_mcap(v: float) -> str:
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v:,.0f}"
