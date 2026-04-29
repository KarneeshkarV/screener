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
from rich.console import Console

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from .buildup import (
    DEFAULT_MIN_SCORE as DEFAULT_BUILDUP_MIN,
    DEFAULT_WINDOW as DEFAULT_BUILDUP_WINDOW,
    compute_buildup_score,
    scan_buildups,
)
from .delivery import load_delivery_panel, overlay_events, quiet_accumulation_events
from .detector import (
    DEFAULT_MIN_RVOL,
    DEFAULT_MIN_Z,
    Event,
    detect_market,
)
from .enrich import attach_sector, deep_enrich_india, fetch_sector_map
from .filters import (
    fetch_fno_ban_list,
    passes_market_cap,
    passes_volume_floor,
)
from .output import render_rich, sort_events, write_json, write_markdown


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
    from run_pinescript_strategies import load_universe  # lazy import; pulls TV
    return load_universe(market)


def _fetch_bars(
    tickers: list[str], market: str, as_of: date, console: Console
) -> dict[str, pd.DataFrame]:
    fetcher = YFinancePriceFetcher()
    start = as_of - timedelta(days=400)
    end = as_of + timedelta(days=1)

    yf_map = {t: tv_to_yf(t, market) for t in tickers}
    out: dict[str, pd.DataFrame] = {}

    def _fetch_one(tv_sym: str) -> tuple[str, Optional[pd.DataFrame]]:
        yf_sym = yf_map[tv_sym]
        try:
            frames = fetcher.fetch([yf_sym], start, end)
        except Exception:
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
    json_path: Optional[str],
    md_path: Optional[str],
    no_output_files: bool,
    limit: int,
    buildup_enabled: bool,
    buildup_window: int,
    buildup_min_score: float,
) -> None:
    """Detect abnormal trading volume across a market on a given day."""
    console = Console()
    as_of: date = (
        as_of_arg.date() if isinstance(as_of_arg, datetime) else (as_of_arg or date.today())
    )

    universe = _resolve_universe(market, tickers, universe_file)
    if not universe:
        raise click.UsageError("Empty universe — pass --tickers or --universe-file.")
    console.print(
        f"[dim]Scanning {len(universe)} {market.upper()} tickers as of {as_of}…[/dim]"
    )

    bars_by_tv = _fetch_bars(universe, market, as_of, console)
    if not bars_by_tv:
        console.print("[red]No OHLCV data fetched. Aborting.[/red]")
        sys.exit(1)

    # India: drop F&O ban-list tickers up front so we don't waste downstream work.
    ban_set: set[str] = set()
    if market == "india" and not include_fno_ban:
        ban_set = fetch_fno_ban_list()
        if ban_set:
            before = len(bars_by_tv)
            bars_by_tv = {
                tv_sym: df
                for tv_sym, df in bars_by_tv.items()
                if _india_symbol(tv_sym) not in ban_set
            }
            console.print(
                f"[dim]F&O ban filter: dropped {before - len(bars_by_tv)} ticker(s) "
                f"({len(ban_set)} symbols in ban list).[/dim]"
            )

    # Liquidity floor — runs against the full window before we hit detection.
    liquid: dict[str, pd.DataFrame] = {
        tv_sym: df
        for tv_sym, df in bars_by_tv.items()
        if passes_volume_floor(df, min_avg_volume, as_of)
    }
    console.print(
        f"[dim]Volume floor (≥{int(min_avg_volume):,} 20d avg): "
        f"{len(liquid)}/{len(bars_by_tv)} survive.[/dim]"
    )
    if not liquid:
        console.print("[yellow]No tickers passed the volume floor.[/yellow]")
        return

    events = detect_market(liquid, as_of, min_rvol=min_rvol, min_z=min_z)
    console.print(f"[dim]Detector emitted {len(events)} candidate events.[/dim]")

    # Index events back to their TradingView symbol so India delivery / enrich
    # can use the right key shape.
    by_tv: dict[str, str] = {}
    if market == "india":
        for tv_sym in liquid.keys():
            by_tv[_india_symbol(tv_sym)] = tv_sym
        # Detector stored the universe symbol on the event; align it to the bare
        # NSE symbol so it matches the bhavcopy SYMBOL column.
        for ev in events:
            ev.symbol = _india_symbol(ev.symbol)

    # India delivery overlay + quiet-accumulation pass.
    panel: pd.DataFrame = pd.DataFrame()
    if market == "india":
        india_syms = [_india_symbol(s) for s in liquid.keys()]
        try:
            panel = load_delivery_panel(india_syms, as_of, history_days=40)
        except Exception as exc:
            console.print(
                f"[yellow]Delivery overlay failed: {exc}. Continuing without it.[/yellow]"
            )
            panel = pd.DataFrame()
        if not panel.empty:
            overlay_events(events, panel)
            # Re-key bars by NSE symbol for quiet-accumulation pass.
            bars_for_quiet = {
                _india_symbol(tv): df for tv, df in liquid.items()
            }
            quiet = quiet_accumulation_events(
                bars_for_quiet, panel, as_of, min_rvol_skip=min_rvol
            )
            if quiet:
                console.print(
                    f"[dim]Quiet-accumulation pass added {len(quiet)} event(s).[/dim]"
                )
            events.extend(quiet)

    # Strength filter.
    floor_rank = _STRENGTH_RANK[strength_floor.upper()]
    events = [e for e in events if _STRENGTH_RANK[e.strength] >= floor_rank]

    # Build-up overlay — annotates surviving events AND surfaces standalone
    # build-ups (tickers with no volume spike yet but a clean accumulation
    # footprint over the prior window).
    if buildup_enabled:
        delivery_for_buildup = panel if (market == "india" and not panel.empty) else None
        bars_for_buildup = (
            {_india_symbol(tv): df for tv, df in liquid.items()}
            if market == "india"
            else dict(liquid)
        )
        # 1) Annotate already-detected events.
        annotated = 0
        for ev in events:
            score = compute_buildup_score(
                ev.symbol,
                bars_for_buildup.get(ev.symbol),
                as_of,
                delivery_panel=delivery_for_buildup,
                window=buildup_window,
            )
            if score is None:
                continue
            ev.buildup_score = score.composite
            ev.buildup_flags = list(score.flags)
            annotated += 1
        # 2) Surface standalone build-ups not already in the events list.
        existing = {(e.symbol, e.direction) for e in events}
        existing_syms = {e.symbol for e in events}
        scores = scan_buildups(
            bars_for_buildup,
            as_of,
            delivery_panel=delivery_for_buildup,
            window=buildup_window,
            min_score=buildup_min_score,
        )
        added = 0
        for s in scores:
            if s.symbol in existing_syms:
                continue
            bars = bars_for_buildup.get(s.symbol)
            if bars is None or bars.empty:
                continue
            df_s = bars.sort_index() if isinstance(bars.index, pd.DatetimeIndex) else bars
            last = df_s.iloc[-1]
            prev_close = float(df_s["close"].iloc[-2]) if len(df_s) >= 2 else float(last["close"])
            close_v = float(last["close"])
            pct_change = (
                (close_v - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0
            )
            ev = Event(
                symbol=s.symbol,
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
                buildup_score=s.composite,
                buildup_flags=list(s.flags),
                notes="multi-week build-up: " + ", ".join(s.flags) if s.flags else "multi-week build-up",
            )
            events.append(ev)
            added += 1
        console.print(
            f"[dim]Build-up pass: annotated {annotated} event(s); "
            f"added {added} standalone build-up(s) at score >= {buildup_min_score}.[/dim]"
        )

    # Sector + market-cap enrichment.
    if events:
        sector_map = fetch_sector_map(market, [e.symbol for e in events])
        if sector_map:
            attach_sector(events, sector_map)

    # Market-cap floor (after enrichment so we have a value to compare).
    resolved_min_mcap = (
        _DEFAULT_MIN_MCAP.get(market, 0.0)
        if min_market_cap is None
        else float(min_market_cap)
    )
    if resolved_min_mcap > 0:
        before = len(events)
        events = [
            e for e in events if passes_market_cap(e.market_cap, resolved_min_mcap)
        ]
        console.print(
            f"[dim]Market-cap floor (≥{_human_mcap(resolved_min_mcap)}): "
            f"{len(events)}/{before} survive.[/dim]"
        )

    if market == "india" and deep_india and events:
        console.print("[dim]Running openscreener deep enrichment for India events…[/dim]")
        deep_enrich_india(events)

    if not events:
        console.print(
            f"[yellow]No unusual-volume events on {as_of} for {market.upper()}.[/yellow]"
        )
        return

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
