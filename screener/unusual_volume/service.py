"""Reusable unusual-volume scan workflow."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from rich.console import Console

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from .buildup import BuildupScore, compute_buildup_score, scan_buildups
from .delivery import load_delivery_panel, overlay_events, quiet_accumulation_events
from .detector import Event, detect_market
from .enrich import attach_sector, deep_enrich_india, fetch_sector_map
from .filters import fetch_fno_ban_list, passes_market_cap, passes_volume_floor


_DEFAULT_MIN_MCAP = {"us": 300_000_000.0, "india": 5_000_000_000.0}
_STRENGTH_RANK = {"MODERATE": 1, "HIGH": 2, "EXTREME": 3}


@dataclass(frozen=True)
class UnusualVolumeRequest:
    market: str
    as_of: date
    universe: list[str]
    min_rvol: float
    min_z: float
    strength_floor: str
    min_avg_volume: float
    min_market_cap: Optional[float]
    include_fno_ban: bool
    deep_india: bool
    buildup_enabled: bool
    buildup_window: int
    buildup_min_score: float


@dataclass(frozen=True)
class UnusualVolumeResult:
    events: list[Event]
    fetched_count: int
    liquid_count: int


def fetch_bars(
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

    from concurrent.futures import ThreadPoolExecutor, as_completed

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


def india_symbol(tv_sym: str) -> str:
    """Return the NSE bhavcopy symbol for a TradingView-style symbol."""
    if ":" in tv_sym:
        _exch, rest = tv_sym.split(":", 1)
        return rest.upper()
    return tv_sym.upper()


def bars_on_or_before_as_of(bars: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            return pd.DataFrame()
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"]).values))
    df = df.sort_index()
    return df[df.index <= pd.Timestamp(as_of).normalize()]


def standalone_buildup_event(
    score: BuildupScore, bars: pd.DataFrame, as_of: date
) -> Optional[Event]:
    df_s = bars_on_or_before_as_of(bars, as_of)
    if df_s.empty:
        return None
    last = df_s.iloc[-1]
    prev_close = float(df_s["close"].iloc[-2]) if len(df_s) >= 2 else float(last["close"])
    close_v = float(last["close"])
    pct_change = (
        (close_v - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0
    )
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


def run_unusual_volume_scan(
    request: UnusualVolumeRequest,
    console: Console,
) -> UnusualVolumeResult:
    console.print(
        f"[dim]Scanning {len(request.universe)} {request.market.upper()} "
        f"tickers as of {request.as_of}...[/dim]"
    )
    bars_by_tv = fetch_bars(request.universe, request.market, request.as_of, console)
    if not bars_by_tv:
        return UnusualVolumeResult(events=[], fetched_count=0, liquid_count=0)

    if request.market == "india" and not request.include_fno_ban:
        ban_set = fetch_fno_ban_list()
        if ban_set:
            before = len(bars_by_tv)
            bars_by_tv = {
                tv_sym: df
                for tv_sym, df in bars_by_tv.items()
                if india_symbol(tv_sym) not in ban_set
            }
            console.print(
                f"[dim]F&O ban filter: dropped {before - len(bars_by_tv)} ticker(s) "
                f"({len(ban_set)} symbols in ban list).[/dim]"
            )

    liquid = {
        tv_sym: df
        for tv_sym, df in bars_by_tv.items()
        if passes_volume_floor(df, request.min_avg_volume, request.as_of)
    }
    console.print(
        f"[dim]Volume floor (>={int(request.min_avg_volume):,} 20d avg): "
        f"{len(liquid)}/{len(bars_by_tv)} survive.[/dim]"
    )
    if not liquid:
        return UnusualVolumeResult(
            events=[],
            fetched_count=len(bars_by_tv),
            liquid_count=0,
        )

    events = detect_market(
        liquid,
        request.as_of,
        min_rvol=request.min_rvol,
        min_z=request.min_z,
    )
    console.print(f"[dim]Detector emitted {len(events)} candidate events.[/dim]")

    panel = _overlay_india_delivery(request, liquid, events, console)
    floor_rank = _STRENGTH_RANK[request.strength_floor.upper()]
    events = [e for e in events if _STRENGTH_RANK[e.strength] >= floor_rank]

    if request.buildup_enabled:
        _apply_buildup_overlay(request, liquid, panel, events, console)

    if events:
        sector_map = fetch_sector_map(request.market, [e.symbol for e in events])
        if sector_map:
            attach_sector(events, sector_map)

    resolved_min_mcap = (
        _DEFAULT_MIN_MCAP.get(request.market, 0.0)
        if request.min_market_cap is None
        else float(request.min_market_cap)
    )
    if resolved_min_mcap > 0:
        before = len(events)
        events = [
            e for e in events if passes_market_cap(e.market_cap, resolved_min_mcap)
        ]
        console.print(
            f"[dim]Market-cap floor (>={_human_mcap(resolved_min_mcap)}): "
            f"{len(events)}/{before} survive.[/dim]"
        )

    if request.market == "india" and request.deep_india and events:
        console.print("[dim]Running openscreener deep enrichment for India events...[/dim]")
        deep_enrich_india(events)

    return UnusualVolumeResult(
        events=events,
        fetched_count=len(bars_by_tv),
        liquid_count=len(liquid),
    )


def _overlay_india_delivery(
    request: UnusualVolumeRequest,
    liquid: dict[str, pd.DataFrame],
    events: list[Event],
    console: Console,
) -> pd.DataFrame:
    panel = pd.DataFrame()
    if request.market != "india":
        return panel
    for ev in events:
        ev.symbol = india_symbol(ev.symbol)
    india_syms = [india_symbol(s) for s in liquid.keys()]
    try:
        panel = load_delivery_panel(india_syms, request.as_of, history_days=40)
    except Exception as exc:
        console.print(
            f"[yellow]Delivery overlay failed: {exc}. Continuing without it.[/yellow]"
        )
        return pd.DataFrame()
    if panel.empty:
        return panel
    overlay_events(events, panel)
    bars_for_quiet = {india_symbol(tv): df for tv, df in liquid.items()}
    quiet = quiet_accumulation_events(
        bars_for_quiet,
        panel,
        request.as_of,
        min_rvol_skip=request.min_rvol,
        existing_events=events,
    )
    if quiet:
        console.print(f"[dim]Quiet-accumulation pass added {len(quiet)} event(s).[/dim]")
    events.extend(quiet)
    return panel


def _apply_buildup_overlay(
    request: UnusualVolumeRequest,
    liquid: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    events: list[Event],
    console: Console,
) -> None:
    delivery_for_buildup = (
        panel if (request.market == "india" and not panel.empty) else None
    )
    bars_for_buildup = (
        {india_symbol(tv): df for tv, df in liquid.items()}
        if request.market == "india"
        else dict(liquid)
    )
    annotated = 0
    for ev in events:
        score = compute_buildup_score(
            ev.symbol,
            bars_for_buildup.get(ev.symbol),
            request.as_of,
            delivery_panel=delivery_for_buildup,
            window=request.buildup_window,
        )
        if score is None:
            continue
        ev.buildup_score = score.composite
        ev.buildup_flags = list(score.flags)
        annotated += 1

    existing_syms = {e.symbol for e in events}
    scores = scan_buildups(
        bars_for_buildup,
        request.as_of,
        delivery_panel=delivery_for_buildup,
        window=request.buildup_window,
        min_score=request.buildup_min_score,
    )
    added = 0
    for score in scores:
        if score.symbol in existing_syms:
            continue
        bars = bars_for_buildup.get(score.symbol)
        if bars is None or bars.empty:
            continue
        standalone = standalone_buildup_event(score, bars, request.as_of)
        if standalone is None:
            continue
        events.append(standalone)
        added += 1
    console.print(
        f"[dim]Build-up pass: annotated {annotated} event(s); "
        f"added {added} standalone build-up(s) at score >= "
        f"{request.buildup_min_score}.[/dim]"
    )


def _human_mcap(v: float) -> str:
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v:,.0f}"
