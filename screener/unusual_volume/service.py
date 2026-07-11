"""Reusable unusual-volume scan workflow."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any, ClassVar, Optional, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import requests
from rich.console import Console

from screener.backtester.data import build_price_fetcher, tv_to_yf
from screener.symbols import tv_to_nse
from .buildup import (
    DEFAULT_MIN_SCORE as DEFAULT_BUILDUP_MIN,
    DEFAULT_WINDOW as DEFAULT_BUILDUP_WINDOW,
    BuildupScore,
    compute_buildup_score,
    scan_buildups,
)
from .classify import STRENGTH_RANK
from .delivery import load_delivery_panel, overlay_events, quiet_accumulation_events
from .detector import (
    DEFAULT_MIN_RVOL,
    DEFAULT_MIN_Z,
    Event,
    bars_on_or_before_as_of,
    detect_market,
)
from .enrich import attach_sector, deep_enrich_india, fetch_sector_map
from .enrichment import (
    MICROSTRUCTURE_ENRICHMENTS,
    Enrichment,
    EnrichmentDiagnostic,
)
from .filters import fetch_fno_ban_list, passes_market_cap, passes_volume_floor
from .microstructure import run_microstructure_enrichments


_DEFAULT_MIN_MCAP = {"us": 300_000_000.0, "india": 5_000_000_000.0}
DEFAULT_MIN_AVG_VOLUME = 100_000.0


def _live_nse_snapshot_date() -> date:
    """Return the trading date represented by live NSE-only endpoints."""
    today = date.today()
    try:
        from screener.operator.fetch import latest_trading_day

        return latest_trading_day(today)
    except Exception:
        return today


class UnusualVolumeRequest(BaseModel):
    market: str
    as_of: date
    universe: list[str]
    min_rvol: float = Field(default=DEFAULT_MIN_RVOL, ge=0.0)
    min_z: float = Field(default=DEFAULT_MIN_Z, ge=0.0)
    strength_floor: str = "moderate"
    min_avg_volume: float = Field(default=DEFAULT_MIN_AVG_VOLUME, ge=0.0)
    min_market_cap: Optional[float] = Field(default=None, ge=0.0)
    include_fno_ban: bool = False
    enrichments: frozenset[Enrichment] = frozenset()
    buildup_window: int = Field(default=DEFAULT_BUILDUP_WINDOW, ge=1)
    buildup_min_score: float = Field(default=DEFAULT_BUILDUP_MIN, ge=0.0)
    refresh: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")

    _LEGACY_ENRICHMENT_FLAGS: ClassVar[dict[str, Enrichment]] = {
        "buildup_enabled": Enrichment.BUILDUP,
        "deep_india": Enrichment.DEEP_INDIA,
        "option_chain": Enrichment.OPTION_CHAIN,
        "fii_dii": Enrichment.FII_DII,
        "pledge": Enrichment.PLEDGE,
    }
    # Remove this flag shim once CLI/tests use ``enrichments`` enum values only.

    @model_validator(mode="before")
    @classmethod
    def _collect_enrichment_flags(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        selected = set(data.get("enrichments", ()))
        for flag, enrichment in cls._LEGACY_ENRICHMENT_FLAGS.items():
            if data.pop(flag, False):
                selected.add(enrichment)
        data["enrichments"] = selected
        return data

    @field_validator("market", "strength_floor")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @field_validator("universe")
    @classmethod
    def _normalize_universe(cls, value: list[str]) -> list[str]:
        normalized = [ticker.strip() for ticker in value if ticker.strip()]
        if not normalized:
            raise ValueError("universe must include at least one ticker")
        return normalized

    def includes(self, enrichment: Enrichment) -> bool:
        return enrichment in self.enrichments

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update and self._LEGACY_ENRICHMENT_FLAGS.keys() & update.keys():
            data = self.model_dump()
            data.update(update)
            return self.__class__.model_validate(data)
        return super().model_copy(update=dict(update) if update else None, deep=deep)

    @property
    def buildup_enabled(self) -> bool:
        return self.includes(Enrichment.BUILDUP)

    @property
    def deep_india(self) -> bool:
        return self.includes(Enrichment.DEEP_INDIA)

    @property
    def option_chain(self) -> bool:
        return self.includes(Enrichment.OPTION_CHAIN)

    @property
    def fii_dii(self) -> bool:
        return self.includes(Enrichment.FII_DII)

    @property
    def pledge(self) -> bool:
        return self.includes(Enrichment.PLEDGE)


class UnusualVolumeResult(BaseModel):
    events: list[Event] = Field(default_factory=list)
    fetched_count: int = Field(ge=0)
    liquid_count: int = Field(ge=0)
    diagnostics: list[EnrichmentDiagnostic] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


def fetch_bars(
    tickers: list[str],
    market: str,
    as_of: date,
    console: Console,
    *,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    fetcher = build_price_fetcher(refresh=refresh)
    start = as_of - timedelta(days=400)
    end = as_of + timedelta(days=1)

    yf_map = {t: tv_to_yf(t, market) for t in tickers}
    reverse_map = {yf_sym: tv_sym for tv_sym, yf_sym in yf_map.items()}
    out: dict[str, pd.DataFrame] = {}
    try:
        frames = fetcher.fetch(list(yf_map.values()), start, end)
    except (
        requests.RequestException,
        ConnectionError,
        TimeoutError,
        KeyError,
        ValueError,
    ):
        return out
    for yf_sym, df in frames.items():
        tv_sym = reverse_map.get(yf_sym)
        if tv_sym and df is not None and not df.empty:
            out[tv_sym] = df
    console.print(
        f"  [{market}] fetched {len(frames)}/{len(tickers)} ({len(out)} usable)",
        style="dim",
    )
    return out


def india_symbol(tv_sym: str) -> str:
    """Return the NSE bhavcopy symbol for a TradingView-style symbol."""
    return tv_to_nse(tv_sym)


def standalone_buildup_event(
    score: BuildupScore, bars: pd.DataFrame, as_of: date
) -> Optional[Event]:
    df_s = bars_on_or_before_as_of(bars, as_of)
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


def run_unusual_volume_scan(
    request: UnusualVolumeRequest,
    console: Console,
) -> UnusualVolumeResult:
    console.print(
        f"[dim]Scanning {len(request.universe)} {request.market.upper()} "
        f"tickers as of {request.as_of}...[/dim]"
    )
    bars_by_tv = fetch_bars(
        request.universe,
        request.market,
        request.as_of,
        console,
        refresh=request.refresh,
    )
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
    floor_rank = STRENGTH_RANK[request.strength_floor.upper()]
    events = [e for e in events if STRENGTH_RANK[e.strength] >= floor_rank]

    if request.buildup_enabled:
        _apply_buildup_overlay(request, liquid, panel, events, console)

    diagnostics: list[EnrichmentDiagnostic] = []
    if request.market == "india":
        diagnostics = _overlay_india_microstructure(request, events, console) or []

    if events:
        sector_map = fetch_sector_map(
            request.market,
            [e.symbol for e in events],
            refresh=request.refresh,
        )
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
        console.print(
            "[dim]Running openscreener deep enrichment for India events...[/dim]"
        )
        deep_enrich_india(events)

    return UnusualVolumeResult(
        events=events,
        fetched_count=len(bars_by_tv),
        liquid_count=len(liquid),
        diagnostics=diagnostics,
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
    except (
        requests.RequestException,
        OSError,
        RuntimeError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
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
        console.print(
            f"[dim]Quiet-accumulation pass added {len(quiet)} event(s).[/dim]"
        )
    events.extend(quiet)
    return panel


def _overlay_india_microstructure(
    request: UnusualVolumeRequest,
    events: list[Event],
    console: Console,
) -> list[EnrichmentDiagnostic]:
    """Run selected India overlay stages and render their diagnostics."""
    selected = frozenset(request.enrichments & MICROSTRUCTURE_ENRICHMENTS)
    if not events or not selected:
        return []
    snap_date = _live_nse_snapshot_date()
    if request.as_of != snap_date:
        console.print(
            "[dim]Live NSE overlays use "
            f"{snap_date}; preserving historical scan date {request.as_of}.[/dim]"
        )
    diagnostics = run_microstructure_enrichments(
        events,
        selected,
        scan_date=request.as_of,
        snapshot_date=snap_date,
        refresh=request.refresh,
    )
    for diagnostic in diagnostics:
        style = "yellow" if diagnostic.status == "failed" else "dim"
        console.print(f"[{style}]{diagnostic.message}[/{style}]")
    return diagnostics


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
