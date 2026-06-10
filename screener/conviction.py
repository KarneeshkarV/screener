"""Single-ticker conviction card.

Fuses the existing screen modules into one composite scored card. Each
pillar reuses the signal math already shipped elsewhere in the package
(no duplication):

* ``trend``        — EMA20/50/200 alignment, RSI(14) and 63-day relative
                     strength vs the market benchmark (SPY / ^NSEI), on
                     bars from a :class:`PriceFetcher`.
* ``breakout``     — proximity to a valid breakout using the
                     :mod:`screener.rs_breakout` helpers (distance to the
                     52-week high, previous-week pivot, SuperTrend, RS55,
                     base length).
* ``volume``       — :func:`screener.unusual_volume.detector.detect_ticker`
                     RVOL/z-score/direction for the single ticker; India
                     additionally folds in NSE delivery when available.
* ``smart_money``  — :mod:`screener.insiders`: FMP Form 4 net buying (US,
                     when an API key is configured) or screener.in promoter
                     holding changes (India).
* ``fundamentals`` — GARP criteria from :mod:`screener.garp` evaluated for
                     the single ticker against the market thresholds.
* ``risk``         — India-only penalty pillar from promoter share pledge
                     (:mod:`screener.pledge`).

Every pillar returns a 0-100 score, one evidence line and an
``ok`` / ``skipped(reason)`` status. Missing keys, providers or data yield
clearly-labelled skipped pillars — nothing is fabricated. The composite is
the weighted average over the *available* (non-skipped) pillars with the
weights renormalized, so the card always reflects only real evidence.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from screener.backtester.data import PriceFetcher, tv_to_yf
from screener.cache import cached_json_call
from screener.garp import (
    GarpThresholds,
    INDIA_THRESHOLDS,
    US_THRESHOLDS,
    _fetch_fmp_us_cached,
    _fetch_india_sections,
    _fmp_us_row,
    _india_row,
    _num,
    _us_row,
)
from screener.indicators.plugins.ema import ema
from screener.indicators.plugins.rsi import rsi
from screener.insiders import (
    _fetch_fmp_insider_one,
    _fetch_openscreener_one,
    _fmp_api_key,
)
from screener.pledge import resolve_pledge_pct
from screener.rs_breakout import (
    DEFAULT_BENCHMARKS,
    delivery_lookup,
    india_symbol,
    normalize_bars,
    previous_completed_week_high,
    relative_strength_55,
    required_history_bars,
    supertrend,
)
from screener.unusual_volume.delivery import load_delivery_panel
from screener.unusual_volume.detector import EXTREME_RVOL, EXTREME_Z, detect_ticker


# Composite weights per pillar. Skipped pillars are dropped and the
# remaining weights renormalized, so e.g. a US run without an FMP key
# averages trend/breakout/volume/fundamentals only. ``risk`` is an
# India-only penalty pillar (high promoter pledge drags the composite).
PILLAR_WEIGHTS: dict[str, float] = {
    "trend": 0.25,
    "breakout": 0.20,
    "volume": 0.15,
    "smart_money": 0.15,
    "fundamentals": 0.15,
    "risk": 0.10,
}

# Calendar days of history fetched: ~280 trading bars so EMA200 / the
# 252-bar high / the 90-bar volume z-score all have a full window.
HISTORY_DAYS = 420
# Trend needs the 63-day relative-strength lookback plus RSI(14) warm-up.
TREND_MIN_BARS = 64
RS_TREND_WINDOW = 63


class PillarResult(BaseModel):
    name: str
    score: Optional[float] = None
    evidence: str = ""
    status: Literal["ok", "skipped"]
    reason: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    @property
    def label(self) -> str:
        return "ok" if self.status == "ok" else f"skipped({self.reason})"


class ConvictionCard(BaseModel):
    symbol: str
    market: str
    as_of: date
    composite: Optional[float]
    pillars: list[PillarResult]

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "as_of": self.as_of.isoformat(),
            "composite": self.composite,
            "weights": dict(PILLAR_WEIGHTS),
            "pillars": [p.model_dump(mode="json") for p in self.pillars],
        }


def _ok(name: str, score: float, evidence: str) -> PillarResult:
    return PillarResult(
        name=name, score=round(float(score), 1), evidence=evidence, status="ok"
    )


def _skipped(name: str, reason: str) -> PillarResult:
    return PillarResult(name=name, status="skipped", reason=reason)


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def compose(pillars: list[PillarResult]) -> Optional[float]:
    """Weighted average over available pillars, weights renormalized."""
    weighted = [
        (PILLAR_WEIGHTS.get(p.name, 0.0), p.score)
        for p in pillars
        if p.status == "ok" and p.score is not None
    ]
    total = sum(w for w, _ in weighted)
    if total <= 0:
        return None
    return round(sum(w * s for w, s in weighted) / total, 1)


# ── trend / momentum ─────────────────────────────────────────────────────────


def _rsi_points(value: float) -> float:
    """0-30 points: rewards a healthy 55-75 zone, penalizes overheated."""
    if value >= 85:
        return 5.0
    if value >= 75:
        return 15.0
    if value >= 55:
        return 30.0
    if value >= 45:
        return 20.0
    if value >= 35:
        return 10.0
    return 0.0


def score_trend(bars: pd.DataFrame, benchmark_close: pd.Series) -> PillarResult:
    """EMA stack (40) + RSI14 (30) + 63d relative strength (30)."""
    name = "trend"
    if bars is None or bars.empty or len(bars) < TREND_MIN_BARS:
        return _skipped(name, f"insufficient price history (<{TREND_MIN_BARS} bars)")
    close = bars["close"].astype(float).to_numpy()
    last = float(close[-1])
    e20 = float(ema(close, 20)[-1])
    e50 = float(ema(close, 50)[-1])
    e200 = float(ema(close, 200)[-1])
    stack = sum(
        10.0 for check in (last > e20, e20 > e50, e50 > e200, last > e200) if check
    )
    rsi_value = float(rsi(close, 14)[-1])
    rsi_pts = _rsi_points(rsi_value)

    rel: Optional[float] = None
    if benchmark_close is not None and not benchmark_close.empty:
        aligned = pd.concat(
            [bars["close"].astype(float), benchmark_close.astype(float)],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) > RS_TREND_WINDOW:
            stock_ret = (
                aligned.iloc[-1, 0] / aligned.iloc[-1 - RS_TREND_WINDOW, 0] - 1.0
            ) * 100.0
            bench_ret = (
                aligned.iloc[-1, 1] / aligned.iloc[-1 - RS_TREND_WINDOW, 1] - 1.0
            ) * 100.0
            rel = float(stock_ret - bench_ret)

    stack_count = int(stack // 10)
    if rel is None:
        # No benchmark data: renormalize the 70 EMA+RSI points to 100
        # instead of silently treating missing RS as RS == 0.
        score = (stack + rsi_pts) / 70.0 * 100.0
        rs_note = "RS63 n/a"
    else:
        # -10pp underperformance → 0, +20pp outperformance → full 30.
        score = stack + rsi_pts + _clamp((rel + 10.0) / 30.0, 0.0, 1.0) * 30.0
        rs_note = f"RS63 {rel:+.1f}pp vs benchmark"
    evidence = f"EMA stack {stack_count}/4, RSI14 {rsi_value:.0f}, {rs_note}"
    return _ok(name, _clamp(score), evidence)


# ── breakout setup ───────────────────────────────────────────────────────────


def score_breakout(
    bars: pd.DataFrame, benchmark_close: pd.Series, as_of: date
) -> PillarResult:
    """Proximity to a valid breakout, reusing the rs_breakout signal math.

    40 pts proximity to the 52-week high (full at the high, 0 at 25% below),
    20 pts previous-completed-week pivot cleared (10 within 3% below),
    20 pts price above SuperTrend, 20 pts RS55 > 0 (renormalized away when
    no benchmark data is available).
    """
    name = "breakout"
    df = normalize_bars(bars, as_of)
    min_bars = required_history_bars()
    if df.empty or len(df) < min_bars:
        return _skipped(name, f"insufficient price history (<{min_bars} bars)")

    close = float(df["close"].iloc[-1])
    window = df["high"].astype(float).tail(252)
    high_52w = float(window.max())
    dist_pct = max(0.0, (high_52w - close) / high_52w * 100.0)
    base_len = len(window) - 1 - int(window.to_numpy().argmax())
    proximity_pts = _clamp(1.0 - dist_pct / 25.0, 0.0, 1.0) * 40.0

    pivot = previous_completed_week_high(df, df.index[-1].date())
    if pivot is None:
        pivot_pts, pivot_note = 0.0, "no pivot"
    elif close > pivot:
        pivot_pts, pivot_note = 20.0, "above pivot"
    elif close >= pivot * 0.97:
        pivot_pts, pivot_note = 10.0, "near pivot"
    else:
        pivot_pts, pivot_note = 0.0, "below pivot"

    st = supertrend(df)
    st_last = float(st.iloc[-1]) if not st.empty else float("nan")
    st_ok = math.isfinite(st_last) and close > st_last
    st_pts = 20.0 if st_ok else 0.0

    rs_pts: Optional[float] = None
    rs_note = "RS55 n/a"
    if benchmark_close is not None and not benchmark_close.empty:
        rs = relative_strength_55(df["close"], benchmark_close)
        if not rs.empty and not pd.isna(rs.iloc[-1]):
            rs_last = float(rs.iloc[-1])
            rs_pts = 20.0 if rs_last > 0 else 0.0
            rs_note = f"RS55 {rs_last:+.1f}"

    raw = proximity_pts + pivot_pts + st_pts
    score = raw + rs_pts if rs_pts is not None else raw / 80.0 * 100.0
    evidence = (
        f"{dist_pct:.1f}% below 52w high, base {base_len}d, {pivot_note}, "
        f"SuperTrend {'support' if st_ok else 'overhead'}, {rs_note}"
    )
    return _ok(name, _clamp(score), evidence)


# ── volume confirmation ──────────────────────────────────────────────────────

_DIRECTION_BONUS: dict[str, float] = {
    "BUYING": 20.0,
    "QUIET_ACCUMULATION": 20.0,
    "BUILDUP": 15.0,
    "REVERSAL": 10.0,
    "CHURN": 5.0,
    "SELLING": 0.0,
}


def score_volume(
    symbol: str,
    bars: pd.DataFrame,
    as_of: date,
    delivery: tuple[Optional[float], Optional[float]] | None = None,
) -> PillarResult:
    """RVOL/z-score/direction from the unusual-volume detector.

    Thresholds are bypassed (``min_rvol=0``) so the detector always emits the
    metrics for a scoring read instead of acting as a pass/fail filter.
    India delivery (latest vs previous DELIV_PER) adds/removes 10 points.
    """
    name = "volume"
    event = detect_ticker(symbol, bars, as_of, min_rvol=0.0, min_z=-1e9)
    if event is None:
        return _skipped(name, "insufficient volume history or stale bars")
    rvol = event.rvol if math.isfinite(event.rvol) else 0.0
    z = event.z_score if math.isfinite(event.z_score) else 0.0
    pts = (
        _clamp(rvol / EXTREME_RVOL, 0.0, 1.0) * 60.0
        + _clamp(max(z, 0.0) / EXTREME_Z, 0.0, 1.0) * 20.0
        + _DIRECTION_BONUS.get(str(event.direction), 0.0)
    )
    evidence = f"RVOL20 {rvol:.1f}x, z {z:.1f}, {event.direction} ({event.strength})"
    if delivery is not None and delivery[0] is not None:
        latest, prev = delivery
        if prev is not None:
            pts += 10.0 if latest > prev else -10.0
            evidence += f", delivery {prev:.1f}%→{latest:.1f}%"
        else:
            evidence += f", delivery {latest:.1f}%"
    return _ok(name, _clamp(pts), evidence)


# ── smart money ──────────────────────────────────────────────────────────────


def _score_smart_money_us(payload: dict[str, Any]) -> PillarResult:
    bought = float(payload.get("fmp_buy_shares_6m") or 0.0)
    sold = float(payload.get("fmp_sell_shares_6m") or 0.0)
    net = float(payload.get("fmp_net_shares_6m") or 0.0)
    buys = int(payload.get("fmp_buy_trans_6m") or 0)
    sells = int(payload.get("fmp_sell_trans_6m") or 0)
    total = bought + sold
    score = 50.0 if total <= 0 else bought / total * 100.0
    evidence = f"Form 4 6m: net {net:+,.0f} sh ({buys} buys / {sells} sells)"
    return _ok("smart_money", score, evidence)


def _score_smart_money_india(payload: dict[str, Any]) -> PillarResult:
    change = _num(payload.get("promoter_change"))
    if change is None:
        return _skipped("smart_money", "no promoter change data")
    latest = _num(payload.get("promoter_pct_latest"))
    prev = _num(payload.get("promoter_pct_prev"))
    quarter = payload.get("latest_quarter")
    # ±2pp promoter-holding change maps to the 0-100 extremes; 0pp → 50.
    score = _clamp(50.0 + 25.0 * change)
    evidence = f"promoter {change:+.2f}pp"
    if latest is not None and prev is not None:
        evidence = f"promoter {prev:.2f}%→{latest:.2f}% ({change:+.2f}pp)"
    if quarter:
        evidence += f", qtr {quarter}"
    return _ok("smart_money", score, evidence)


def _load_smart_money_us(
    symbol: str, api_key: str, *, cache_ttl: float | None, refresh: bool
) -> Optional[dict[str, Any]]:
    return _fetch_fmp_insider_one(
        symbol, symbol, api_key=api_key, cache_ttl=cache_ttl, refresh=refresh
    )


def _load_smart_money_india(
    symbol: str, *, cache_ttl: float | None, refresh: bool
) -> Optional[dict[str, Any]]:
    return _fetch_openscreener_one(
        india_symbol(symbol), cache_ttl=cache_ttl, refresh=refresh
    )


def _smart_money_pillar(
    symbol: str, market: str, *, cache_ttl: float | None, refresh: bool
) -> PillarResult:
    name = "smart_money"
    if market == "us":
        api_key = _fmp_api_key()
        if not api_key:
            return _skipped(name, "FMP_API_KEY not configured")
        try:
            payload = _load_smart_money_us(
                tv_to_yf(symbol, market), api_key, cache_ttl=cache_ttl, refresh=refresh
            )
        except Exception as exc:  # provider failure must not sink the card
            return _skipped(name, f"FMP error: {exc}")
        if not payload:
            return _skipped(name, "no Form 4 buy/sell activity in window")
        return _score_smart_money_us(payload)
    try:
        payload = _load_smart_money_india(symbol, cache_ttl=cache_ttl, refresh=refresh)
    except Exception as exc:
        return _skipped(name, f"promoter data error: {exc}")
    if not payload:
        return _skipped(name, "no promoter shareholding data")
    return _score_smart_money_india(payload)


# ── fundamentals (GARP) ──────────────────────────────────────────────────────

# (label, row key, comparator vs GarpThresholds) — same criteria the GARP
# screen gates on, minus the absolute size floors (size is not conviction).
_GARP_MIN_METRICS = 3


def _garp_checks(
    row: dict[str, Any], thresholds: GarpThresholds
) -> list[tuple[str, Optional[bool]]]:
    def check(key: str, predicate: Any) -> Optional[bool]:
        value = _num(row.get(key))
        if value is None:
            return None
        return bool(predicate(value))

    return [
        ("PEG", check("peg", lambda v: 0 < v < thresholds.peg_max)),
        (
            "sales5y",
            check("sales_growth_5y", lambda v: v > thresholds.sales_growth_5y_min),
        ),
        (
            "opg",
            check(
                "operating_profit_growth",
                lambda v: v > thresholds.operating_profit_growth_min,
            ),
        ),
        ("eps5y", check("eps_growth_5y", lambda v: v > thresholds.eps_growth_5y_min)),
        ("roe5y", check("roe_5y", lambda v: v > thresholds.roe_5y_min)),
        (
            "roce/roic",
            check("roce_or_roic", lambda v: v > thresholds.roce_or_roic_min),
        ),
        ("qtr-profit", check("quarterly_profit_growth", lambda v: v > 0)),
    ]


def score_fundamentals(row: dict[str, Any], thresholds: GarpThresholds) -> PillarResult:
    """Fraction of evaluable GARP criteria passed, scaled to 0-100."""
    name = "fundamentals"
    checks = _garp_checks(row, thresholds)
    evaluated = [(label, passed) for label, passed in checks if passed is not None]
    if len(evaluated) < _GARP_MIN_METRICS:
        return _skipped(
            name,
            f"insufficient fundamental data ({len(evaluated)}/{len(checks)} metrics)",
        )
    passed = [label for label, ok in evaluated if ok]
    failed = [label for label, ok in evaluated if not ok]
    score = len(passed) / len(evaluated) * 100.0
    evidence = f"GARP {len(passed)}/{len(evaluated)} checks"
    if failed:
        evidence += f" (missed: {', '.join(failed)})"
    return _ok(name, score, evidence)


def _load_fundamentals(
    symbol: str, market: str, *, cache_ttl: float | None, refresh: bool
) -> Optional[dict[str, Any]]:
    if market == "india":
        sym = india_symbol(symbol)
        payload = cached_json_call(
            "garp_india",
            ("india", sym),
            ttl_seconds=cache_ttl,
            refresh=refresh,
            fetch=lambda: _fetch_india_sections(sym),
        )
        if not isinstance(payload, dict):
            return None
        return _india_row(sym, "", payload)
    yf_sym = tv_to_yf(symbol, market)
    api_key = _fmp_api_key()
    if api_key:
        fmp_payload = _fetch_fmp_us_cached(
            yf_sym, api_key, cache_ttl=cache_ttl, refresh=refresh
        )
        if isinstance(fmp_payload, dict):
            row = _fmp_us_row(yf_sym, "", fmp_payload)
            if row is not None:
                return row
    return _us_row(yf_sym, "")


def _fundamentals_pillar(
    symbol: str, market: str, *, cache_ttl: float | None, refresh: bool
) -> PillarResult:
    try:
        row = _load_fundamentals(symbol, market, cache_ttl=cache_ttl, refresh=refresh)
    except Exception as exc:
        return _skipped("fundamentals", f"provider error: {exc}")
    if not row:
        return _skipped("fundamentals", "no fundamental data")
    thresholds = INDIA_THRESHOLDS if market == "india" else US_THRESHOLDS
    return score_fundamentals(row, thresholds)


# ── risk flags (India promoter pledge) ───────────────────────────────────────


def score_pledge(pledge_pct: float) -> PillarResult:
    """Penalty pillar: 0% pledge → 100; 40%+ promoter pledge → 0."""
    score = _clamp(100.0 - 2.5 * pledge_pct)
    return _ok("risk", score, f"promoter pledge {pledge_pct:.1f}%")


def _load_pledge(symbol: str, *, refresh: bool) -> Optional[float]:
    sym = india_symbol(symbol)
    return resolve_pledge_pct(sym, sym, refresh=refresh)


def _risk_pillar(symbol: str, *, refresh: bool) -> PillarResult:
    try:
        pledge = _load_pledge(symbol, refresh=refresh)
    except Exception as exc:
        return _skipped("risk", f"pledge provider error: {exc}")
    if pledge is None:
        return _skipped("risk", "no promoter pledge data")
    return score_pledge(float(pledge))


# ── delivery helper (India volume overlay) ───────────────────────────────────


def _load_delivery(
    symbol: str, as_of: date
) -> tuple[Optional[float], Optional[float]] | None:
    sym = india_symbol(symbol)
    try:
        panel = load_delivery_panel([sym], as_of, history_days=14)
    except Exception:
        return None
    return delivery_lookup(panel).get(sym)


# ── card assembly ────────────────────────────────────────────────────────────


def build_conviction_card(
    symbol: str,
    market: str,
    as_of: date,
    fetcher: PriceFetcher,
    *,
    cache_ttl: float | None = 86400,
    refresh: bool = False,
) -> ConvictionCard:
    yf_sym = tv_to_yf(symbol, market)
    benchmark = DEFAULT_BENCHMARKS[market]
    start = as_of - timedelta(days=HISTORY_DAYS)
    end = as_of + timedelta(days=1)
    data = fetcher.fetch([yf_sym, benchmark], start, end)
    bars = normalize_bars(data.get(yf_sym, pd.DataFrame()), as_of)
    bench = normalize_bars(data.get(benchmark, pd.DataFrame()), as_of)
    bench_close = (
        bench["close"].astype(float) if not bench.empty else pd.Series(dtype=float)
    )

    pillars: list[PillarResult] = []
    if bars.empty:
        pillars.append(_skipped("trend", "no price data"))
        pillars.append(_skipped("breakout", "no price data"))
        pillars.append(_skipped("volume", "no price data"))
    else:
        pillars.append(score_trend(bars, bench_close))
        pillars.append(score_breakout(bars, bench_close, as_of))
        delivery = _load_delivery(symbol, as_of) if market == "india" else None
        display_symbol = india_symbol(symbol) if market == "india" else yf_sym
        pillars.append(score_volume(display_symbol, bars, as_of, delivery=delivery))
    pillars.append(
        _smart_money_pillar(symbol, market, cache_ttl=cache_ttl, refresh=refresh)
    )
    pillars.append(
        _fundamentals_pillar(symbol, market, cache_ttl=cache_ttl, refresh=refresh)
    )
    if market == "india":
        pillars.append(_risk_pillar(symbol, refresh=refresh))

    return ConvictionCard(
        symbol=symbol,
        market=market,
        as_of=as_of,
        composite=compose(pillars),
        pillars=pillars,
    )


def render_card(card: ConvictionCard, console: Console) -> None:
    console.print(
        f"[bold]{card.symbol} conviction[/bold] "
        f"[dim]{card.market.upper()} as of {card.as_of}[/dim]"
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("Pillar")
    table.add_column("Weight", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    table.add_column("Evidence")
    for pillar in card.pillars:
        table.add_row(
            pillar.name,
            f"{PILLAR_WEIGHTS.get(pillar.name, 0.0):.2f}",
            "-" if pillar.score is None else f"{pillar.score:.1f}",
            pillar.label,
            pillar.evidence or "-",
        )
    console.print(table)
    available = sum(1 for p in card.pillars if p.status == "ok")
    if card.composite is None:
        console.print("[yellow]Composite: n/a (all pillars skipped)[/yellow]")
    else:
        console.print(
            f"[bold]Composite conviction: {card.composite:.1f}/100[/bold] "
            f"[dim](weighted over {available} of {len(card.pillars)} pillars)[/dim]"
        )
