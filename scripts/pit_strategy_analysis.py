"""Per-strategy verdicts for the India PIT HTML report."""

from __future__ import annotations

from typing import Any

import pandas as pd

UNIVERSES = ("midsmall", "n500", "n50", "mid", "small")
PRIMARY = ("midsmall", "mid", "small")
LABEL = {
    "mid": "mid",
    "small": "small",
    "midsmall": "combined",
    "n50": "nifty 50",
    "n500": "nifty 500",
}
ROLE_ORDER = ("core", "overlay", "ok", "thin", "avoid", "broken")

WHAT: dict[str, str] = {
    "momentum_12_1": "Jegadeesh-Titman 12-1 cross-section. Long names with a positive 12-month return, skip the last month.",
    "momentum_12_1_trend": "Same 12-1 momentum, plus a dual-momentum gate: price must sit above the 200-day SMA.",
    "tsmom_12_1": "Moskowitz time-series momentum. Long a name only when its own 12-1 return is positive.",
    "nifty_momentum": "NSE Momentum 50 recipe. Rank by 6-month and 12-month return, skip the last month.",
    "nifty_momentum_trend": "NSE Momentum 50 recipe plus a 200-day SMA trend gate.",
    "hurst_trend_quality": "Hurst persistence. Long names whose price path trends (H > 0.55) and whose 12-month return is positive.",
    "ema150_200_revenue_up_3q": "Price above EMA150 and EMA200, and revenue up for three quarters in a row.",
    "earnings_momentum": "Sequential earnings growth. Long names with rising EPS, hold about two quarters.",
    "pead_drift": "Post-earnings announcement drift. Enter after a report and hold about one quarter.",
    "quality_lowvol": "Nifty Quality-style gate (ROE, leverage, EPS growth), then rank by lowest 252-day volatility.",
    "quality_lowbeta": "Same quality gate as quality_lowvol, then rank by lowest 252-day beta.",
    "quality_stability": "Quality gate plus three-quarter revenue growth and a shallow drawdown, then rank by low downside vol.",
    "quality_value": "Quality gate plus PE and PB caps, then rank by lowest volatility.",
    "quality_mom_lowvol": "Quality gate plus 12-1 momentum, then rank by low volatility.",
    "momentum_quality": "Quality-gated 12-1 momentum. Almost the same book as quality_mom_lowvol.",
    "momentum_quality_pe": "Quality-gated momentum with a PE cap.",
    "momentum_quality_pb": "Quality-gated momentum with a PB cap.",
    "momentum_quality_pe60": "Quality-gated momentum with a PE <= 60 cap.",
    "max_avoidance": "Skip lottery names (high MAX daily return). Rank the rest by low crash risk.",
    "seasonal_strong_trend": "Hold only in historically strong India months, and only when the name is in a long-term uptrend.",
    "conservative_investment": "Fama-French CMA. Long low asset-growth names inside a value screen, with a Z-score floor.",
    "garp": "Growth at a reasonable price. Quality plus a valuation cap.",
    "value_rank": "Cheap on PE / PB / similar value ranks. Few quality gates.",
    "value_momentum_harness": "Cheap names that also have positive 12-1 momentum.",
    "deep_value": "Deep cheap. Fewer quality gates than GARP, so the book is thin.",
    "qmj_quality": "Asness Quality Minus Junk. Needs a full fundamental stack.",
    "gross_profitability": "Novy-Marx gross profit over assets, then rank by 6-month momentum.",
    "sloan_low_accruals": "Sloan accruals anomaly. Needs operating cash flow. Fail-closed when cash-flow is missing.",
    "piotroski_value": "Piotroski F-score inside cheap value. Needs a complete statement set. Fail-closed when missing.",
    "fcf_yield_value": "Free-cash-flow yield. Needs cash-flow statements. Fail-closed when missing.",
    "low_idio_vol": "Ang low idiosyncratic volatility. Long the quiet residual-vol names.",
    "betting_against_beta": "Frazzini-Pedersen BAB. Long low-beta names.",
    "downside_risk": "Long names with low downside deviation versus the market.",
    "low_volatility": "Ang-Hodrick low realized volatility. No quality gate.",
    "mom_lowvol_combo": "12-1 momentum blended with low realized volatility. Always invested.",
    "lt_reversal_path": "De Bondt-Thaler long-term reversal. Long the 3-to-5 year losers.",
    "str_reversal_trend": "Jegadeesh 1-month reversal, only when the longer trend is still up.",
    "gw52_proximity": "George-Hwang. Long names sitting near a 52-week high.",
    "hs_same_month": "Heston-Sadka. Long names that were strong in this calendar month in prior years.",
    "kama_trend": "Kaufman adaptive moving average. Enter when efficiency ratio is high and price is above KAMA.",
    "ma_timing_200": "Binary timing. Long when price is above the 200-day SMA.",
    "rs_trend": "Relative-strength trend versus the universe. Always invested.",
    "rs_breakout": "Relative-strength breakout. Short hold, high turnover.",
    "rs_momentum_regime": "Relative strength rising and price above EMA200.",
    "vwap_trend": "Price above VWAP and VWAP rising.",
    "vwap_reversion": "Mean-revert toward VWAP. Short hold.",
    "chandelier_breakout": "ATR chandelier breakout. Long hold (250 sessions).",
    "turtle_breakout": "Original Turtle Donchian dual-channel breakout.",
    "supertrend_expr": "SuperTrend long-only. Stay long while the indicator points up.",
    "vcp_breakout": "Minervini-style volatility contraction, then a breakout.",
    "vol_expansion_breakout": "Donchian breakout confirmed by an ATR expansion.",
    "keltner_squeeze_breakout": "TTM-style Keltner squeeze, then a breakout.",
    "vol_target_lowvol": "Inverse-vol targeting on a low-vol universe. Sizing, not alpha.",
    "volume_surge": "Gervais high-volume return premium. Short hold.",
    "obv_flow_trend": "Granville on-balance volume in an uptrend.",
    "cmf_flow_factor": "Chaikin money flow, cross-section rank.",
    "delivery_accumulation": "India delivery percent rising, with price confirmation.",
    "tom_window_trend": "Ariel turn-of-month window, with a trend gate.",
    "pre_holiday_trend": "Pre-holiday drift. Hold about 10 sessions.",
    "nov_apr_trend": "November to April seasonality (sell in May), with a trend gate.",
    "breakout": "52-week / channel breakout. Hold 20 sessions.",
    "ema_trend": "Ride EMA20 above EMA200. Exit on a close under EMA20.",
    "mark_minervini": "Minervini trend template. Stage-2 growth, short hold.",
    "minervini_growth_in": "Minervini template plus an India growth fundamental gate.",
    "minervini_pro_in": "Stricter Minervini plus India fundamentals.",
    "vivek_equity_tool": "Custom Pine tool. Short-hold expression set.",
}

# Prefer the left-hand name when two books are the same.
CLONE_CANONICAL = {
    "quality_lowbeta": "quality_lowvol",
    "quality_mom_lowvol": "momentum_quality",
}
ROLE_FORCE = {
    "ema150_200_revenue_up_3q": "core",
}


def _f(n: Any) -> float | None:
    if n is None:
        return None
    try:
        v = float(n)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None
    return v


def _cell(g: pd.DataFrame) -> dict[str, Any]:
    by = {int(r.years): r for r in g.itertuples()}

    def get(y: int, col: str) -> float | None:
        r = by.get(y)
        return _f(getattr(r, col, None)) if r is not None else None

    sharpes = [s for s in (get(1, "sharpe"), get(2, "sharpe"), get(3, "sharpe"), get(5, "sharpe")) if s is not None]
    s3, s5 = get(3, "sharpe"), get(5, "sharpe")
    s35 = None if s3 is None or s5 is None else (s3 + s5) / 2
    return {
        "s1": get(1, "sharpe"),
        "s2": get(2, "sharpe"),
        "s3": s3,
        "s5": s5,
        "s35": s35,
        "mean": sum(sharpes) / len(sharpes) if sharpes else None,
        "min_s": min(sharpes) if sharpes else None,
        "cagr5": get(5, "cagr"),
        "dd5": get(5, "max_drawdown"),
        "exp5": get(5, "exposure"),
        "t5": int(get(5, "n_trades") or 0),
        "hit5": get(5, "hit_rate"),
    }


def _all_zero(rec: dict[str, Any]) -> bool:
    cells = rec.get("by") or {}
    if not cells:
        return False
    return all((c.get("t5") or 0) == 0 for c in cells.values())


def _clone_map(by_name: dict[str, dict[str, Any]]) -> dict[str, str]:
    found = dict(CLONE_CANONICAL)
    names = sorted(by_name)
    for i, a in enumerate(names):
        if _all_zero(by_name[a]):
            continue
        va = by_name[a]["by"].get("midsmall")
        if not va:
            continue
        vec_a = [va.get(k) for k in ("s1", "s2", "s3", "s5")]
        if any(x is None for x in vec_a):
            continue
        if max(abs(x) for x in vec_a) < 0.02:
            continue
        for b in names[i + 1 :]:
            if a in found or b in found or _all_zero(by_name[b]):
                continue
            vb = by_name[b]["by"].get("midsmall")
            if not vb:
                continue
            vec_b = [vb.get(k) for k in ("s1", "s2", "s3", "s5")]
            if any(x is None for x in vec_b):
                continue
            if max(abs(x - y) for x, y in zip(vec_a, vec_b)) < 0.02:
                found[b] = a
    return found


def _mean_s(cells: list[dict[str, Any]]) -> float | None:
    vals = [c["mean"] for c in cells if c.get("mean") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _classify(rec: dict[str, Any]) -> str:
    by = rec["by"]
    if _all_zero(rec):
        return "broken"
    primary = [by[u] for u in PRIMARY if u in by]
    if not primary:
        primary = list(by.values())
    t5p = [c["t5"] for c in primary]
    mean_primary = _mean_s(primary)
    primary_invested = [
        c
        for c in primary
        if (c.get("exp5") or 0) >= 0.80 and c["t5"] >= 20 and c.get("s35") is not None
    ]
    if not primary_invested:
        if mean_primary is not None and mean_primary < 0.15:
            return "avoid"
        if max(t5p or [0]) < 12:
            return "thin"
        return "overlay"
    best = max(primary_invested, key=lambda c: c["s35"] or -99)
    s35 = best["s35"] or 0.0
    mean_s = best["mean"] or 0.0
    dd = best["dd5"] if best["dd5"] is not None else 0.0
    s1, s2 = best.get("s1") or 0.0, best.get("s2") or 0.0
    hold = int(rec.get("hold") or 0)
    if s35 < 0.45 or mean_s < 0.20:
        return "avoid"
    if dd < -0.38 and s35 < 1.15:
        return "avoid"
    if s35 >= 1.20:
        if s1 < 0.20 and s2 < 0.30:
            return "ok"
        if best["t5"] >= 500 and hold <= 20 and (s1 < 0.5 or s2 < 0.4):
            return "ok"
        return "core"
    if s35 >= 0.75:
        return "ok"
    return "avoid"


def _best_book(rec: dict[str, Any]) -> dict[str, Any] | None:
    invested = []
    rest = []
    for u, c in rec["by"].items():
        if c.get("s35") is None:
            continue
        item = (u, c)
        if (c.get("exp5") or 0) >= 0.80 and c["t5"] >= 20:
            invested.append(item)
        else:
            rest.append(item)
    pool = invested or rest
    if not pool:
        return None

    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, float]:
        u, c = item
        bonus = 0.08 if u in PRIMARY else 0.0
        return ((c["s35"] or -99) + bonus, c.get("exp5") or 0.0)

    u, c = max(pool, key=key)
    return {"universe": u, **c}


def _fmt(n: float | None, d: int = 2) -> str:
    if n is None:
        return "-"
    return f"{n:.{d}f}"


def _pct(n: float | None) -> str:
    if n is None:
        return "-"
    return f"{n * 100:.0f}%"


def _regime(cell: dict[str, Any] | None) -> str:
    if not cell:
        return ""
    s1, s2, s3, s5 = cell.get("s1"), cell.get("s2"), cell.get("s3"), cell.get("s5")
    bull = (s3 or 0) > 1.2 and (s5 or 0) > 1.1
    flat_ok = (s1 or 0) >= 0.5 and (s2 or 0) >= 0.4
    flat_bad = (s1 or 0) < 0.3 or (s2 or 0) < 0.2
    if bull and flat_ok:
        return "Holds in the flat 1y/2y window and in the 2023-2024 bull."
    if bull and flat_bad:
        return "The 3y/5y numbers ride the 2023-2024 bull. 1y and 2y are weak."
    if flat_ok and not bull:
        return "Better in the flat 1y/2y window than in the long bull."
    return ""


def _verdict(rec: dict[str, Any]) -> str:
    name = rec["name"]
    role = rec["role"]
    best = rec.get("best")
    clone = rec.get("clone")
    hold = rec["hold"]
    bits: list[str] = []
    if clone:
        bits.append(f"This is the same book as {clone} on combined mid+small. Keep one.")
        return " ".join(bits)
    if role == "broken":
        bits.append(
            "Zero trades on every universe. Live FMP is 401 and the local cache has no cash-flow statements, so the gate fails closed."
        )
        return " ".join(bits)
    if best:
        lab = LABEL.get(best["universe"], best["universe"])
        bits.append(
            f"Best book: {lab}. "
            f"3y+5y Sharpe {_fmt(best.get('s35'))}. "
            f"5y CAGR {_pct(best.get('cagr5'))}. "
            f"5y drawdown {_pct(best.get('dd5'))}. "
            f"5y exposure {_pct(best.get('exp5'))}. "
            f"{best.get('t5') or 0} trades. Hold {hold} sessions."
        )
        bits.append(_regime(best))
    if role == "core":
        bits.append("This can be a book. It stays invested and the 3y+5y edge is large enough.")
    elif role == "overlay":
        bits.append("Sharpe is high because the book sits in cash. Do not size it as the core.")
    elif role == "ok":
        bits.append("Usable, but weaker than the momentum core on the same universe.")
    elif role == "avoid":
        bits.append("Do not use. The return does not pay for the drawdown, or the mean Sharpe is too low.")
    elif role == "thin":
        bits.append("Too few completed trades to trust.")
    extra = OVERRIDE.get(name)
    if extra:
        bits.append(extra)
    return " ".join(b for b in bits if b)


OVERRIDE: dict[str, str] = {
    "momentum_12_1_trend": "This is the pick for combined mid+small. The SMA200 gate cuts 1y/2y damage versus raw 12-1 without cutting 5y CAGR.",
    "momentum_12_1": "Use this if the book is small-cap only. On combined, prefer the trend-gated twin.",
    "hurst_trend_quality": "Highest 1y Sharpe among fully invested names on combined, and 27% 5y CAGR. Accept a 40% hole or do not use it.",
    "tsmom_12_1": "Highest 5y CAGR on combined (29%). Fewer trades than 12-1. Drawdown is in the Hurst zone. A cousin, not a second book.",
    "nifty_momentum": "Best fully invested mid-only name on 3y+5y. Weak 1y/2y. Use as a mid book, not as 1y defense.",
    "nifty_momentum_trend": "Same NSE recipe plus SMA200. Prefer this over raw nifty_momentum when you want a milder 1y/2y.",
    "ema150_200_revenue_up_3q": "Tightest Sharpe range on combined. Always invested. A conservative fully invested alternative, not a higher-CAGR book.",
    "earnings_momentum": "Works on combined and stays mostly invested. Needs fundamentals. Do not pair it with 12-1 as a second full book. The overlap is large.",
    "pead_drift": "Event book, fully invested, enough trades. Add it as a sleeve next to momentum. Do not replace momentum with it.",
    "quality_lowvol": "The only quality name that stays mostly invested. Combined exposure is about 70%. Nifty 500 is the invested book. Use as a quieter tilt, not as a replacement for 12-1.",
    "minervini_growth_in": "High turnover (hold 20, 700+ trades). 1y/2y are weak. Do not use as the core book.",
    "quality_value": "Highest mean Sharpe on combined. 30% exposure and 12% CAGR. The cash is the Sharpe.",
    "momentum_quality": "Almost the same trades as quality_mom_lowvol. Keep one.",
    "seasonal_strong_trend": "Calendar timing. 33% exposure. Overlay only.",
    "max_avoidance": "Best as a Nifty 500 tilt. Combined exposure is 64%. Do not make it the core.",
    "sloan_low_accruals": "Re-run when a working FMP key can fill cash-flow.",
    "piotroski_value": "Led some earlier Nifty 500 studies. This run cannot test it.",
    "fcf_yield_value": "Re-run when cash-flow yield is available.",
    "vwap_reversion": "Worst name in the set. Negative Sharpe and deep holes.",
    "keltner_squeeze_breakout": "Negative mean, long hold, deep drawdown.",
    "kama_trend": "Overtrading. 1,000+ trades and a negative mean.",
    "low_volatility": "Low-vol without a quality gate fails 1y/2y on this universe.",
    "vol_target_lowvol": "A sizer, not an alpha rule. Negative here with equal slots.",
    "tom_window_trend": "High floor on small, about 20% exposure, tiny CAGR. Not a book.",
}


def analyze(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for (univ, strat), g in df.groupby(["universe", "strategy"]):
        rec = out.setdefault(
            strat,
            {
                "name": strat,
                "family": g["family"].iloc[0] if "family" in g.columns else "",
                "hold": int(g["hold"].iloc[0] or 0) if "hold" in g.columns else 0,
                "what": WHAT.get(strat, "Named strategy in this study."),
                "by": {},
            },
        )
        rec["by"][univ] = _cell(g)
    clones = _clone_map(out)
    rows: list[dict[str, Any]] = []
    for name, rec in out.items():
        rec["clone"] = clones.get(name)
        rec["role"] = "overlay" if rec["clone"] else ROLE_FORCE.get(name) or _classify(rec)
        rec["best"] = _best_book(rec)
        rec["verdict"] = _verdict(rec)
        rows.append(rec)
    rows.sort(
        key=lambda r: (
            ROLE_ORDER.index(r["role"]) if r["role"] in ROLE_ORDER else 9,
            -(r["best"]["s35"] if r.get("best") and r["best"].get("s35") is not None else -99),
            r["name"],
        )
    )
    return rows
