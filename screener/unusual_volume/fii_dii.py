"""Market-wide FII/DII net-flow overlay.

NSE publishes a single market-wide FII and DII provisional figure per trading
day (no per-symbol breakdown, current day only). We fetch today's record,
append it to an accumulating panel (``~/.screener/panels/fii_dii.parquet``),
derive 5-day net / trend metrics from the accumulated history, and broadcast
the *same* values onto every event in the scan (they are market-level signals,
not per-symbol).

Because NSE has no historical FII/DII archive on this endpoint, the 5d/trend
metrics are partial/None until enough daily runs accumulate (≥5 rows for the
5-day sums, ≥20 for the trend baseline). This is expected cold-start behaviour.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from screener.cache import append_panel_snapshot, panel_path, read_frame

from .detector import Event
from .nse_client import nse_cached_json

_FIIDII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
_PANEL = "fii_dii"


def fetch_fii_dii_today(*, refresh: bool = False) -> Optional[list]:
    raw = nse_cached_json(
        "nse_fii_dii",
        ("fiidii", str(date.today())),
        _FIIDII_URL,
        "fii/dii activity",
        refresh=refresh,
    )
    return raw if isinstance(raw, list) else None


def _as_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_fii_dii(raw: list, as_of: date) -> Optional[dict]:
    """Reduce the 2-row NSE payload to {date, fii_net, dii_net}."""
    if not raw:
        return None
    fii_net: Optional[float] = None
    dii_net: Optional[float] = None
    for row in raw:
        if not isinstance(row, dict):
            continue
        category = str(row.get("category", "")).upper()
        net = _as_float(row.get("netValue"))
        if net is None:
            buy = _as_float(row.get("buyValue"))
            sell = _as_float(row.get("sellValue"))
            net = (buy - sell) if buy is not None and sell is not None else None
        if "FII" in category or "FPI" in category:
            fii_net = net
        elif "DII" in category:
            dii_net = net
    if fii_net is None and dii_net is None:
        return None
    return {"date": as_of, "fii_net": fii_net, "dii_net": dii_net}


_METRIC_COLUMNS = ("fii_5d_net", "dii_5d_net", "fii_trend")


def fii_dii_metric_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Return live-equivalent FII/DII metrics indexed by normalized date."""
    if panel is None or panel.empty:
        return pd.DataFrame(columns=_METRIC_COLUMNS)
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = (
        df.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    if df.empty:
        return pd.DataFrame(columns=_METRIC_COLUMNS)
    rows: list[dict[str, float | pd.Timestamp | None]] = []
    for current in df["date"]:
        hist = df[df["date"] <= current]
        fii = pd.to_numeric(hist["fii_net"], errors="coerce").dropna()
        dii = pd.to_numeric(hist["dii_net"], errors="coerce").dropna()
        fii_5d = float(fii.tail(5).sum()) if not fii.empty else None
        dii_5d = float(dii.tail(5).sum()) if not dii.empty else None
        fii_trend: Optional[float] = None
        if len(fii) >= 5:
            baseline = float(fii.tail(20).mean())
            today = float(fii.iloc[-1])
            # A negative baseline preserves the sign of the ratio; only a zero
            # or NaN baseline is undefined.
            if baseline != 0.0 and not pd.isna(baseline):
                fii_trend = round(today / baseline, 4)
        rows.append(
            {
                "date": current,
                "fii_5d_net": fii_5d,
                "dii_5d_net": dii_5d,
                "fii_trend": fii_trend,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def compute_fii_dii_metrics(panel: pd.DataFrame, as_of: date) -> dict:
    """Derive fii_5d_net / dii_5d_net / fii_trend from the accumulated panel."""
    empty = {"fii_5d_net": None, "dii_5d_net": None, "fii_trend": None}
    metrics = fii_dii_metric_series(panel)
    if metrics.empty:
        return empty
    cutoff = pd.Timestamp(as_of).normalize()
    metrics = metrics[metrics.index <= cutoff]
    if metrics.empty:
        return empty
    latest = metrics.iloc[-1]
    return {
        col: None if pd.isna(latest[col]) else float(latest[col])
        for col in _METRIC_COLUMNS
    }


def overlay_fii_dii(
    events: list[Event], as_of: date, *, refresh: bool = False
) -> Optional[dict]:
    """Fetch + persist today's FII/DII, broadcast metrics onto every event."""
    raw = fetch_fii_dii_today(refresh=refresh)
    record = parse_fii_dii(raw, as_of) if raw else None
    panel: pd.DataFrame
    if record is not None:
        panel = append_panel_snapshot(
            _PANEL, pd.DataFrame([record]), dedupe_keys=["date"]
        )
    else:
        existing = read_frame(panel_path(_PANEL))
        panel = existing if existing is not None else pd.DataFrame()
    metrics = compute_fii_dii_metrics(panel, as_of)
    for ev in events:
        ev.fii_5d_net = metrics["fii_5d_net"]
        ev.dii_5d_net = metrics["dii_5d_net"]
        ev.fii_trend = metrics["fii_trend"]
    return metrics
