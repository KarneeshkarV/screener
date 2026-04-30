"""NSE bhavcopy delivery overlay.

Pulls daily ``sec_bhavdata_full`` files via jugaad-data, computes delivery_pct,
delivery_rvol, and conviction_score for each (symbol, date) pair, and joins
the result onto a detector ``Event``.

The bhavcopy CSVs ship column names with a leading space (`` SERIES`` rather
than ``SERIES``). We strip that quirk on load. SERIES is filtered to cash
equity (`EQ`, `BE`, `BZ`) so government securities and SME segments are
excluded — those have very different delivery profiles and would distort
the SMA window.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .detector import Event


CACHE_DIR = Path.home() / ".screener" / "bhavcopy"
CASH_SERIES = {"EQ", "BE", "BZ"}
DELIVERY_SMA_WINDOW = 20
QUIET_DELIVERY_RVOL = 2.0
HIGH_DELIVERY_PCT = 50.0
LOW_DELIVERY_PCT = 25.0
LONG_HOLDER_DELIVERY_PCT = 60.0


def _load_one_day(dt: date) -> Optional[pd.DataFrame]:
    """Return a (date, symbol)-indexed DataFrame for one trading day.

    Returns ``None`` on any failure (404 = market holiday, network glitch,
    parse error). Caller is expected to skip silently and move on.
    """
    from jugaad_data.nse import full_bhavcopy_save  # lazy import

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path = full_bhavcopy_save(dt, str(CACHE_DIR))
    except Exception:
        return None
    if not path or not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    needed = {"SYMBOL", "SERIES", "DATE1", "TTL_TRD_QNTY", "DELIV_QTY", "DELIV_PER"}
    if not needed.issubset(df.columns):
        return None
    for col in ("SYMBOL", "SERIES", "DATE1"):
        df[col] = df[col].astype(str).str.strip()
    df = df[df["SERIES"].isin(CASH_SERIES)].copy()
    df["date"] = pd.to_datetime(df["DATE1"], format="%d-%b-%Y", errors="coerce").dt.date
    df["TTL_TRD_QNTY"] = pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce")
    df["DELIV_QTY"] = pd.to_numeric(df["DELIV_QTY"], errors="coerce")
    df["DELIV_PER"] = pd.to_numeric(df["DELIV_PER"], errors="coerce")
    return df[["SYMBOL", "date", "TTL_TRD_QNTY", "DELIV_QTY", "DELIV_PER"]]


def load_delivery_panel(
    symbols: Iterable[str],
    as_of: date,
    history_days: int = 40,
) -> pd.DataFrame:
    """Return the (symbol, date)-indexed delivery panel for the given window.

    ``history_days`` is in calendar days; pulled across roughly that many
    business days backwards from ``as_of`` so the 20-bar delivery SMA has a
    clean window. Holidays and weekends silently produce no rows for that
    date — the consumer downstream computes RVOL on whatever bars exist.
    """
    sym_set = {s.upper() for s in symbols}
    frames: list[pd.DataFrame] = []
    cur = as_of
    earliest = as_of - timedelta(days=history_days)
    while cur >= earliest:
        if cur.weekday() < 5:  # skip weekends locally; jugaad still hits archives though
            day = _load_one_day(cur)
            if day is not None and not day.empty:
                frames.append(day[day["SYMBOL"].isin(sym_set)])
        cur -= timedelta(days=1)
    if not frames:
        return pd.DataFrame(
            columns=["SYMBOL", "date", "TTL_TRD_QNTY", "DELIV_QTY", "DELIV_PER"]
        )
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["SYMBOL", "date"]).reset_index(drop=True)
    return panel


def compute_delivery_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """Add delivery_rvol + conviction_score columns to a delivery panel."""
    if panel.empty:
        out = panel.copy()
        for col in ("delivery_rvol", "conviction_score"):
            out[col] = pd.Series(dtype=float)
        return out
    panel = panel.copy()
    panel["delivery_sma_20"] = (
        panel.groupby("SYMBOL")["DELIV_QTY"]
        .transform(
            lambda s: s.shift(1).rolling(DELIVERY_SMA_WINDOW, min_periods=5).mean()
        )
    )
    panel["delivery_rvol"] = panel["DELIV_QTY"] / panel["delivery_sma_20"]
    # Rolling mean of DELIV_PER for build-up detection — the per-bar
    # delivery_pct is too noisy on its own; sustained-elevation over weeks is
    # what marks accumulation.
    panel["delivery_pct_sma_20"] = (
        panel.groupby("SYMBOL")["DELIV_PER"]
        .transform(
            lambda s: s.rolling(DELIVERY_SMA_WINDOW, min_periods=5).mean()
        )
    )
    return panel


def _delivery_notes(rvol: float, delivery_pct: Optional[float], direction: str) -> str:
    if delivery_pct is None or pd.isna(delivery_pct):
        return ""
    notes: list[str] = []
    if rvol >= 3.0 and delivery_pct >= HIGH_DELIVERY_PCT:
        notes.append("strong institutional footprint")
    elif rvol >= 3.0 and delivery_pct < LOW_DELIVERY_PCT:
        notes.append("speculative/operator-driven; low conviction")
    if (
        direction == "SELLING"
        and rvol >= 3.0
        and delivery_pct > LONG_HOLDER_DELIVERY_PCT
    ):
        notes.append("long-holder distribution")
    return "; ".join(notes)


def overlay_events(
    events: list[Event], panel: pd.DataFrame
) -> list[Event]:
    """Mutate ``events`` in place with delivery_qty / pct / rvol / conviction."""
    if not events:
        return events
    panel = compute_delivery_metrics(panel)
    if panel.empty:
        return events
    panel = panel.drop_duplicates(subset=["SYMBOL", "date"], keep="last")
    indexed = panel.set_index(["SYMBOL", "date"])
    for ev in events:
        key = (ev.symbol.upper(), ev.date)
        if key not in indexed.index:
            continue
        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        ev.delivery_qty = (
            float(row["DELIV_QTY"]) if not pd.isna(row["DELIV_QTY"]) else None
        )
        ev.delivery_pct = (
            float(row["DELIV_PER"]) if not pd.isna(row["DELIV_PER"]) else None
        )
        ev.delivery_rvol = (
            float(row["delivery_rvol"]) if not pd.isna(row["delivery_rvol"]) else None
        )
        if ev.delivery_pct is not None and ev.rvol == ev.rvol:  # not NaN
            ev.conviction_score = round(ev.rvol * (ev.delivery_pct / 100.0), 4)
        notes = _delivery_notes(ev.rvol, ev.delivery_pct, ev.direction)
        if notes:
            ev.notes = (ev.notes + "; " + notes).strip("; ") if ev.notes else notes
    return events


def quiet_accumulation_events(
    bars_by_symbol: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    as_of: date,
    min_rvol_skip: float,
    existing_events: Optional[Iterable[Event]] = None,
) -> list[Event]:
    """Surface 'quiet accumulation' bars: delivery RVOL >= 2 even though
    raw volume RVOL is below the unusual-volume threshold.

    These are events the regular detector would discard. They need to be
    re-built from scratch because we explicitly want bars that *failed*
    the volume thresholds. Symbols already emitted by the regular detector
    are skipped so a z-score-only detector event is not duplicated.
    """
    if panel.empty:
        return []
    existing_symbols = {ev.symbol.upper() for ev in existing_events or []}
    panel = compute_delivery_metrics(panel)
    as_of_ts = pd.Timestamp(as_of).normalize()
    out: list[Event] = []
    today = panel[
        (panel["date"] == as_of)
        & (panel["delivery_rvol"] >= QUIET_DELIVERY_RVOL)
    ]
    for _, row in today.iterrows():
        sym = str(row["SYMBOL"]).upper()
        if sym in existing_symbols:
            continue
        bars = bars_by_symbol.get(sym)
        if bars is None or bars.empty:
            continue
        df = bars.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"]).values))
            else:
                continue
        df = df[df.index <= as_of_ts]
        if df.empty:
            continue
        last = df.iloc[-1]
        v = float(last["volume"])
        avg20 = (
            float(df["volume"].rolling(20, min_periods=20).mean().shift(1).iloc[-1])
            if len(df) >= 21
            else float("nan")
        )
        rvol = v / avg20 if avg20 and avg20 > 0 else float("nan")
        if not pd.isna(rvol) and rvol >= min_rvol_skip:
            continue  # already covered by the regular detector
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else float(last["close"])
        pct_change = (
            (float(last["close"]) - prev_close) / prev_close * 100.0
            if prev_close > 0
            else 0.0
        )
        delivery_pct = (
            float(row["DELIV_PER"]) if not pd.isna(row["DELIV_PER"]) else None
        )
        delivery_rvol = (
            float(row["delivery_rvol"]) if not pd.isna(row["delivery_rvol"]) else None
        )
        conviction = (
            round((rvol if not pd.isna(rvol) else 1.0) * (delivery_pct or 0.0) / 100.0, 4)
            if delivery_pct is not None
            else None
        )
        ev = Event(
            symbol=sym,
            date=as_of,
            close=float(last["close"]),
            pct_change=round(pct_change, 4),
            volume=v,
            avg_volume_20d=avg20 if not pd.isna(avg20) else 0.0,
            rvol=round(rvol, 4) if not pd.isna(rvol) else float("nan"),
            rvol_5d=float("nan"),
            rvol_50d=float("nan"),
            rvol_90d=float("nan"),
            z_score=float("nan"),
            pct_rank_252d=float("nan"),
            direction="QUIET_ACCUMULATION",
            strength="MODERATE",
            delivery_qty=(
                float(row["DELIV_QTY"]) if not pd.isna(row["DELIV_QTY"]) else None
            ),
            delivery_pct=delivery_pct,
            delivery_rvol=delivery_rvol,
            conviction_score=conviction,
            notes="quiet accumulation: delivery surge without volume spike",
        )
        out.append(ev)
    return out
