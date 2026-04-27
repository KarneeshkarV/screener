"""NSE Bhavcopy fetchers for the Operator Intent screener.

Two raw NSE feeds are consumed:

  Cash Bhavcopy with delivery: ``sec_bhavdata_full_<DD><MMM><YYYY>bhav.csv``
    Provides per-symbol PREV_CLOSE, CLOSE_PRICE, AVG_PRICE (= official daily
    VWAP), DELIV_QTY and DELIV_PER. Fetched via
    ``jugaad_data.nse.NSEArchives.full_bhavcopy_save``.

  F&O UDiff Bhavcopy:        ``BhavCopy_NSE_FO_0_0_0_<YYYYMMDD>_F_0000.csv``
    Provides per-contract OpnIntrst by expiry. Fetched via
    ``NSEArchives.daily_reports.download_file('FO-UDIFF-BHAVCOPY-CSV', …)``.
    The legacy ``fo<DD><MMM><YYYY>bhav.csv.zip`` URL was retired by NSE in
    mid-2024; ``bhavcopy_fo_save`` against that URL now 404s.

Raw downloads are cached at ``~/.screener/nse_bhavcopy/<YYYY-MM-DD>/`` so
re-runs on the same date hit disk only.

52-week High/Low is *not* re-fetched — the existing autoresearch parquet
cache at ``./.autoresearch/ohlcv/india/<SYMBOL>__*.parquet`` already holds
~4 years of daily closes per ticker.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

LOG = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".screener" / "nse_bhavcopy"
INDIA_OHLCV_CACHE = Path(".autoresearch/ohlcv/india")
TRADING_DAY_LOOKBACK = 7  # how far back to walk if today's bhavcopy is missing


# ── trading-day resolution ─────────────────────────────────────────────


def latest_trading_day(d: date, *, lookback: int = TRADING_DAY_LOOKBACK) -> date:
    """Return the most recent NSE trading day on or before ``d``.

    NSE's cash bhavcopy archive is sticky on weekends/holidays — requesting
    the file for a Sunday silently returns Friday's CSV. We therefore parse
    the DATE1 column to learn the *actual* trading day represented by the
    bytes we got back, and trust that, rather than the URL date.
    """
    for delta in range(lookback + 1):
        candidate = d - timedelta(days=delta)
        try:
            df = _read_cash_bhavcopy_raw(candidate)
        except Exception as exc:
            LOG.debug("cash bhavcopy fetch failed for %s: %s", candidate, exc)
            continue
        if df is None or df.empty:
            continue
        actual = _parse_bhavcopy_date(df)
        if actual is None:
            continue
        if actual != d:
            LOG.warning("bhavcopy for %s unavailable; using %s (NSE returned data for that day)", d, actual)
        return actual
    raise RuntimeError(
        f"no NSE cash bhavcopy found within {lookback} days of {d}"
    )


def _parse_bhavcopy_date(df: pd.DataFrame) -> Optional[date]:
    if "DATE1" not in df.columns or df.empty:
        return None
    raw = str(df["DATE1"].iloc[0]).strip()
    try:
        return pd.to_datetime(raw, dayfirst=True).date()
    except Exception:
        return None


# ── cash bhavcopy ──────────────────────────────────────────────────────


def _cash_cache_path(d: date) -> Path:
    """Cache path keyed by the *URL* date (which is what was requested).

    Because NSE silently serves the previous trading day's file on weekends,
    two URL dates can map to identical bytes — we cache by the URL date and
    let ``latest_trading_day`` resolve the actual trading day from DATE1.
    """
    day_dir = CACHE_ROOT / d.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    # Match jugaad's filename convention so the helper finds it on second run
    return day_dir / f"sec_bhavdata_full_{d.strftime('%d%b%Y')}bhav.csv"


def _read_cash_bhavcopy_raw(d: date) -> pd.DataFrame:
    """Download (or load) the raw cash bhavcopy CSV with no filtering."""
    from jugaad_data.nse import NSEArchives  # lazy import for tests

    path = _cash_cache_path(d)
    if not path.exists():
        n = NSEArchives()
        n.full_bhavcopy_save(d, str(path.parent))
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def fetch_cash_bhavcopy(d: date) -> pd.DataFrame:
    """Cleaned cash bhavcopy keyed by SYMBOL (SERIES == 'EQ').

    Returns the delivery + VWAP columns the screener needs, with numeric
    columns coerced to floats (NaN on failure).
    """
    df = _read_cash_bhavcopy_raw(d)
    df = df[df["SERIES"] == "EQ"].copy()
    numeric_cols = [
        "PREV_CLOSE", "CLOSE_PRICE", "AVG_PRICE",
        "TTL_TRD_QNTY", "DELIV_QTY", "DELIV_PER",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["SYMBOL"] + numeric_cols
    return df[keep].reset_index(drop=True)


# ── F&O UDiff bhavcopy ─────────────────────────────────────────────────


def _fo_cache_path(d: date) -> Path:
    day_dir = CACHE_ROOT / d.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"BhavCopy_NSE_FO_{d.strftime('%Y%m%d')}.csv"


FO_ARCHIVE_URL = (
    "https://nsearchives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)


def fetch_fo_bhavcopy(d: date) -> pd.DataFrame:
    """Download (or load from cache) the F&O UDiff Bhavcopy.

    Returns one row per (SYMBOL, EXPIRY_DT) for stock futures (FinInstrmTp =
    'STF'), with the cumulative OI per contract. Stocks not in the F&O list
    will simply be absent from the frame.

    The fetch hits the historical archive URL directly (which serves all
    dates back to 2024-07-08 when NSE switched to the UDiff format). The
    ``daily_reports`` API is only fronted CurrentDay+PreviousDay, so it
    fails for older dates — we skip it entirely.
    """
    from jugaad_data.nse import NSEArchives

    path = _fo_cache_path(d)
    if not path.exists():
        n = NSEArchives()
        url = FO_ARCHIVE_URL.format(yyyymmdd=d.strftime("%Y%m%d"))
        r = n.s.get(url, timeout=10)
        if r.status_code != 200 or r.content[:2] != b"PK":
            raise RuntimeError(
                f"FO bhavcopy fetch failed for {d}: HTTP {r.status_code}, "
                f"content-type={r.headers.get('content-type')!r}"
            )
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fp:
                path.write_bytes(fp.read())
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["FinInstrmTp"] == "STF"].copy()  # stock futures only
    df["XpryDt"] = pd.to_datetime(df["XpryDt"], errors="coerce")
    df["OpnIntrst"] = pd.to_numeric(df["OpnIntrst"], errors="coerce")
    df = df.rename(columns={"TckrSymb": "SYMBOL", "OpnIntrst": "OI", "XpryDt": "EXPIRY"})
    return df[["SYMBOL", "EXPIRY", "OI"]].reset_index(drop=True)


def near_month_oi(fo_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-expiry rows to (SYMBOL, Current_OI, Next_OI, Cumulative_OI).

    For each symbol take the two earliest expiries by date — these are the
    "current month" and "next month" futures contracts that drive the
    Operator Intent signal. Symbols with only one expiry contribute
    Current_OI only; Next_OI is NaN.
    """
    fo_df = fo_df.sort_values(["SYMBOL", "EXPIRY"])
    rows = []
    for sym, g in fo_df.groupby("SYMBOL"):
        ois = g["OI"].tolist()[:2]
        cur = ois[0] if len(ois) >= 1 else float("nan")
        nxt = ois[1] if len(ois) >= 2 else float("nan")
        # Cumulative_OI = Current_OI + Next_OI (per spec). If next is missing,
        # treat Cumulative_OI as just Current_OI rather than NaN, since the
        # signal logic only needs a comparable per-symbol total day-over-day.
        cumulative = (cur or 0.0) + (nxt if pd.notna(nxt) else 0.0)
        rows.append({"SYMBOL": sym, "Current_OI": cur, "Next_OI": nxt,
                     "Cumulative_OI": cumulative})
    return pd.DataFrame(rows)


# ── 52-week High / Low from the parquet cache ──────────────────────────


def _resolve_parquet(symbol: str) -> Optional[Path]:
    """Pick the deepest-history parquet file for ``symbol`` from the cache.

    Files are named ``<SYMBOL>__<start>__<end>.parquet`` and many tickers
    have multiple snapshots. We pick the one with the earliest start date
    (most history) to ensure 252+ trading days are available.
    """
    if not INDIA_OHLCV_CACHE.exists():
        return None
    matches = sorted(INDIA_OHLCV_CACHE.glob(f"{symbol}__*.parquet"))
    if not matches:
        return None
    # Earliest start date = smallest second token after the first '__'
    return min(matches, key=lambda p: p.name.split("__")[1])


def fifty_two_week_hl(symbols: Iterable[str], as_of: date) -> pd.DataFrame:
    """Compute 52-week High/Low using the autoresearch parquet cache.

    Only symbols with at least 200 cached trading days are returned with
    populated values; the rest get NaN H/L (the screener tolerates missing
    52W metrics — Operator_Action does not depend on them).
    """
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=400)  # buffer for non-trading days
    ts_as_of = pd.Timestamp(as_of)
    rows = []
    for sym in symbols:
        path = _resolve_parquet(sym)
        if path is None:
            rows.append({"SYMBOL": sym, "_52W_High": float("nan"),
                         "_52W_Low": float("nan")})
            continue
        try:
            df = pd.read_parquet(path, columns=["date", "high", "low", "close"])
        except Exception:
            rows.append({"SYMBOL": sym, "_52W_High": float("nan"),
                         "_52W_Low": float("nan")})
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        window = df[(df["date"] >= cutoff) & (df["date"] <= ts_as_of)]
        if len(window) < 200:
            rows.append({"SYMBOL": sym, "_52W_High": float("nan"),
                         "_52W_Low": float("nan")})
            continue
        # Use intraday high/low if available, else close — matches the NSE
        # convention of "52-week high" being the highest traded price.
        rows.append({
            "SYMBOL": sym,
            "_52W_High": float(window["high"].max()),
            "_52W_Low": float(window["low"].min()),
        })
    return pd.DataFrame(rows)
