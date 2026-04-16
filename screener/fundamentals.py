"""Historical point-in-time fundamentals via yfinance quarterly reports.

Metrics are reconstructed using a 45-day reporting lag:
a quarter's data is treated as available on (period_end + 45 days),
so there is no lookahead bias from using unreported earnings.

Cached as JSON under ~/.screener/fundamentals/{market}/{symbol}.json.
Cache is valid for CACHE_MAX_AGE_DAYS (7 by default).

Returned snapshot dict keys:
    pe          — Price / TTM Diluted EPS  (None if unavailable)
    roe         — TTM Net Income / Common Equity × 100  (%)
    de          — Total Debt / Common Equity
    div_yield   — Trailing 12-month dividends / Price × 100  (%)
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

FUND_CACHE_DIR = Path.home() / ".screener" / "fundamentals"
CACHE_MAX_AGE_DAYS = 7
_REPORTING_LAG = pd.Timedelta(days=45)

# Row-name candidates for each metric (checked in order; first match wins).
_EPS_ROWS   = ["Diluted EPS", "Basic EPS"]
_NI_ROWS    = ["Net Income", "Net Income From Continuing Operation Net Minority Interest",
               "Net Income From Continuing And Discontinued Operation"]
_DEBT_ROWS  = ["Total Debt", "Long Term Debt"]
_EQ_ROWS    = ["Common Stock Equity", "Stockholders Equity",
               "Total Equity Gross Minority Interest"]
_SHARES_ROWS = ["Ordinary Shares Number", "Share Issued"]


def _yf_sym(tv_symbol: str, market: str) -> str:
    sym = tv_symbol.split(":", 1)[-1] if ":" in tv_symbol else tv_symbol
    return f"{sym}.NS" if market == "india" else sym


def _cache_path(market: str, tv_symbol: str) -> Path:
    d = FUND_CACHE_DIR / market
    d.mkdir(parents=True, exist_ok=True)
    safe = tv_symbol.replace(":", "__").replace("/", "_").replace("&", "_")
    return d / f"{safe}.json"


def _df_to_records(df: pd.DataFrame) -> Optional[dict]:
    if df is None or df.empty:
        return None
    return {
        "index": list(df.index),
        "columns": [str(c) for c in df.columns],
        "data": [[None if pd.isna(v) else v for v in row] for row in df.values.tolist()],
    }


def _records_to_df(rec: dict) -> pd.DataFrame:
    df = pd.DataFrame(rec["data"], index=rec["index"], columns=pd.to_datetime(rec["columns"]))
    return df


def _series_to_records(s) -> Optional[list]:
    if s is None:
        return None
    # Newer yfinance may return a DataFrame instead of a plain Series for dividends.
    if isinstance(s, pd.DataFrame):
        if s.empty:
            return None
        s = s.iloc[:, 0]  # take first column as Series
    if s.empty:
        return None
    return [[str(idx), float(val)] for idx, val in s.items()]


def _records_to_series(rec: list) -> pd.Series:
    result = {}
    for r in rec:
        ts = pd.Timestamp(r[0])
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None).normalize()
        else:
            ts = ts.normalize()
        result[ts] = r[1]
    return pd.Series(result)


def fetch_fundamentals(
    tv_symbol: str,
    market: str,
    refresh: bool = False,
) -> Optional[dict]:
    """Return cached quarterly fundamentals, fetching from yfinance if needed.

    Returns a dict ``{income, balance, dividends}`` with DataFrames/Series,
    or ``None`` if data is unavailable.
    """
    path = _cache_path(market, tv_symbol)

    if not refresh and path.exists():
        try:
            cached = json.loads(path.read_text())
            age = datetime.now() - datetime.fromisoformat(cached["fetched_at"])
            if age < timedelta(days=CACHE_MAX_AGE_DAYS):
                return _load_cache(cached)
        except Exception:
            pass

    yf_sym = _yf_sym(tv_symbol, market)
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(yf_sym)
            inc = t.quarterly_income_stmt
            bal = t.quarterly_balance_sheet
            divs = t.dividends
    except Exception:
        return None

    if inc is None or inc.empty:
        return None

    payload = {
        "fetched_at": datetime.now().isoformat(),
        "income":  _df_to_records(inc),
        "balance": _df_to_records(bal),
        "dividends": _series_to_records(divs) if divs is not None and len(divs) > 0 else None,
    }
    try:
        path.write_text(json.dumps(payload))
    except Exception:
        pass

    return _load_cache(payload)


def _load_cache(payload: dict) -> Optional[dict]:
    result: dict = {}
    for key in ("income", "balance"):
        rec = payload.get(key)
        result[key] = _records_to_df(rec) if rec else None
    divs_rec = payload.get("dividends")
    result["dividends"] = _records_to_series(divs_rec) if divs_rec else None
    return result


def _first_valid(df: pd.DataFrame, row_names: list[str]):
    """Return the first row from *df* whose name is in *row_names*, or None."""
    for name in row_names:
        if name in df.index:
            return df.loc[name]
    return None


def fundamental_snapshot(
    fund_data: Optional[dict],
    as_of_ts: pd.Timestamp,
    close_price: float,
) -> Optional[dict]:
    """Compute point-in-time fundamentals as of *as_of_ts* with a 45-day lag.

    Returns ``{"pe": float|None, "roe": float|None,
               "de": float|None, "div_yield": float}``
    or ``None`` if there is no data at all.
    """
    if fund_data is None:
        return None

    inc = fund_data.get("income")
    bal = fund_data.get("balance")
    divs = fund_data.get("dividends")

    pe = roe = de = None
    div_yield = 0.0

    # ── filter quarterly columns that were likely reported by as_of_ts ───────
    def _valid_cols(df):
        if df is None or df.empty:
            return []
        return sorted(
            [c for c in df.columns if pd.Timestamp(c) + _REPORTING_LAG <= as_of_ts],
            reverse=True,
        )

    inc_cols = _valid_cols(inc)
    bal_cols = _valid_cols(bal)

    if not inc_cols:
        return None  # no usable data before as_of

    # ── TTM EPS → P/E ────────────────────────────────────────────────────────
    ttm_eps: Optional[float] = None
    ttm_ni: Optional[float] = None

    eps_row = _first_valid(inc, _EPS_ROWS)
    if eps_row is not None:
        vals = pd.to_numeric(eps_row[inc_cols[:4]], errors="coerce").dropna()
        if len(vals) >= 2:
            ttm_eps = float(vals.sum())

    ni_row = _first_valid(inc, _NI_ROWS)
    if ni_row is not None:
        vals = pd.to_numeric(ni_row[inc_cols[:4]], errors="coerce").dropna()
        if len(vals) >= 2:
            ttm_ni = float(vals.sum())

    # Fall back: compute EPS from Net Income / shares
    if ttm_eps is None and ttm_ni is not None and bal is not None:
        sh_row = _first_valid(bal, _SHARES_ROWS)
        if sh_row is not None and bal_cols:
            sh_vals = pd.to_numeric(sh_row[bal_cols[:1]], errors="coerce").dropna()
            if not sh_vals.empty:
                shares = float(sh_vals.iloc[0])
                if shares > 0:
                    ttm_eps = ttm_ni / shares

    if ttm_eps is not None and ttm_eps > 0 and close_price > 0:
        pe = close_price / ttm_eps

    # ── ROE & D/E from balance sheet ─────────────────────────────────────────
    if bal is not None and bal_cols:
        eq_row = _first_valid(bal, _EQ_ROWS)
        debt_row = _first_valid(bal, _DEBT_ROWS)

        if eq_row is not None:
            eq_vals = pd.to_numeric(eq_row[bal_cols[:2]], errors="coerce").dropna()
            if not eq_vals.empty:
                avg_equity = float(eq_vals.mean())
                if avg_equity > 0:
                    if ttm_ni is not None:
                        roe = (ttm_ni / avg_equity) * 100.0

                    if debt_row is not None:
                        dt_vals = pd.to_numeric(debt_row[bal_cols[:1]], errors="coerce").dropna()
                        if not dt_vals.empty:
                            de = float(dt_vals.iloc[0]) / avg_equity

    # ── Dividend yield ───────────────────────────────────────────────────────
    if divs is not None and not divs.empty and close_price > 0:
        one_yr_ago = as_of_ts - pd.Timedelta(days=365)
        recent = divs[(divs.index >= one_yr_ago) & (divs.index <= as_of_ts)]
        annual_div = float(recent.sum())
        div_yield = (annual_div / close_price) * 100.0

    return {"pe": pe, "roe": roe, "de": de, "div_yield": div_yield}
