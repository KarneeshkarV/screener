from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def enrich_fundamentals(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market != "india":
        return df

    try:
        from openscreener import Stock
    except ImportError:
        return df

    symbols = df["name"].tolist()
    if not symbols:
        return df

    try:
        batch = Stock.batch(symbols)
        ratios_data = batch.fetch("ratios")
    except (AttributeError, RuntimeError, ConnectionError, TimeoutError):
        return df

    rows = []
    for symbol in symbols:
        data = ratios_data.get(symbol, {})
        rows.append(
            {
                "name": symbol,
                "P/E": data.get("stock_p_e"),
                "ROCE%": data.get("roce_percent"),
                "ROE%": data.get("return_on_equity"),
            }
        )

    fundamentals = pd.DataFrame(rows)
    return df.merge(fundamentals, on="name", how="left")


def _screen_symbols(df: pd.DataFrame) -> list[str]:
    """Prefer ``name`` (TradingView scanner), fall back to ``ticker``."""
    if df.empty:
        return []
    if "name" in df.columns:
        return [str(s) for s in df["name"].tolist()]
    if "ticker" in df.columns:
        return [str(s) for s in df["ticker"].tolist()]
    return []


def enrich_days_to_earnings(
    df: pd.DataFrame,
    market: str,
    *,
    as_of: date | None = None,
    provider: Callable[..., dict[str, date | None]] | None = None,
) -> pd.DataFrame:
    """Attach a ``days_to_earnings`` column for final screen result rows.

    Only the rows in *df* are queried (post-screen). Provider failures leave the
    column as all-``None`` and log a warning rather than failing the screen.
    """
    out = df.copy()
    out["days_to_earnings"] = None
    symbols = _screen_symbols(out)
    if not symbols:
        return out

    as_of_d = as_of or date.today()
    fetch = provider
    if fetch is None:
        from screener.earnings_backtest.earnings_dates import fetch_next_earnings_dates

        fetch = fetch_next_earnings_dates

    try:
        next_dates = fetch(symbols, market, as_of=as_of_d)
    except TypeError:
        # Allow simpler stubs: provider(symbols, market) without as_of.
        try:
            next_dates = fetch(symbols, market)
        except Exception as exc:
            logger.warning(
                "days_to_earnings_enrich_failed",
                extra={"market": market, "error": str(exc)},
            )
            return out
    except Exception as exc:
        logger.warning(
            "days_to_earnings_enrich_failed",
            extra={"market": market, "error": str(exc)},
        )
        return out

    days: list[Any] = []
    for sym in symbols:
        nxt = next_dates.get(sym) if next_dates else None
        if nxt is None:
            days.append(None)
        else:
            days.append(int((pd.Timestamp(nxt).normalize().date() - as_of_d).days))
    # Object dtype keeps unknown as None rather than coercing to float NaN.
    out["days_to_earnings"] = pd.Series(days, index=out.index, dtype=object)
    return out


def filter_earnings_buffer(
    df: pd.DataFrame,
    buffer_days: int,
) -> pd.DataFrame:
    """Drop rows whose ``days_to_earnings`` is known and ``<= buffer_days``.

    Rows with unknown (null) earnings dates are kept.
    """
    if buffer_days < 0:
        raise ValueError("buffer_days must be >= 0")
    if df.empty or "days_to_earnings" not in df.columns:
        return df
    dte = pd.to_numeric(df["days_to_earnings"], errors="coerce")
    keep = dte.isna() | (dte > buffer_days)
    return df.loc[keep].reset_index(drop=True)
