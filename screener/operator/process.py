"""Build the per-symbol Operator Intent dataset for a single trading day.

Pulls today's Cash + F&O bhavcopies, the prior trading day's F&O bhavcopy
(for OI day-over-day change), the trailing 5 days of cash bhavcopies (for
``5_Day_Avg_Delivery``), and the autoresearch parquet cache (for 52W H/L),
then computes the spec's six derived columns.

The screener label itself lives in ``screen.py`` — this module stops at the
arithmetic so the calculated frame is also useful standalone.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from collections.abc import Mapping
from typing import cast

import pandas as pd

from screener.options.metrics import compute_chain_metrics
from screener.options.models import OptionChain
from screener.options.nse_bhavcopy import load_bhavcopy_chains

from .fetch import (
    fetch_cash_bhavcopy,
    fetch_fo_bhavcopy,
    fifty_two_week_hl,
    latest_trading_day,
    near_month_oi,
)
from .universe import combined_universe

LOG = logging.getLogger(__name__)

DELIVERY_LOOKBACK = 5  # 5-day avg per spec


def _options_oi_confirmation(chains: Mapping[str, OptionChain]) -> pd.DataFrame:
    """Collapse exact option OI/premium changes into an ATM confirmation row."""
    columns = [
        "SYMBOL",
        "ATM_Call_Writing_OI",
        "ATM_Put_Writing_OI",
        "Options_OI_Confirmation",
    ]
    if not chains:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for symbol, chain in chains.items():
        metrics = compute_chain_metrics(chain)
        call_writing = metrics.call_writing_near_spot
        put_writing = metrics.put_writing_near_spot
        if call_writing is None or put_writing is None:
            confirmation = None
        elif put_writing > call_writing and put_writing > 0:
            confirmation = "Bullish: put writing"
        elif call_writing > put_writing and call_writing > 0:
            confirmation = "Bearish: call writing"
        else:
            confirmation = "Neutral"
        rows.append(
            {
                "SYMBOL": symbol,
                "ATM_Call_Writing_OI": call_writing,
                "ATM_Put_Writing_OI": put_writing,
                "Options_OI_Confirmation": confirmation,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _trailing_trading_days(d: date, n: int) -> list[date]:
    """Walk back from ``d - 1`` collecting ``n`` distinct prior trading days.

    Uses ``latest_trading_day`` to skip weekends/holidays. Each returned date
    has its cash bhavcopy already fetched (and cached) as a side effect.
    """
    days: list[date] = []
    cursor = d - timedelta(days=1)
    while len(days) < n:
        td = latest_trading_day(cursor)
        days.append(td)
        cursor = td - timedelta(days=1)
    return days


def _five_day_avg_delivery(as_of: date) -> pd.DataFrame:
    """Mean DELIV_QTY over the 5 trading days *prior to* ``as_of``.

    Per the spec: ``5_Day_Avg_Delivery`` is the rolling 5-day average. We use
    the 5 days strictly before ``as_of`` so today's delivery is being
    compared against a clean baseline (avoids self-reference).
    """
    days = _trailing_trading_days(as_of, DELIVERY_LOOKBACK)
    frames = []
    for td in days:
        df = fetch_cash_bhavcopy(td)[["SYMBOL", "DELIV_QTY"]].copy()
        df["_d"] = td
        frames.append(df)
    stacked = pd.concat(frames, ignore_index=True)
    avg = stacked.groupby("SYMBOL", as_index=False)["DELIV_QTY"].mean()
    avg_frame = cast(pd.DataFrame, avg)
    return avg_frame.rename(columns={"DELIV_QTY": "5_Day_Avg_Delivery"})


def build_dataset(
    as_of: date | None = None, *, universe_mode: str = "fo+cash"
) -> tuple[pd.DataFrame, date]:
    """Build the screener dataset for ``as_of`` (defaults to today).

    Returns ``(df, actual_trading_day)``. ``actual_trading_day`` is the date
    of the bhavcopy actually used — useful when ``as_of`` lands on a
    weekend/holiday and we walked back.
    """
    today = as_of or date.today()
    today = latest_trading_day(today)
    LOG.info("operator scan for trading day %s", today)

    universe, fno_set = combined_universe(today, mode=universe_mode)
    LOG.info("universe size: %d (F&O: %d)", len(universe), len(fno_set))

    cash_today = fetch_cash_bhavcopy(today)
    avg_deliv = _five_day_avg_delivery(today)

    fo_today_raw = fetch_fo_bhavcopy(today)
    options_confirmation = _options_oi_confirmation(load_bhavcopy_chains(today))
    fo_today = near_month_oi(fo_today_raw)
    prev_day = _trailing_trading_days(today, 1)[0]
    fo_prev = near_month_oi(fetch_fo_bhavcopy(prev_day))[["SYMBOL", "Cumulative_OI"]]
    fo_prev = fo_prev.rename(columns={"Cumulative_OI": "Prev_Cumulative_OI"})

    hl = fifty_two_week_hl(universe, today)

    # Restrict to chosen universe — symbols outside it (e.g. obscure SME
    # listings present in the bhavcopy) are dropped.
    base = pd.DataFrame({"SYMBOL": universe})
    df = base.merge(cash_today, on="SYMBOL", how="left")
    df = df.merge(avg_deliv, on="SYMBOL", how="left")
    df = df.merge(fo_today, on="SYMBOL", how="left")
    df = df.merge(fo_prev, on="SYMBOL", how="left")
    df = df.merge(hl, on="SYMBOL", how="left")
    df = df.merge(options_confirmation, on="SYMBOL", how="left")

    # ── derived columns (spec Step 2) ──────────────────────────────────
    # %_Change_Price = (close − prev_close) / prev_close × 100
    df["%_Change_Price"] = (df["CLOSE_PRICE"] / df["PREV_CLOSE"] - 1.0) * 100.0

    # %_Change_OI = day-over-day change in Cumulative_OI; NaN for non-F&O
    df["%_Change_OI"] = (df["Cumulative_OI"] / df["Prev_Cumulative_OI"] - 1.0) * 100.0

    # %_Change_Delivery = today's delivery / 5-day avg × 100. The spec
    # interprets >100 as "today is above the trailing baseline".
    df["%_Change_Delivery"] = (df["DELIV_QTY"] / df["5_Day_Avg_Delivery"]) * 100.0

    # Dist_From_52W_High = % below 52W high (>=0). NaN if H/L unavailable.
    df["Dist_From_52W_High"] = (
        (df["_52W_High"] - df["CLOSE_PRICE"]) / df["_52W_High"]
    ) * 100.0

    # Mark which rows are F&O eligible — used by screen.py
    df["_is_fno"] = df["SYMBOL"].isin(fno_set)

    return df, today
