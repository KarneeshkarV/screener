"""Screener logic — assigns Operator_Action labels to each row.

Implements the four classic Operator Intent buckets (spec Step 3):

  Long Build-up    price↑  OI↑   delivery > 5-day avg
                   → fresh long positioning, with cash market participation
  Short Covering   price↑  OI↓   delivery > 5-day avg
                   → shorts buying back into a rising tape
  Short Build-up   price↓  OI↑   delivery > 5-day avg
                   → fresh short positioning, with cash market participation
  Long Unwinding   price↓  OI↓   delivery > 5-day avg
                   → longs exiting on a falling tape

The ``%_Change_Delivery > 100`` gate ensures we only label a stock when
the cash market is actively confirming the F&O move — without it, a small
illiquid F&O contract can flap signals every day on noise.

A separate boolean ``High_Momentum_Watch`` flags Long Build-ups within
15% of the 52-week high — the spec's bonus filter that catches momentum
plays before they break out.
"""
from __future__ import annotations

import pandas as pd

ACTIONS = (
    "Long Build-up",
    "Short Covering",
    "Short Build-up",
    "Long Unwinding",
)


def _classify(row) -> str | None:
    if not row.get("_is_fno", False):
        return None
    p = row.get("%_Change_Price")
    oi = row.get("%_Change_OI")
    d = row.get("%_Change_Delivery")
    if pd.isna(p) or pd.isna(oi) or pd.isna(d):
        return None
    # Delivery confirmation gate — spec Step 3
    if d <= 100:
        return None
    if p > 0 and oi > 0:
        return "Long Build-up"
    if p > 0 and oi < 0:
        return "Short Covering"
    if p < 0 and oi > 0:
        return "Short Build-up"
    if p < 0 and oi < 0:
        return "Long Unwinding"
    # Flat price (== 0) is rare on liquid futures and not part of the spec
    return None


def label(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Operator_Action`` and ``High_Momentum_Watch`` columns in place
    and return the frame.
    """
    df = df.copy()
    df["Operator_Action"] = df.apply(_classify, axis=1)
    # High Momentum Watch: Long Build-up + within 15% of 52-week high.
    # NaN dist (cache miss) is treated as not-near-high.
    near_high = df["Dist_From_52W_High"].le(15.0).fillna(False)
    df["High_Momentum_Watch"] = (
        (df["Operator_Action"] == "Long Build-up") & near_high
    )
    return df
