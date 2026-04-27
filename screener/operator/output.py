"""CSV writer for the daily Operator Intent screen.

Output columns (per spec Step 4 emphasis on Operator_Action / Close / VWAP):

  SYMBOL, Operator_Action, High_Momentum_Watch,
  Close, VWAP, %_Change_Price, %_Change_OI, %_Change_Delivery,
  Dist_From_52W_High, 52W_High, 52W_Low,
  Deliv_Qty, Deliv_Pct, 5_Day_Avg_Delivery,
  Current_OI, Next_OI, Cumulative_OI, Prev_Close

Rows sort by High_Momentum_Watch desc, then Operator_Action priority
(Long Build-up first), then %_Change_Delivery desc — so the most
actionable rows surface at the top of the file.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ACTION_RANK = {
    "Long Build-up": 0,
    "Short Covering": 1,
    "Short Build-up": 2,
    "Long Unwinding": 3,
}

OUTPUT_COLUMNS = [
    "SYMBOL",
    "Operator_Action",
    "High_Momentum_Watch",
    "Close",
    "VWAP",
    "%_Change_Price",
    "%_Change_OI",
    "%_Change_Delivery",
    "Dist_From_52W_High",
    "52W_High",
    "52W_Low",
    "Deliv_Qty",
    "Deliv_Pct",
    "5_Day_Avg_Delivery",
    "Current_OI",
    "Next_OI",
    "Cumulative_OI",
    "Prev_Close",
]


def _format(df: pd.DataFrame) -> pd.DataFrame:
    """Rename internal columns to the public output names."""
    rename = {
        "CLOSE_PRICE": "Close",
        "AVG_PRICE": "VWAP",
        "_52W_High": "52W_High",
        "_52W_Low": "52W_Low",
        "DELIV_QTY": "Deliv_Qty",
        "DELIV_PER": "Deliv_Pct",
        "PREV_CLOSE": "Prev_Close",
    }
    return df.rename(columns=rename)


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_action_rank"] = df["Operator_Action"].map(ACTION_RANK).fillna(99).astype(int)
    df["_hmw_rank"] = (~df["High_Momentum_Watch"].fillna(False)).astype(int)
    df = df.sort_values(
        ["_hmw_rank", "_action_rank", "%_Change_Delivery"],
        ascending=[True, True, False],
        na_position="last",
    )
    return df.drop(columns=["_action_rank", "_hmw_rank"])


def write_csv(df: pd.DataFrame, as_of: date, out_path: Path | None = None,
              *, only_actions: bool = False) -> Path:
    """Write ``df`` to ``daily_operator_data_YYYYMMDD.csv``.

    ``out_path`` overrides the default filename (still relative to CWD if not
    absolute). ``only_actions=True`` filters to rows with a non-null
    Operator_Action — handy for terminal review of just the signalled names.
    """
    df = _format(df)
    if only_actions:
        df = df[df["Operator_Action"].notna()].copy()
    df = _sort(df)
    df = df[OUTPUT_COLUMNS]
    out_path = out_path or Path(f"daily_operator_data_{as_of.strftime('%Y%m%d')}.csv")
    df.to_csv(out_path, index=False, float_format="%.4f")
    return out_path
