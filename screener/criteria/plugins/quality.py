"""High ROE (>15%) with low debt."""

from __future__ import annotations

from tradingview_screener import col

from screener.criteria import criterion


@criterion("quality")
def quality() -> list:
    return [
        col("return_on_equity") > 15,
        col("debt_to_equity") < 1,
    ]
