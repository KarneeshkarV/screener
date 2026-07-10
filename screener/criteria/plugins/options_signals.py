"""Panel-backed options signal pipeline criteria."""

from __future__ import annotations

from typing import Any

from screener.criteria import criterion
from screener.options.criteria import run_options_criterion


def _run(name: str, *, market: str, limit: int, output_csv: bool) -> None:
    run_options_criterion(
        name,
        market=market,
        limit=limit,
        output_csv=output_csv,
    )


@criterion("unusual_options", pipeline=True)
def unusual_options(*, market: str, limit: int, output_csv: bool, **_: Any) -> None:
    _run("unusual_options", market=market, limit=limit, output_csv=output_csv)


@criterion("bullish_oi_buildup", pipeline=True)
def bullish_oi_buildup(*, market: str, limit: int, output_csv: bool, **_: Any) -> None:
    _run("bullish_oi_buildup", market=market, limit=limit, output_csv=output_csv)


@criterion("high_iv_rank", pipeline=True)
def high_iv_rank(*, market: str, limit: int, output_csv: bool, **_: Any) -> None:
    _run("high_iv_rank", market=market, limit=limit, output_csv=output_csv)


@criterion("low_iv_rank", pipeline=True)
def low_iv_rank(*, market: str, limit: int, output_csv: bool, **_: Any) -> None:
    _run("low_iv_rank", market=market, limit=limit, output_csv=output_csv)


@criterion("cheap_earnings_vol", pipeline=True)
def cheap_earnings_vol(*, market: str, limit: int, output_csv: bool, **_: Any) -> None:
    _run("cheap_earnings_vol", market=market, limit=limit, output_csv=output_csv)


__all__ = [
    "bullish_oi_buildup",
    "cheap_earnings_vol",
    "high_iv_rank",
    "low_iv_rank",
    "unusual_options",
]
