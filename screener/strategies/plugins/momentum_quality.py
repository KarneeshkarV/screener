"""Quality-filtered momentum: Nifty Momentum-Quality style combination.

Source: NSE Indices "Momentum Quality" strategy index family (Nifty Midcap150
Momentum Quality 100, Nifty Smallcap250 Momentum Quality 100, Nifty Total
Market Momentum Quality 50) plus the quality-factor literature for India
("Firm quality and stock returns: Evidence from India"; MSCI quality factor).

Nifty's quality score uses return on equity (high), financial leverage (low)
and EPS growth stability over five years. This strategy approximates the same
idea with the backtester's dated fundamental columns:

    quality gate: roe_ttm >= 12  and  debt_to_equity <= 2.0  and eps_growth_yoy > 0
    momentum rank: same Normalized Momentum Score as ``nifty_momentum``

Fundamentals are merged into the bars *after* the strategy prepare hook (see
``screener.backtester.price_panel.build_price_panel``), so the quality filter
cannot be part of ``rank_score``; it is applied as an entry gate on the merged
columns and selection ranks the passing names by momentum. This matches the
index family's construction: a quality screen first, momentum selection within.

The workflow auto-enables the fundamentals provider when the entry expression
references fundamental fields (FMP for US, openscreener for India).
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.plugins.nifty_momentum import _prepare_momentum
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_ROE_MIN = 12.0
_DEBT_MAX = 2.0

ENTRY = "mom_z > 0 and roe_ttm >= 12 and debt_to_equity <= 2 and eps_growth_yoy > 0"
# Valuation-capped variants: keep momentum-quality winners but exclude
# frothy multiples (value/quality tilt, India GARP style).
ENTRY_PE = (
    "mom_z > 0 and roe_ttm >= 12 and debt_to_equity <= 2 and eps_growth_yoy > 0 "
    "and pe_ttm <= 50"
)
ENTRY_PB = (
    "mom_z > 0 and roe_ttm >= 12 and debt_to_equity <= 2 and eps_growth_yoy > 0 "
    "and pb_ttm <= 8"
)
ENTRY_PE40 = (
    "mom_z > 0 and roe_ttm >= 12 and debt_to_equity <= 2 and eps_growth_yoy > 0 "
    "and pe_ttm <= 40"
)
ENTRY_PE60 = (
    "mom_z > 0 and roe_ttm >= 12 and debt_to_equity <= 2 and eps_growth_yoy > 0 "
    "and pe_ttm <= 60"
)
ENTRY_PE55 = (
    "mom_z > 0 and roe_ttm >= 12 and debt_to_equity <= 2 and eps_growth_yoy > 0 "
    "and pe_ttm <= 55"
)


def _prepare_quality_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    return _prepare_momentum(ctx)


def _lookback() -> int:
    return 252


register_expression_strategy(
    "momentum_quality",
    entry=ENTRY,
    exit=None,
    prepare_bars=_prepare_quality_momentum,
    required_lookback=_lookback,
)

register_expression_strategy(
    "momentum_quality_pe",
    entry=ENTRY_PE,
    exit=None,
    prepare_bars=_prepare_quality_momentum,
    required_lookback=_lookback,
)

register_expression_strategy(
    "momentum_quality_pb",
    entry=ENTRY_PB,
    exit=None,
    prepare_bars=_prepare_quality_momentum,
    required_lookback=_lookback,
)

register_expression_strategy(
    "momentum_quality_pe40",
    entry=ENTRY_PE40,
    exit=None,
    prepare_bars=_prepare_quality_momentum,
    required_lookback=_lookback,
)

register_expression_strategy(
    "momentum_quality_pe60",
    entry=ENTRY_PE60,
    exit=None,
    prepare_bars=_prepare_quality_momentum,
    required_lookback=_lookback,
)

register_expression_strategy(
    "momentum_quality_pe55",
    entry=ENTRY_PE55,
    exit=None,
    prepare_bars=_prepare_quality_momentum,
    required_lookback=_lookback,
)
