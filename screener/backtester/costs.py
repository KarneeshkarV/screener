"""Statutory trading-cost models (fees), separate from slippage.

Fees reduce cash on entry/exit; they do **not** move the fill price. Slippage
models in :mod:`screener.backtester.slippage` handle price impact. Both feed
the portfolio cash accounting in :mod:`screener.backtester.portfolio`.

Also hosts the Corwin-Schultz (2012) high-low half-spread estimator used as an
optional fill-price spread proxy.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

Side = Literal["buy", "sell"]


@runtime_checkable
class CostModel(Protocol):
    def side_cost_fraction(self, side: Side, notional: float) -> float:
        """Total fees as a non-negative fraction of ``notional`` for ``side``."""


def _bps_fraction(bps: float) -> float:
    return bps / 10_000.0


class FlatCommission(BaseModel):
    """Legacy flat commission: ``commission_bps`` of notional on every fill."""

    model_config = ConfigDict(frozen=True)

    bps: float = 0.0

    def side_cost_fraction(self, side: Side, notional: float) -> float:
        return _bps_fraction(self.bps)


class IndiaDeliveryCosts(BaseModel):
    """NSE equity **delivery** statutory fee stack (cash market, not F&O).

    Rates are constructor fields so future regulatory changes need no code edits.
    Defaults match common discount-broker delivery schedules as of 2024–2025:

    * brokerage often 0 for delivery on discount brokers
    * STT 0.1% both sides (delivery)
    * stamp duty 0.015% on buy only
    * NSE transaction charge ~0.00297% both sides
    * SEBI turnover fee 0.0001% both sides
    * GST 18% on (brokerage + exchange txn + SEBI)
    * IPFT 0.0001% both sides
    """

    model_config = ConfigDict(frozen=True)

    # Brokerage in bps of notional; discount delivery default is 0.
    brokerage_bps: float = 0.0
    # Securities Transaction Tax — 0.1% of notional on BOTH buy and sell (delivery).
    stt_rate: float = 0.001
    # Stamp duty — 0.015% of notional on BUY side only.
    stamp_duty_rate: float = 0.00015
    # NSE equity cash transaction charge — 0.00297% of notional, both sides.
    exchange_txn_rate: float = 0.0000297
    # SEBI turnover fee — 0.0001% of notional, both sides.
    sebi_turnover_rate: float = 0.000001
    # GST — 18% levied on (brokerage + exchange transaction charge + SEBI fee).
    gst_rate: float = 0.18
    # Investor Protection Fund Trust (IPFT) — 0.0001% of notional, both sides.
    ipft_rate: float = 0.000001

    def side_cost_fraction(self, side: Side, notional: float) -> float:
        brokerage = _bps_fraction(self.brokerage_bps)
        exchange = self.exchange_txn_rate
        sebi = self.sebi_turnover_rate
        gst = self.gst_rate * (brokerage + exchange + sebi)
        total = brokerage + self.stt_rate + exchange + sebi + gst + self.ipft_rate
        if side == "buy":
            total += self.stamp_duty_rate
        return total


def build_cost_model(
    name: str = "flat",
    *,
    commission_bps: float = 0.0,
) -> CostModel:
    """Construct a :class:`CostModel` from a config name.

    ``"flat"`` — :class:`FlatCommission` using ``commission_bps`` (legacy).
    ``"india"`` — :class:`IndiaDeliveryCosts` with published delivery defaults.
    """
    key = (name or "flat").strip().lower()
    if key == "flat":
        return FlatCommission(bps=float(commission_bps))
    if key == "india":
        return IndiaDeliveryCosts()
    raise ValueError(f"unknown cost_model {name!r}; expected 'flat' or 'india'")


def cost_model_from_config(cfg: object) -> CostModel:
    """Resolve the fee model from a :class:`~screener.backtester.models.BacktestConfig`."""
    name = getattr(cfg, "cost_model", "flat") or "flat"
    commission_bps = float(getattr(cfg, "commission_bps", 0.0) or 0.0)
    return build_cost_model(str(name), commission_bps=commission_bps)


def corwin_schultz_half_spread(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Corwin-Schultz (2012) high-low **half-spread** estimator.

    For each consecutive-day pair ending at ``t``:

    * ``beta`` = sum over the 2 days of ``ln(H/L)^2``
    * ``gamma`` = ``ln(H2/L2)^2`` over the 2-day joint high/low
    * ``alpha`` = ``(sqrt(2*beta)-sqrt(beta))/(3-2*sqrt(2))
      - sqrt(gamma/(3-2*sqrt(2)))``
    * ``spread`` = ``2*(e^alpha - 1)/(1 + e^alpha)``

    Negative raw spreads are floored at 0, then a rolling ``window`` mean is
    taken. Returns the **half-spread** (``spread / 2``) as a fraction of price.

    Day-``t`` values use only high/low through ``t`` (no lookahead). Missing or
    non-positive high/low yield 0 for that observation.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0; got {window}")

    high_f = pd.to_numeric(high, errors="coerce").astype(float)
    low_f = pd.to_numeric(low, errors="coerce").astype(float)
    index = high_f.index

    valid = (
        high_f.notna()
        & low_f.notna()
        & (high_f > 0.0)
        & (low_f > 0.0)
        & (high_f >= low_f)
    )
    log_hl = pd.Series(np.nan, index=index, dtype=float)
    log_hl.loc[valid] = np.log(high_f.loc[valid] / low_f.loc[valid])
    sq = log_hl**2
    # beta_t uses days t-1 and t (available at close of t).
    beta = sq + sq.shift(1)

    high_2 = high_f.where(valid).rolling(window=2, min_periods=2).max()
    low_2 = low_f.where(valid).rolling(window=2, min_periods=2).min()
    gamma = pd.Series(np.nan, index=index, dtype=float)
    pair_ok = high_2.notna() & low_2.notna() & (high_2 > 0.0) & (low_2 > 0.0)
    gamma.loc[pair_ok] = np.log(high_2.loc[pair_ok] / low_2.loc[pair_ok]) ** 2

    denom = 3.0 - 2.0 * math.sqrt(2.0)
    # Guard negative radicands from floating noise / missing pairs.
    beta_safe = beta.clip(lower=0.0)
    gamma_safe = gamma.clip(lower=0.0)
    sqrt_beta = np.sqrt(beta_safe)
    alpha = (math.sqrt(2.0) * sqrt_beta - sqrt_beta) / denom - np.sqrt(
        gamma_safe / denom
    )

    # np.exp on a Series returns a Series at runtime, but numpy stubs widen it
    # to an ndarray; cast keeps `spread` typed as a pandas Series so `.clip` and
    # `.where` resolve to the pandas overloads below.
    exp_a = cast("pd.Series[float]", np.exp(alpha))
    spread = 2.0 * (exp_a - 1.0) / (1.0 + exp_a)
    # Floor negative spreads at 0 (CS paper recommendation).
    spread = spread.clip(lower=0.0)
    spread = spread.where(beta.notna() & gamma.notna(), 0.0).fillna(0.0)

    averaged = spread.rolling(window=window, min_periods=1).mean()
    half = pd.Series((averaged / 2.0).fillna(0.0), index=index, dtype=float)
    half.name = "half_spread"
    return half
