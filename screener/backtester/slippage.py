"""Slippage models for fill-price adjustment.

Each model exposes
``adverse_fraction(side, shares, adv, sigma_daily, half_spread=0) -> float``
returning a non-negative fraction by which the reference price is widened
against the trader. ``apply_slippage`` multiplies that through a reference
price, with direction handled for buys vs sells.

Models:
  * ``FixedBpsSlippage`` — constant basis-point adverse fill (legacy behaviour).
  * ``HalfSpreadSlippage`` — quoted half-spread charged on every fill.
  * ``EstimatedHalfSpreadSlippage`` — per-fill half-spread fraction (e.g. CS).
  * ``VolumeImpactSlippage`` — Almgren-Chriss sqrt-law impact:
    ``k * sigma_daily * sqrt(shares / adv_shares)``.
  * ``CompositeSlippage`` — sums adverse fractions from component models.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


Side = Literal["buy", "sell"]


@runtime_checkable
class SlippageModel(Protocol):
    def adverse_fraction(
        self,
        side: Side,
        shares: float,
        adv: float,
        sigma_daily: float,
        half_spread: float = 0.0,
    ) -> float:
        """Return the adverse price adjustment as a non-negative fraction."""


def needs_liquidity_inputs(model: SlippageModel) -> bool:
    """Return whether ``model`` may depend on shares, ADV, or volatility.

    Built-in constant-cost models declare that liquidity preparation can be
    skipped. Unknown third-party implementations conservatively return true so
    their existing access to the full slippage protocol is preserved.
    """
    if isinstance(model, (FixedBpsSlippage, HalfSpreadSlippage)):
        return False
    if isinstance(model, VolumeImpactSlippage):
        return True
    if isinstance(model, CompositeSlippage):
        return any(needs_liquidity_inputs(component) for component in model.models)
    return True


def apply_slippage(
    model: SlippageModel,
    reference_price: float,
    side: Side,
    shares: float = 0.0,
    adv: float = 0.0,
    sigma_daily: float = 0.0,
    half_spread: float = 0.0,
) -> float:
    frac = float(
        model.adverse_fraction(side, shares, adv, sigma_daily, half_spread=half_spread)
    )
    if frac < 0.0:
        frac = 0.0
    if side == "buy":
        return reference_price * (1.0 + frac)
    return reference_price * (1.0 - frac)


def _bps_fraction(bps: float) -> float:
    return bps / 10_000.0


class FixedBpsSlippage(BaseModel):
    model_config = ConfigDict(frozen=True)

    bps: float = 0.0

    def adverse_fraction(
        self,
        side: Side,
        shares: float,
        adv: float,
        sigma_daily: float,
        half_spread: float = 0.0,
    ) -> float:
        return _bps_fraction(self.bps)


class HalfSpreadSlippage(FixedBpsSlippage):
    model_config = ConfigDict(frozen=True)

    half_spread_bps: float = 0.0

    def adverse_fraction(
        self,
        side: Side,
        shares: float,
        adv: float,
        sigma_daily: float,
        half_spread: float = 0.0,
    ) -> float:
        return _bps_fraction(self.half_spread_bps)


class EstimatedHalfSpreadSlippage(BaseModel):
    """Charge a precomputed per-fill half-spread fraction on every fill.

    The fraction is supplied at fill time via ``half_spread`` (same path as
    ``sigma_daily`` / ``adv``). Missing estimates should pass 0.0.
    """

    model_config = ConfigDict(frozen=True)

    def adverse_fraction(
        self,
        side: Side,
        shares: float,
        adv: float,
        sigma_daily: float,
        half_spread: float = 0.0,
    ) -> float:
        if half_spread <= 0.0 or not math.isfinite(half_spread):
            return 0.0
        return float(half_spread)


class VolumeImpactSlippage(BaseModel):
    """Almgren-Chriss square-root-law market impact.

    ``adv`` is expected in shares (not dollars). When ADV is unknown or zero
    the model returns 0 rather than raising, so the caller can fall through
    to other components in a composite.
    """

    model_config = ConfigDict(frozen=True)

    k: float = 0.1

    def adverse_fraction(
        self,
        side: Side,
        shares: float,
        adv: float,
        sigma_daily: float,
        half_spread: float = 0.0,
    ) -> float:
        if adv <= 0.0 or shares <= 0.0 or sigma_daily <= 0.0:
            return 0.0
        return self.k * sigma_daily * math.sqrt(shares / adv)


class CompositeSlippage(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    models: tuple[SlippageModel, ...] = Field(default_factory=tuple)

    def adverse_fraction(
        self,
        side: Side,
        shares: float,
        adv: float,
        sigma_daily: float,
        half_spread: float = 0.0,
    ) -> float:
        total = 0.0
        for m in self.models:
            total += float(
                m.adverse_fraction(
                    side, shares, adv, sigma_daily, half_spread=half_spread
                )
            )
        return total
