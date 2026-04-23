"""Slippage models for fill-price adjustment.

Each model exposes ``adverse_fraction(side, shares, adv, sigma_daily) -> float``
returning a non-negative fraction by which the reference price is widened
against the trader. ``apply_slippage`` multiplies that through a reference
price, with direction handled for buys vs sells.

Models:
  * ``FixedBpsSlippage`` — constant basis-point adverse fill (legacy behaviour).
  * ``HalfSpreadSlippage`` — quoted half-spread charged on every fill.
  * ``VolumeImpactSlippage`` — Almgren-Chriss sqrt-law impact:
    ``k * sigma_daily * sqrt(shares / adv_shares)``.
  * ``CompositeSlippage`` — sums adverse fractions from component models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


Side = Literal["buy", "sell"]


@runtime_checkable
class SlippageModel(Protocol):
    def adverse_fraction(
        self, side: Side, shares: float, adv: float, sigma_daily: float
    ) -> float:
        """Return the adverse price adjustment as a non-negative fraction."""


def apply_slippage(
    model: SlippageModel,
    reference_price: float,
    side: Side,
    shares: float = 0.0,
    adv: float = 0.0,
    sigma_daily: float = 0.0,
) -> float:
    frac = float(model.adverse_fraction(side, shares, adv, sigma_daily))
    if frac < 0.0:
        frac = 0.0
    if side == "buy":
        return reference_price * (1.0 + frac)
    return reference_price * (1.0 - frac)


@dataclass(frozen=True)
class FixedBpsSlippage:
    bps: float = 0.0

    def adverse_fraction(
        self, side: Side, shares: float, adv: float, sigma_daily: float
    ) -> float:
        return self.bps / 10_000.0


@dataclass(frozen=True)
class HalfSpreadSlippage:
    half_spread_bps: float = 0.0

    def adverse_fraction(
        self, side: Side, shares: float, adv: float, sigma_daily: float
    ) -> float:
        return self.half_spread_bps / 10_000.0


@dataclass(frozen=True)
class VolumeImpactSlippage:
    """Almgren-Chriss square-root-law market impact.

    ``adv`` is expected in shares (not dollars). When ADV is unknown or zero
    the model returns 0 rather than raising, so the caller can fall through
    to other components in a composite.
    """

    k: float = 0.1

    def adverse_fraction(
        self, side: Side, shares: float, adv: float, sigma_daily: float
    ) -> float:
        if adv <= 0.0 or shares <= 0.0 or sigma_daily <= 0.0:
            return 0.0
        return self.k * sigma_daily * math.sqrt(shares / adv)


@dataclass(frozen=True)
class CompositeSlippage:
    models: tuple[SlippageModel, ...] = field(default_factory=tuple)

    def adverse_fraction(
        self, side: Side, shares: float, adv: float, sigma_daily: float
    ) -> float:
        total = 0.0
        for m in self.models:
            total += float(m.adverse_fraction(side, shares, adv, sigma_daily))
        return total
