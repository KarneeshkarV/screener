# ADR 0001: Size entries at the pre-impact reference price

## Status

Accepted.

## Decision

When a slippage model depends on order size, `FillModel` calculates the entry share count from the order's pre-slippage reference price and the configured buy-side `CostModel`, including one fixed-reference fee quote for capped schedules.

It passes that exact count through `_SlotState` to `Portfolio.open`.

`Portfolio.open` charges the same model's actual fees and does not recompute shares from the impacted fill price.

Fixed-price slippage models retain their established portfolio sizing because their fill price does not depend on shares.

## Context

Impact slippage creates a cycle between shares, fill price, and fees.

Sizing at the reference price makes the convention deterministic and lets a volume-impact model use the intended order size without a hidden second sizing rule.

## Rejected alternative

We rejected a convergent joint quote-and-size solve that repeatedly recalculates shares from the impacted fill price.

The single fixed-reference fee quote is not that rejected alternative because it never recalculates the slippage price.

That approach makes sizing depend on iteration details and obscures which price is the budget convention.

The fixed pre-impact convention is easier to explain, reproduce, and test.

## Consequences

Impact can leave residual cash or a cash shortfall relative to the pre-impact allocation because the acquired shares are not changed after the fill price changes.

Flat costs retain their existing arithmetic, while non-flat schedules now use one cost model for both the quote and portfolio charge.
