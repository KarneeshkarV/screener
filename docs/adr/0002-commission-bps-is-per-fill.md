# ADR 0002: `commission_bps` is a per-fill rate

## Status

Accepted.

## Decision

`commission_bps` means the flat model's commission rate for each fill.

A completed long trade therefore receives the rate once on entry and once on exit.

The equity, earnings-drift, and PEAD engines all construct `FlatCommission` with this unchanged per-fill value and apply their round-trip fees through `apply_round_trip_costs` in `screener.backtester.costs`.

## Context

The equity engine has always modelled flat commission per fill.

Earnings-drift preserved a separate legacy convention by halving its configured value before creating `FlatCommission`.

PEAD separately subtracted its configured value once per round trip.

Those special cases made the same option mean different things and prevented PEAD from using statutory schedules consistently.

## Consequences

For the flat model, earnings-drift and PEAD now charge twice the previous drag for the same positive `commission_bps` value.

For example, `commission_bps=10` changes from 10 basis points per completed trade to 20 basis points per completed trade.

`execution.py` remains only as a compatibility re-export for external imports.

New code uses the shared cost and slippage modules directly.
