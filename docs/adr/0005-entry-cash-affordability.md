# ADR 0005: Apply a cash limit after the entry quote

## Status

Accepted.
This supersedes the cash-shortfall consequence and exact-share-through-portfolio rule in ADR 0001.
The pre-impact sizing convention and rejection of a joint slippage-and-size solve remain in force.

## Decision

The fill model uses the pre-impact share quote when calculating share-dependent slippage.
The portfolio then limits the acquired shares at that unchanged fill price so that the notional plus exact buy-side fees does not exceed the assigned budget or available cash.
India entries retain whole-share rounding.

Fee schedules must have a nondecreasing total buy cost as shares increase.
The affordability calculation retains the feasible endpoint instead of rounding a midpoint that could overdraw cash.
It does not recalculate the slippage price.

## Evidence and consequence

The correctness review reproduced a cash overdraft under the capped US fee schedule.
Actual fees depend on the filled notional and shares, so a fixed-reference fee estimate is not sufficient to enforce a cash limit.
The engine does not model borrowing for a long cash account.
A fill can therefore acquire fewer shares than the pre-impact quote, while retaining the conservatively quoted impact price.
