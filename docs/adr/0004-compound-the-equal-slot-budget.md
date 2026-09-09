# ADR 0004: Compound the equal-slot budget from realized equity

## Status

Accepted.

## Decision

`equal_slot`, the default sizing rule, sizes each entry from a ceiling recomputed per entry as `realized_equity / slot_count`, where realized equity is cash plus the cost basis of open positions.

Available cash remains the hard cap, as it was before.

The same realized equity is the equity the risk rules (`fixed_fraction`, `fixed_risk`, `atr_risk`, `inverse_vol`) size against.

`reinvested_equal_slot` is unchanged and keeps sizing from marked-to-market equity.

`--no-compounding` restores the previous ceiling, a `initial_capital / slot_count` constant held for the life of the run, and restores `initial_capital` as the risk rules' equity.

## Context

The previous default froze the per-slot ceiling at day one.
A run that profited could not redeploy its gains: realized proceeds above the frozen slot stayed as idle cash, so the invested fraction of the book fell monotonically as the strategy worked.

That is not a conservative default, it is a silent one.
The measured volatility of a book that is 60% cash decays toward zero, Sharpe drifts up, and drawdown shrinks, all for reasons that have nothing to do with the signal.
Every long-window comparison between two strategies was therefore also a comparison of how much cash each had accidentally accumulated.

## Rejected alternative: size from marked-to-market equity

Sizing the default from marked equity would deploy more capital sooner and is what `reinvested_equal_slot` already does.

We rejected it as the default because it sizes new entries off marks a later exit may not realize.
An open position sitting on a large unrealized gain would enlarge every new slot, and a subsequent reversal would leave the book both larger and losing.
Realized equity moves only when a position closes, so the ceiling is backed by cash and closed P&L, never by a paper mark.

Keeping the marked variant as a separate named rule preserves the choice without making it the number everyone gets by default.

## Consequences

Every backtest number moves. That is the point, and the pinned characterization goldens were regenerated to match.

The ceiling is recomputed from one shared pool, so slots opened at the same moment are equal.
Slots opened at different moments are not: an open lot keeps the basis it was bought at, so a book that has realized a gain holds older lots at a smaller basis than newer ones.
That spread closes as each lot recycles at the current ceiling.
It is the price of not resizing open positions, which would mean trading to rebalance and is a separate decision.

Because realized equity excludes unrealized gains, a run whose winners are all still open compounds more slowly than a marked-equity run would.
This is deliberate and is the distinction between `equal_slot` and `reinvested_equal_slot`.

`--no-compounding` exists to reproduce a pinned pre-change baseline.
It is not a risk setting: freezing the slot understates a long window's volatility and drawdown rather than reducing them.
