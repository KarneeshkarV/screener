# ADR 0005: Queue entries and enforce exact buy affordability

## Status

Accepted.
Supersedes the unchanged-share and cash-shortfall consequences of ADR 0001.
The pre-impact sizing convention in ADR 0001 remains in force.

## Decision

Both equity engines queue entry orders on the completed signal bar without opening a position or debiting cash.
An order reserves a slot until its execution bar, cancellation, or the end of available history.
Market-on-open orders execute at the next available ticker bar's open.
Market-on-close orders execute at that bar's close, after the existing positions' exit sweep.
Limit orders are tested only against the current execution bar.

Market selection uses completed signal data, never a future fill price.
India candidates that cannot buy one whole share at the completed signal close can be rejected before queuing.
A later unaffordable market fill cancels the order rather than selecting a lower-ranked stock on the earlier signal day.
Rolling market selections record queued decisions even when the eventual order cannot fill.
Public result columns remain unchanged.

Entry budgets are calculated at execution time.
Open and limit fills use marks strictly before the current bar timestamp.
Close fills can use the current completed close.
The marked equal-slot sizing rule divides available cash among orders fillable in the current batch, excluding untouched limits and market orders awaiting a later bar.

A pre-impact share quote remains the upper bound used to calculate size-dependent slippage.
The fill price is then fixed.
Exact buy fees at that price must fit both the entry budget and available cash.
Shares can decrease for affordability, including fractional-share markets, but slippage is not recalculated.
Whole-share markets round down.
An affordability search retains its feasible endpoint, not a tolerance-rounded result that can leave negative cash.
Supported fee schedules must have nondecreasing total buy cost as share count increases.

## Context

Quoting the next available open while selecting candidates allows a later sparse-ticker price to change an earlier executed trade.
Debiting queued entries also spends future cash before their execution bar.
A fixed-reference fee fraction is not exact for capped fee schedules.
The previous two-pass fee correction could exceed a budget, and quoted shares were capped against cash rather than the explicit entry budget.

## Consequences

Trade sizes and selection records can change when ticker calendars differ, a queued market order becomes unaffordable, or exact fees constrain a fill.
Queued entries do not appear as holdings or reduce cash before execution.
This is an affordability solve at one fixed price, not the joint price-and-size iteration rejected in ADR 0001.
Daily bars still cannot establish the sequence of every intrabar limit fill and exit.
The existing conservative protection-first exit convention remains unchanged.
