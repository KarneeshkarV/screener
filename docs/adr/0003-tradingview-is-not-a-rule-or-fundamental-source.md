# ADR 0003: TradingView is not a rule or fundamental source

## Status

Accepted.

## Decision

Local bars are the canonical source for every eligibility rule and every
fundamental value the screen and the backtest use.

TradingView keeps exactly two jobs.
It is a field cutter, narrowing the ticker list before bars are downloaded.
It supplies display and ranking columns for a screen's output table.

A strategy is defined once, as an `ExpressionStrategySpec` with a
`StrategyProfile`, and that one definition drives the screen, the rolling
backtest and the optimizer.
Where a criterion name and a strategy name are the same word - `breakout`,
`mark_minervini`, `momentum_12_1` - the criterion is an alias onto the
strategy, so `screen --criteria breakout` and a rolling backtest of
`breakout` ask the same question of the same day.

A criterion's TradingView filters survive only as a declared
`tv_prefilter`, and only under one invariant: a prefilter may never drop a
name the bar rule would have kept.
`--universe` is the exact path and applies no prefilter at all.

## Context

Before this, the same strategy had two definitions that could not agree.
`screen --criteria breakout` sent TradingView a set of snapshot filters and
took the vendor's answer.
`backtest --strategy breakout` evaluated a Pine expression over local bars.
Nothing checked that the two selected the same names, and they did not.

Three concrete gaps drove the decision.

TradingView columns are snapshots that carry one as-of-today value, so
ranking a past day by them is lookahead.
The vendor's spellings are approximations: the `mark_minervini` criterion
silently drops the SMA200-rising and RS-rank legs because TradingView has no
column for them.
And a vendor column's definition is not ours - `price_52_week_high` is the
52-week extreme of highs, so fronting a `highest(close, 252)` rule with it
dropped names the rule would have kept.

The third one was found by the reconciliation tests this work added, after
the flip had already shipped it, which is the argument for the tests rather
than against them.

## Consequences

The three aliased criterion names changed meaning, so the `runs.criteria`
label changed with them at cutover.
A stored run from before the flip and one from after are not comparable, and
the label makes that visible rather than silent.

The default screen is slower, because the vendor cut is now narrower than
the rule and bars have to be fetched for the survivors.
D4 accepts a 60 second uncached screen.

The screen refuses what it cannot honestly answer.
Four strategies whose trade generation is not a per-bar boolean
(`heikin_ashi`, `shooting_star`, `bb_pattern`, `rsi_pattern`) are rejected by
kind, with a message, rather than answered by a vendor approximation.

Two definitions of the Minervini trend template remain outside this
guarantee.
`minervini.py:evaluate_symbol` is a hand-written re-implementation and D11
freezes it; `historical.py` keeps its own candidate definition and D12
schedules it for deletion.

The agreement is enforced, not asserted.
`tests/correctness/test_screen_backtest_reconciliation.py` is driven off
`registry.items()`, so a new strategy is covered the day it is registered.
