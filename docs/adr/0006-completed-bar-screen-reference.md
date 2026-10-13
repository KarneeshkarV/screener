# ADR 0006: Preserve completed-bar screen reference populations

## Status

Accepted.
Supersedes ADR 0003's use of TradingView as a pre-download field cutter for bar-based screens.
Local bars remain the source of eligibility rules.

## Decision and evidence

TradingView supplies the default unfiltered ticker field and display columns.
Bar-based screens do not apply live snapshot prefilters, even when a strategy profile declares one.
The review reproduced a valid completed-session breakout removed by today's incomplete volume snapshot.
A wider trend filter also changed Minervini's RS percentile by removing stronger-return reference stocks before ranking.

Cross-sectional ranks use the full declared reference population, masked by membership on each signal date.
Pre-membership price history remains available for indicator warmup.
Minervini declares no prefilter because its reference population is part of the rule.

## Consequences

Default screens can require more price downloads.
A capped vendor scan still warns when it cannot supply the complete ticker field.
Use an exact universe when the declared field must not depend on that cap.
