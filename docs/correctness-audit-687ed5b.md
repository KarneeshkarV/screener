# Screening and backtest correctness fixes

This change addresses the 29 defects reproduced in the correctness review of commit `687ed5b97d5b9f31af4836a2fd8db57eab302f47`.
The fixes include regression tests for provider conventions and public screening, backtest, and optimization paths.
All verification uses offline data.

## Data and signal timing

| Review item | Correct behavior |
| --- | --- |
| 1 | Yahoo prices, volume, and dividends already adjusted for splits are not adjusted again. |
| 2 | Cache extension and backfill compare overlapping prices before merging; an unverified or changed scale requires a fresh full window. |
| 11 | Intraday bars become available only after their full interval ends. |
| 12 | Fundamental fields become available at their own source section's publication date plus the configured lag. |
| 13 | Missing and nonfinite values stay unknown through expressions and cannot create an entry signal. |
| 18 | Fetch warmup includes dynamic membership, regime, liquidity, and sizing consumers without adding entry gates for new listings. |
| 19 | Fundamental fetches retain the latest pre-window observation for each field. |
| 20 | India benchmark indexes use the India session completion clock. |

`split_factor` specifies the split adjustment still required by the canonical frame.
It is `1.0` when the provider already adjusts splits.
`stock_splits` retains corporate-action events independently.
Yahoo cache keys change to prevent reuse of old frames with ambiguous adjustment contracts or mixed scales.
Old cache files remain on disk but are not read by the new Yahoo keys.

Undated fundamental sections cannot inherit the publication date of another section.
Their fields remain unavailable unless their own publication date is known.
This can reduce the number of eligible fundamental signals.

## Execution and accounting

| Review item | Correct behavior |
| --- | --- |
| 3 | The tighter active protective stop controls a long position, with the configured gap-fill behavior. |
| 4 | Reachable targets execute in price order, and a full exit prevents higher targets from executing. |
| 5 | A partial tier that closes the position stops further tier processing. |
| 6 | Prior-bar lookup uses an exclusive bound rather than subtracting a nanosecond from an index timestamp. |
| 7 | Market orders are selected on the signal bar and checked for affordability on their actual execution bar. |
| 16 | Simultaneously fillable pending orders share available cash under reinvested sizing. |
| 17 | Historical data covers the full simulation horizon, including reserve trades. |
| 27 | Buy sizing checks exact fee schedules and cannot overdraw cash. |
| 28 | Runtime accounting and curve reconstruction both reject nonfinite dividends. |

A future fill price must not change an earlier candidate selection.
If a queued market order cannot buy a share on its execution bar, the engine does not replace it with a lower-ranked candidate on the earlier signal date.
This corrects the earlier use of future affordability information.

## Screening and history

| Review item | Correct behavior |
| --- | --- |
| 8 | Minervini RS uses the full reference field, not a trend-prefiltered subset. |
| 9 | Dated membership masks the RS reference population while retaining price history for indicator warmup. |
| 10 | A live TradingView filter cannot remove a candidate from a completed-bar screen. |
| 21 | US share-class symbols map to Yahoo's hyphenated form, such as `NYSE:BRK.B` to `BRK-B`. |
| 22 | New history rows retain the canonical symbol, including its exchange. |
| 23 | Exact-universe screens return a UTC-aware `as_of` timestamp. |
| 24 | History labels distinguish resolved dynamic-universe policies. |

Existing history rows remain readable.
An old bare symbol cannot identify one exchange when several canonical symbols share that display name.
The history comparison must not silently choose an exchange in that case.

## Optimization

| Review item | Correct behavior |
| --- | --- |
| 14 | Validated slippage updates rebuild the effective fixed-bps component. |
| 15 | Research reports retain period-wide point-in-time membership windows. |
| 25 | Unknown parameters and invalid parameter combinations fail before execution or trial writes. |
| 26 | Fundamental optimization passes a resolved fetcher through preparation and engine execution. |
| 29 | Combined walk-forward metrics use the selected risk-free rate. |

A slippage-bps update with an ambiguous custom model requires an explicit replacement model.
Fundamental collaborators must match the configured market and publication lag.
Fundamental result caching is disabled when the frozen input identity does not authenticate fundamental data.

## Research impact

Backtest numbers are expected to change where the old behavior affected signals, prices, fills, sizing, fees, or membership.
Re-run affected research before using its results.
This includes split-crossing windows, warm-cache runs, Minervini studies, fundamental strategies, partial exits, reinvested pending orders, and optimization reports.
Passing regression tests does not validate live provider availability or establish that a strategy will be profitable.
