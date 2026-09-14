# India backtest integrity

This change repairs measurement and execution defects found in the cash-equity swing and position review.
It does not establish a profitable strategy or validate leverage.
The default cash hurdle remains zero; use `--risk-free-rate` to declare the annual rate assumed for excess-return metrics.
This rate changes Sharpe, Sortino, PSR, and DSR, but does not credit interest to idle cash.

## Execution and ranking rules

- An entry at the open receives stop and target protection on its entry bar.
  A close entry does not use that day's earlier high or low.
  An intraday limit entry can trigger its protective stop, but cannot claim that the day's high happened after entry.
- An existing protective stop wins if the same daily bar also crosses a profit target.
  A stop raised after a partial sale starts on the next bar.
  This is a conservative daily-bar assumption; intraday sequencing is unknown.
- Indian entries and partial sales use whole shares.
  An order that cannot buy one share after costs is skipped.
  A partial target that would sell less than one share does not raise the stop.
  Affordability trimming retains the conservative impact quoted for the larger order.
- Indian daily exit expressions use the completed bar and fill at the next available bar's open.
  Scheduled time exits and rank exits retain their existing execution rules.
  Other markets and intraday expression exits retain their prior timing.
- Exit volume, volatility, and estimated spread use the previous completed bars, including at forced closure.
- Composite percentile ranks and component z-scores exclude stocks outside dated membership before blending.
  Price history remains available for indicator warmup.
  The reference group is dated membership, not the later price or liquidity filter result.
  Sector score normalization uses the eligible candidate set.

## Metrics and research controls

`exposure` retains its serialized key but is displayed as average slot occupancy.
`avg_capital_exposure` and `max_capital_exposure` use holdings value divided by equity at each close.
They use the same valuation function as the equity curve and handle partial sales and idle cash.
They do not measure intraday peak exposure, and are not reconstructed for stitched walk-forward output.

`calendar_cagr` uses elapsed calendar years; `cagr` retains the existing bar-count convention.
`max_drawdown_duration_days` includes the current unrecovered drawdown.
`expected_shortfall_95` is the mean return at or below the fifth percentile, expressed as a signed return.

The optimizer exposes `--experiment-id`, `--trial-db`, and `--n-trials-effective`.
Reuse the family ID and database when comparing related research choices.
Research reports store the full-period descriptive grid under `<family>::descriptive`.
Each walk-forward training period uses `<family>::train:<start>:<end>`.
This preserves repeated searches within that period and prevents descriptive or later-period statistics from changing earlier training decisions.
The full-report effective-trial estimate applies to its descriptive grid; its walk-forward folds use their own nominal counts.
A manual outer research register and untouched final assessment remain necessary.

Walk-forward reports state calendar-day lengths and forced-flat fold boundaries.
Holding-period warnings account for the maximum weekdays in a fold; exchange holidays can reduce the sessions further.
An integrity PASS remains a minimum data check.
Factor reports add a Bartlett Newey-West mean t-statistic with horizon-based lag and a finite-sample correction.
The classical iid statistic remains available for comparison.
Factor reports still use a fixed universe and close-to-close labels.

## Fixed-data replay

Three previously reviewed configurations were rerun without downloads or parameter selection, from 2022-09-01 through 2026-09-01.
Each starts with INR 100,000 and ten slots, with compounding, the saved India delivery cost model, and five basis points of slippage per side.
Sharpe below uses a zero annual hurdle to permit comparison with the original results.
The price snapshot SHA-256 is `2874eec74a8c9e0e738448409ff27ec1b4e245e208b375fc8d6ddd63ddaa790e`.

| Previously reviewed case | Hold sessions | Old Sharpe | Corrected Sharpe | Old maximum drawdown | Corrected maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Momentum 12-1 | 63 | 0.763 | 0.864 | -38.70% | -35.87% |
| Momentum and low volatility | 21 | 0.751 | 0.832 | -32.07% | -24.92% |
| EMA stack and low volatility | 21 | 0.997 | 0.791 | -15.05% | -15.10% |

All three ledgers contain whole shares and reconcile with final equity within INR 0.000001.
The lower EMA result shows why earlier strategy rankings need reassessment.
These changes combine share rounding, dated ranks, and execution fixes; the comparison does not isolate each effect.
This is previously inspected data with known coverage gaps, not an independent final test.
Full replay files are retained locally under `reports/india_integrity_fix_2026-09-13/`; the frozen provider dataset is not included in this PR.

## Status of all review findings

| Finding | Status after this change | Remaining requirement |
| --- | --- | --- |
| Entry-day stops skipped | Fixed in both engines | Intraday data for exact sequencing |
| Optimistic partial-target order | Fixed with conservative stop priority | Intraday data for alternative sequencing |
| Incomplete membership observations | Gap and stale-observation diagnostics added | Complete official membership history |
| Missing provider prices | Bounded warning added | Delisted, renamed, and missing price histories |
| Future members alter composite ranks | Fixed for dated membership and generic blends | Complete membership input |
| Fractional Indian shares | Fixed | Corporate-action share transformations still depend on adjusted-price mode |
| Zero-hurdle Sharpe and misleading exposure | Configurable hurdle and capital exposure added | Choose a justified historical hurdle before research selection |
| Frozen budgets hide account growth | Compounding was already the default and remains so | Use frozen budgets only for explicit comparisons |
| Incomplete and undated costs | Remains open | Broker DP fees, date-aware schedules, and measured slippage |
| Repeated search overfits the study | Trial controls exposed; train periods isolated | Outer research register and untouched final data |
| Unreliable publication dates | FMP period-end fallback removed; estimated India dates labeled | Actual announcements and original statement vintages |
| Nifty 50 price benchmark mismatch | Remains open | Verified Nifty 500 TRI and appropriate factor/passive benchmarks |
| Sector normalization mistaken for a risk cap | Eligible normalization fixed; no portfolio cap added | Dated sector/group classifications and portfolio limits |
| ATR sizing mistaken for an ATR stop | Remains an allocation rule | Define a separate stop policy and measure gap losses |
| Circuit limits and blocked exits | Remains open | Security-level bands, tradability data, partial-fill and queue assumptions |
| Stale exit impact inputs | Fixed | Calibrate impact against actual orders |
| Same-close expression exits | Fixed for Indian daily runs | Other intervals require separate order-timing rules |
| Stale marks and assumed tail liquidation | Remains open | Suspension, delisting, and recovery-value policy |
| Misleading dollar-turnover label in India | CLI identifies market currency and INR | Calibrate order participation to intended capital |
| Adjusted prices used for execution | Existing modes retained | Separate raw execution and total-return signal data with corporate actions |
| Long holds in short walk-forward folds | Calendar and weekday-fit warnings added | Choose adequate windows or design continuous portfolio validation |
| Small minimum evidence gate | Reports explicitly label the integrity limitation | Predeclare statistical acceptance requirements |
| Overlapping factor-return statistics | HAC statistic added | Dated factor universe and executable return labels |
| Correlated strategy families | Remains a portfolio research task | Holding overlap, daily correlations, and joint-loss tests |
| Entry filters confused with ranking | Existing turnover fallback retained | Test gates and rank rules as separate research choices |
| Leverage absent | Outside this cash-equity model | Broker funding, haircuts, margin calls, and forced-sale model |

The remaining items above must not be treated as fixed by a warning or a passing test.
Existing saved research should be rerun before selecting a strategy or changing capital.
