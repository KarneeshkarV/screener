# Connors crash short strategy on available US and India data

This is the dated exploratory baseline from the original simulator.
The current simulator resets indicators on missing sessions, blocks incomplete windows by default, and uses a fixed optional FMP fallback instead of selecting by future coverage.
The historical tables and commands below describe the original experiment and do not reproduce its numbers with the revised policy.
See `docs/connors-pit-research.md` for the timestamped evidence contract and remaining external-data requirements.
No complete verified PIT bundle has been obtained or run.

## Conclusion

The available-data test does not establish a profitable, executable strategy.
The US result is slightly negative in every requested window before borrow and execution charges.
India results are too sparse and sensitive to price-quality events to support a conclusion.
The final one-year India window generates no entries.
Small portfolio drawdowns reflect very low investment, not proven control of short-squeeze risk.

## Scope and dates

The common end date is 2026-10-09, the latest session shared by the cached SPY and Nifty 50 benchmark histories.
This is not a run through the machine date, 2026-10-20.
The windows begin on 2021-10-09, 2023-10-09, 2024-10-09, and 2025-10-09.
Each window starts with a new account, no positions, and no pending orders.
US capital is USD 100,000 and India capital is INR 100,000.

The US universe is quarterly observed S&P 500 membership, not all liquid US equities.
India uses the committed archive-observed Nifty 500 history, not all Indian equities or exact official index-effective dates.
The daily price and previous-session turnover filters apply inside these dated membership sets.
Price must exceed 5 and turnover must exceed 1,000,000 in local market currency.
The India thresholds are therefore INR 5 and INR 1 million, not USD-equivalent thresholds.

## Results

Returns below are cumulative portfolio returns, not annualized returns or average trade returns.
Gross includes short dividend obligations where explicit action data exists, but excludes borrow charges, commissions, taxes, cash interest, and execution charges.
The cost sensitivity adds a declared 10% annual borrow rate and 5 basis points per side.
These are assumptions, not measured historical costs.

| Market | Years | Gross return | Gross calendar CAGR | Gross maximum drawdown | Closed trades | Win rate | Return with assumed costs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| US | 5 | -0.035% | -0.007% | -1.228% | 16 | 43.75% | -0.088% |
| US | 3 | -0.216% | -0.072% | -1.106% | 14 | 42.86% | -0.306% |
| US | 2 | -0.632% | -0.317% | -1.115% | 12 | 33.33% | -0.708% |
| US | 1 | -0.294% | -0.294% | -0.777% | 8 | 25.00% | -0.342% |
| India | 5 | +0.115% | +0.023% | -0.541% | 10 | 70.00% | +0.051% |
| India | 3 | +0.062% | +0.021% | -0.541% | 7 | 71.43% | +0.020% |
| India | 2 | +0.322% | +0.161% | -0.082% | 3 | 100.00% | +0.310% |
| India | 1 | 0.000% | 0.000% | 0.000% | 0 | Not defined | 0.000% |

The five-year US average gross short exposure is only 0.133% of equity.
The five-year India average is only 0.087%.
The 40-position limit is not an observed diversification level.
The 100% volatility and CRSI above 90 requirements produce very few entries in these index universes.
US five-year trades occur only in EPAM, GL, SMCI, and SNDK.

## Price-quality sensitivity

GPIL, CGCL, and Motilal Oswal have apparent adjusted-price scale breaks on 2024-01-01 with close-to-close changes of approximately -79.5%, -74.8%, and -74.6%.
These observations are not verified economic returns.
They inflate the volatility filter and affect five of the ten India trades.
Raymond and ABFRL also have large price changes around corporate restructurings that require verified total-return and corporate-action treatment.
The three India two-year trades are two Raymond shorts and one ABFRL short.
Their apparent 100% win rate is not reliable evidence of a trading edge.

A separate causal sensitivity blocks new entries and cancels pending entries for 100 benchmark sessions after an observed absolute adjusted return above 50%.
It also requires 101 consecutive valid benchmark-session bars so missing data cannot leave an extreme return inside the valid-observation window after quarantine expires.
It does not repair prices or use knowledge of future events.
This rule also excludes real crashes, so it is a stress test, not a corrected production strategy.

| Market | Years | Gross return after quarantine | Closed trades |
| --- | ---: | ---: | ---: |
| US | 5 | -0.130% | 15 |
| US | 3 | -0.359% | 13 |
| US | 2 | -0.632% | 12 |
| US | 1 | -0.294% | 8 |
| India | 5 | +0.053% | 3 |
| India | 3 | 0.000% | 0 |
| India | 2 | 0.000% | 0 |
| India | 1 | 0.000% | 0 |

The remaining three India five-year trades are in Adani Enterprises during 2023.
Their combined gross profit is approximately INR 53 on INR 100,000 starting equity.

## Missing prices and incomplete histories

A member-session is one dated index member expected to have a price on one available benchmark session.
These counts do not prove the stock should have traded on every such date.
IPO timing, mergers, suspensions, ticker changes, stale membership observations, and provider gaps can all cause missing member-sessions.
The report records the location of a gap, not an unsupported explanation for it.

The fuller-cache proxy audit covers 608 US symbols and 781 India symbols observed during the five-year window.
It uses 53 FMP total-return proxy histories in the US and 47 in India where FMP has more bars than the action-capable Yahoo cache.
One source is selected per symbol; histories with different scales are not spliced.
FMP files lack explicit split and dividend histories, so they cannot verify nominal-share execution or the trade-date price cutoff.
Provider selection uses coverage over the full requested window, not a historically available provider-selection rule.
This is retrospective dataset assembly and must not be presented as a causal live data-selection process.
Their adjusted-price return is a synthetic-share approximation, not a proven cash-equity ledger.
The added histories do not change the reported trades or returns in these runs, but they reduce coverage gaps substantially.

| Market | Years | Missing member-sessions | Symbols with at least one missing session |
| --- | ---: | ---: | ---: |
| US | 5 | 8,363 | 69 |
| US | 3 | 3,213 | 42 |
| US | 2 | 1,457 | 36 |
| US | 1 | 847 | 23 |
| India | 5 | 9,983 | 561 |
| India | 3 | 4,675 | 524 |
| India | 2 | 3,330 | 513 |
| India | 1 | 793 | 24 |

Five-year missing coverage is 1.324% of 631,568 US member-sessions and 1.613% of 619,036 India member-sessions.
A high number of affected India symbols partly reflects common single-date gaps, rather than hundreds of completely absent histories.

### US gaps

COG has no usable selected cache.
T, VZ, EIX, CMCSA, and CTVA each lack 813 eligible sessions before their selected history starts on 2025-01-06.
BMY lacks 809 sessions before 2024-12-30.
LNC lacks all 496 of its observed eligible sessions; its available prices start after its membership window.
SBNY lacks all 313 eligible sessions; its selected history starts after its index membership window.
FB and INFO also have available history only after their observed eligibility, which raises ticker-identity concerns.
EA, EQR, and AVB have fuller FMP histories but end on 2025-12-31, leaving later eligible dates uncovered.
Removed securities can have legitimate price-history ends, while stale quarterly membership continues to expect them.

The five-year US gap ledger contains 5,985 sessions before first usable history, 2,039 after last usable history, 281 internal gaps, and 58 with no usable history.

### India gaps

There is no selected usable cache for these observed symbols:

```text
ADANIGAS.NS, CADILAHC.NS, DUMMYABFRL.NS, DUMMYDBRLT.NS,
DUMMYHDLVR.NS, DUMMYRAYMN.NS, DUMMYSIEMS.NS, DUMMYVEDL1.NS,
DUMMYVEDL2.NS, DUMMYVEDL3.NS, DUMMYVEDL4.NS, ESSELPACK.NS,
HEXAWARE.NS, IBVENTURES.NS, INFRATEL.NS, JUBILANT.NS, LTI.NS,
MINDAIND.NS, MINDTREE.NS, MOTHERSUMI.NS, NBVENTURES.NS,
NIITTECH.NS, ORIENTREF.NS, PHILIPCARB.NS, SHRIRAMCIT.NS,
SRTRANSFIN.NS, STRTECH.NS, WABCOINDIA.NS
```

Some names are former tickers or merger entities.
DUMMY symbols are corporate-action placeholders and must not be assumed tradeable stocks.
A verified security master is needed to resolve these identities without guessing.

L&TFH has 862 missing eligible sessions after invalid bars are excluded.
NMDC has 572 missing eligible sessions before usable selected history starts on 2024-08-01.
SRTRANSFIN and LTI each lack 368 eligible sessions.
MINDAIND, MINDTREE, SHRIRAMCIT, and SPICEJET each lack 247 eligible sessions.

The benchmark has a 2025-03-18 date on which 488 observed India members lack usable bars even with the FMP proxy extension.
This suggests a market-wide calendar or cache inconsistency, not 488 proven suspensions.
An independent verified exchange calendar is not installed or supplied.
The simulation uses each cached benchmark's session dates and does not fabricate stock bars.
Raw indicators advance on available valid stock bars, so a return across a missing interval is treated as one observation.
This can inflate annualized volatility and leave stale windows ready.
The coverage files count ready but incomplete 101-session windows, while the quality sensitivity blocks them.
The fuller-cache five-year audit finds 365 such eligible symbol-days in the US and 70,516 in India.
These are potentially affected indicator windows, not additional missing bars.

The five-year India gap ledger contains 4,037 sessions without usable history, 3,143 after last history, 1,770 internal gaps, and 1,033 before first history.

## Missing membership history

US snapshots are generally quarterly.
Actual additions and removals between snapshots are not resolved to their official effective dates.
The last US observation is 2026-08-28 and is carried through 2026-10-09.

India has three large observed-membership gaps relevant to this study:

| Previous observation | Next observation | Calendar-day gap |
| --- | --- | ---: |
| 2020-07-25 | 2022-05-04 | 648 |
| 2023-04-04 | 2024-02-07 | 309 |
| 2024-02-07 | 2025-06-16 | 495 |

The five-year run begins inside the first gap.
The last India observation is 2026-08-27 and is carried through 2026-10-09.
These are dated historical observations, not complete PIT-effective universes.
Neither market's index snapshots cover the full liquid-stock universe from the supplied strategy.

## Execution assumptions and remaining missing data

CRSI follows the inspected current LEAN implementation, including the unchanged streak on equal closes and rank against the previous 100 returns.
Volatility follows the supplied code: 100 closes, 99 log returns, population standard deviation, and annualization by the square root of 252.
All entry and exit signals use the previous completed benchmark session.
Existing shorts cover at the next available open after CRSI drops below 30 or observed universe removal.
Exit orders keep their original reason and wait if the next stock bar is missing.
Missing prices never create invented fills.

New sell limits are 3% above the prior nominal close, rounded to two decimals.
A daily high touching the limit fills the full order at the higher of the limit and open.
Daily data cannot prove queue priority, exact touch fills, or partial fills.
Entries use integer shares and a 2.5% prior-marked-equity target.
A gap above the limit can make actual entry exposure exceed 2.5%.
Open positions are not reweighted each day, matching the supplied code rather than a continuously equal-weight book.
Pending exits still occupy a slot that morning.

Unfilled entry limits are GTC, matching the absence of cancellation code in the supplied strategy.
In the five-year run, the longest wait between entry-order submission and fill is 77 calendar days in the US and 60 in India.
The model can therefore open a short long after its original overbought signal disappears.
The declared stop-trading-before-end variable in the source is unused, so this research does not invent an end-period trading halt.
No forced end liquidation is added; final equity is marked to the last available close.
All requested raw runs happen to finish without open positions.

The following records are missing and prevent a verified executable backtest:

- Historical US locates, stock availability, recall events, borrow rates, short-sale restrictions, and forced buy-ins.
- India SLB eligibility, actual lending contracts, tenor, fees, recalls, and settlement obligations.
- Broker margin rules, margin calls, and account-level forced liquidation.
- Intraday quotes, spreads, queue positions, partial fills, market impact, and India circuit-band restrictions.
- Verified security identities through renames, mergers, demergers, and ticker reuse.
- Complete corporate-action and delisting-recovery records.
- Date-aware commissions, taxes, India SLB charges, and broker fees.
- Independent exchange calendars and complete full-market historical universes.

The results are research estimates, not evidence that these shorts were available or executable.
The small trade count is insufficient to estimate crash-tail or short-squeeze risk.

## Files and reproduction

All report paths below are relative to the repository root.
The large `reports/` directory is gitignored.
`findings/connors_crash_summary.csv` retains the compact requested-window results outside it.
`findings/connors_crash_missing_symbols.csv` retains the complete list of symbols with missing member-sessions.

- `reports/connors_crash_2026-10-20/full_cache_proxy/summary.csv` contains the eight requested windows and cost scenarios using fuller available histories.
- `reports/connors_crash_2026-10-20/full_cache_quality_proxy/summary.csv` contains the causal extreme-return quarantine sensitivity.
- `reports/connors_crash_2026-10-20/full_cache_proxy/us_coverage.csv` and `india_coverage.csv` list every symbol, source type, first and last bar, missing sessions, indicator warmup gaps, and invalid bars.
- `reports/connors_crash_2026-10-20/full_cache_proxy/us_missing_sessions.csv` and `india_missing_sessions.csv` list every missing symbol and date.
- `reports/connors_crash_2026-10-20/us_missing_ranges.csv` and `india_missing_ranges.csv` group those missing dates into contiguous benchmark-session ranges.
- `reports/connors_crash_2026-10-20/full_cache_proxy/us_observations.csv` and `india_observations.csv` list all membership observation intervals.
- `reports/connors_crash_2026-10-20/full_cache_proxy/us_anomalies.csv` and `india_anomalies.csv` list absolute adjusted-price changes above 50%.
- Every market, window, and scenario folder contains `trades.csv`, `orders.csv`, `equity.csv`, `open_positions.csv`, and `summary.json`.
- Each `manifest.json` records source paths, SHA-256 hashes, and assumptions.
- The top-level report folder retains the stricter action-capable-cache run before the FMP proxy extension.
- `reports/connors_crash_2026-10-20/all_summaries.csv` combines all 96 runs across four data variants and three cost scenarios.

All 96 account ledgers reconcile to within 0.000001 local-currency units.
The targeted Connors, PIT-membership, and import-boundary regression run passed 92 tests.
An independent review checked signals, corporate actions, cash accounting, and missing data.
Its split-morning sizing and discarded-action findings were fixed and covered by regression tests.
The full offline suite after the accounting fixes passed 2,952 tests with 17 skips.
The additional loader-action regression test passed in the final 92-test targeted run.
Ruff lint and formatting checks passed for both new Python files.

```bash
uv run python -m scripts.connors_crash_research \
  --allow-fmp-proxy \
  --output reports/connors_crash_2026-10-20/full_cache_proxy

uv run python -m scripts.connors_crash_research \
  --allow-fmp-proxy --quarantine-extreme-returns \
  --output reports/connors_crash_2026-10-20/full_cache_quality_proxy

uv run pytest -q tests/test_connors_crash_research.py \
  tests/correctness/test_universes_pit.py tests/test_import_boundaries.py --no-cov
```
