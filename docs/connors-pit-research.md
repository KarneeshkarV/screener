# Connors short research data contract

## Scope

The Connors simulator is a separate short-equity research script.
It does not modify the long-only backtest engines or place orders.

There are two input paths.
The default path reads existing caches and dated index observations.
It always reports `observed_index_proxy`, even if a cache filename contains `pit`.
It cannot prove historical data availability, effective membership, or borrowability.

`--pit-bundles` reads timestamped, nominal-price evidence supplied by the caller.
It reports `timestamped_signal_research`, not certified PIT or executable performance.
The loader checks the evidence contract, not whether a provider's source assertions are true.
An independent audit must verify completeness and provenance.

No code change can recover unavailable delisted histories, official membership dates, corporate-action settlements, or historical lending contracts.
Do not manufacture those records from today's constituents, ticker text, or price jumps.

## Commands

```bash
# Existing incomplete caches, explicitly labeled as index-proxy research.
uv run python -m scripts.connors_crash_research \
  --end 2026-10-09 --output reports/connors_observed_proxy

# Timestamped evidence, with no fallback to existing caches.
uv run python -m scripts.connors_crash_research \
  --pit-bundles /data/connors-evidence \
  --end 2026-10-09 --output reports/connors_timestamped

# Require dated short capacity and rates as well.
uv run python -m scripts.connors_crash_research \
  --pit-bundles /data/connors-evidence --require-short-permissions \
  --end 2026-10-09 --output reports/connors_borrow_evidence
```

The bundle root must contain `us/` and `india/` directories.
An explicit end date is mandatory for timestamped bundles.
The runner creates independent five-, three-, two-, and one-year accounts in both markets.
It includes gross, assumed 10%, and assumed 30% borrow scenarios.
With required short permissions, recorded annual rates replace these scenario rates.
The execution-cost scenarios remain zero or five basis points per side.
The scenario names are retained for output compatibility; inspect `historical_short_permissions_required` before interpreting their borrow assumptions.

## PIT data dependency

The independent loader and full CSV contract are in [Timestamped PIT market data contract](pit-market-data.md).
The strategy imports `PitMarketData` and `load_pit_market_bundle` from `scripts/pit_market_data.py`.
The data module never imports this strategy.
The Connors adapter requires 101 pre-start sessions, then computes CRSI and volatility from contiguous causal `signal_close` segments.

Missing quotes reset the indicator history and block entries until a complete 101-session window exists.
If a held short's prior exit indicator is unavailable, the simulator queues a cover with reason `exit_indicator_unavailable`.
It preserves an existing exit reason and fills only at the next valid open.
`exit_indicator_unavailable_days` counts held position-days without an exit indicator.
`unfilled_exit_days` separately counts queued covers with no valid open.
This is a data-loss exit policy, not an original CRSI signal or proof of executable liquidation.

A delayed prior valuation aborts a held position's simulation rather than using a future mark.
An ordinary missing quote retains a stale mark until a valid quote exists.
A held short that reaches a known delisting without a modeled settlement aborts, including on the final date.

With `--require-short-permissions`, a missing permissions file is an error.
Order size cannot exceed recorded capacity, and pending fills recheck it.
Held permission loss aborts rather than inventing a recall fill or loan renewal.
Recorded rates accrue over the preceding calendar-day interval on the previous close.
This remains a daily-rate approximation, not verified lender billing.
All outputs retain `executable=false`.

## Outputs and checks

Each market exports coverage, missing-session records, evidence classification, and source hashes.
Each window and cost scenario exports orders, trades, daily equity, open positions, and `decisions.csv`.
The decision ledger records why every universe member passed or failed evaluation, including late bars, incomplete history, quality quarantine, and missing short capacity.
A `candidate` row means the signal passed the gates, not that a slot or fill was guaranteed.

The manifest records the input file hashes and both implementation hashes.
Do not compare revised runs with historical reports as if the data policy were unchanged.
The original available-data results in `findings/connors_crash_available_data.md` are retained as a dated exploratory baseline, not as evidence that the new timestamped path was run on real vendor data.

```bash
uv run pytest -q tests/test_connors_crash_research.py \
  tests/test_connors_pit_data.py tests/test_pit_market_data.py --no-cov
uv run ruff check scripts/connors_crash_research.py \
  scripts/pit_market_data.py tests/test_connors_crash_research.py \
  tests/test_connors_pit_data.py tests/test_pit_market_data.py
```

The offline end-to-end test supplies both market bundles, runs the real CLI, and checks all 24 scenario outputs.
Universe validation uses grouped counts and listing-interval counts rather than scanning all universe rows for every session.
Bar listing and decision-availability checks are vectorized, including segmented arrival-time maxima for Wilder history.
A local 1,400-session synthetic benchmark improved from 12.789 to 1.103 seconds for 40 securities and from 26.489 to 2.127 seconds for 80 securities.
The revised loader took 5.323 seconds for 200 securities and 10.600 seconds for 400 securities.
These are measured loader times, excluding fixture generation, not a claim about full-market runtime.
Other tests alter future prices and actions, delay bars, remove universe snapshots, rename tickers, remove prices, and withdraw short permissions.
The tests verify rejection or causal behavior, not market-data completeness.

## Available-cache regression

The revised default policy was rerun through 2026-10-09 without a real timestamped bundle.
These results remain observed-index proxy research.
The US trade counts and returns did not change.
India five-, three-, two-, and one-year gross returns were +0.095%, +0.042%, +0.302%, and 0.000%, with 9, 6, 2, and 0 closed trades.
Compared with the original baseline, the revised data checks remove one India trade in each nonempty window.
The existing corporate-action uncertainties still invalidate a claim of verified profit.
`findings/connors_crash_pit_controls_summary.csv` retains the revised summary.
Detailed local ledgers are in `reports/connors_pit_controls_2026-10-20/`, which is gitignored.
All 24 account ledgers reconcile within 0.000001 local-currency units.
A rerun after the missing-exit-indicator fix left all 24 returns and trade counts unchanged.
No held exit-indicator gaps occurred in these available-cache ledgers.
The post-review local ledgers are in `reports/connors_claude_review_fixes_2026-10-20/`.

## External work still required

Obtain licensed or otherwise authorized historical sources for both markets.
Verify source coverage before presenting a complete PIT result.
The required records are the full historical listed-security master, effective daily universe, official sessions, nominal prices including removed securities, corporate actions and settlements, and dated lending evidence.
No such complete bundle is included in this repository.
