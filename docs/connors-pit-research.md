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

## Bundle files

CSV files use UTF-8 and header rows.
Empty event tables must retain their headers.
`session` is a date-only `YYYY-MM-DD` exchange-session label.
Every timestamp must include `Z` or a numeric timezone offset.
The loader normalizes timestamps to UTC.
`available_at` means earliest historical availability to the strategy, not the date of the current download.

### metadata.json

```json
{
  "schema_version": 1,
  "market": "us",
  "universe_kind": "all_listed",
  "calendar_source": "source URI and version",
  "universe_source": "source URI and version",
  "security_source": "source URI and version",
  "price_source": "source URI and version",
  "action_source": "source URI and version",
  "actions_complete_through": "2026-10-09"
}
```

Set `market` to `india` for the India directory.
`universe_kind` is `all_listed` or `index`.
An index bundle must not be described as the original full liquid-stock universe.
`actions_complete_through` is a source assertion, not something the loader can independently establish.
Supply `borrow_source` when a permissions file exists.

### sessions.csv

Columns: `session,decision_at,open_at,close_at,available_at`.

Provide an independent exchange calendar, including indicator warmup.
Do not derive it from which benchmark prices happen to exist.
The calendar must be sorted, unique, and cover the requested window.
It must include at least 101 exchange sessions strictly before the earliest test start.
The loader rejects a short calendar warmup instead of silently testing an inactive opening period.
New listings and stock-specific gaps can still lack valid price warmup; the decision and coverage ledgers report those blocks.
The calendar record must be available before the decision.
The previous session must close before the next decision, and the decision must precede or equal the open.
Decision, open, and close timestamps must match the session label in `America/New_York` for the US or `Asia/Kolkata` for India.
Only same-local-day preopen decisions are supported; shifted-year timestamps cannot masquerade as historical sessions.

### securities.csv

Columns: `security_id,available_at,listed_on,delisted_on`.

`security_id` is a permanent provider or exchange identifier, not a ticker.
Each ID appears once.
`listed_on` and `delisted_on` are inclusive listing bounds.
Leave `delisted_on` empty only when the security remains listed through the evidence horizon.
Do not derive universe changes from delisting dates alone; the daily universe must independently establish when changes were known.
Keep removed and delisted securities in the security master and price history.
The loader rejects quote rows before listing or after delisting.
A held short that reaches a known delisting without a modeled settlement aborts, including a delisting on the final date.

### universe_sessions.csv

Columns: `session,available_at,member_count`.

Supply exactly one row for every calendar session, including sessions with zero members.
This makes a missing snapshot distinct from a genuinely empty universe.
The number must match `universe.csv`.

### universe.csv

Columns: `session,security_id,ticker,available_at`.

Supply the complete eligible security set before daily price and turnover filters.
Use effective membership and availability records from a verified source, not quarterly carried-forward observations.
The ticker is a dated display mapping; all prices, actions, orders, and positions use the permanent ID.
Ticker changes therefore do not create new securities.
The loader rejects duplicate IDs, concurrent ticker collisions, DUMMY corporate-action placeholders, and members outside their listing interval.
An `all_listed` universe must include every listed security in the supplied master on each date.
This check cannot detect a security missing from both the master and universe.

### bars.csv

Columns: `session,security_id,open,high,low,close,turnover,available_at`.

OHLC values are unadjusted trade-date nominal prices in local currency.
Turnover is trade-date traded value, not retrospectively adjusted price multiplied by an incompatible volume series.
Full daily bars cannot be available before their session closes.
No source selection depends on future coverage.
A bundle uses its supplied source without silently borrowing prices from legacy caches.

Missing or invalid bars do not create fills.
They reset daily indicators and require a fresh complete 101-session window before new entries or pending entry fills.
The loader checks that all observations contributing to a signal were available before the decision.
This includes older observations retained by Wilder smoothing.
A held position with a delayed, unavailable prior valuation aborts the run instead of using an undisclosed future mark for sizing.
Ordinary missing quotes retain a stale mark and appear in the equity ledger.
If a held short's prior exit indicator is unavailable, the simulator queues a cover with reason `exit_indicator_unavailable`.
It preserves any exit reason already queued and fills only at the next valid open.
It does not wait another 100 sessions for indicator warmup or invent a fill during a halt.
This is an explicit data-loss exit policy, not an original CRSI signal or proof of executable liquidation.
`exit_indicator_unavailable_days` counts held position-days without an available exit indicator, including the eventual fill day.
`unfilled_exit_days` separately counts queued covers that have no valid open.

Signals use a causal total-return index built from nominal closes and effective split/dividend events.
Future action records never rescale earlier signals.
This differs from unverified retrospectively adjusted caches and must be treated as an explicit signal-price definition.

### actions.csv

Columns: `session,security_id,split,dividend,unsupported,available_at`.

The table is sparse and allows one combined action record per security and session.
`split` is new shares per old share; use 1 for no split.
`dividend` is local-currency cash per post-split share on the entitlement session.
Corporate-action events are independent of quote validity.
A missing quote cannot erase a dividend owed or a split obligation.
The evidence must establish each action before that morning's decision.

Set `unsupported` to 1 for a merger, demerger, delisting settlement, distributed asset, or another event that cannot be represented by the supported split/cash schema.
The loader refuses the bundle rather than treating that price discontinuity as a crash or discarding the obligation.
A verified settlement model is prerequisite work for those cases.
Do not mark such an event as supported merely to make a run pass.

### short_permissions.csv

Optional columns: `session,security_id,max_quantity,annual_rate,contract_id,valid_until,available_at`.

When `--require-short-permissions` is enabled, absence of this file is an error.
An absent daily security record means zero capacity, never assumed availability.
Capacity is in that session's nominal shares and must remain valid through the close.
`annual_rate` is a nonnegative decimal annual borrow rate, not a percentage number.
India records require a nonempty SLB contract ID.
A contract ID is a reference for external verification, not proof that a loan existed.

New order size cannot exceed capacity.
Pending orders recheck capacity before each possible fill.
If capacity disappears or falls below a held position, the simulation aborts.
It does not invent an immediate recall fill or silently extend a lending contract.
The model accrues the session's recorded annual rate over the preceding calendar-day interval on the previous close.
Use these figures as a declared daily-rate approximation, not verified lender billing or term-loan accounting.

Short-sale restrictions, circuit bands, loan settlement, margin, taxes, partial fills, recalls, and forced buy-ins still require additional evidence and execution models.
Every result therefore retains `executable=false`, even with permissions supplied.

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
  tests/test_connors_pit_data.py --no-cov
uv run ruff check scripts/connors_crash_research.py \
  scripts/connors_pit_data.py tests/test_connors_crash_research.py \
  tests/test_connors_pit_data.py
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
