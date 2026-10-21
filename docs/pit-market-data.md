# Timestamped PIT market data contract

## Scope

`scripts/pit_market_data.py` loads timestamped market evidence without importing strategy code.
It validates supplied records, not their source truth or completeness.
It does not calculate CRSI, run backtests, or place orders.
No complete historical vendor dataset is included in this PR.

Each bundle represents one market, `us` or `india`.
Use permanent security IDs, independent exchange calendars, nominal prices, and dated availability evidence.
The result is labeled `timestamped_signal_research` with `source_assertions_verified=false` and `executable=false`.
No code change can recover missing delisted histories, effective membership dates, settlement records, or lending contracts.

## Load a bundle

```python
from pathlib import Path

import pandas as pd

from scripts.pit_market_data import load_pit_market_bundle

market = load_pit_market_bundle(
    Path("/data/evidence/us"),
    "us",
    end=pd.Timestamp("2026-10-09"),
    earliest_start=pd.Timestamp("2021-10-09"),
    history_sessions=101,
)
```

`history_sessions` defaults to one and must be positive.
Set it to the strategy's required number of contiguous closes.
The loader returns `PitMarketData`, including price and evidence arrays, membership, coverage, missing-session records, source SHA-256 hashes, and evidence limits.
It never falls back to legacy caches or selects prices using future coverage.

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
It must include at least `history_sessions` exchange sessions strictly before the earliest requested start.
The loader rejects a short calendar warmup instead of silently testing an inactive opening period.
New listings and stock-specific gaps can still lack valid price warmup; the coverage ledger reports incomplete history.
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
The loader returns delisting flags and dates so callers can reject unsupported held settlements.

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

Missing or invalid bars remain missing in the returned arrays.
The loader records complete contiguous windows using the caller's `history_sessions` value.
It does not calculate strategy indicators or create orders, positions, or fills.
`prior_quote_available` checks the previous valid quote against the decision timestamp.
`prior_bar_available` checks every observation since the most recent missing bar, for callers with recursive indicators.
Both checks preserve nanosecond timestamp precision.

`signal_close` is a causal total-return index built from nominal closes and effective split/dividend events.
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

The loader reads this file when present; callers decide whether to require it.
An absent daily security record means zero capacity, never assumed availability.
Capacity is in that session's nominal shares and must remain valid through the close.
`annual_rate` is a nonnegative decimal annual borrow rate, not a percentage number.
India records require a nonempty SLB contract ID.
A contract ID is a reference for external verification, not proof that a loan existed.

The loader returns `short_capacity` and `borrow_rate` arrays.
It does not size orders, accrue borrow costs, or implement recalls.
A strategy must enforce its own lending requirements before using those arrays.

Short-sale restrictions, circuit bands, loan settlement, margin, taxes, partial fills, recalls, and forced buy-ins still require additional evidence and execution models.
Every result therefore retains `executable=false`, even with permissions supplied.

## Verification

```bash
uv run pytest -q tests/test_pit_market_data.py --no-cov
uv run ruff check scripts/pit_market_data.py tests/test_pit_market_data.py
uv run ruff format --check scripts/pit_market_data.py tests/test_pit_market_data.py
```

The standalone tests use synthetic evidence and need no strategy files.
They test identities, listing intervals, daily membership, missing quotes, timestamp precision, corporate actions, and optional lending records.
Grouped counts, listing-interval counts, and segmented availability maxima avoid scanning the full universe for each session.
Tests establish contract behavior, not market-data completeness.
