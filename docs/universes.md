# Backtest universes

Rolling backtests support built-in index snapshots, custom point-in-time
snapshots, and rule-based dynamic universes. List them with:

```bash
uv run screener universes list
```

Built-ins are `sp500`, `nifty50`, `nifty500`, and `sensex`. Applying a current
index snapshot to older dates has survivorship bias. Use custom snapshot history
or a dynamic universe when historical eligibility must be reproducible.

Capture named-index membership automatically with an idempotent command:

```bash
uv run screener universes sync nifty50
uv run screener universes sync sensex
```

Run this daily or weekly from cron/CI. It writes a complete snapshot only when
membership changes, so the resulting CSV becomes point-in-time history without
manual addition/removal maintenance.

## Backfilling history you did not capture

`universes sync` only builds history forward from the day you start running it.
To recover history that predates that, `universes backfill` reads the Internet
Archive's crawls of the same NSE constituent CSV and turns each distinct copy
into a dated snapshot.

```bash
uv run screener universes backfill nifty500 \
  --output data/universes/nifty500_pit_snapshots.csv
```

Both commands write the same format, so the normal setup is to backfill once
and then let `sync` extend the file forward.
The backfill never touches a date the file already carries; it reports the
conflict and keeps the existing rows, because a `sync` snapshot is a first-hand
observation and an archived crawl of the same day is not.
Pass `--replace-existing` to overwrite those dates, which is what you want
after a bad backfill wrote wrong rows.

`--min-symbols` rejects a crawl that parsed into an implausibly short list.
A truncated capture would otherwise erase most of the index for the whole
window that snapshot covers, which reads as a plausible backtest result rather
than as the fetch failure it is.
It defaults to the index's own floor: 400 for the Nifty 500, 40 for the Nifty
50.

A count alone does not prove a crawl is the right document, because the archive
can serve a different list of the same length from the same path.
A crawl that keeps less than half of the previous snapshot's membership is
therefore rejected as well and reported as a warning.
A real rebalance moves tens of names out of 500, so the check only fires on a
wrong file.

### What the reconstruction can and cannot tell you

- **It is lookahead-free.** Each snapshot is dated at the crawl that observed
  it, which is on or after the day NSE published that membership, never before.
  No name becomes eligible earlier than it really was in the index.
- **It is only as fine as the crawl cadence.** A membership change is dated at
  the first crawl that saw it, not at its true effective date. An addition
  enters late, and a deleted name stays eligible until the next crawl. It was a
  real, tradeable listing over that stretch, so this is a resolution limit
  rather than a bias toward names that turned out well.
- **Check the printed dates for gaps** before trusting a window. The command
  prints every snapshot it kept with its symbol count and source URL.
- **A backtest that starts before the first snapshot trades nothing** over that
  stretch, because no name is eligible yet. The run's universe note says so
  explicitly rather than letting the gap read as a strategy result.

## Shipped `nifty500_pit`

`universes.yaml` in the repository root defines `nifty500_pit` over a committed
backfill at `data/universes/nifty500_pit_snapshots.csv`. It carries 13
snapshots from 2018-10 to 2026-08 and 850 names that were ever members, against
the 500 the current NSE list reports.

```bash
uv run screener backtest-rolling -m india \
  --universe nifty500_pit --universe-config universes.yaml \
  --strategy rs_breakout
```

Crawl coverage is uneven: 2018-10, 2019-02, 2020-07, then nothing until
2022-05, after which snapshots land roughly two to six months apart. Treat the
2020-07 to 2022-05 stretch as a single frozen membership rather than as
resolved history.

## Dynamic universe

Dynamic membership is ranked by average dollar volume calculated from prior
bars only and held until the next rebalance.

```bash
uv run screener backtest-rolling -m india \
  --universe dynamic --dynamic-base nifty500 \
  --universe-size 100 --universe-lookback 60 \
  --universe-rebalance monthly --strategy rs_breakout
```

## Point-in-time membership

`backtest-rolling` runs with `--point-in-time` on by default, so a US run reconstructs S&P 500 membership from the article revisions current at each quarterly sample and a symbol is a candidate only inside its own membership windows.
Pass `--no-point-in-time` to run against today's list instead.

The flag downgrades itself rather than aborting when it is not typed.
A universe that serves no membership history, or a `--tickers` / `--universe-file` list, runs survivorship-biased and says so in the universe note.
For S&P 500 the first fallback is the weaker "date added" column, which keeps post-as-of additions out but cannot bring removed ex-members back.
A typed `--point-in-time` fails loudly instead of downgrading.

The default candidate pool is the current Nifty 500 for India runs, which publishes no machine-readable membership history and so remains survivorship-biased.
Ranking is causal either way.
For rigorous history on a universe without one, supply dated snapshots containing removed and delisted securities.

## Custom universes

Pass a TOML, YAML, or JSON definition with `--universe-config`.

```toml
[universes.my_watchlist]
type = "static"
market = "india"
benchmark = "^NSEI"
symbols = ["NSE:RELIANCE", "NSE:TCS", "NSE:INFY"]
```

Snapshot YAML:

```yaml
universes:
  my_index:
    type: snapshots
    market: us
    benchmark: SPY
    path: my_index_members.csv
```

The CSV contains a complete constituent snapshot per effective date:

```csv
effective_date,symbol
2024-01-01,AAA
2024-01-01,BBB
2024-07-01,BBB
2024-07-01,CCC
```

Custom dynamic YAML:

```yaml
universes:
  liquid_india_100:
    type: dynamic
    market: india
    benchmark: ^NSEI
    base: nifty500
    size: 100
    lookback: 60
    rebalance: monthly
```

```bash
uv run screener backtest-rolling -m india \
  --universe liquid_india_100 \
  --universe-config universes.yaml \
  --strategy rs_breakout
```

Configuration content is hashed into the run's universe note. Unchanged input
therefore retains an auditable identity.

## Point-in-time Nifty 500

`--point-in-time` needs dated membership, which NSE does not publish: it serves
only a current constituent CSV.
A reconstructed history from the Internet Archive's copies of that CSV ships in
`data/universes/`.

```bash
uv run screener backtest-rolling -m india \
  --universe nifty500_pit --universe-config data/universes/india_pit.toml \
  --point-in-time --years 5 --strategy momentum_12_1 --cost-model india
```

Regenerate or extend it with `uv run python scripts/build_nifty500_history.py`.

The bias this removes is large.
Of today's 500 members, 164 were not in the index in May 2022, and those are
disproportionately the names that rallied enough to be promoted.
Measured on a 5-year rolling backtest, correcting membership cuts
`mark_minervini` CAGR by 28% and `momentum_12_1` CAGR by 12%.

Three residual biases remain, all upward, so corrected results are a lower
bound rather than an exact answer:

- Snapshot dates are Archive capture dates, not NSE's semi-annual rebalance
  dates, so changes inside a gap collapse onto the next observed date.
- The earliest snapshot is backdated to 2021-01-01, so windows starting before
  May 2022 assume constant membership.
- Roughly 7% of historical members (merged, renamed, or delisted) have no
  usable price history upstream and so cannot be traded even when membership is
  correct.

For periods from today onward, prefer `screener universes sync nifty500` on a
cron.
It records real membership changes as they happen and needs no archaeology.

## Point-in-time cap tiers (mid, small, micro)

The Archive holds only one or two captures of NSE's mid-, small-, and microcap
constituent CSVs, so the approach above cannot be repeated for them.
These tiers are reconstructed from NSE's own methodology instead: each index is
a contiguous rank band of the listed pool by trailing six-month average full
market capitalisation, reconstituted at the end of March and September.

| Universe | Rank band | NSE index |
| --- | --- | --- |
| `nifty_midcap150_pit` | 101-250 | Nifty Midcap 150 |
| `nifty_smallcap250_pit` | 251-500 | Nifty Smallcap 250 |
| `nifty_microcap250_pit` | 501-750 | Nifty Microcap 250 |
| `nifty_smid650_pit` | 101-750 | the three stacked |

```bash
uv run screener backtest-rolling -m india \
  --universe nifty_smallcap250_pit --universe-config data/universes/india_pit.toml \
  --point-in-time --years 5 --strategy momentum_12_1_defensive --cost-model india
```

Regenerate with:

```bash
uv run python scripts/build_cap_tier_history.py \
  --refresh --cap-start 2014-06-01 --start 2015-01-01 --backfill-from 2015-06-01
```

`--cap-start` sets how far back the FMP market-cap pull reaches; keeping it six
months earlier than `--start` gives the first reconstitution a full averaging
window instead of a truncated one.
Month-end market caps come from FMP and are cached in
`data/universes/nse_marketcap_monthly.csv.gz`, so re-runs need no network.

Ranks 101-500 are taken within the Nifty 500 membership in force on each date,
because the mid and small indices are defined as bands of the Nifty 500 rather
than of the raw listed pool.
That anchor inherits NSE's listing-history and trading-frequency screens for
those two tiers.
Microcap has no such anchor and is ranked over everything outside the Nifty 500.

Against NSE's live constituent files, the 2026-03-31 snapshot reproduces 85% of
Nifty Midcap 150, 84% of Nifty Smallcap 250, and 60% of Nifty Microcap 250.
Anchoring is what buys the small-cap accuracy: ranking over the raw pool instead
scores 76%.

Residual biases, all upward:

- FMP's candidate pool is a current listing, so companies delisted before its
  coverage began are absent. This concentrates in the microcap tier.
- Ranking uses six month-end observations rather than NSE's daily average, so
  names within a few ranks of a band edge can land on the wrong side.
- FMP market cap is shares outstanding times price, not NSE's free-float
  adjusted figure, which shifts ranks for closely held companies.
- Snapshots before 2021 are anchored to a reconstructed Nifty 500 rather than
  the archived one; see below.

## Ten-year point-in-time Nifty 500

The archived Nifty 500 history starts in 2021, so `nifty500_pit` cannot support
a window that opens earlier.
`nifty500_extended_pit` covers 2015 onward instead.

```bash
uv run screener backtest-rolling -m india \
  --universe nifty500_extended_pit --universe-config data/universes/india_pit.toml \
  --point-in-time --years 10 --strategy momentum_12_1 --cost-model india
```

It is two histories joined at the archive's first capture:

- From 2021-01-01 onward it copies the archived snapshots verbatim, so it is
  identical to `nifty500_pit` over that stretch, capture dates included.
- Before then it reconstructs membership as the top 500 of the listed pool by
  trailing six-month average market cap, on NSE's semi-annual schedule - the
  same reconstruction the cap tiers use, applied to the whole index.

The reconstructed half is weaker than the archived half in the same three ways
the cap tiers are, and in one more: NSE's listing-history and trading-frequency
screens are not applied, so a large but thinly traded name can enter the
reconstructed index where the real one would have excluded it. Read a run that
starts before 2021 as a lower bound with a wider error bar than one that starts
after, and prefer `nifty500_pit` whenever the whole window is inside the
archive.

Rebuilding it needs no separate command: `build_cap_tier_history.py` writes
`nifty_500_extended_history.csv` alongside the four cap tiers.

## Data-source limitations

- Nifty current constituents come from NSE's published index CSV files.
- Nifty historical constituents come from Internet Archive crawls of those same
  CSV files, parsed by the same code as the live fetch. NSE publishes no
  machine-readable membership history.
- S&P 500 current membership comes from the live Wikipedia constituent table.
- S&P 500 membership at a past date is reconstructed from the Wikipedia revision
  that was current on that date, read through the MediaWiki API.
  The `oldid` of that revision is recorded as the universe source, so every
  point-in-time run names the exact article revision it used.
  Revision content never changes, so parsed revisions are cached permanently.
- When no revision can be read, the loader falls back to today's members, marks
  the result as not point-in-time and warns.
  That fallback is survivorship-biased and must not be reported as a
  point-in-time result.
- Sensex uses the public Wikipedia constituent table because BSE does not expose
  a stable documented free constituent API.
- yfinance may lack usable prices for removed or delisted securities. Missing
  history is reported and never synthesized.
