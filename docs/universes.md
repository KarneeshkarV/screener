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

## Dynamic universe

Dynamic membership is ranked by average dollar volume calculated from prior
bars only and held until the next rebalance.

```bash
uv run screener backtest-rolling -m india \
  --universe dynamic --dynamic-base nifty500 \
  --universe-size 100 --universe-lookback 60 \
  --universe-rebalance monthly --strategy rs_breakout
```

The default candidate pool is the current S&P 500 for US runs and current Nifty
500 for India runs. Ranking is causal, but that candidate pool remains
survivorship-biased. For rigorous history, supply dated snapshots containing
removed and delisted securities.

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
- S&P 500 current and reconstructed membership uses the existing Wikipedia
  constituent/change tables and records that provenance.
- Sensex uses the public Wikipedia constituent table because BSE does not expose
  a stable documented free constituent API.
- yfinance may lack usable prices for removed or delisted securities. Missing
  history is reported and never synthesized.
