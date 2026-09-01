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
  --output data/universes/nifty500_pit_snapshots.csv --min-symbols 400
```

Both commands write the same format and merge rather than overwrite, so the
normal setup is to backfill once and then let `sync` extend the file forward.

`--min-symbols` rejects a crawl that parsed into an implausibly short list.
A truncated capture would otherwise erase most of the index for the whole
window that snapshot covers, which reads as a plausible backtest result rather
than as the fetch failure it is.

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

## Data-source limitations

- Nifty current constituents come from NSE's published index CSV files.
- Nifty historical constituents come from Internet Archive crawls of those same
  CSV files, parsed by the same code as the live fetch. NSE publishes no
  machine-readable membership history.
- S&P 500 current and reconstructed membership uses the existing Wikipedia
  constituent/change tables and records that provenance.
- Sensex uses the public Wikipedia constituent table because BSE does not expose
  a stable documented free constituent API.
- yfinance may lack usable prices for removed or delisted securities. Missing
  history is reported and never synthesized.
