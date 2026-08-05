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

## Data-source limitations

- Nifty current constituents come from NSE's published index CSV files.
- S&P 500 current and reconstructed membership uses the existing Wikipedia
  constituent/change tables and records that provenance.
- Sensex uses the public Wikipedia constituent table because BSE does not expose
  a stable documented free constituent API.
- yfinance may lack usable prices for removed or delisted securities. Missing
  history is reported and never synthesized.
