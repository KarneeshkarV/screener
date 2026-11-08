# India momentum, filter by filter, on survivorship-free data

Research, not financial advice.

## What was tested

A retail momentum model for Indian stocks, one rule at a time:

- stage-2 trend (close above the 30, 50 and 100-day SMA),
- within 20% of the 52-week high,
- rank by 12-month rate of change divided by 12-month volatility,
- skip names stuck in price-band locks,
- blend in the distance to the 52-week high,
- top 20 equal weight, monthly rank exit, top-50 hold buffer.

Each stage adds one rule to the stage before it.
A second set ranks on 12-1 momentum (the 12-month return that skips the last month) in place of 12-month rate of change.

## Data

yfinance has no bars for most delisted Indian names, and the repo's NSE universes start in 2018.
A test on them keeps only the survivors.
`scripts/india_pit_bhavcopy.py` builds the data from NSE's daily bhavcopy instead.

- 3430 sessions, 2013-01-01 to 2026-11-06, old `cm*.zip` and UDiFF formats, including 14 weekend Muhurat and special sessions.
- Symbols link to one entity through ISIN and dated renames from NSE's `symbolchange.csv` plus two manual ones.
  A rename links only a symbol that traded before the rename date, so a reused symbol stays separate.
- Universes: the top 500 and top 750 entities by 126-session median traded value, ranked on the session before each month's first session, at least 80% traded sessions.
  160 monthly snapshots each, 2013-08 to 2026-11, with 1360 and 1819 entities ever a member.
- `nifty500_pit_entity`: the repo's `nifty500_pit` renamed to entity symbols, 2018-10 to 2026-09.
- Prices are back-adjusted per entity.
  Adjustments applied: 711 listed splits and bonuses, 5 inferred clean splits, 181 rights issues, 118 demergers, 173 one-day bad prints dropped.
  14 rights issues with no single stated price are not adjusted.
  758 large gaps stay unexplained and are kept as real moves.
- Check against the yfinance cache on 1508 names: total return within 2% for 90.6%, within 10% for 94.8%, off by more than 25% for 49.
  Most of the 49 are demergers, which yfinance does not adjust.

## Backtest setup

- `backtest-rolling` engine, point-in-time membership, bhavcopy prices for every `.NS` name.
- India cost model plus 10 bps slippage, `reinvested_equal_slot` sizing, total-return prices.
- 2019-04-01 to 2026-11-06, and 2014-04-01 to 2026-11-06 for the long run on the top 750.

## Results

CAGR / Sharpe / max drawdown.

| stage | rule added | Nifty 500 PIT | turnover top 500 | turnover top 750 | top 750 from 2014 |
|---|---|---|---|---|---|
| s0 | rank by 52-week-high distance | 16.3 / 0.94 / -27.8 | 20.4 / 1.09 / -28.2 | 24.0 / 1.21 / -28.9 | 16.9 / 0.95 / -46.4 |
| s1 | rank by 12-month ROC | 29.7 / 1.15 / -39.0 | 32.0 / 1.16 / -41.8 | 38.0 / 1.34 / -42.3 | 31.7 / 1.14 / -63.6 |
| s1b | s1 + top-50 hold buffer | 33.5 / 1.27 / -41.6 | 35.9 / 1.28 / -38.2 | 41.6 / 1.43 / -49.9 | 32.2 / 1.16 / -62.2 |
| s2 | + stage-2 trend | 20.8 / 0.90 / -37.6 | 26.4 / 1.02 / -40.3 | 32.4 / 1.20 / -41.5 | 29.5 / 1.13 / -64.4 |
| s3 | + within 20% of high | 22.0 / 0.95 / -36.4 | 30.8 / 1.16 / -39.2 | 34.8 / 1.26 / -40.9 | 27.6 / 1.08 / -63.5 |
| s4 | + rank by ROC / vol | 20.7 / 0.96 / -34.5 | 27.1 / 1.10 / -37.2 | 35.6 / 1.37 / -38.5 | 32.6 / 1.30 / -55.8 |
| s5 | + band-lock filter | 18.8 / 0.89 / -33.7 | 28.0 / 1.14 / -35.9 | 28.9 / 1.16 / -37.5 | 28.6 / 1.22 / -47.3 |
| s6 | + top-50 hold buffer | 16.9 / 0.82 / -36.7 | 30.2 / 1.22 / -35.9 | 27.6 / 1.13 / -36.9 | 26.4 / 1.15 / -49.7 |
| s7 | + 52-week-high in the rank | 16.8 / 0.86 / -31.2 | 24.9 / 1.11 / -29.3 | 26.9 / 1.17 / -30.0 | 20.9 / 1.02 / -41.1 |
| s8 | + above 200-day SMA | 16.4 / 0.84 / -31.1 | 24.7 / 1.10 / -29.3 | 27.1 / 1.18 / -30.0 | 20.8 / 1.01 / -41.6 |

What each rule adds:

- Momentum ranking is the whole edge.
  Ranking by 52-week-high distance alone (s0) earns 16-24% CAGR; ranking by 12-month ROC (s1) earns 30-38%.
- A hold buffer on the plain ROC rank (s1b) gives the best Sharpe on all three 2019 universes, but keeps a 38% to 50% drawdown.
- The trend, near-high and vol-scaling rules (s2 to s4) cut drawdown by 4 to 8 points.
  They cost 9 points of CAGR on the Nifty 500, 5 on the top 500, 2 on the top 750 and nothing from 2014.
- s4 has the best Sharpe on the top 750 from 2014.
- The band-lock filter and the hold buffer (s5, s6) do not help in a consistent way.
- The composite rank (s7) cuts drawdown by another 5 to 9 points for 0 to 6 points of CAGR.
  The 200-day SMA (s8) adds nothing on top of it.
- No stage avoids a deep drawdown over the long window: 41% to 64% from 2014, from a 2018 peak to the March-April 2020 low.

## 12-1 momentum

Same stages, ranked on `close[t-21] / close[t-252] - 1`, with `mom_12_1 > 0` as an extra entry floor.
Each row shows the 12-1 stage minus its 12-month twin.

| 12-1 stage | twin | Nifty 500 PIT | turnover top 500 | turnover top 750 | top 750 from 2014 |
|---|---|---|---|---|---|
| m1 | s1 | 32.3 / 1.25 / -39.5 (+2.6 / +0.10 / -0.5) | 35.1 / 1.26 / -41.9 (+3.1 / +0.10 / -0.1) | 33.5 / 1.23 / -44.5 (-4.5 / -0.10 / -2.2) | 33.6 / 1.20 / -57.1 (+1.9 / +0.07 / +6.5) |
| m4 | s4 | 19.8 / 0.95 / -35.2 (-1.0 / -0.01 / -0.7) | 27.0 / 1.13 / -35.8 (-0.1 / +0.03 / +1.3) | 29.5 / 1.22 / -41.1 (-6.1 / -0.15 / -2.6) | 30.7 / 1.27 / -53.1 (-1.9 / -0.03 / +2.7) |
| m6 | s6 | 18.0 / 0.89 / -35.2 (+1.2 / +0.07 / +1.4) | 27.3 / 1.15 / -36.5 (-2.9 / -0.07 / -0.5) | 26.0 / 1.11 / -39.5 (-1.6 / -0.02 / -2.6) | 23.8 / 1.09 / -49.8 (-2.6 / -0.06 / -0.1) |
| m7 | s7 | 17.2 / 0.88 / -33.2 (+0.4 / +0.02 / -2.1) | 29.5 / 1.28 / -29.7 (+4.6 / +0.17 / -0.3) | 28.2 / 1.23 / -34.6 (+1.3 / +0.06 / -4.7) | 20.8 / 1.04 / -38.9 (-0.1 / +0.02 / +2.2) |

- 12-1 helps the plain momentum rank (m1) on 3 of 4 universes, by 2 to 3 points of CAGR and 0.07 to 0.10 Sharpe.
  On the top 750 from 2019 it is 4.5 points worse.
- With the trend, near-high and vol filters on (m4), 12-1 is the same or worse on every universe.
- With the band-lock filter and hold buffer (m6), 12-1 is worse on 3 of 4 universes.
- With the composite rank (m7), 12-1 is better on the 2019 turnover universes and the same on the others.
- Verdict: 12-1 is not a clear improvement on the full model.
  Most differences are within the rebalance-day effect in the caveats.

## Allocation

s4 on the turnover top 750, 2019-04 to 2026-11, with monthly rebalance.
Gold is GOLDBEES, REIT is the mean of EMBASSY and MINDSPACE, debt is a flat 6% a year.

| mix | CAGR | Sharpe | max drawdown |
|---|---|---|---|
| 100% momentum | 34.9% | 1.37 | -38.5% |
| 70 momentum / 30 gold | 32.2% | 1.65 | -28.9% |
| 80 momentum / 10 gold / 10 REIT | 32.3% | 1.51 | -33.9% |
| 80 / 20 gold+REIT, moved in on drawdown | 33.0% | 1.43 | -35.5% |
| 70 / 30 gold, moved in on drawdown | 30.6% | 1.40 | -33.5% |
| 70 momentum / 30 debt | 26.7% | 1.50 | -27.4% |
| Nifty 50 (price) | 10.6% | 0.67 | -38.4% |
| GOLDBEES | 20.6% | 1.17 | -24.4% |

A fixed 30% gold sleeve gives the best Sharpe and cuts drawdown by 10 points for 2.7 points of CAGR.
Moving the sleeve into momentum on drawdown does not beat the fixed mix.
Gold had an unusually strong 2019-2026; the sleeve's benefit depends on that.

## Caveats

- One path per universe, no walk-forward or parameter sweep.
- Results depend on the rebalance day.
  The last data rebuild added 14 weekend sessions, which moved each monthly rebalance by one session, and fixed some rights adjustments.
  Together they changed s2 to s8 on the Nifty 500 by 3 to 8 points of CAGR and up to 0.3 Sharpe, but s0 and s1 by 2 points or less.
  Treat differences smaller than that as noise.
- The band-lock count reads closes at the day's high or low with a move of 4.9% or more; 2% band locks are not caught.
- The 758 unexplained gaps are kept as real moves; some will be missed corporate actions.
- Demerger adjustment uses the ex-date open against the prior close, so it also absorbs that day's market move.
- Bhavcopy carries no index constituents before 2018, so the long window uses the turnover universe only.

## Reproduce

```bash
uv run python -m scripts.india_pit_bhavcopy download
uv run python -m scripts.india_pit_bhavcopy parse
uv run python -m scripts.india_pit_bhavcopy universe --top 500 --top 750
uv run python -m scripts.india_pit_bhavcopy actions
uv run python -m scripts.india_pit_bhavcopy prices
uv run python -m scripts.india_pit_bhavcopy check
uv run python -m scripts.run_india_momentum_ablation stages --universe india_turnover750_pit --bhavcopy-prices
uv run python -m scripts.run_india_momentum_ablation allocate s4_volpen
```
