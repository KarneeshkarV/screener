# On the S&P 500 a wide ATR stop is a small gain, and a tight stop still hurts

Date: windows end 2026-09-12. Run 2026-11-08 on main at `2ff190e`.
Scope: 960 rolling backtests, 0 failures.
Universe: `sp500`, point-in-time membership (611 candidate symbols, 11073 membership windows, Wikipedia revision snapshots).
Benchmark: `SPY`.
Reproduce with `uv run python scripts/run_momentum_atr_stop_sweep.py --market us`.

This is the same sweep as [the Nifty 500 study](atr-stop-momentum-nifty500-2026-09-12.md), on the US market.
Same 15 strategies, same 16 arms, same four windows, same `hold=20`, `top=10`, `equal_slot` sizing, zero costs.
Only the universe and the benchmark change.
The cross-market comparison is at the end.

## One symbol is excluded: `PARA`

FMP serves corrupt history for `PARA`.
After Paramount delisted, FMP gave the ticker to "Parabolic Technologies, Inc.".
Its bars before 2025 are near $1,000,000 with volume 1 and repeated 10x one-day jumps.
FMP's live API serves the same data, so a cache refresh does not fix it.
On the first run this put +904% trades into every momentum strategy: `ha_momentum` showed a 60% 5-year CAGR at 165% volatility.

The sweep script drops `PARA` for `--market us` by serving it no bars.
The engine then lists it with the other vendor gaps in each run's warnings (`PARA, SATS`), so the exclusion is visible.
After the fix, the highest annual volatility in all 960 runs is 57%.
That run (`momentum_12_1`, 1 year, +104% CAGR) is real: its best trades are `MU`, `SNDK`, `STX` and `WDC` in the 2026 memory rally.

A scan of the cached US FMP files flags 79 other symbols with the same jump pattern.
Only `SBNY` is in the S&P 500 candidate list, and its jumps are the real 2023 collapse, so it stays in.
The other 78 are outside this universe.
Fixing the provider is out of scope here.

## The answer

**Tight stops hurt, as on Nifty 500. Wide ATR stops (3x to 4x) are a small gain.**
49 of 60 cells in the delta table are negative, but 8 of the 11 positive cells are in the 5-year column, and 3 of them are `atr_3` to `atr_4`.

| Arm | Cells where it beats `no_stop` (of 60) |
|---|---|
| `atr_4` | 38 |
| `atr_3` | 34 |
| `pct_20` | 30 |
| `atr_2.5` | 29 |
| `atr_3.5`, `pct_15` | 25 |
| `atr_2` | 20 |
| `pct_12` | 19 |
| `pct_10` | 17 |
| `pct_04` | 15 |
| `atr_1.5` | 13 |
| `atr_0.5` | 12 |
| `pct_06` | 10 |
| `pct_08` | 9 |
| `atr_1` | 6 |

`atr_4` is the most frequent best arm (14 of 60 cells), then `no_stop` (11) and `atr_3` (8).
`atr_4` beats `no_stop` on mean Sharpe for 12 of the 15 strategies.

### Sharpe against the no-stop control

Delta in median Sharpe, arm minus `no_stop`:

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| atr_0.5 | -0.648 | -0.597 | -0.161 | -0.066 |
| atr_1 | -0.461 | -0.364 | -0.233 | -0.012 |
| atr_1.5 | -0.307 | -0.297 | -0.252 | +0.002 |
| atr_2 | -0.223 | -0.183 | -0.183 | +0.020 |
| atr_2.5 | -0.051 | -0.069 | -0.177 | +0.022 |
| atr_3 | +0.065 | -0.044 | -0.133 | +0.101 |
| atr_3.5 | -0.013 | -0.027 | -0.081 | +0.109 |
| atr_4 | +0.128 | +0.017 | -0.093 | +0.114 |
| pct_04 | -0.223 | -0.327 | -0.120 | -0.005 |
| pct_06 | -0.242 | -0.362 | -0.158 | -0.104 |
| pct_08 | -0.483 | -0.219 | -0.247 | -0.031 |
| pct_10 | -0.179 | -0.214 | -0.139 | -0.076 |
| pct_12 | -0.095 | -0.085 | -0.115 | +0.018 |
| pct_15 | -0.007 | -0.043 | -0.027 | -0.081 |
| pct_20 | -0.038 | -0.037 | -0.004 | +0.124 |

The 3-year column is negative for every arm.
The 5-year column is where wide stops help: the unstopped 5-year median is only 0.746, and that window holds the 2022 bear market.

### Against the benchmark

SPY over these windows: +17.5% (1y), +18.3%/yr (2y), +21.1%/yr (3y), +12.9%/yr (5y).
Median annualised alpha:

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| no_stop | +0.116 | +0.154 | +0.091 | +0.049 |
| atr_4 | **+0.181** | +0.182 | +0.078 | +0.092 |
| pct_20 | +0.119 | **+0.189** | **+0.106** | **+0.093** |
| atr_2 | +0.102 | +0.128 | +0.064 | +0.055 |
| atr_0.5 | +0.015 | +0.031 | +0.063 | +0.043 |

Median 5-year CAGR: 13.9% with `no_stop`, 18.4% with `atr_4`, 18.6% with `pct_20`.
Share of strategies that beat SPY on CAGR at 5y: 67% with `no_stop`, 60% with `atr_4`, `atr_3` and `pct_20`, 53% with `atr_0.5`.
The median rises with a wide stop, but fewer strategies clear the index.

## Drawdown: stops do not deepen it on the US

Median max drawdown, relative to no stop (negative is worse):

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| atr_0.5 | -0.018 | +0.055 | +0.011 | +0.031 |
| atr_1 | -0.024 | +0.032 | -0.018 | -0.003 |
| atr_2 | -0.007 | +0.041 | -0.006 | +0.036 |
| atr_3 | +0.001 | +0.045 | +0.018 | +0.023 |
| atr_4 | +0.028 | +0.043 | +0.000 | +0.012 |
| pct_04 | +0.022 | +0.070 | +0.022 | +0.047 |
| pct_20 | +0.005 | +0.017 | -0.002 | +0.007 |

Over all 15 stop arms, deeper cells per window: 4 at 1y, 0 at 2y, 3 at 3y, 1 at 5y.
The deeper cells are all small (at most 2.4 points).
This is the opposite of Nifty 500, where all 15 arms were deeper at 5 years.

| 5-year median | no_stop | atr_0.5 | atr_2 | atr_4 |
| --- | --- | --- | --- | --- |
| beta | 0.85 | 0.63 | 0.76 | 0.80 |
| annual vol | 0.226 | 0.188 | 0.212 | 0.227 |
| max drawdown | -0.286 | **-0.254** | **-0.250** | -0.274 |
| CAGR | 0.139 | 0.117 | 0.143 | **0.184** |

On the US, the stop cuts the worst case together with the average risk.
The churn is still there (5-year median trades: 602 unstopped, 2108 at 0.5x ATR, 660 at 4x), but it does not show up as a deeper drawdown.

## ATR versus a fixed stop: on the US, ATR wins

Ranked by mean of the median Sharpe over the four windows, with the median share of trades the stop closed:

| arm | fires | 1y | 2y | 3y | 5y | mean |
|---|---|---|---|---|---|---|
| **atr_4** | 15% | 1.364 | 1.207 | 1.077 | 0.860 | **1.127** |
| pct_20 | 3% | 1.199 | 1.153 | 1.167 | 0.870 | 1.097 |
| no_stop | 0% | 1.236 | 1.190 | 1.171 | 0.746 | 1.086 |
| atr_3 | 26% | 1.301 | 1.147 | 1.037 | 0.847 | 1.083 |
| atr_3.5 | 20% | 1.223 | 1.164 | 1.090 | 0.854 | 1.083 |
| pct_15 | 10% | 1.229 | 1.147 | 1.143 | 0.665 | 1.046 |
| atr_2.5 | 35% | 1.185 | 1.121 | 0.994 | 0.767 | 1.017 |
| pct_12 | 16% | 1.141 | 1.105 | 1.055 | 0.764 | 1.016 |
| atr_2 | 46% | 1.013 | 1.008 | 0.988 | 0.765 | 0.944 |
| pct_10 | 22% | 1.057 | 0.976 | 1.031 | 0.669 | 0.933 |
| pct_04 | 59% | 1.014 | 0.863 | 1.051 | 0.741 | 0.917 |
| atr_1.5 | 56% | 0.929 | 0.894 | 0.919 | 0.747 | 0.872 |
| pct_06 | 43% | 0.994 | 0.828 | 1.013 | 0.641 | 0.869 |
| pct_08 | 33% | 0.753 | 0.972 | 0.923 | 0.714 | 0.841 |
| atr_1 | 69% | 0.775 | 0.826 | 0.938 | 0.734 | 0.818 |
| atr_0.5 | 84% | 0.588 | 0.593 | 1.010 | 0.680 | 0.718 |

### Matched on firing rate

| fixed | fires | matched ATR | fires | Sharpe fixed | Sharpe ATR | gap | ATR wins (of 60 cells) |
|---|---|---|---|---|---|---|---|
| pct_04 | 0.59 | atr_1.5 | 0.56 | 0.917 | 0.872 | -0.045 | 35 |
| pct_06 | 0.43 | atr_2 | 0.46 | 0.869 | 0.944 | +0.074 | 35 |
| pct_08 | 0.33 | atr_2.5 | 0.35 | 0.841 | 1.017 | +0.176 | **50** |
| pct_10 | 0.22 | atr_3.5 | 0.20 | 0.933 | 1.083 | +0.149 | 43 |
| pct_12 | 0.16 | atr_4 | 0.14 | 1.016 | 1.127 | +0.111 | 44 |
| pct_15 | 0.10 | atr_4 | 0.14 | 1.046 | 1.127 | +0.081 | 35 |
| pct_20 | 0.03 | atr_4 | 0.14 | 1.097 | 1.127 | +0.030 | 33 |

ATR wins 6 of 7 pairs on the mean, and 275 of 420 matched cells (65%).
Mean absolute gap between a matched pair: 0.095 Sharpe.
Gap from no stop to the average stop arm: 0.127.
On Nifty 500 the mechanism mattered three times less than the choice to stop at all.
On the S&P 500 it matters almost as much.

### Firing rate still explains most of it

```
corr(firing rate, Sharpe) = -0.859        R^2 = 0.738
  ATR arms only:   -0.997
  fixed arms only: -0.775

residual once firing rate is accounted for:
  ATR   +0.034
  fixed -0.039
```

Firing rate explains 74% of the variance across arms, and the relation is monotone for ATR (-0.997).
The family label adds 0.073 Sharpe on top, against 0.016 on Nifty 500.
The fit gives +1.116 for a stop that never fires, and the measured no-stop result is +1.086, a little below the trend.
On Nifty 500 no stop was above the trend.

Dispersion of Sharpe across the 15 strategies: `no_stop` 0.39, `atr_3` 0.35, `atr_4` 0.37, `pct_15` 0.37, `pct_20` 0.39.
Wide ATR stops compress dispersion a little on the US, which is the claim made for ATR.
On Nifty 500 they did not.

## Which strategy, then

Unstopped, ranked by mean Sharpe over the four windows:

| strategy | 1y | 2y | 3y | 5y | mean | worst |
|---|---|---|---|---|---|---|
| **bb_breakout** | 1.263 | 1.625 | 1.561 | 0.706 | **1.289** | 0.71 |
| momentum_12_1_trend | 1.623 | 1.270 | 1.188 | 0.965 | 1.262 | 0.97 |
| rs_momentum_regime | 1.236 | 1.347 | 1.279 | 1.088 | 1.237 | **1.09** |
| momentum_12_1 | 1.531 | 1.292 | 1.163 | 0.962 | 1.237 | 0.96 |
| momentum_12_1_riskadj | 1.490 | 1.174 | 1.317 | 0.961 | 1.236 | 0.96 |
| supertrend_flip | 1.335 | 1.190 | 1.171 | 1.021 | 1.179 | 1.02 |
| momentum_12_1_ema10 | 1.259 | 1.192 | 1.306 | 0.917 | 1.169 | 0.92 |
| rs_breakout | 0.979 | 1.360 | 1.482 | 0.746 | 1.142 | 0.75 |
| ema_trend | 1.412 | 1.145 | 1.029 | 0.840 | 1.107 | 0.84 |
| breakout | 0.761 | 1.246 | 1.071 | 0.725 | 0.951 | 0.73 |
| mom_lowvol_combo | 0.926 | 1.015 | 1.305 | 0.547 | 0.948 | 0.55 |
| donchian_breakout | 1.114 | 0.622 | 1.000 | 0.734 | 0.868 | 0.62 |
| ha_momentum | 0.458 | 0.834 | 0.630 | 0.606 | 0.632 | 0.46 |
| supertrend_rsi | 0.639 | 0.361 | 0.366 | 0.287 | 0.413 | 0.29 |
| supertrend | 0.512 | -0.451 | 0.323 | 0.468 | 0.213 | -0.45 |

Five-year detail for the leaders by Sharpe, against SPY at 12.9%/yr:

| strategy | Sharpe | CAGR | alpha | beta | max DD | profit factor |
|---|---|---|---|---|---|---|
| rs_momentum_regime | **1.088** | 26.9% | +14.6% | 0.97 | -28.5% | 1.53 |
| supertrend_flip | 1.021 | 24.3% | +12.0% | 0.98 | **-26.4%** | 1.48 |
| momentum_12_1_trend | 0.965 | 34.1% | **+19.6%** | 1.37 | -36.3% | 1.73 |
| momentum_12_1 | 0.962 | **34.4%** | +19.3% | 1.42 | -36.3% | 1.73 |
| momentum_12_1_riskadj | 0.961 | 29.1% | +17.2% | 1.08 | -36.1% | **1.81** |

`rs_momentum_regime` is the most stable: best 5-year Sharpe and the best worst-window Sharpe.
`bb_breakout` has the best mean, but its 5-year Sharpe is near the bottom of the leaders, so it is a recent-window result.
The `momentum_12_1` family takes four of the top seven places, at a beta above 1.
The gaps across the top nine (1.11 to 1.29) are too small to trust the order.

## Cross-market comparison

| | Nifty 500 | S&P 500 |
|---|---|---|
| Negative delta cells (of 60) | 54 | 49 |
| Best arm by mean Sharpe | `pct_20` (+0.009 over `no_stop`) | `atr_4` (+0.041 over `no_stop`) |
| `atr_4` beats `no_stop` (cells of 60) | 24 | 38 |
| Stop arms with deeper 5y drawdown (of 15) | 15 | 1 |
| corr(firing rate, Sharpe) | -0.917 | -0.859 |
| ATR wins matched cells | 53% | 65% |
| Matched gap / stop-vs-no-stop gap | 0.073 / 0.249 | 0.095 / 0.127 |
| Tightest stop (0.5x ATR): hit rate, share of trades closed | 13%, 86% | 15%, 84% |

What holds in both markets:

- Firing rate is the main variable, and the relation is monotone.
  Every stop that closes more than a third of trades is below no stop on mean Sharpe in both markets.
- Tight stops (0.5x to 2x ATR, 4% to 10% fixed) are worse than no stop.
- `pct_20` is close to `no_stop` because it closes only 3% of trades.

What does not hold:

- On Nifty 500 every stop deepens the 5-year drawdown.
  On the S&P 500 none do by more than 0.3 points.
- On Nifty 500 ATR and fixed are a coin flip.
  On the S&P 500 ATR wins 6 of 7 matched pairs.
- On Nifty 500 the best stop costs nothing.
  On the S&P 500 `atr_3` to `atr_4` add +0.10 to +0.11 Sharpe at 5 years, and lose 0.08 to 0.13 at 3 years.

Strategy ranks do not carry over (Spearman 0.28 on mean Sharpe):

| strategy | Nifty 500 rank | S&P 500 rank |
|---|---|---|
| ha_momentum | 1 | 13 |
| momentum_12_1_ema10 | 2 | 7 |
| momentum_12_1 | 3 | 4 |
| momentum_12_1_riskadj | 4 | 5 |
| momentum_12_1_trend | 5 | 2 |
| rs_momentum_regime | 9 | 3 |
| bb_breakout | 8 | 1 |
| supertrend_flip | 13 | 6 |
| supertrend | 15 | 15 |
| supertrend_rsi | 7 | 14 |

Only the `momentum_12_1` family is in the top half in both markets.
`supertrend` is last in both.
`ha_momentum`, the Nifty 500 leader, is third from last on the US, and `supertrend_flip`, which loses money on Nifty 500, is sixth on the US.
Do not carry a strategy choice from one market to the other without a run.

## Caveats

**`atr_4` is the best of 15 arms on one dataset.**
Its +0.041 mean Sharpe edge is selection-biased, and the windows are nested, so the 60 cells are not independent.
The edge is about one tenth of the spread between strategies, and it reverses at 3 years.
Read it as "a wide ATR stop does not hurt on the US", not as a proven gain.

**Costs were zero.**
`atr_4` adds about 10% turnover at 5 years (660 trades against 602), so a small cost would take only part of its edge.
The tight arms multiply turnover by up to 3.5x and would lose more.

**`PARA` is excluded and `SATS` has no FMP history.**
Both appear in each run's warnings.

## Recommendation

Keep the stop off by default for momentum in both markets.
On the US, if a stop is wanted, use a wide ATR stop (3x to 4x, window 14).
`atr_4` and `pct_20` are the only arms above no stop on mean Sharpe, and `atr_4` beats the fixed stops that fire at a similar rate.
Do not use a tight stop of either kind in either market.

The next experiment is the same as for Nifty 500: a trailing ATR stop, and `--sizing atr_risk` with the stop on.
