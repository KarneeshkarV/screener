# Stops that fire make momentum worse on Nifty 500, and ATR versus fixed barely matters

Date: windows end 2026-09-12. Rerun 2026-11-06 on main at `2ff190e`.
Scope: 960 rolling backtests, 0 failures.
Universe: `nifty500_pit`, point-in-time membership (841 candidate symbols, 6507 membership windows).
Reproduce with `uv run python scripts/run_momentum_atr_stop_sweep.py`.

The same sweep on the S&P 500 is in [atr-stop-momentum-sp500-2026-09-12.md](atr-stop-momentum-sp500-2026-09-12.md).
Tight stops hurt there too, but wide ATR stops do not deepen drawdown and beat a fixed stop matched on firing rate, so the conclusions below are for Nifty 500 only.

The first version of this doc ran on an older main (850 symbols, 6520 windows).
Main moved between the two runs, and the numbers below replace the old ones.
Two conclusions changed: `pct_20` now ties `no_stop` on mean Sharpe, and the drawdown damage is clear only at 3 and 5 years.
The old version is in git history.

## What was tested

Every momentum-family strategy in the registry (15 of them) against an ATR stop grid and a fixed stop grid, over nested trailing windows.

| Axis | Values |
|---|---|
| Strategies | `bb_breakout`, `breakout`, `donchian_breakout`, `ema_trend`, `ha_momentum`, `mom_lowvol_combo`, `momentum_12_1`, `momentum_12_1_ema10`, `momentum_12_1_riskadj`, `momentum_12_1_trend`, `rs_breakout`, `rs_momentum_regime`, `supertrend`, `supertrend_flip`, `supertrend_rsi` |
| ATR arms | `--stop-atr` at 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 (window 14) |
| Fixed arms | `--stop-loss` at 4%, 6%, 8%, 10%, 12%, 15%, 20% |
| Control | `no_stop` |
| Windows | 1, 2, 3, 5 years, all ending 2026-09-12 |

Fixed: `hold=20`, `top=10`, `equal_slot` sizing, benchmark `^NSEI`, daily bars.

Windows are nested, so the 1-year sits inside the 5-year.
Read them as "did this hold up as the window shortened", not as four independent samples.

## The answer

**No stop wins, or ties with a stop that almost never fires.**
`no_stop` is the most frequent best arm, in 15 of 60 strategy x window cells.
The next are `pct_15` and `pct_20`, with 6 each.
No stop arm beats `no_stop` in half of the cells:

| Arm | Cells where it beats `no_stop` (of 60) |
|---|---|
| `atr_4`, `pct_20` | 24 |
| `atr_3.5`, `pct_15` | 19 |
| `atr_3` | 16 |
| `atr_2.5` | 15 |
| `pct_06` | 14 |
| `atr_2`, `pct_10` | 13 |
| `pct_08` | 11 |
| `pct_04` | 10 |
| `atr_1.5`, `pct_12` | 8 |
| `atr_0.5` | 7 |
| `atr_1` | 6 |

Median Sharpe, all 15 strategies:

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| no_stop | 0.126 | **-0.211** | **0.685** | **0.707** |
| pct_20 | **0.311** | -0.260 | 0.644 | 0.649 |
| atr_4 | 0.136 | -0.254 | 0.526 | 0.611 |
| atr_3.5 | 0.011 | -0.279 | 0.407 | 0.621 |
| pct_08 | -0.329 | -0.435 | 0.184 | 0.535 |
| atr_0.5 | -0.295 | -0.583 | -0.089 | 0.246 |

`no_stop` has the best median Sharpe at 2, 3 and 5 years.
At 1 year, wide stops lead, and `pct_20` is best.
The tight end is very bad: at 0.5x ATR the median hit rate falls from 50% to 13%, because the stop is inside the daily noise of a Nifty 500 name.
It closes 86% of trades.

### Sharpe against the no-stop control

Delta in median Sharpe, arm minus `no_stop`.
**54 of 60 cells are negative.**
All 6 positive cells are in the 1-year column.
At 2, 3 and 5 years all 45 cells are negative.

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| atr_0.5 | -0.420 | -0.371 | -0.774 | -0.462 |
| atr_1 | -0.147 | -0.461 | -0.644 | -0.370 |
| atr_1.5 | -0.185 | -0.411 | -0.482 | -0.402 |
| atr_2 | -0.397 | -0.324 | -0.451 | -0.329 |
| atr_2.5 | -0.290 | -0.242 | -0.348 | -0.192 |
| atr_3 | +0.068 | -0.215 | -0.428 | -0.226 |
| atr_3.5 | -0.115 | -0.068 | -0.277 | -0.086 |
| atr_4 | +0.011 | -0.043 | -0.159 | -0.096 |
| pct_04 | -0.080 | -0.247 | -0.503 | -0.340 |
| pct_06 | +0.053 | -0.288 | -0.443 | -0.315 |
| pct_08 | -0.455 | -0.224 | -0.501 | -0.172 |
| pct_10 | -0.095 | -0.233 | -0.422 | -0.162 |
| pct_12 | +0.116 | -0.176 | -0.316 | -0.348 |
| pct_15 | +0.059 | -0.060 | -0.293 | -0.177 |
| pct_20 | +0.185 | -0.049 | -0.041 | -0.058 |

Averaged over the four windows, `no_stop` scores +0.327 and `pct_20` scores +0.336.
That gap (0.009) is noise, and `pct_20` closes only 3% of trades.
The best stop that fires on more than 10% of trades is `atr_4` at +0.255.
Cells with Sharpe > 1.0: `no_stop` 5 of 60, best stop arm (`atr_4`) 3 of 60.

### Against the benchmark

^NSEI over these windows: -6.8% (1y), -4.0%/yr (2y), +5.3%/yr (3y), +6.2%/yr (5y).
Median annualised alpha:

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| no_stop | +0.062 | **-0.011** | +0.080 | **+0.121** |
| pct_20 | **+0.103** | -0.018 | **+0.085** | +0.081 |
| pct_15 | +0.087 | -0.025 | +0.024 | +0.090 |
| atr_3.5 | +0.055 | -0.022 | +0.028 | +0.091 |
| atr_4 | +0.082 | -0.016 | +0.040 | +0.074 |
| atr_2 | -0.002 | -0.051 | -0.004 | +0.033 |

At 5 years `no_stop` has the highest alpha, by 3 points over the next arm.
Share of strategies that beat the index on CAGR at 5y: 93% with `no_stop`, 93% with `pct_20`, 87% with `atr_4`, and 40% to 47% for the tight ATR arms.

## Drawdown: the stop does not prevent it

Median max drawdown, relative to no stop (negative is worse):

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| atr_1.5 | -0.030 | -0.026 | -0.074 | **-0.067** |
| atr_2.5 | -0.003 | -0.043 | -0.078 | -0.042 |
| atr_3 | -0.014 | -0.007 | **-0.085** | -0.046 |
| atr_4 | -0.004 | -0.022 | -0.049 | -0.034 |
| pct_08 | -0.008 | -0.010 | -0.042 | -0.031 |
| pct_15 | -0.016 | +0.022 | -0.039 | -0.035 |
| pct_20 | +0.008 | +0.001 | +0.003 | -0.052 |

At 5 years, all 15 stop arms have a deeper median drawdown than no stop.
At 3 years, 14 of 15 are deeper, and the one exception (`pct_20`, +0.003) is flat.
At 1 and 2 years, 11 of 15 are deeper, and the few positives do not agree in sign across widths, so they are noise.

The likely mechanism is the one already seen with entry-only risk gates.
The stop sells into a pullback, and the strategy re-enters after the bounce has started.
You realize the loss and buy back higher, and doing that again and again through a drawdown adds to it.
The turnover agrees: median 5-year trade count goes from 591 with no stop to 2240 at 0.5x ATR, and is still 646 at 4x.

The stop does take risk off, but the worst case does not improve:

| 5-year median | no_stop | atr_0.5 | atr_2 | atr_4 |
| --- | --- | --- | --- | --- |
| beta | 0.94 | 0.71 | 0.86 | 0.92 |
| annual vol | 0.225 | 0.172 | 0.212 | 0.230 |
| max drawdown | **-0.325** | -0.326 | -0.358 | -0.359 |

At 0.5x ATR, beta falls from 0.94 to 0.71 and volatility from 22.5% to 17.2%, but the drawdown stays the same, and median CAGR falls by 10.6 points.
An exit that only cut exposure would cut the worst case with the average risk.
This one cuts the average risk and keeps the worst case, and that gap is the churn.

## ATR versus a fixed stop: the mechanism is not the variable

Ranked by mean of the median Sharpe over the four windows, with the median share of trades the stop closed:

| arm | fires | 1y | 2y | 3y | 5y | mean |
|---|---|---|---|---|---|---|
| pct_20 | 3% | 0.311 | -0.260 | 0.644 | 0.649 | **+0.336** |
| **no_stop** | 0% | 0.126 | -0.211 | 0.685 | 0.707 | +0.327 |
| atr_4 | 15% | 0.136 | -0.254 | 0.526 | 0.611 | +0.255 |
| pct_15 | 10% | 0.184 | -0.272 | 0.392 | 0.530 | +0.209 |
| atr_3.5 | 19% | 0.011 | -0.279 | 0.407 | 0.621 | +0.190 |
| pct_12 | 18% | 0.242 | -0.387 | 0.369 | 0.359 | +0.146 |
| atr_3 | 27% | 0.193 | -0.426 | 0.257 | 0.482 | +0.126 |
| pct_10 | 26% | 0.030 | -0.444 | 0.263 | 0.545 | +0.099 |
| pct_06 | 50% | 0.178 | -0.500 | 0.242 | 0.392 | +0.078 |
| atr_2.5 | 36% | -0.164 | -0.453 | 0.337 | 0.515 | +0.059 |
| pct_04 | 67% | 0.046 | -0.458 | 0.182 | 0.368 | +0.034 |
| pct_08 | 38% | -0.329 | -0.435 | 0.184 | 0.535 | -0.011 |
| atr_1.5 | 59% | -0.060 | -0.623 | 0.203 | 0.306 | -0.043 |
| atr_2 | 48% | -0.271 | -0.535 | 0.234 | 0.378 | -0.049 |
| atr_1 | 73% | -0.022 | -0.672 | 0.041 | 0.338 | -0.079 |
| atr_0.5 | 86% | -0.295 | -0.583 | -0.089 | 0.246 | -0.180 |

### Matched on firing rate

Comparing the best of eight ATR widths against one fixed level is not fair.
Two stops are comparable when they close a similar share of trades, so each fixed level is paired with the ATR width that fires at the nearest rate:

| fixed | fires | matched ATR | fires | Sharpe fixed | Sharpe ATR | gap | ATR wins (of 60 cells) |
|---|---|---|---|---|---|---|---|
| pct_04 | 0.67 | atr_1 | 0.73 | +0.034 | -0.079 | -0.113 | 26 |
| pct_06 | 0.50 | atr_2 | 0.48 | +0.078 | -0.049 | -0.127 | 30 |
| pct_08 | 0.38 | atr_2.5 | 0.36 | -0.011 | +0.059 | +0.070 | 34 |
| pct_10 | 0.26 | atr_3 | 0.26 | +0.099 | +0.126 | +0.028 | 32 |
| pct_12 | 0.18 | atr_3.5 | 0.19 | +0.146 | +0.190 | +0.044 | **45** |
| pct_15 | 0.10 | atr_4 | 0.15 | +0.209 | +0.255 | +0.046 | 35 |
| pct_20 | 0.03 | atr_4 | 0.15 | +0.336 | +0.255 | -0.081 | 20 |

ATR wins 4 of 7 pairs on the mean.
Per cell, ATR wins 222 of 420 matched comparisons (53%), close to a coin flip.
ATR does a little better in the middle of the grid (`atr_3.5` against `pct_12` wins 45 of 60), and fixed does better at the tight end.

The scale settles it.
Mean absolute gap between a matched ATR and fixed pair: **0.073 Sharpe**.
Gap from no stop to the average stop arm: **0.249**.
The choice of mechanism matters about three times less than the choice to run a stop at all.

### One variable explains it

Pool all 15 stop arms, both families together, and regress mean Sharpe on the share of trades the stop closed:

```
corr(firing rate, Sharpe) = -0.917        R^2 = 0.840
  ATR arms only:   -0.966
  fixed arms only: -0.811

residual once firing rate is accounted for:
  ATR   -0.007
  fixed +0.009
```

Firing rate alone explains 84% of the variance across arms, and the family label adds 0.016 Sharpe on top of it, which is noise.
The fit gives +0.273 for a stop that never fires, and the measured no-stop result is +0.327, better than the trend predicts.
The relation is monotone over the grid and the optimum is at zero.

`pct_20` at the top is the same point again: it closes 3% of trades, so it is nearly the no-stop arm with a stop's name.

### ATR's own claim also fails

The case for ATR is not a higher mean.
It is that a stop scaled to each name's range should make outcomes more uniform across names.
Dispersion of Sharpe across the 15 strategies, matched pairs: ATR tighter in 3, fixed tighter in 4.
Stops of either family do not compress dispersion below no stop: the wide arms sit at 0.43 to 0.45 and `no_stop` is at 0.44.

## Which strategy, then

The sweep was built to test the stop, but it also ranks the 15 strategies against each other on one setup.
Unstopped, `hold=20`, `top=10`, ranked by mean Sharpe over the four windows:

| strategy | 1y | 2y | 3y | 5y | mean | worst |
|---|---|---|---|---|---|---|
| **ha_momentum** | 1.397 | -0.166 | 1.078 | 0.932 | **0.810** | -0.17 |
| momentum_12_1_ema10 | 1.141 | 0.239 | 0.960 | 0.519 | 0.715 | **+0.24** |
| momentum_12_1 | 0.881 | -0.062 | 1.141 | 0.794 | 0.689 | -0.06 |
| momentum_12_1_riskadj | 0.900 | -0.194 | 0.756 | 0.771 | 0.558 | -0.19 |
| momentum_12_1_trend | 0.859 | -0.486 | 0.795 | 0.747 | 0.479 | -0.49 |
| breakout | 0.402 | -0.105 | 0.685 | 0.707 | 0.422 | -0.11 |
| supertrend_rsi | -0.308 | 0.052 | 1.104 | 0.827 | 0.419 | -0.31 |
| bb_breakout | 0.476 | -0.423 | 0.410 | 0.678 | 0.285 | -0.42 |
| rs_momentum_regime | -0.122 | 0.158 | 0.425 | 0.668 | 0.282 | -0.12 |
| mom_lowvol_combo | 0.126 | -0.626 | 0.712 | 0.816 | 0.257 | -0.63 |
| ema_trend | -0.176 | -0.511 | 0.595 | 0.437 | 0.086 | -0.51 |
| rs_breakout | -0.873 | -0.387 | 0.625 | 0.962 | 0.082 | -0.87 |
| supertrend_flip | -0.084 | -0.211 | 0.050 | -0.034 | -0.070 | -0.21 |
| donchian_breakout | -0.645 | -0.816 | -0.104 | 0.460 | -0.276 | -0.82 |
| supertrend | -1.130 | -0.823 | -0.100 | 0.417 | -0.409 | -1.13 |

Five-year detail for the leaders by Sharpe, against a benchmark that returned 6.2%/yr:

| strategy | Sharpe | CAGR | alpha | beta | max DD | profit factor |
|---|---|---|---|---|---|---|
| rs_breakout | **0.962** | 18.3% | +13.3% | 0.90 | -30.9% | 1.35 |
| ha_momentum | 0.932 | 19.3% | +14.3% | 0.94 | -30.2% | **1.39** |
| supertrend_rsi | 0.827 | 13.5% | +8.7% | 0.82 | -24.0% | 1.24 |
| mom_lowvol_combo | 0.816 | 11.0% | +6.3% | **0.76** | **-21.8%** | 1.38 |
| momentum_12_1 | 0.794 | 18.6% | +13.7% | 1.08 | -35.7% | 1.33 |

`ha_momentum` has the best mean over the four windows, and it is second at 5 years by 0.03 Sharpe.
`rs_breakout` leads at 5 years but posts -0.87 at 1 year, so it is not stable as the window shortens.
`momentum_12_1_ema10` is the only strategy that is positive in all four windows, but its 5-year Sharpe is low (0.519).
`breakout` is worth a note: at 5 years it has the highest CAGR (27.5%) and the lowest drawdown (-21.1%) of the group, with a Sharpe of only 0.707.

Three limits.
Best-of-15 on one dataset is selection bias, and the gaps at the top (`ha_momentum`, `momentum_12_1_ema10`, `momentum_12_1`) are too small to trust the order.
The gap to the bottom of the table is large enough to trust.
The 2-year column is negative almost everywhere because the index returned -4.0%/yr over that window; that is the market, not instability.
And every stop arm costs `ha_momentum` at least 0.355 mean Sharpe, which fits: its own exits already work, and the stop fights them.

Avoid `supertrend`, `supertrend_flip` and `donchian_breakout`.
All three have a negative mean Sharpe, and `supertrend_flip` loses money over five years (-3.0%/yr, profit factor 0.95) while the index made 6.2%.
`supertrend_rsi` is not in that group on this run: it is strong at 3 and 5 years, but -0.31 at 1 year.

## Two caveats on reading this

**The per-strategy "best ATR" numbers are selection bias.**
Picking the best of 8 ATR widths per strategy will beat a single arm by chance alone.
The table is here for completeness, not as a result:

| strategy | no_stop 5y | best ATR 5y | best arm | delta |
|---|---|---|---|---|
| supertrend_flip | -0.034 | 0.299 | atr_0.5 | +0.333 |
| ema_trend | 0.437 | 0.667 | atr_1.5 | +0.230 |
| momentum_12_1_ema10 | 0.519 | 0.672 | atr_2 | +0.153 |
| momentum_12_1_riskadj | 0.771 | 0.873 | atr_2.5 | +0.102 |
| momentum_12_1 | 0.794 | 0.875 | atr_4 | +0.080 |
| rs_breakout | 0.962 | 0.845 | atr_3.5 | -0.117 |
| breakout | 0.707 | 0.528 | atr_1 | -0.179 |
| ha_momentum | 0.932 | 0.674 | atr_3.5 | -0.258 |
| bb_breakout | 0.678 | 0.388 | atr_4 | -0.291 |

Even with the selection advantage, 5 of 15 strategies are still worse off.
The best widths are scattered from 0.5x to 4x, which is what chance looks like.

**Costs were zero.**
The sweep ran `slippage_bps=0`, `commission_bps=0`, `cost_model=flat`.
A stop that multiplies turnover by 1.1x to 3.8x would lose more under the India cost model.
So the negative result is conservative.

## What I did not test

- A **trailing** ATR stop (chandelier).
  Every arm here is a static level set at entry.
  A stop that ratchets up may not have the "sell the dip, buy the bounce" failure in the same way, and it is the obvious next experiment.
- ATR windows other than 14.
- Pairing the stop with `--sizing atr_risk`, which now reads the same distance.
  Every run used `equal_slot`, so position size was constant across arms and the comparison isolates the exit.
- Non-momentum strategies.

## Recommendation

Do not put a stop of either kind on these momentum strategies on Nifty 500.
"ATR or fixed percent" is the wrong question.
Matched on how often they fire, the two are within 0.07 Sharpe of each other and split the head-to-head cells 53/47, while the average stop gives up 0.25 Sharpe against no stop.
Firing rate is the variable, the relation is monotone, and the best value is zero.

`pct_20` ties no stop on mean Sharpe only because it almost never fires.
It is behind no stop at 2, 3 and 5 years, and its 5-year drawdown is 5 points deeper.
It is not a reason to add a stop.

The feature is still worth having.
It fixes a real disagreement where `atr_risk` sizing assumed a stop the engine never applied.
On this evidence it should stay off by default for momentum, which is what it is.

If a stop is required for reasons outside the backtest (a mandate, a risk limit), take the widest one that meets the constraint.
`pct_20` and `atr_4` cost the least, and they cost the least because they seldom fire.
