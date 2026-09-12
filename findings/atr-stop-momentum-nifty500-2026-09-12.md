# ATR stops make momentum worse on Nifty 500, and deepen the drawdown they are meant to prevent

Date: 2026-09-12.
Scope: 660 rolling backtests, 0 failures.
Universe: `nifty500_pit`, point-in-time membership (850 candidate symbols, 6520 membership windows).
Reproduce with `uv run python scripts/run_momentum_atr_stop_sweep.py`.

## What was tested

Every momentum-family strategy in the registry (15 of them) against an ATR stop grid, over nested trailing windows.

| Axis | Values |
|---|---|
| Strategies | `bb_breakout`, `breakout`, `donchian_breakout`, `ema_trend`, `ha_momentum`, `mom_lowvol_combo`, `momentum_12_1`, `momentum_12_1_ema10`, `momentum_12_1_riskadj`, `momentum_12_1_trend`, `rs_breakout`, `rs_momentum_regime`, `supertrend`, `supertrend_flip`, `supertrend_rsi` |
| ATR arms | `--stop-atr` at 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 (window 14) |
| Controls | `no_stop`, and flat `--stop-loss` at 8% and 12% |
| Windows | 1, 2, 3, 5 years, all ending 2026-09-12 |

Fixed: `hold=20`, `top=10`, `equal_slot` sizing, benchmark `^NSEI`, daily bars.

Windows are nested, so the 1-year sits inside the 5-year. Read them as "did this hold up as the window shortened", not as four independent samples.

## The answer

**No stop wins.** It is the single most frequent best arm, in 22 of 60 strategy x window cells. No ATR width beats it in even a third of cells.

| Arm | Cells where it beats `no_stop` (of 60) |
|---|---|
| `atr_2.5` | 20 |
| `atr_4` | 19 |
| `atr_3`, `atr_3.5` | 17 |
| `atr_2` | 15 |
| `pct_08` | 15 |
| `pct_12` | 14 |
| `atr_1` | 9 |
| `atr_0.5` | 8 |
| `atr_1.5` | 6 |

Median Sharpe, all 15 strategies:

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| no_stop | 0.367 | -0.229 | **0.562** | **0.578** |
| atr_2.5 | 0.175 | -0.608 | 0.097 | 0.226 |
| atr_3.5 | 0.203 | -0.326 | 0.299 | 0.461 |
| pct_08 | 0.052 | -0.229 | 0.300 | 0.213 |

Every stop arm loses to no stop at 3 and 5 years, on both Sharpe and CAGR. The tight end is catastrophic: at 0.5x ATR the median hit rate collapses from 50% to 17%, because the stop is inside the daily noise of a Nifty 500 name and fires on almost everything it touches (83% of trades exit by stop).

### Sharpe against the no-stop control

Delta in median Sharpe, arm minus `no_stop`. **All 40 cells are negative** - there is no width and no window where the stop pays for itself in risk-adjusted terms.

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| atr_0.5 | -1.142 | -0.588 | -0.599 | -0.480 |
| atr_1 | -0.178 | -0.345 | -0.568 | -0.429 |
| atr_1.5 | -0.514 | -0.253 | -0.457 | -0.463 |
| atr_2 | -0.407 | -0.070 | -0.266 | -0.369 |
| atr_2.5 | -0.192 | -0.379 | -0.466 | -0.352 |
| atr_3 | -0.287 | -0.023 | -0.273 | -0.194 |
| atr_3.5 | -0.164 | -0.097 | -0.264 | **-0.116** |
| atr_4 | -0.269 | -0.057 | -0.179 | -0.295 |
| pct_08 | -0.315 | -0.000 | -0.263 | -0.365 |
| pct_12 | -0.285 | -0.159 | -0.284 | -0.359 |

Averaged over the four windows, `no_stop` scores +0.320 against +0.159 for the best stop arm (`atr_3.5`) - twice the risk-adjusted return. Cells clearing Sharpe > 1.0: `no_stop` 8 of 60, best stop arm 3 of 60.

### Against the benchmark

^NSEI over these windows: -6.8% (1y), -4.0%/yr (2y), +5.3%/yr (3y), +6.2%/yr (5y). Median annualised alpha:

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| **no_stop** | **+0.121** | -0.017 | **+0.033** | **+0.061** |
| atr_2 | +0.038 | -0.026 | +0.002 | -0.017 |
| atr_3 | +0.069 | -0.018 | +0.003 | +0.017 |
| atr_3.5 | +0.097 | -0.035 | +0.004 | +0.037 |
| atr_4 | +0.070 | -0.023 | +0.024 | -0.003 |
| pct_08 | +0.074 | -0.018 | +0.007 | -0.015 |

At five years only `atr_3` and `atr_3.5` hold positive alpha at all, and both sit well below no stop. Share of strategies beating the index on CAGR at 5y: 60% with no stop, 53% for the best stop arm.

## The finding that matters

**The stop deepens the drawdown it exists to prevent.** Median max drawdown, relative to no stop (negative is worse):

| arm | 1y | 2y | 3y | 5y |
| --- | --- | --- | --- | --- |
| atr_1.5 | -0.027 | -0.070 | **-0.096** | -0.083 |
| atr_1 | +0.001 | -0.025 | -0.028 | **-0.060** |
| atr_2 | -0.021 | -0.039 | -0.035 | -0.050 |
| atr_2.5 | +0.007 | -0.063 | -0.040 | -0.035 |
| atr_4 | +0.004 | -0.031 | -0.030 | -0.040 |
| pct_08 | +0.001 | -0.030 | -0.025 | -0.029 |

Every arm, every width, every window of 2 years or more, is worse. The 1-year column has a few tiny positives and they are not consistent in sign across widths, so they are noise.

The mechanism looks like the one already seen with entry-only risk gates: the stop sells into a pullback, and the strategy re-enters after the bounce has already started. You realize the loss and buy back higher. Doing that repeatedly through a drawdown compounds it rather than truncating it. The turnover confirms it - median trade count over 5 years goes from 590 with no stop to 1740 at 0.5x ATR, and is still above 630 even at 4x.

What rules out the boring explanation - "the stop just reduces exposure" - is that the de-risking clearly works, and the drawdown gets worse anyway:

| 5-year median | no_stop | atr_0.5 | atr_2 | atr_4 |
| --- | --- | --- | --- | --- |
| beta | 0.94 | 0.73 | 0.85 | 0.94 |
| annual vol | 0.207 | 0.186 | 0.200 | 0.215 |
| max drawdown | **-0.343** | -0.399 | -0.393 | -0.383 |

Beta and volatility both fall, so the stop is genuinely taking risk off. Sharpe still drops, because return falls faster than risk. An exit that merely cut exposure would improve the worst case and leave Sharpe roughly flat; this worsens both. That gap between lower average risk and a deeper worst case is the churn.

## Which strategy, then

The sweep was built to test the stop, but it also ranks the 15 strategies against each other on one consistent setup. Unstopped, `hold=20`, `top=10`, ranked by mean Sharpe over the four windows:

| strategy | 1y | 2y | 3y | 5y | mean | worst |
|---|---|---|---|---|---|---|
| **ha_momentum** | 1.282 | -0.032 | 1.126 | **1.099** | **0.869** | **-0.03** |
| momentum_12_1 | 1.022 | 0.049 | 1.109 | 0.764 | 0.736 | +0.05 |
| momentum_12_1_ema10 | 0.929 | 0.229 | 1.078 | 0.482 | 0.679 | +0.23 |
| momentum_12_1_trend | 1.155 | -0.361 | 0.879 | 0.796 | 0.617 | -0.36 |
| breakout | 0.388 | 0.102 | 0.752 | 0.683 | 0.481 | +0.10 |
| momentum_12_1_riskadj | 0.838 | -0.295 | 0.613 | 0.748 | 0.476 | -0.30 |
| bb_breakout | 0.931 | -0.403 | 0.384 | 0.578 | 0.373 | -0.40 |
| ema_trend | 0.149 | 0.132 | 0.375 | 0.345 | 0.250 | +0.13 |
| rs_momentum_regime | 0.367 | -0.113 | 0.392 | 0.297 | 0.235 | -0.11 |
| mom_lowvol_combo | -0.108 | -0.655 | 0.562 | 0.912 | 0.178 | -0.66 |
| rs_breakout | -1.308 | -0.069 | 0.755 | 1.023 | 0.100 | -1.31 |
| donchian_breakout | -1.005 | -0.229 | 0.365 | 0.074 | -0.199 | -1.01 |
| supertrend_flip | -0.854 | -0.847 | 0.324 | 0.109 | -0.317 | -0.85 |
| supertrend_rsi | -1.159 | -0.428 | 0.084 | 0.155 | -0.337 | -1.16 |
| supertrend | -1.366 | -1.056 | -0.324 | -0.199 | -0.736 | -1.37 |

Five-year detail for the leaders, against a benchmark that returned 6.2%/yr:

| strategy | Sharpe | CAGR | alpha | beta | max DD | profit factor |
|---|---|---|---|---|---|---|
| **ha_momentum** | **1.099** | **21.7%** | **+17.0%** | 0.84 | **-22.2%** | **1.57** |
| rs_breakout | 1.023 | 20.0% | +15.0% | 0.90 | -31.5% | 1.40 |
| mom_lowvol_combo | 0.912 | 13.7% | +8.6% | 0.83 | -25.9% | 1.40 |
| momentum_12_1_trend | 0.796 | 18.9% | +13.9% | 1.10 | -37.3% | 1.32 |
| momentum_12_1 | 0.764 | 18.3% | +13.3% | 1.14 | -41.7% | 1.33 |

`ha_momentum` leads on every axis at once - Sharpe, CAGR, alpha, drawdown, profit factor - with a beta under 1, and its worst window is barely negative where `rs_breakout` posts -1.31 and `mom_lowvol_combo` -0.66. For a second, differently-constructed pick, `momentum_12_1_trend` beats plain `momentum_12_1`: same return, four points less drawdown.

Three qualifications. Best-of-15 on one dataset is selection bias, and the gap between `ha_momentum` and `momentum_12_1` is not wide enough to be confident of the ordering - the gap to the supertrend family is. The 2-year column is negative almost everywhere because the index itself returned -4.0%/yr over that window; that is the market, not instability. And `ha_momentum` is the strategy the ATR stop damaged most (-0.430 Sharpe), which fits: its own exits already work, and the stop fights them.

The supertrend family should be avoided outright. All three are negative to flat on every window with a profit factor at or below 1.02, and plain `supertrend` loses money over five years (-5.2%/yr, profit factor 0.90) while the index made 6.2%.

## Two caveats on reading this

**The per-strategy "best ATR" numbers are selection bias.** Picking the best of 8 ATR widths per strategy will beat a single fixed arm by chance alone. The table below is presented for completeness, not as a result:

| strategy | no_stop 5y | best ATR 5y | best arm | delta |
|---|---|---|---|---|
| donchian_breakout | 0.074 | 0.488 | atr_2.5 | +0.414 |
| momentum_12_1_ema10 | 0.482 | 0.789 | atr_2.5 | +0.307 |
| momentum_12_1_riskadj | 0.748 | 0.886 | atr_2.5 | +0.138 |
| momentum_12_1 | 0.764 | 0.787 | atr_2.5 | +0.023 |
| rs_breakout | 1.023 | 0.955 | atr_4 | -0.067 |
| bb_breakout | 0.578 | 0.283 | atr_4 | -0.295 |
| ha_momentum | 1.099 | 0.669 | atr_4 | -0.430 |

Even with the selection advantage, 6 of 15 strategies are still worse off.

**Costs were zero.** The sweep ran `slippage_bps=0`, `commission_bps=0`, `cost_model=flat`. A stop that multiplies turnover by 1.4x to 3x would be penalized considerably harder under the India cost model. The negative result is therefore conservative.

## What I did not test

- A **trailing** ATR stop (chandelier). Every arm here is a static level set at entry. Whether a stop that ratchets up behaves differently is a genuinely open question and the obvious next experiment - it does not have the "sell the dip, buy the bounce" failure mode in the same way.
- ATR windows other than 14.
- Pairing the stop with `--sizing atr_risk`, which now reads the same distance. Every run used `equal_slot`, so position size was constant across arms and the comparison isolates the exit.
- Non-momentum strategies.

## Recommendation

Do not put an ATR stop on these momentum strategies on Nifty 500. The feature is still worth having - it fixes a real disagreement where `atr_risk` sizing assumed a stop the engine never applied - but on this evidence it should stay off by default for momentum, which is what it is.

If someone wants a stop on this book anyway, `atr_2.5` and `atr_3.5` are the least bad widths, and both still lose to no stop at 3 and 5 years.
