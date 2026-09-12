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
