# Momentum study: verified findings

Computed directly from the corrected 2,720-cell run set, independent of the per-strategy analyses.
Where an analysis claim conflicts with a number here, this file wins, because these are reproducible from the run JSONs.

See [momentum-study-meta-analysis.md](momentum-study-meta-analysis.md) for the synthesis built on top of this.

## 1. Episode concentration: the framing matters

Most of the 16 analyses claim India's decade "collapses toward flat" once the 2020-21 advance (+48.3% benchmark over 343 days) and the 2023-24 advance (+39.2% over 468 days) are removed.
That comparison strips the episodes from the strategy but leaves them in the benchmark.
Removing them from both, which is 22% of trading days:

| Strategy | Full | Ex-2 adv | Index ex-2 | Excess |
|---|---|---|---|---|
| momentum_12_1_riskadj | 32.2% | 15.6% | 4.0% | +11.6pp |
| momentum_6_6 | 27.8% | 13.6% | 4.0% | +9.6pp |
| momentum_12_1 | 31.4% | 13.0% | 4.0% | +9.1pp |
| momentum_12_1_trend | 30.3% | 12.5% | 4.0% | +8.5pp |
| dual_momentum_daa | 25.1% | 8.4% | 4.0% | +4.4pp |
| tsmom_12 | 22.1% | 5.0% | 4.0% | +1.1pp |

The absolute return roughly halves, but the excess over the index, which is what an allocator is actually buying, survives for the core momentum strategies.
Only `tsmom_12` genuinely deflates.
"The edge is two lucky years" is not supported; "the *level* of return is two lucky years" is.

## 2. Capture ratios: pro-cyclical is correct, with one exception

Median per-segment ratio over 4 advance and 8 decline episodes per market.
Declines are measured peak-to-trough rather than peak-to-recovery: segments are back-dated to the prior peak and run through recovery, so their net return is near zero and a start-to-end ratio would be meaningless.

| Market | Strategy | Upside | Downside | Character |
|---|---|---|---|---|
| India | momentum_12_1_defensive | 2.94 | 0.76 | **defensive** |
| India | momentum_12_1_riskadj | 2.61 | 1.01 | pro-cyclical |
| India | momentum_12_1 | 2.57 | 1.11 | pro-cyclical |
| India | momentum_6_6 | 2.33 | 1.27 | pro-cyclical |
| India | dual_momentum_gem | 2.31 | 1.47 | pro-cyclical |
| India | tsmom_12 | 2.09 | 1.56 | pro-cyclical |
| India | faber_sma10 | 1.84 | 1.54 | pro-cyclical |
| US | momentum_12_1 | 1.17 | 1.27 | pro-cyclical |
| US | dual_momentum_gem | 1.14 | 1.26 | pro-cyclical |
| US | momentum_12_1_riskadj | 1.03 | 1.07 | pro-cyclical |
| US | momentum_6_6 | 1.01 | 1.02 | pro-cyclical |
| US | tsmom_12 | 0.93 | 0.97 | neutral |
| US | faber_sma10 | 0.80 | 1.02 | pro-cyclical |
| US | momentum_12_1_volmanaged | 0.53 | 0.84 | defensive, too costly |

The unanimous "pro-cyclical, not defensive" verdict holds.
What the analyses mostly missed is that `momentum_12_1_defensive` in India is the only strategy in the study with a genuinely asymmetric profile, at the highest upside capture (2.94x) and the lowest downside (0.76x).

The US table explains structurally why beating SPY is hard there.
Upside capture at or below 1.0 against downside capture at or above 1.0 is the wrong asymmetry, and SPY compounded 15.3% over the decade.

## 3. The recent US drawdown is real, and was misread

Five analyses flagged a 28-35% drawdown in the final four months at 40-63% annualized volatility, several describing it as losing a third of value in a rising market.
It is an intra-period drawdown on books that ended up:

| Run | Apr 2026 | Aug 2026 | Intra dd | Benchmark |
|---|---|---|---|---|
| us__momentum_6_6__1y | 134,013 | 175,857 | -30.7% | +18.3% |
| us__dual_momentum_daa__1y | 121,006 | 132,313 | -35.1% | +18.3% |
| us__momentum_12_1__1y | 112,950 | 139,932 | -31.3% | +18.3% |

Underlying trades are clean, with MU +107.5%, INTC +112.2% and STX +91.0% against losers of -8% to -24%.
A 20-name book concentrated into semiconductors, melted up, and unwound.
The volatility is genuine, and compounding is what makes it visible, since frozen slot sizing flattened this path into a much tamer curve.

## 4. Levers, on corrected data

`invvol` cuts max drawdown in 100% of cells in every window, with a median of 5-12pp, but improves Sharpe in only 38-59%.
It is a pure risk lever, not a free lunch.

`top10` deepens drawdown in every window and clears both metrics in 0-19% of cells.
Concentration is not rewarded.

`top50` cuts drawdown in 78-94% of cells at a flat-to-negative Sharpe cost.

`sectorneutral` is the only lever with a strong both-metric window, at 66% for 2y and 59% for 3y, but it does not hold at 1y, 5y or 10y, where it sits at 34%.

`trail25` fails everywhere, at 12-41%.

Regime overlays all fail as a class: `bull` at 9-28%, `nonbear` at 3-33% and `breadth` at 23-46%.

Holding period is the most robust lever in the study.
India's median Sharpe at 5y is 1.34 at h126 against 0.92 at h21, and at 10y it is 1.19 against 1.00.
The US prefers h126 at 5y and 10y as well.
Only the 1-year window favours h21, in both markets.

## 5. Trust boundaries

India's pre-2021 membership is reconstructed, so the 10y India window is the weakest input in the study.
The 5y window uses archived membership and is the honest number.
Every analysis independently flagged this, correctly.

Where a strategy's max drawdown is identical across all five windows, it is a single recent event inside every window rather than a bug.
This was verified for `dual_momentum_paa`, which peaked on 2026-06-22 and troughed on 2026-07-29, an event present in the 1y through 10y windows alike.
