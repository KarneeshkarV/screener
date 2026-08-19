# Momentum study: meta-analysis

Synthesis of 16 independent per-strategy analyses over a 2,720-cell sweep: 16 strategies x 2 point-in-time universes x 5 windows x holding period x regime overlay x construction lever.
Long-only, no leverage, costs at 5bps slippage plus 5bps commission.

Where an agent's claim conflicted with the run data, the run data wins and the disagreement is called out.
Three such corrections appear below.

## The one-line answer

Momentum works in both markets, but for opposite reasons, and almost every "improvement" in the sweep is noise.
**Holding period is the only lever that robustly helps.**
India pays you for taking amplified risk in advances; the US pays a smaller, steadier edge that a 15.3% SPY makes hard to beat.

## 1. Every strategy is pro-cyclical. One is not.

All 16 analyses independently concluded their strategy is pro-cyclical rather than defensive.
The capture ratios confirm it, measured as the median per-segment ratio over 4 advance and 8 decline episodes per market, with declines measured peak-to-trough:

| Market | Upside capture | Downside capture |
|---|---|---|
| India | 1.84x - 2.94x | 1.01x - 1.56x |
| US | 0.53x - 1.17x | 0.84x - 1.27x |

India momentum is a leveraged bet on advances: it captures roughly 2.5x the index's upside and slightly more than all of its downside.
That is the whole edge.
None of these are risk reducers, and none should be sized as one.

**The exception the agents missed:** `momentum_12_1_defensive` in India is the only strategy in the study with genuine asymmetry, at the highest upside capture (2.94x) and the lowest downside (0.76x).
It is the one strategy whose name matches its behaviour.

The US table is the structural explanation for why the US is hard.
Upside capture at or below 1.0 against downside capture at or above 1.0 is the wrong asymmetry to be paying costs for, and the benchmark compounded 15.3% a year.

## 2. India vs US: 14 of 16 say India, and the two exceptions matter

Fourteen analyses recommend India and reject the US.
The reasoning is consistent: India offers several strategies clearing an 11.1% index by 10-20pp, while the US asks you to beat 15.3% with at-best-market participation.

The two exceptions are the interesting ones.
`momentum_6_6` recommends the US as a core holding, at 21.3% against SPY's 15.3% over ten years, with alpha spread across the decade rather than concentrated, and no reconstructed-universe caveat.
`dual_momentum_market` and `industry_trend_breakout` reject both markets outright at default settings.

## 3. Holding period is the only robust lever

This is the study's most reusable finding, and it is unanimous across the sweep and the analyses.
The papers' 21-day default is wrong for long horizons:

| Market | 5y median Sharpe | 10y median Sharpe |
|---|---|---|
| India h21 | 0.92 | 1.00 |
| India h126 | **1.34** | **1.19** |
| US h21 | 0.72 | 0.55 |
| US h126 | **0.81** | **0.65** |

Only the 1-year window prefers h21, in both markets.
Thirteen of 16 verdicts specify h63 or h126, and several call it the single change that matters.
For `tsmom_blend` in India, h63 improves both Sharpe and drawdown in all five windows, the only variant in the entire 2,720-cell sweep to do that.

## 4. Everything else fails

Both-metric win rates, meaning the share of cells improving Sharpe *and* max drawdown against the same baseline:

| Lever | 1y | 2y | 3y | 5y | 10y | Verdict |
|---|---|---|---|---|---|---|
| `invvol` | 56% | 47% | 59% | 53% | 38% | risk lever only |
| `sectorneutral` | 34% | 66% | 59% | 34% | 34% | inconsistent |
| `top50` | 50% | 41% | 25% | 41% | 34% | inconsistent |
| `trail25` | 41% | 12% | 12% | 31% | 12% | fails |
| `top10` | 19% | 9% | 0% | 6% | 9% | actively harmful |
| `breadth` | 30% | 25% | 27% | 23% | 46% | fails |
| `bull` | 11% | 9% | 10% | 28% | 18% | fails |
| `nonbear` | 3% | 12% | 11% | 33% | 22% | fails |

Three readings worth stating plainly.

`invvol` is a pure risk lever, not a free lunch.
It cuts max drawdown in 100% of cells in every window, with a median of 5-12pp, but improves Sharpe in only 38-59%.
Use it to buy drawdown, and expect to pay for it.

Concentration is punished.
`top10` deepens drawdown in every window and clears both metrics in 0-19% of cells.
More names, not fewer.

Regime overlays fail as a class.
Entry gates cannot fix a drawdown profile when the strategy holds through the decline anyway, because a gate that does not also govern the exit rides the crash down and then sits out the rebound.

Strategy-specific exceptions exist: `tsmom_12` in India with the breadth overlay passes 4 of 5 windows, and `tsmom_blend` in the US with the bull overlay passes all 5.
The agents flagged these themselves as likely overfit across 2,720 backtests.
Treat them as hypotheses, not findings.

## 5. Three corrections to the analyses

### The episode-concentration claim is framed wrong

Most analyses argue India's decade "collapses toward flat" without the 2020-21 and 2023-24 advances.
That comparison strips the episodes from the strategy while leaving them in the benchmark.
Removing them from both, which is 22% of trading days:

| Strategy | Full | Ex-2 adv | Index ex-2 | Excess |
|---|---|---|---|---|
| momentum_12_1_riskadj | 32.2% | 15.6% | 4.0% | +11.6pp |
| momentum_6_6 | 27.8% | 13.6% | 4.0% | +9.6pp |
| momentum_12_1 | 31.4% | 13.0% | 4.0% | +9.1pp |
| momentum_12_1_trend | 30.3% | 12.5% | 4.0% | +8.5pp |
| dual_momentum_daa | 25.1% | 8.4% | 4.0% | +4.4pp |
| tsmom_12 | 22.1% | 5.0% | 4.0% | +1.1pp |

The return *level* is episode-driven.
The *excess* is not, for the core momentum strategies.
Only `tsmom_12` genuinely deflates to nothing.

### The recent US drawdown was misread

Five analyses flagged a 28-35% drawdown in the final four months at 40-63% annualized volatility, several describing it as losing a third of value in a rising market.
It is an intra-period drawdown on books that ended *up*: `momentum_6_6` went from 134,013 to 175,857, a gain of 31%, while drawing down 30.7% along the way.
Underlying trades are clean, with MU +107.5%, INTC +112.2% and STX +91.0% against losers of -8% to -24%.
A 20-name book concentrated into semiconductors, melted up, and unwound.
The volatility is real and was measured correctly for the first time here.

### Identical drawdowns across windows are not bugs

Where a strategy shows the same max drawdown in all five windows, one recent event sits inside every window.
Verified for `dual_momentum_paa`, which peaked on 2026-06-22 and troughed on 2026-07-29.

## 6. What to trust

The 5-year window is the honest one.
India's pre-2021 membership is reconstructed, so 10-year India results are the weakest input in the study.
All 16 analyses flagged this independently and correctly, and every India 10y figure here should be read as an upper bound.

The 1-year and 2-year US numbers are the last 12-18 months of a momentum regime, not an edge.
CAGR decaying monotonically with window length is the tell.

Turnover matters.
At h21 the cost drag runs 2-4% a year, and several strategies' entire edge sits inside that band, which is a second argument for longer holds.

## 7. If you had to allocate

For India, the core is `momentum_12_1_riskadj` at h126 with no overlay and no lever.
It returns 32.8% CAGR at 1.46 Sharpe over 10y and 27.1% at 1.20 on the clean 5y window, against an 11.1% index.
Add `invvol` only to buy drawdown, knowingly.

For the US, the core is `momentum_6_6` at h126 with `sectorneutral`, at 21.3% against 15.3%.
It is the only US case where the excess is spread across the decade.

If you want asymmetry rather than amplification, `momentum_12_1_defensive` in India is the sole strategy that delivers it.

Size all of these as an aggressive sleeve.
Expect roughly 2x the index's drawdown in Indian corrections, and expect two flat years to be normal, since India's 2-year window is near zero for almost every strategy in the study.

## Method notes

Results come from a corrected run set.
Two engine bugs were fixed mid-study.
Position sizing did not compound, which de-levered every long run, and at 10y the corrected figures are CAGR x1.46, volatility x1.53 and drawdown x1.31.
Same-day round trips were double-counted in the equity curve, affecting 334 of 2,720 cells.
Sharpe rank correlation before and after is 0.93-0.99, so relative ordering was never wrong; the risk levels were.

The first round of analyses was discarded and re-run.
It had been given a context file written from pre-fix numbers, asserting that no US strategy beat SPY.
That claim propagated into 11 of 16 analyses and flipped at least one verdict outright, namely `momentum_6_6`, which now recommends the US as a core holding rather than rejecting it.

See [momentum-study-findings.md](momentum-study-findings.md) for the underlying quantitative work, which is reproducible from the run JSONs and independent of the agents.
