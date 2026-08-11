# Golden Cross — SMA50 / SMA200 Trend Following

**Strategy key:** `golden_cross_50_200`

## Research source

**Paper:** Han, Yang & Zhou, *"A New Anomaly: The Cross-Sectional Profitability of Technical Analysis"*, Journal of Financial and Quantitative Analysis 48(5), 2013.

- DOI: https://doi.org/10.1017/S0022109013000586
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1656460

The paper is the modern academic treatment of moving-average trading rules. It
shows that the 50/200-day moving-average configuration — the "golden cross" —
is not folklore: stocks trading above their long-run MA earn statistically
significant abnormal returns that *reverse* at horizons beyond the MA window,
which is exactly what a slow trend-following rule harvests. The result holds
cross-sectionally (MA rank predicts the cross-section of returns) and survives
risk adjustments.

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `crossover(sma(close, 50), sma(close, 200))` — the fast MA crosses above the slow MA |
| Exit | `crossunder(sma(close, 50), sma(close, 200))` — the fast MA crosses back below |

## Implementation notes

- Pure Pine expression; no `prepare_bars` hook needed.
- `required_lookback = 200` bars so the SMA200 is well-defined at the first
  signal bar.
- Suggested holding semantics: this is a low-frequency rule. Run with a long
  `--hold` so the MA exit (not the time exit) closes positions.

## Expected behaviour

- Long holding periods (months), few trades, large winners.
- Deep drawdowns in bear regimes — no shorting, no regime gate.
- All alpha concentrated in bull phases; flat-to-negative in sideways/bear
  years. Best evaluated on 3-5 year windows.
