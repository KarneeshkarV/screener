# Short-Term Reversal — Jegadeesh (1990)

**Strategy key:** `short_term_reversal`

## Research source

**Paper:** Jegadeesh, *"Evidence of Predictable Behavior of Security Returns"*, Journal of Finance 45(3), 1990.

- DOI: https://doi.org/10.1111/j.1540-6261.1990.tb05110.x
- Wiley page: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1990.tb05110.x

Jegadeesh's companion to the 12-1 momentum paper: returns over the *most recent month* are strongly negatively autocorrelated — last month's losers bounce this month. This is precisely why Jegadeesh-Titman (1993) skip the last month in their momentum signal; buying the skip window's losers is the mirror trade.

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `ret_21 < 0` — trailing one-month return negative (eligibility gate) |
| Ranking | `rank_score = −ret_21` — the biggest one-month losers fill the slots first |

with `ret_21[t] = close[t]/close[t−21] − 1`.

## Implementation notes

- Cross-sectional factor portfolio via the `rank_score` mechanism (like `low_volatility`/`momentum_12_1`).
- `required_lookback = 22`.
- Suggested sizing: high turnover — 20 slots, ~21-day holds to match the monthly rebalance cadence.

## Expected behaviour

- Buys the most oversold names of the past month — a real factor, not a timing rule.
- Fails in strong momentum regimes (2023-24 US mega-caps); expected to shine in choppy/correction windows.
- Sharp contrast with `momentum_12_1`: same paper family, opposite sign, different horizon.
