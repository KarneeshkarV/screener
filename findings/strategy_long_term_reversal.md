# Long-Term Reversal — De Bondt & Thaler (1985)

**Strategy key:** `long_term_reversal`

## Research source

**Paper:** De Bondt & Thaler, *"Does the Stock Market Overreact?"*, Journal of Finance 40(3), 1985.

- DOI: https://doi.org/10.1111/j.1540-6261.1985.tb05004.x
- Wiley page: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1985.tb05004.x

The founding contrarian paper (and the first big behavioural-finance hit): extreme 3- to 5-year losers subsequently beat extreme winners by ~25 percentage points over 36 months. De Bondt & Thaler attribute it to investor overreaction; the effect is one of the most replicated anomalies and motivates the entire value/contrarian literature (Lakonishok-Shleifer-Vishny 1994 built on it).

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `ret_756 < 0` — trailing three-year return negative (eligibility gate) |
| Ranking | `rank_score = −ret_756` — the biggest three-year losers fill the slots first |

with `ret_756[t] = close[t]/close[t−756] − 1`.

## Implementation notes

- Cross-sectional factor portfolio via the `rank_score` mechanism.
- `required_lookback = 757` — the engine fetches ~3 extra years of history to define the signal from the first day of the window.
- Suggested sizing: low turnover — 10 slots, longer holds.

## Expected behaviour

- Buys multi-year laggards (deep value, PSU banks, old-economy) — the opposite of every momentum strategy in the repo.
- Long holding periods, low trade count; gate `ret_756 < 0` filters heavily in bull years (few 3-year losers exist after a long rally — e.g., the 1y window on nifty500 produced zero trades).
- Pre-2021 Nifty 500 had deep 3-year losers (banks, capital goods) — expect the 5y backtest to be the informative one.
