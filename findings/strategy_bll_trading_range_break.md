# Trading-Range Break (TRB) — Brock, Lakonishok & LeBaron (1992)

**Strategy key:** `bll_trading_range_break`

## Research source

**Paper:** Brock, Lakonishok & LeBaron, *"Simple Technical Trading Rules and the Stochastic Properties of Stock Returns"*, Journal of Finance 47(5), 1992.

- DOI: https://doi.org/10.1111/j.1540-6261.1992.tb04681.x
- Wiley page: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04681.x

This is the landmark academic study of technical rules, run on the Dow Jones
Industrial Average from 1897 to 1986. BLL tested two families — variable-length
moving averages (VMA) and trading-range breaks (TRB) — and found the rules
generated returns a random-walk null could not explain, with the TRB rule
among the strongest: buy when the price penetrates the 150-day high
(resistance), sell when it penetrates the 150-day low (support). Their work
triggered the entire modern "technical analysis anomalies" literature.

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `close > high_150_prev` — close breaks the prior 150-day closing high |
| Exit | `close < low_150_prev` — close breaks the prior 150-day closing low |

## Implementation notes

- `prepare_bars` computes `high_150_prev = close.rolling(150).max().shift(1)`
  and the mirror `low_150_prev`, so today's close is never part of its own
  channel (strictly prior resistance/support).
- `required_lookback = 150` bars.
- The repo's `donchian_breakout` (20-day high / 10-day low, the Turtle rule) is
  a different channel and a different research lineage; BLL's 150-day channel
  is this strategy.
- Suggested holding semantics: trend following — run with a long `--hold`.

## Expected behaviour

- Trend-following profile like the golden cross but more reactive (150-day vs
  200-day reference).
- Long holding periods, "let winners run" — chops in sideways markets.
- BLL found the rule profitable on the Dow across 90 years of data; the
  question for this repo is whether the edge survives in liquid Indian/US
  universes with modern costs.
