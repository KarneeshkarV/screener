# 52-Week High Momentum — George & Hwang (2004)

**Strategy key:** `fifty_two_week_high`

## Research source

**Paper:** George & Hwang, *"The 52-Week High and Momentum Investing"*, Journal of Finance 59(5), 2004.

- DOI: https://doi.org/10.1111/j.1540-6261.2004.00695.x
- Wiley page: https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x

George & Hwang show that a stock's price **nearness to its 52-week high** is a
stronger predictor of future returns than raw Jegadeesh-Titman 12-month
momentum, and the effect survives after controlling for the momentum premium.
Their explanation is anchoring: investors anchor to the 52-week high, prices
cluster below it, and stocks that push through the high carry positive drift.
The 52-week-high variable is one of the most replicated anomalies in the
momentum literature (see also the Q-theory literature it spawned).

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `close > high_252_prev` — today's close breaks the previous 252-day closing high (fresh 52-week high) |
| Exit | `crossunder(close, sma(close, 50))` — momentum breaks the 50-day trend |

## Implementation notes

- `prepare_bars` computes `high_252_prev = close.rolling(252).max().shift(1)` —
  the rolling window is **shifted by one bar** so today's close is never part
  of its own reference peak. The naive `close > highest(close, 252)` includes
  today in the maximum and can never trigger.
- The paper rebalances monthly on the price/52-week-high ratio; here the
  signal is a daily fresh-high breakout with an SMA50 trend exit, which is the
  practical daily translation.
- `required_lookback = 252` bars.

## Expected behaviour

- Breakout momentum: enters as stocks make new yearly highs, exits on trend
  breaks. Captures sustained rallies (RELIANCE 2020-21 style, US mega-caps
  2023-24) and cuts losers when the 50-day line is lost.
- Can whipsaw badly in range-bound years when the market keeps printing
  marginal new highs and then reversing.
