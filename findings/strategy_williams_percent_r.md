# Williams %R Mean Reversion — Larry Williams

**Strategy key:** `williams_percent_r`

## Research source

**Book:** Williams, *"How I Made One Million Dollars Last Year Trading Commodities"*, 1979, Windsor Books.

- https://www.amazon.com/Million-Dollars-Last-Trading-Commodities/dp/0934233128
- Modern reference: https://school.stockcharts.com/doku.php?id=technical_indicators:williams_percent_r

%R is Larry Williams' inversion of the fast stochastic: it measures how close today's close is to the 14-period high, expressed negatively. Williams traded it as a counter-trend signal — buy the -80 oversold extreme, sell the -20 overbought extreme.

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `%R < −80` — deep 14-day oversold |
| Exit | `%R > −20` — 14-day overbought |

with `%R = −100·(highest(high,14) − close) / (highest(high,14) − lowest(low,14))`.

## Implementation notes

- Pure Pine expression.
- `required_lookback = 14`.
- Suggested sizing: mean reversion — 20 slots, short holds.

## Expected behaviour

- Same family as the stochastic but opposite polarity and stricter extremes: fewer, deeper signals.
- High win rate in bull markets, falling-knife losses in sustained downtrends.
- Historically the %R(14) < −80 → > −20 rule was published as one of the highest win-rate commodity rules of the 1970s; the question is how it survives modern costs and equity markets.
