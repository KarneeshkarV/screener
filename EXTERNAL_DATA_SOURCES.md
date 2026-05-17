# External Data Sources for ML Signal Improvement

## Current Honest Ceiling: AUC = 0.5733
Every experiment with just OHLCV data has failed to beat this. You need external signal.

---

## 1. Options Flow (Unusual Options Activity)

**Why it works:** Call buying before breakouts is strong directional signal.

### Free Tier
| Source | Endpoint | Limits |
|--------|----------|--------|
| **Unusual Whales** API | `https://api.unusualwhales.com/api/...` | 100 req/day free |
| **Cheddar Flow** | Web scraping | Unofficial |
| **Yahoo Finance (yfinance)** | `get_option_chain()` | Historical OI only |

### Paid Tier
| Source | Price | Quality |
|--------|-------|---------|
| Unusual Whales API | $49/mo | Best-in-class |
| Cheddar Flow API | $39/mo | Good for retail |
| Polygon.io Options | $200/mo | Institutional-grade |

### Feature to Compute
```python
# Call/Put volume ratio vs 20d avg
options_signal = (call_volume_1d / put_volume_1d) / (call_volume_20d_avg / put_volume_20d_avg)
# > 1.5 = unusual call buying
```

### Integration
```python
from screener.data_providers.unusual_whales import get_options_flow
```

---

## 2. Social Sentiment (Twitter/X, StockTwits, Reddit)

**Why it works:** Crowd positioning ahead of breakouts. Retail FOMO is real.

### Free Tier
| Source | Endpoint | Limits |
|--------|----------|--------|
| **StockTwits API** | `https://api.stocktwits.com/api/2/streams/symbol/AAPL.json` | 200 req/hr |
| **Reddit (PRAW)** | `r/wallstreetbets` posts | Free but rate-limited |
| **Twitter/X API v2** | Academic/basic tier | Expensive for real-time |

### Paid Tier
| Source | Price | Quality |
|--------|-------|---------|
| **Sentimentrader** | $500/mo | Professional sentiment indices |
| **SwaggyStocks** | Free web + $API | WSB sentiment tracker |
| **HypeIndex** | $99/mo | Multi-platform aggregation |

### Feature to Compute
```python
# Sentiment score: bullish mentions / total mentions
sentiment_score = bullish_count / (bullish_count + bearish_count)
# Sentiment momentum: 1d change in sentiment score
sentiment_delta = sentiment_today - sentiment_5d_avg
```

---

## 3. Sector Breadth (% Stocks Above 50-day MA)

**Why it works:** Individual breakouts fail in weak sectors. Sector confirmation = higher WR.

### Free Tier
| Source | Endpoint | Limits |
|--------|----------|--------|
| **TradingView Screener** | Screener API | Already used! |
| **FRED (St. Louis Fed)** | `SPXEW` equal-weight data | Free |
| **Yahoo Finance** | Download sector ETFs | Free |

### How to Compute (Using Existing Screener!)
```python
# For each signal, compute:
sector_breadth = pct_of_sector_stocks_above_50ma
# Using TradingView screener on the signal's sector
# > 60% = strong sector tailwind
```

This is **free** — you already have the TradingView screener API integrated.

---

## 4. Insider Buying (SEC Form 4)

**Why it works:** Insiders buy before good news. "Smart money" signal.

### Free Tier
| Source | Endpoint | Limits |
|--------|----------|--------|
| **SEC EDGAR** | `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent` | Free, parse XML |
| **OpenInsider** | `http://openinsider.com/screener` | Web scraping |
| **FMP API** | `https://financialmodelingprep.com/api/v4/insider-trading` | 250 req/day free |

### Paid Tier
| Source | Price | Quality |
|--------|-------|---------|
| **BamSEC** | $99/mo | Clean EDGAR API |
| **FMP API** | $19/mo | 300k req/month |

### Feature to Compute
```python
# Net insider buy ratio
insider_score = (insider_buys_30d - insider_sells_30d) / total_insider_transactions_30d
# Positive = net buying
```

---

## 5. Short Interest / Borrow Cost

**Why it works:** High short interest + breakout = short squeeze. Low short interest = institutional accumulation.

### Free Tier
| Source | Endpoint | Limits |
|--------|----------|--------|
| **FINRA (exchange-reported)** | Monthly CSV dumps | Lagged 15 days |
| **FMP API** | `/api/v3/float` + short interest | 250 req/day free |
| **Yahoo Finance** | Key statistics page | Web scraping |

### Paid Tier
| Source | Price | Quality |
|--------|-------|---------|
| **S3 Partners** | $500/mo | Real-time short interest + borrow cost |
| **IHS Markit** | Enterprise | Best data, expensive |

### Feature to Compute
```python
# Short interest as % of float
short_pct_float = short_interest / float
# Short interest trend (increasing = bearish, decreasing = bullish)
short_trend = short_pct_float_current / short_pct_float_30d_ago
```

---

## 6. Earnings Surprise / Guidance

**Why it works:** Post-earnings drift is real. Beat + raised guidance = momentum.

### Free Tier
| Source | Endpoint | Limits |
|--------|----------|--------|
| **FMP API** | `/api/v3/earnings-surprises` | 250 req/day free |
| **Earnings Whispers** | Web scraping | Unofficial |
| **Yahoo Finance** | `get_earnings_dates()` | Limited history |

### Feature to Compute
```python
# Earnings surprise magnitude
earnings_surprise = (actual_eps - estimated_eps) / estimated_eps
# Days since earnings (momentum decays)
days_since_earnings = (signal_date - earnings_date).days
```

---

## 7. Institutional Ownership Changes (13F)

**Why it works:** Whale accumulation precedes breakouts.

### Free Tier
| Source | Endpoint | Limits |
|--------|----------|--------|
| **SEC EDGAR (13F)** | Quarterly filings | 45-day lag |
| **Whale Wisdom** | Web scraping | Aggregated data |
| **FMP API** | `/api/v3/institutional-ownership` | 250 req/day free |

---

## Recommended Integration Priority

Based on signal strength vs implementation cost:

| Priority | Source | Cost | Signal Strength | Effort |
|----------|--------|------|-----------------|--------|
| **1** | Sector breadth (TradingView) | **FREE** | Medium | 2 hours |
| **2** | Insider buying (FMP free) | **FREE** | High | 4 hours |
| **3** | Short interest (FMP free) | **FREE** | Medium | 3 hours |
| **4** | Earnings surprise (FMP free) | **FREE** | High | 3 hours |
| **5** | Options flow (Unusual Whales) | $49/mo | **Very High** | 4 hours |
| **6** | Social sentiment (StockTwits) | **FREE** | Medium | 6 hours |
| **7** | Institutional (13F) | **FREE** | Low | 8 hours |

---

## Quick Wins: Start with FMP Free Tier

FMP provides 5 of 7 sources in one API:
- Insider trading
- Short interest
- Earnings surprises
- Institutional ownership
- Sector/industry classification

**Free tier:** 250 requests/day (enough for 50 tickers × 5 endpoints = 250 req).

**API Key:** Sign up at https://financialmodelingprep.com/developer/docs/
