## Price Data Provider

The default price provider is `yfinance`. To use Financial Modeling Prep
instead, set these environment variables before running a command:

```bash
export FMP_API_KEY="your_fmp_api_key"
export SCREENER_PRICE_PROVIDER=fmp
```

Then run the project through `uv`, for example:

```bash
uv run screener backtest-historical --tickers AAPL,MSFT --entry "close > sma(close, 20)"
```

FMP responses are cached under `~/.screener/fmp_prices`. Use a command's
existing `--refresh` option where available to bypass cached price data.
