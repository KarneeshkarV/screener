## Price Data Provider

The default price provider is `yfinance` with Financial Modeling Prep fallback
when `FMP_API_KEY` is available. Set this environment variable before running a
command:

```bash
export FMP_API_KEY="your_fmp_api_key"
```

Then run the project through `uv`, for example:

```bash
uv run screener backtest-historical --tickers AAPL,MSFT --entry "close > sma(close, 20)"
```

FMP responses are cached under `~/.screener/fmp_prices`. Use a command's
existing `--refresh` option where available to bypass cached price data.

To force one provider instead of fallback mode, set `SCREENER_PRICE_PROVIDER`
to `yfinance` or `fmp`.
