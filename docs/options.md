# Options data layer

The options package normalizes free US and India option data into immutable
contracts/chains, derives source-neutral signals, and accumulates one daily row
per underlying in the existing atomic Parquet panel store.

Internal implied volatility values are decimals: `0.25` means 25%. Existing
earnings-sentiment output remains percentage based for backward compatibility.

## Data sources and coverage

| Market | Source | What it supplies | Historical coverage / freshness |
|---|---|---|---|
| India | NSE F&O UDiff + legacy bhavcopy | Exact end-of-day contract OI, OI change, volume, price, strike, expiry, underlying price, and board lot | Immutable daily archive. The UDiff format begins 2024-07-08; earlier dates fall back to the legacy archive (roughly 2000 onward) normalized to the same schema. Unavailable dates are skipped and reported. This is the primary backtest source. |
| India | NSE equity/index option-chain API | Intraday OI, OI change, volume, IV, bid/ask and last price | Live snapshot only. Equity and NIFTY/BANKNIFTY/FINNIFTY-family index endpoints use the existing primed NSE session and 15-minute cache. |
| India | NSE participant OI archive | Client/DII/FII/Pro long and short positioning across index/stock futures and options | Exact daily archive, backfillable with `options participants`. |
| India | NSE all-index close archive / `allIndices` | India VIX close or current value | Exact daily archive plus a live fallback. |
| India | NSE `fo_mktlots.csv` | Current lot size by underlying/expiry | Cached daily. Historical UDiff rows use their point-in-time `NewBrdLotQty` instead. |
| US | CBOE delayed quotes JSON | Full chain, all expiries, IV, greeks, OI, volume, bid/ask, last price | Roughly 15-minute delayed and live-only. Normalized snapshots are cached intraday and accumulated daily. Index symbols use CBOE's underscore form. |
| US | yfinance option chains | OI, volume, IV, bid/ask and last price | Live-only fallback when CBOE is unavailable. Black-Scholes greeks are computed when spot/IV/time are usable. |
| US | CBOE daily market statistics | Total, index, ETP, equity, VIX, and SPX put/call ratios | Official daily page; parsed records are cached per date. |
| US | FRED `VIXCLS` / `VXVCLS` | VIX and CBOE 3-month volatility index | Daily history. The panel also stores term spread/ratio and a simple volatility regime. |

No paid options provider is required. FMP is not used for option chains.

### Historical lot sizes

Legacy (pre-July-2024) bhavcopy rows carry no board lot, and NSE does not
publish historical lot sizes reliably, so they are never fabricated. To get
correct pre-2024 rupee notionals, maintain an optional point-in-time CSV at
`~/.screener/lot_sizes_history.csv`:

```csv
symbol,effective_from,lot_size
RELIANCE,2020-01-01,505
RELIANCE,2023-06-01,250
TCS,2022-01-01,150
```

`effective_from` is an ISO date marking when that lot took effect. For any
`as_of`, the latest row per symbol with `effective_from <= as_of` is used;
symbols whose first record starts later are omitted. A missing or malformed
file is simply ignored (no lot fallback). An embedded `NewBrdLotQty` on a row
always takes precedence over this file, so it only affects legacy dates.

## Commands

Run all commands through `uv`.

Build the India historical panel from point-in-time UDiff files:

```bash
uv run screener options build-panel -m india --start 2026-06-01 --end 2026-07-08
uv run screener options build-panel -m india --start 2026-07-01 --tickers RELIANCE,TCS
```

Snapshot US chains. CBOE is attempted first and yfinance is the fallback:

```bash
uv run screener options snapshot -m us --tickers AAPL,NVDA,SPY
uv run screener options snapshot -m us --universe-size 50 --workers 4
```

Inspect accumulated history. Output always includes the source, as-of date,
contract coverage, and history length:

```bash
uv run screener options show -m india --symbol RELIANCE
uv run screener options show -m us --symbol AAPL --days 60
uv run screener options show -m us --symbol AAPL --csv
```

Build participant and market-regime panels:

```bash
uv run screener options participants --start 2026-07-01 --end 2026-07-08
uv run screener options regime -m india --start 2026-07-01 --end 2026-07-08
uv run screener options regime -m us --start 2026-07-01 --end 2026-07-08
```

`--refresh` bypasses the relevant provider cache. NSE fan-out remains modest;
one unavailable archive day does not abort a multi-day backfill.

## Deep historical backfill

The India panel now reaches back to the pre-UDiff era (roughly 2000 onward)
using NSE's legacy F&O archive. All range commands accept arbitrary multi-year
`--start`/`--end` windows; a missing archive day is skipped and reported, so a
long run degrades gracefully rather than aborting. `just` wraps the recommended
order:

```bash
just options-backfill 2020-01-01 2024-12-31          # per-contract chains + metrics
just options-backfill-context 2020-01-01 2024-12-31  # participant OI, then India VIX regime
```

Run the panel build first (it is the base layer), then the participant OI and
regime context. A five-year build spans a bit over a thousand trading days;
because each day is one throttled archive download plus local normalization,
expect on the order of a few tens of minutes per year on a cold cache, and near
instant re-runs once cached.

Two caveats specific to the pre-2024 era. Legacy bhavcopies carry no underlying
price, so spot is filled from that day's NSE cash bhavcopy (equity closes);
index underlyings such as NIFTY have no matching equity symbol and keep a null
spot. And legacy files carry no IV, so IV and greeks are derived point-in-time
by inverting the daily settle price (see Metrics), rather than being read from
the feed.

## Metrics

Aggregates cover all unexpired contracts in a snapshot:

- OI PCR, volume PCR, and call/put OI ratio;
- call/put OI and volume, their daily change, and India notionals using lot size;
- IV median, expanding IV rank/percentile, and per-symbol trailing option volume;
- snapshot-difference OI changes for US rows when the source has no daily change.

Strike-sensitive fields use the front expiry:

- maximum-pain strike and distance from spot;
- highest-OI put support strikes below spot and call resistance strikes above it;
- ATM IV, 25-delta skew (nearest-OTM proxy when greeks are absent), and ATM
  straddle implied move;
- near-spot call/put writing inferred from exact India OI plus option-premium
  changes.

Term structure is next-expiry ATM IV minus front-expiry ATM IV. Zero/missing OI
denominators produce `null`, never infinity. IV rows must be between 0 and 500%
and have nonzero OI or volume.

## Panels

Panels live under `~/.screener/panels/` and use
`cache.append_panel_snapshot`, including its file lock, atomic replace, and
keep-last deduplication.

| Panel | Dedupe key | Purpose |
|---|---|---|
| `options_metrics_india` | `as_of, SYMBOL` | Historical bhavcopy metrics and any accumulated India snapshots |
| `options_metrics_us` | `as_of, SYMBOL` | CBOE/yfinance daily snapshots |
| `participant_oi` | `as_of, participant` | Client/DII/FII/Pro positioning |
| `india_vix` | `as_of` | India VIX archive/live series |
| `pcr_market_us` | `as_of` | CBOE market PCR plus FRED VIX/VIX3M |

History-derived values are causal. For example, today's unusual-volume ratio
uses a trailing mean shifted by one row, and an IV rank on date D uses only IV
observations dated on or before D.

## Screen criteria

The following are registered pipeline criteria and therefore cannot be mixed
with another `-c` value:

```bash
uv run screener screen -m us -c unusual_options -n 20
uv run screener screen -m india -c bullish_oi_buildup -n 20
uv run screener screen -m us -c high_iv_rank -n 20
uv run screener screen -m us -c low_iv_rank -n 20
uv run screener screen -m us -c cheap_earnings_vol -n 20
```

- `unusual_options`: current option volume is at least 2x that symbol's own
  prior trailing average. It needs six daily snapshots at minimum.
- `bullish_oi_buildup`: exact put-writing confirmation for India; a clearly
  labelled consecutive-snapshot OI proxy for US when writing direction is not
  observable.
- `high_iv_rank` / `low_iv_rank`: expanding rank at or above 80 / at or below
  20, with the usable IV-day count emitted on every row.
- `cheap_earnings_vol`: front-straddle implied move below the median absolute
  historical earnings move, requiring at least two prior events. Earnings dates
  and OHLCV come from the existing earnings-backtest providers.

Thin or missing panels produce an actionable message instead of silently
returning a fabricated signal.

## Backtests and no-lookahead behavior

Options columns can be referenced directly in historical and rolling Pine-like
expressions:

```bash
uv run screener backtest-rolling -m us --tickers AAPL,MSFT \
  --entry "close > sma(close,20) and pcr < 0.8 and iv_rank < 50" \
  --years 2
```

The engines inspect the parsed AST. A price-only expression does not read an
options panel. When an options identifier is present, each symbol's panel is
forward-filled onto price bars with an as-of join: only rows whose `as_of` is
less than or equal to the bar timestamp can appear. Dates before first coverage
and symbols without coverage remain `NaN`, so comparisons evaluate false. The
run warning reports covered tickers and requested fields.

US chain history begins only when snapshots have been accumulated, so it cannot
support a deep historical chain backtest retroactively. India UDiff history is
the high-confidence point-in-time path. Daily panel data should not be treated
as an intraday signal before that session's close.

## Testing

All automated tests are offline. Verified, truncated NSE CSV and CBOE JSON/HTML
fixtures pin provider shapes; fetchers and cache providers are injected in unit
tests. Live access is used only for explicit smoke checks.
