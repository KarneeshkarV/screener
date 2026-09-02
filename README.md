## Screener

A Python CLI for screening, ranking, and backtesting US and Indian equities with technical, fundamental, relative-strength, unusual-volume, and operator-intent workflows.

Run commands through `uv`:

```bash
uv run screener --help
uv run screener screen -m india -n 30
```

The repo also has `just` shortcuts that use the local virtualenv:

```bash
just --list
just screen -m us -n 20 --csv
just backtest -m us --as-of 2026-03-20 --entry "close > 0" --tickers AAPL,MSFT
```

## Use as a library

Another codebase can install this straight from GitHub; there is no PyPI release.

```bash
uv add "screener @ git+https://github.com/KarneeshkarV/screener@main"
```

That installs both the `screener` console script and the importable package.
The supported import surface is `screener.api`, re-exported from `screener` itself.
Everything else under `screener.` is internal and may move in any commit.

```python
from screener import screen

outcome = screen(market="india", criteria="ema", limit=30)
print(outcome.total)   # total TradingView matches
print(outcome.df)      # result frame
```

By default an embedded call has no side effects: it returns the frame and writes neither a `~/.screener/history.db` row nor an HTML report.
Pass `persist=True` for the CLI's full behaviour.

```python
from screener import ScreenRequest, run_screen_workflow, list_criteria, list_markets

list_criteria()   # ['breakout', 'cheap_quality', 'dividend', 'ema', ...]
list_markets()    # ['india', 'us']

# Full control: build the request yourself.
outcome = run_screen_workflow(ScreenRequest(
    market="us", criteria_names=("ema", "breakout"), limit=50,
    order_by="setup_score", output_csv=True, detail=False,
    refresh=False, cache_ttl="15m", report_path=None,
))
```

`import screener` is lazy, so it pulls in neither pandas nor the scanner until you touch one of those names.

### Extras

The default install carries the screen path only.
Other workflows need their extra:

| Extra | Installs | Needed for |
|---|---|---|
| `report` | plotly | HTML reports and tear-sheets, i.e. `screen` without `--csv`, `dashboard`, `factor-tearsheet` |
| `prices` | yfinance, openscreener | every backtest, `garp`, `conviction`, `options` |
| `india` | jugaad-data | `unusual-volume`, `operator-scan` |
| `usage` | libsql-client | Turso history mirror and feature-usage tracking |
| `all` | all of the above | the pre-split behaviour |

```bash
uv add "screener[report] @ git+https://github.com/KarneeshkarV/screener@main"
```

A missing extra raises an `ImportError` naming the one to install, rather than a bare `ModuleNotFoundError`.
Nothing degrades silently: a workflow either has its dependency or says which extra supplies it.

Two consequences worth knowing before you pick an install:

- **Using the CLI?** Install `screener[all]`.
  A bare `screener screen` writes an HTML report, so it needs `report`; only `screener screen --csv` runs on the core install.
- **Embedding `screen()`?** The core install is enough.
  The default `persist=False` renders no report, so it needs no extra.

Working on the repo itself still needs everything: `uv sync --all-extras --all-groups`.

## Commands

### `screen`

TradingView-based technical screener.

```bash
uv run screener screen -m us -c ema -n 50
uv run screener screen -m india -c ema -c breakout --detail
uv run screener screen -m us -c intraday_momentum --csv
```

Features:

- Markets: `us`, `india`.
- Composable `-c` criteria: `ema`, `breakout`, `ema_breakout`, `value`, `quality`, `cheap_quality`, `undervalued`, `dividend`, `momentum_value`, `intraday_momentum`, `intraday_breakout`, `near_52_high` (repeat `-c` to combine).
- Full workflows are their own top-level commands, not `-c` criteria: `garp`, `mark-minervini`, `rs-breakout`, `promoter-buys`, `unusual-volume`, `vol-breakout-live`, `obv-trend-live`, and `options signals -c <unusual_options|bullish_oi_buildup|high_iv_rank|low_iv_rank|cheap_earnings_vol>`.
- Local `setup_score` ranking by default — each criterion has its own philosophy score (trend for `ema`, cheapness for `value`, yield quality for `dividend`, etc.); combined `-c` criteria average their scores.
- Optional CSV output with `--csv`.
- Optional fundamentals with `--detail`.
- TradingView cache controls with `--cache-ttl` and `--refresh`.
- Saves non-CSV runs to `~/.screener/history.db` and prints added/removed tickers versus the previous run.

### `history`

Lists the screen runs persisted to `~/.screener/history.db`, newest first, with the run id used by `backtest-historical --from-run`.

```bash
uv run screener history
uv run screener history -m india -c ema --limit 10
uv run screener history --csv
```

### `history-backup`

Mirrors the local screen-run history to Turso (or pulls remote runs missing locally with `--restore`). Reads `TURSO_DATABASE_URL`/`TURSO_AUTH_TOKEN`.

```bash
uv run screener history-backup
uv run screener history-backup --restore
just history-backup
```

### `conviction`

One composite 0-100 conviction card per ticker, fusing trend, breakout, volume, smart-money, fundamentals, and risk pillars. Point-in-time aware: pillars without dated data are skipped for stale `--as-of` dates and weights renormalize over survivors.

```bash
uv run screener conviction AAPL -m us
uv run screener conviction RELIANCE -m india --as-of 2026-06-30 --json
just conviction AAPL -m us
```

### `garp`

Finds GARP stocks using market-specific fundamental data.

```bash
uv run screener garp -m india -n 30
uv run screener garp -m us --universe-size 300 --workers 8 --csv
just garp -m india -n 30
```

### `promoter-buys`

Finds stocks where promoter or insider holdings increased.

```bash
uv run screener promoter-buys -m india --min-change 0.5
uv run screener promoter-buys -m us --min-yf-net-pct 0.01
just promoter-buys -m india --min-change 0.5
```

India mode uses screener.in promoter data with optional yfinance cross-checks. US mode uses yfinance insider transaction data.

### `institutional`

Shows FMP institutional (13F) ownership per US ticker, ranked by quarter-over-quarter change. Requires `FMP_API_KEY`.

```bash
uv run screener institutional --tickers AAPL,MSFT
just institutional --tickers AAPL,MSFT --csv
```

### `filings`

Reads US SEC filings via FMP. `filings list` shows recent filings (10-K/10-Q/8-K) with document links; `filings report` reads a filed 10-K/10-Q by section (case-insensitive `--section` substring match). Requires `FMP_API_KEY`.

```bash
uv run screener filings list AAPL --type 10-K --limit 5
uv run screener filings report AAPL --year 2024 --list-sections
uv run screener filings report AAPL --year 2024 --section "balance sheet"
just filings report AAPL --year 2024 --period Q3 --section revenue
```

`filings report` with no `--section` lists the available sections. Add `--json PATH` to dump the matched section JSON.

### `index-inclusion`

Event study of post-addition excess drift for S&P 500 additions versus SPY.

```bash
uv run screener index-inclusion --years 5
just index-inclusion --years 5 --csv
```

### `seasonality`

Monthly, turn-of-month, and day-of-week seasonality statistics for a ticker.

```bash
uv run screener seasonality AAPL --years 10
uv run screener seasonality RELIANCE -m india --csv
just seasonality AAPL
```

### `rs-breakout`

Screens for relative strength, SuperTrend, breakout, and volume setups.

```bash
uv run screener rs-breakout -m india -n 50
uv run screener rs-breakout -m us --tickers AAPL,MSFT,NVDA --no-output-files
uv run screener rs-breakout -m india --json rs.json --md rs.md
just rs-breakout -m india -n 50
```

### `unusual-volume`

Detects abnormal trading volume across a market or a ticker list.

```bash
uv run screener unusual-volume -m us --tickers AAPL,MSFT
just unusual-volume -m india
```

### `operator-scan`

NSE Operator Intent screener. It combines NSE Cash Bhavcopy delivery/VWAP data with F&O open interest changes, labels operator action, and writes a CSV.

```bash
uv run screener operator-scan
uv run screener operator-scan --date 2026-05-08 --only-actions --verbose
uv run screener operator-scan --universe fo --output operator.csv
just operator-scan --only-actions
```

Action labels include Long Build-up, Short Covering, Short Build-up, Long Unwinding, and High_Momentum_Watch.

### `options`

Build and accumulate free US/India option-chain data, inspect PCR/IV/max-pain
signals, screen panel-backed criteria, and use point-in-time options fields in
backtests.

```bash
uv run screener options build-panel -m india --start 2026-07-01
uv run screener options snapshot -m us --tickers AAPL,NVDA,SPY
uv run screener options show -m us --symbol AAPL
uv run screener screen -m us -c high_iv_rank -n 20
```

See [docs/options.md](docs/options.md) for sources, freshness/coverage limits,
panel schemas, criteria, and no-lookahead behavior.

## Backtesting

### `backtest-historical`

Runs a historical point-in-time backtest. This is wrapped by `just backtest`.

```bash
uv run screener backtest-historical -m us --as-of 2026-03-20 --entry "close > 0" --tickers AAPL,MSFT --hold 5 --top 2
just backtest -m india --as-of 2026-03-20 --entry "close > 0" --tickers RELIANCE,TCS --hold 5 --top 2
```

### Screen → backtest replay (`--from-run`)

Every non-CSV `screen` run is persisted to `~/.screener/history.db`. `backtest-historical --from-run` replays one of those runs as a point-in-time backtest: the universe is the stored tickers, `--as-of` is the run date, `--entry` defaults to `close > 0` (buy what the screen picked), and `--top` defaults to the snapshot size.

```bash
uv run screener history                                            # find run ids
uv run screener backtest-historical --from-run 42 --hold 5         # replay run #42
uv run screener backtest-historical --from-run india:ema --run-age-days 7 --hold 5
```

The `MARKET:CRITERIA` form picks the most recent run at least `--run-age-days` calendar days old — useful in cron, where "replay last week's screen" needs no id lookup. All the usual backtest knobs (`--hold`, `--stop-loss`, `--slippage-bps`, `--report`, `--csv`, custom `--entry`/`--strategy`, …) still apply.

Caveats: only the top-N rows shown at screen time were persisted, so the replay covers what the screen displayed, not its full match set, and candidate ranking inside the backtest is by as-of dollar volume, not the screen's original rank.

#### Daily replay cron

`scripts/daily_screen_replay.sh` runs every registered screen criterion on every market (persisting each run), then replays each `market:criteria` pair's most recent run that is at least `REPLAY_AGE_DAYS` (default 7) old, writing logs and HTML tear-sheets to `~/.screener/replay-logs/`. Pairs without an old-enough run are skipped, so the first week after install only accumulates history. Finally it backs up the local `history.db` to Turso via `screener history-backup` (reading `TURSO_DATABASE_URL`/`TURSO_AUTH_TOKEN` from the repo's `.env`); a backup failure is logged but never aborts the run.

```cron
# m h dom mon dow  command
30 11 * * 1-5  /root/screneer_main/screener/scripts/daily_screen_replay.sh >> "$HOME/.screener/replay-logs/cron.log" 2>&1
```

Tunables via environment: `MARKETS` (default `us india`), `CRITERIA` (default: every registered criterion), `REPLAY_AGE_DAYS`, `HOLD` (default 5), `TOP_N` (default 50), `LOG_DIR`, `KEEP_DAYS` (log retention, default 30).

### `backtest-rolling`

Runs a rolling backtest across a date window.

```bash
uv run screener backtest-rolling -m us --years 2 --strategy rs_breakout --top 10
uv run screener backtest-rolling -m india --start 2024-01-01 --end 2026-05-08 --entry "close > sma(close, 20)" --exit false
just backtest-rolling -m us --years 2 --strategy rs_breakout --top 10
```

Supports position sizing slots, holding period, stop loss, take profit, trailing stop, slippage/commission, benchmark, liquidity filters, custom tickers, CSV ledger output, and optional dashboard output.

Two defaults are on and can be turned off:

- `--point-in-time` (`--no-point-in-time` to disable) gates candidates by the membership they held at each signal date instead of by today's index list. It downgrades itself, with a note, for a universe that carries no membership history. See [docs/universes.md](docs/universes.md).
- `--compare-reinvestment` (`--no-compare-reinvestment` to disable) replays the same signal panel under the other equal-slot sizing rule and prints both side by side. The default result is unchanged; the second simulation roughly doubles the run time.

### `backtest-monte-carlo`

Runs the same rolling simulation as `backtest-rolling`, then block-bootstraps the resulting equity curve to show what a typical and a bad run of the same strategy look like.

```bash
uv run screener backtest-monte-carlo -m us --years 2 --strategy rs_breakout --top 10
uv run screener backtest-monte-carlo -m us --years 2 --strategy rs_breakout --iterations 10000 --block 40 --json mc.json
```

It accepts every flag that defines a `backtest-rolling` run (window, universe, signals, costs, sizing, `--interval`), plus:

- `--iterations` (default 5000) synthetic equity paths.
- `--block` (default 20) bars per resampled block. Blocks keep the short-horizon autocorrelation that drives drawdown; `--block 1` makes the draw i.i.d. and understates it.
- `--seed` (default 42) so a run is reproducible, and `--ruin-threshold` (default 0.5) for the fraction of starting capital that counts as ruin.
- `--paths` (default 1000) simulated paths kept for the report's fan chart and percentile bands. Higher is smoother but costs memory and HTML size; `0` skips the chart.
- `--json` to write the raw result next to the usual HTML tear-sheet.

The `MC ...` rows join the realized run's metrics in both the terminal table and the tear-sheet: median/p05/p95 return, the drawdown percentiles, the probability of ending in profit, and the risk of ruin.

The tear-sheet also gets a **Monte Carlo** tab that `backtest-rolling` does not have.
It shows the fan of simulated equity paths with p05/median/p95 bands and the realized run drawn over them, the terminal-return and max-drawdown distributions with the realized value marked on each, an outcome-percentile table (p01 to p99), and a run summary (iterations, bars, block, seed, probability of profit, risk of ruin).

The equity curve is resampled rather than the trade list.
A rolling run holds `--top` positions at once, so its trades overlap in time; chaining them one after another would compound a portfolio that never existed and would misstate the drawdown.
`screener optimize validate` still bootstraps trades, because it scores a trade ledger on its own.

### Position sizing (`--sizing`)

Both backtest commands accept rule-based per-entry position sizing. The default `equal_slot` matches the legacy fixed-slot engine bit-for-bit; every other rule sizes down from the slot budget (never above it, never beyond available cash):

- `fixed_fraction` — `--sizing-position-pct` of initial capital per position (default 0.1).
- `fixed_risk` — risk `--sizing-risk-pct` of initial capital per trade (default 0.01); requires `--stop-loss`.
- `atr_risk` — risk budget divided by `--sizing-atr-multiple` × ATR(`--sizing-atr-window`) per share.
- `inverse_vol` — targets `--sizing-risk-pct` daily volatility using a `--sizing-vol-window` return lookback.

ATR/volatility lookbacks read only up to the signal bar (no lookahead) and fall back to the slot budget during warmup.

```bash
uv run screener backtest-rolling -m us --years 2 --strategy rs_breakout --top 10 --sizing atr_risk --sizing-risk-pct 0.01
uv run screener backtest-historical -m us --as-of 2026-03-20 --tickers AAPL,MSFT --entry "close > 0" --sizing fixed_risk --stop-loss 0.08
```

### Intraday intervals

Both backtest commands accept `--interval` (default `1d`; also `1h`, `30m`, `15m`, `5m`, `1m`). All bar-count parameters (`--hold`, lookbacks in entry/exit expressions) are interpreted in bars of the chosen interval, trades carry full timestamps, and metrics annualize by bars-per-year for the interval.

```bash
uv run screener backtest-rolling -m us --tickers AAPL,MSFT --start 2026-06-22 --end 2026-07-02 --interval 15m --entry "close > sma(close,5)" --hold 4
```

Notes:

- yfinance caps intraday history (1m ≈ last 30 days, 5m–30m ≈ 60 days, 1h ≈ 730 days); requests past the cap log a warning and return what is available.
- With `FMP_API_KEY` set, FMP serves intraday bars (raw, unadjusted) via `historical-chart` — both as the automatic fallback and with `SCREENER_PRICE_PROVIDER=fmp`.
- Intraday timestamps are canonical naive UTC across providers; intraday bars are cached under interval-namespaced keys so daily caches are never polluted.
- Long-warmup strategies (e.g. anything needing SMA200) usually cannot fill their lookback inside the capped intraday windows.

### Backtest delta harness (`scripts/backtest_delta.py`)

Fully offline, deterministic matrix that pins backtest numbers across engines, cost models, sizing rules, and intervals.
Use it as a baseline before refactors that deliberately change fills, costs, force-close windows, or indicator seeding.
Unlike `just backtest-smoke-us`, it never hits the network and does not depend on cache state.

```bash
uv run python scripts/backtest_delta.py --out /tmp/baseline.json
uv run python scripts/backtest_delta.py --compare /tmp/baseline.json
```

`--out` writes sorted JSON (metrics + full trade ledgers per cell).
`--compare` reruns the matrix, prints a per-cell diff, and exits non-zero if anything moved.
Bar factories and the price-fetcher stub live inside the script so a later `tests/` refactor cannot silently move the baseline.

### `backtest-lab`

Launches a local browser UI for comparing rolling backtest strategies.

```bash
uv run screener backtest-lab
uv run screener backtest-lab --host 127.0.0.1 --port 8766
just backtest-lab
```

### `factor-tearsheet`

Computes factor IC and quantile tearsheet for a named strategy (or a `combo:name=w,...` weighting) that emits `rank_score`.

```bash
uv run screener factor-tearsheet -m us --strategy momentum_12_1 --years 3
just factor-tearsheet -m india --strategy momentum_12_1 --universe nifty50
# dual-momentum: 12-1 winners that are also above SMA200
uv run screener backtest-rolling -m us --years 3 --strategy momentum_12_1_trend --top 10
# risk-adjusted: rank by mom_12_1 / vol_252 (volatility-scaled winners)
uv run screener backtest-rolling -m us --years 3 --strategy momentum_12_1_riskadj --top 10
```

### `earnings-backtest`

Backtests sentiment-scored earnings entries: buy one or two sessions before a
report (E-1/E-2), exit at E. Wrapped by `just earnings-backtest`.

```bash
uv run screener earnings-backtest -m us --years 3 --strategy drift --entry-days 1
just earnings-backtest -m india --years 2 --csv > trades.csv
```

### `earnings-pead`

Backtests post-earnings-announcement drift (PEAD): enter at the next open after
a report whose EPS surprise is at least `--min-surprise` (default 5%). Wrapped
by `just earnings-pead`.

Two exit modes via `--exit-mode` (default `fixed`):

- `fixed` — exit at the close `--hold-days` sessions after entry (default 40).
- `dynamic` — stay in the position until a later report fails the surprise
  criterion (exit at the next open, `exit_reason=criteria_failed`) or the price
  history ends (`exit_reason=end_of_data`). `--hold-days` is ignored and the
  CSV ledger gains a trailing `exit_reason` column.

```bash
uv run screener earnings-pead -m us --years 3 --min-surprise 5
uv run screener earnings-pead -m india --years 2 --exit-mode dynamic --csv > pead.csv
just earnings-pead -m us --exit-mode dynamic
```

India surprise data comes from FMP's historical earnings calendar (requires
`FMP_API_KEY`): real NSE announcement dates are preferred and enriched with
FMP's EPS surprise, and quarters NSE lacks fall back to FMP's own
point-in-time dates. The default India earnings path used by
`earnings-backtest` is unchanged.

### Standalone Pine Runner

The standalone Pine strategy runner is not a `uv run screener` subcommand; it is a separate script wrapped by `just pine`.

```bash
just pine --market us --years 3 --limit 50
just pine-india --years 2
uv run python -m screener.research.pine_runner --market us --years 3 --limit 50
```

## Optimization

### `optimize grid`

Runs exhaustive grid search over backtest parameter ranges.

```bash
uv run screener optimize grid -m us --years 2 --strategy rs_breakout --stop-loss 0.05,0.08 --take-profit 0.1,0.15 --hold 5,10
just optimize grid -m us --years 2 --strategy rs_breakout --stop-loss 0.05,0.08 --take-profit 0.1,0.15 --hold 5,10
```

### `optimize walk-forward`

Runs rolling train/test walk-forward optimization.

```bash
uv run screener optimize walk-forward -m india --years 3 --strategy rs_breakout --train-days 252 --test-days 63
just optimize walk-forward -m india --years 3 --strategy rs_breakout --train-days 252 --test-days 63
```

### `optimize validate`

Runs Monte Carlo stress testing on an existing trade ledger.

```bash
uv run screener optimize validate --trades trades.csv --iterations 5000 --json validation.json
just optimize validate --trades trades.csv --iterations 5000 --json validation.json
```

### `optimize research-report`

One-command research pipeline: grid search → walk-forward → Monte Carlo, reusing a single price fetcher across stages. Writes `<out>.json` and `<out>.html` plus a stdout summary.

```bash
uv run screener optimize research-report -m us --years 1 --strategy rs_breakout --top 10
just research-report -m us --years 1 --strategy rs_breakout
```

## Utility Commands

### `usage-report`

Shows successful feature usage counts from Turso.

```bash
uv run screener usage-report
just usage-report
```

### `cache`

Inspects and prunes the screener's on-disk caches under `~/.screener/`.

```bash
uv run screener cache status
uv run screener cache clean --older-than 30
just cache status
```

## Config File

The CLI can load YAML or JSON defaults with `--config`. The repo includes an example at `screener.yaml`.

```bash
uv run screener --config screener.yaml screen
uv run screener --config screener.yaml backtest-historical
uv run screener --config screener.yaml optimize grid
```

Config files must contain a top-level mapping. Top-level keys are global options and command names. For nested Click command groups, such as `optimize`, put the subcommand under the group name.

```yaml
log_level: INFO
log_json: false

screen:
  market: india
  criteria_names:
    - ema
    - breakout
  limit: 30
  order_by: setup_score
  cache_ttl: 15m

backtest-historical:
  market: us
  as_of: "2026-03-20"
  tickers: AAPL,MSFT,NVDA
  entry_expr: close > sma(close, 20)
  exit_expr: "false"
  hold: 5
  top: 2

unusual-volume:
  market: india
  strength_floor: high
  limit: 50
  buildup_enabled: true

optimize:
  grid:
    market: us
    years: 1
    strategy_name: rs_breakout
    hold: 5,10,20
    top: 10
    metric: sharpe
```

Use Click parameter names in config, not always the visible flag name. Most are the flag converted to snake case, for example `--cache-ttl` becomes `cache_ttl`. Some commands use custom internal names:

- `--criteria` -> `criteria_names`
- `--sort` -> `order_by`
- `--entry` -> `entry_expr`
- `--exit` -> `exit_expr`
- `--strategy` -> `strategy_name`
- `--csv` -> `output_csv`
- `--strength` -> `strength_floor`
- `--buildup/--no-buildup` -> `buildup_enabled`
- `--json` -> `json_path`
- `--md` -> `md_path`

Explicit CLI flags override values from the config file.

### Global Options

Every `uv run screener ...` command accepts these top-level options before the subcommand:

```bash
uv run screener --config config.yaml screen -m india
uv run screener --log-level DEBUG screen -m us
uv run screener --log-json screen -m us --csv
```

## Just Shortcuts

Current `justfile` recipes:

```bash
just
just help
just help-<command>          # per-command help: screen, backtest, backtest-rolling,
                             # cache, conviction, earnings-backtest,
                             # earnings-pead, factor-tearsheet, filings, garp, history,
                             # history-backup, index-inclusion, institutional,
                             # operator-scan, optimize, options, pine, promoter-buys,
                             # research-report, rs-breakout, seasonality,
                             # unusual-volume
just screen ...
just screen-us ...
just screen-india ...
just backtest ...
just backtest-rolling ...
just backtest-smoke-us
just backtest-smoke-india
just cache ...
just conviction ...
just earnings-backtest ...
just earnings-pead ...
just factor-tearsheet ...
just history ...
just history-backup ...
just index-inclusion ...
just institutional ...
just options ...
just pine ...
just pine-us ...
just pine-india ...
just research-report ...
just seasonality ...
just unusual-volume ...
just garp ...
just promoter-buys ...
just rs-breakout ...
just operator-scan ...
just optimize ...
just usage-report
just compile
just test ...
just lint
just format-check
just typecheck
just ci
```

All current top-level `uv run screener` commands are wrapped by `just`.

## Price Data Provider

The default price provider is `yfinance` with Financial Modeling Prep fallback when `FMP_API_KEY` is available. Set this environment variable before running a command:

```bash
export FMP_API_KEY="your_fmp_api_key"
```

Then run the project through `uv`, for example:

```bash
uv run screener backtest-historical --tickers AAPL,MSFT --entry "close > sma(close, 20)"
```

FMP responses are cached under `~/.screener/fmp_prices`. Use a command's existing `--refresh` option where available to bypass cached price data.

To force one provider instead of fallback mode, set `SCREENER_PRICE_PROVIDER` to `yfinance` or `fmp`.

