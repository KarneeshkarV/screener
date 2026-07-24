# Intraday + first-class options roadmap

Plan for evolving the screener into minute-resolution equities and first-class
options data, for both screening and backtesting. Phase 1 (minute-bar equity
support) shipped in `12b3cf8`; this document records its status and specifies
the remaining phases.

Two constraints shape everything below:

- **Free minute-level history barely exists.** yfinance serves ~30 days of 1m
  (8 days per request); FMP intraday is shallow on free tiers; historical
  intraday option chains are paid-only for the US (Polygon, ThetaData) and
  simply unavailable for India (NSE publishes EOD bhavcopy only).
- **Therefore two data strategies run in parallel:** *forward capture* (record
  our own snapshots from the free live sources already integrated — yfinance
  1m bars, CBOE delayed chains, the NSE option-chain API) and a *paid-provider
  seam* that can be plugged in later for deep history. Every day a recorder is
  not running is history that can never be recovered — recorders come first
  within each phase.

---

## Phase 1 — Minute-bar equity support ✅ (shipped in `12b3cf8`)

- Interval-partitioned bar store at `~/.screener/bars/{market}/{interval}/{symbol}.parquet`
  (`backtester/bar_store.py`): atomic temp-file writes, naive-UTC
  normalization, `__raw` namespace for unadjusted bars, idempotent appends.
- 1m as the canonical fine archive; 5m/15m/30m/1h served by session-anchored
  local resampling (`price_frames.resample_intraday_bars`) with the same
  tail-freshness TTL discipline as the flat cache.
- `screener bars record` + `scripts/daily_bars_record.sh` cron: appends the
  trailing 1m window for each market's universe daily so the archive outgrows
  the ~30-day provider cap. Registered with `screener cache status`/`clean`.
- Interval-aware costs: unset `--slippage-bps` resolves to 2/3/5/7/10 bps at
  1h/30m/15m/5m/1m (1d stays 0.0 → daily runs byte-identical); Corwin-Schultz
  spread window scales by bars-per-session.
- `day_loop.run_day_loop` generalizes to a session loop (`market_tz` param)
  keyed on `sessions.is_session_last`, with a driver-level
  `flatten_at_session_end` guarantee for `intraday_only` runs; wired into both
  the historical and rolling engines.

**Verified 2026-07-24:** live smoke of `bars record` appended AAPL 1m bars and
a 15m request was served from the stored 1m series with correct 09:30-ET
session anchoring. (See the Phase 3 status block for the current full-suite
figures, which cover Phase 1 and 3 together.)

**Operational item (done):** the recorder cron is installed in the root
crontab, so the 1m archive grows daily:

```
30 1 * * 2-6 /root/screneer_main/screener/scripts/daily_bars_record.sh >> "$HOME/.screener/bars-logs/cron.log" 2>&1
```

---

## Phase 2 — Intraday screener ✅ (shipped in `b8bc895`)

Goal: screens evaluate over our own minute bars, with the same criteria logic
the backtester uses — that identity is what makes intraday backtests
trustworthy.

**Status:** 2.1 (local scanner), 2.2 (live loop), and 2.3 (tests) shipped.
`screener/local_scanner.py` adds `screen --source local --interval {1m,5m,15m,
30m,1h}`; `screener/screen_live.py` adds `screen live --every 5m`, persisting
each pass through the existing `history.save_run` writer (no schema change,
intraday `run_ts`) and diffing to emit only new entrants/exits, with
clock/sleep/refresh/scanner as injectable seams for offline tests. The default
TradingView/daily path is byte-identical. 18 offline-stub tests.

### 2.1 Local scanner path

- A local bar-store-backed scanner alongside the TradingView one: evaluate the
  existing criteria registry (`screener/criteria/…`) over 1m/5m/15m frames
  loaded via `bar_store.load_bars` + `resample_intraday_bars`.
- Criteria that today only exist as TradingView server-side filters
  (`intraday_momentum`, `intraday_breakout`) get local implementations with
  identical semantics, so `screen` results are reproducible offline and in
  backtests.
- CLI: `screener screen … --source local --interval 5m`. Default stays
  TradingView (broader universe coverage); local mode is limited to symbols in
  the bar store.

### 2.2 Live loop mode

- `screener screen live --every 5m`: re-evaluate during the session, diff
  against the previous pass, emit only new entrants/exits.
- Persist each pass to the existing `~/.screener/history.db` (same schema,
  intraday timestamp) so `backtest-historical --from-run` and the Turso backup
  keep working unchanged.
- A trailing-window `bars record --days 1` refresh runs before each pass so
  the store is near-live (yfinance 1m has ~1–2 min lag; acceptable for
  screening, documented as such).

### 2.3 Deliverables / tests

- Offline stub tests: synthetic 1m sessions through the local scanner produce
  the same picks as the equivalent Pine expressions in the backtester.
- Golden test: daily-interval screens byte-identical before/after.

---

## Phase 3 — Options: from EOD panels to a first-class contract store

The core structural change. Today the storage unit is "one metrics row per
underlying per day" (`options/panels.py`). First-class means storing
**contracts as time series**, with the daily panel becoming a derived view so
nothing existing breaks.

**Status:** 3.1 (contract store), 3.2 (snapshot recorder), and the 3.3 provider
seam (stub-only) shipped. `options/contract_store.py` persists every snapshot;
`options record` + `scripts/daily_options_record.sh` capture chains during
session hours; `history_provider.py` defines the `OptionsHistoryProvider`
protocol with the forward-capture store as the only real backend (Polygon /
ThetaData are documented stubs). US intraday options history is therefore
capture-start-forward; India is capture-forward + EOD bhavcopy. 3.4 (derived
intraday panel views) is still open.

**Verified 2026-07-24:** ruff clean, `ruff format --check` clean, mypy strict
clean (218 source files); `pytest tests/` — 1,462 passed, 21 skipped,
coverage 91.28% (≥ 90 floor). 31 new offline-stub tests
(`test_contract_store.py`, `test_options_record.py`,
`test_options_history_provider.py`); the daily EOD panel / signals / backtest
paths are untouched (the contract store is additive and imported nowhere in
them), so their golden tests stay byte-identical. Offline CLI smoke: `options
record --help` renders and the `contracts` area appears in `cache status`.

**Operational item (done):** both options recorders are installed in the root
crontab (market-local `CRON_TZ`, session-open invocation with `--every 15m`),
alongside the Phase 1 bars cron; logs under `~/.screener/options-logs/`:

```
# India — 09:10 IST, weekdays
CRON_TZ=Asia/Kolkata
10 9 * * 1-5 MARKETS=india /root/screneer_main/screener/scripts/daily_options_record.sh >> "$HOME/.screener/options-logs/cron.log" 2>&1
# US — 09:25 ET, weekdays
CRON_TZ=America/New_York
25 9 * * 1-5 MARKETS=us /root/screneer_main/screener/scripts/daily_options_record.sh >> "$HOME/.screener/options-logs/cron.log" 2>&1
```

The live network smoke (actually hitting CBOE/NSE) is opt-in and not yet run:
`screener options record -m us --once --watchlist SPY --max-underlyings 1`.

### 3.1 Contract store (schema) ✅ (shipped)

- **Dimension**: contract identity — underlying, expiry, strike, right, lot
  size (PIT via the existing `lot_history.py` CSV).
- **Facts**: every observed snapshot, timestamped — bid/ask/last, volume, OI,
  IV, greeks. Parquet partitioned by `{market}/{date}/{underlying}.parquet`
  (zstd), atomic writes reusing the `bar_store.py` discipline.
- Normalization already exists: `options/models.py`, `cboe.py`, `nse_live.py`
  produce exactly these objects today; the change is persisting *every*
  snapshot instead of collapsing to one daily panel row.
- EOD sources (UDiff/legacy bhavcopy) load into the same schema as one
  end-of-session snapshot per contract per day, so India's 2000→present
  archive and forward-captured intraday snapshots are queried uniformly.
- IV/greeks enrichment on ingest: reuse the legacy-bhavcopy BS inversion
  (`options/greeks.py`) for any snapshot missing IV, so the store is uniformly
  enriched.

### 3.2 Snapshot recorder — build this first, turn it on immediately ✅ (shipped)

- `screener options record --market us|india --every 15m`: during session
  hours, snapshot CBOE delayed chains (US) and the NSE live chain API (India)
  for a configured watchlist into the contract store.
- Cron wrapper like `daily_bars_record.sh` (session-gated, per-market TZ),
  logs under `~/.screener/options-logs/`, idempotent appends keyed on
  (contract, snapshot_ts).
- Watchlist config: default to index options (SPY/QQQ/NIFTY/BANKNIFTY) plus
  the F&O universe subset actually used by `options signals`; cap watchlist
  size — chain snapshots at 15-min cadence are the storage line-item to watch.
- Storage budget: ~500 contracts/underlying × ~25 snapshots/day; partition by
  date/underlying and monitor via `screener cache status`.

### 3.3 Provider seam for paid history ✅ (stub seam shipped)

- `OptionsHistoryProvider` protocol: `chains(underlying, date) -> snapshots`,
  `contract_bars(contract, interval, start, end)`. Default backend = the
  forward-capture store. Stub adapters for Polygon and ThetaData that
  normalize into the same contract-store schema (implemented only when/if a
  subscription happens).
- India stays capture + bhavcopy — there is no historical intraday source to
  buy at retail; document this explicitly.

### 3.4 Derived views & compatibility ✅ (shipped in `57bc79a`)

- The existing daily panel (`options build-panel`) becomes a reduction over
  the contract store when store data exists, falling back to current behavior
  otherwise. `OPTION_EXPRESSION_FIELDS`, `options signals`, regime and
  participant flows are untouched.
- New intraday-derived panel fields (feeding Phase 4/5): intraday OI change,
  IV change vs. session open, rolling intraday put/call volume ratio.

**Status shipped:** `contract_store.stored_underlyings()` signals the
store-derived path; `panels.py` adds `oi_change_intraday`, `iv_change_intraday`,
`pcr_volume_intraday` (null when only a single/EOD snapshot exists) and
`backtest.py` exposes them to the panel expression registry. No-store output is
byte-identical to the legacy path. 7 offline-stub tests.

---

## Phase 4 — Options-aware backtesting ✅ (4.1–4.3 shipped; 4.4 shipped with 3.4)

**Status:** 4.1 (`26879a4`), 4.2/4.3 (`b35dbc1`) shipped; 4.4 landed with the
3.4 expression registry. Remaining open items are deeper realism (bar-store
bracket resolution between snapshots, full `day_loop` structure slots) noted
inline below.

### 4.1 EOD improvements first (cheap, immediate, no new data needed) ✅ (shipped in `26879a4`)

Extend `options/position_backtest.py` (keeping its strict D-close signal →
D+1 fill causality):

- **Fill models**: mid vs. bid/ask cross with configurable slippage, replacing
  the current `last`-with-`settle`-fallback; per-leg widening for illiquid
  strikes (spread proxied from settle/close dispersion when quotes absent).
- **Margin model** for short options: SPAN-like approximation for India
  (exposure + worst-of scenario grid), Reg-T for US; portfolio-level margin
  utilization tracked so sizing is realistic.
- **Expiry mechanics**: explicit settlement (cash for index, physical flag for
  stock options), ITM assignment, and expiry-day P&L from settlement price.
- **Roll rules**: config-driven roll at DTE/delta thresholds as a first-class
  exit-and-reenter, so calendar-ish strategies stop being one-off scripts.

### 4.2 Intraday options backtesting (on the contract store) ✅ (shipped in `b35dbc1`)

- `options/intraday_backtest.py`: `run_intraday_options_backtest(cfg, provider)`
  walks the store's intraday snapshots (via the `OptionsHistoryProvider` seam),
  entering/marking/exiting at snapshot timestamps; positions flatten at each
  session's last snapshot (`session_end`). Reuses the Phase 4.1 fill/margin/
  premium helpers, so realism knobs behave identically. CLI: `options
  intraday-backtest`.
- Causality: snapshots are processed in strict timestamp order — a decision at
  time T only sees the chain observed at T; a later snapshot never prices an
  earlier fill. Same PIT discipline `position_backtest.py` documents for D/D+1.
- **Open follow-up:** exits are currently checked *at* snapshots; evaluating
  stops/targets *between* snapshots against the underlying's 1m bars (the
  conservative bracket-ambiguity convention the equity engine uses) is the next
  realism increment. Scope stays honest: US intraday is capture-start-forward
  (or a paid provider); India intraday options history is capture-forward only.

### 4.3 Mixed portfolios ✅ (shipped in `b35dbc1`)

- The intraday engine holds an optional signed `equity_hedge_qty` of the
  underlying for the life of each option position, marked from the snapshot
  spot, netting a delta/equity leg into the same portfolio P&L and equity
  curve (recorded in trade `details`). CLI: `--equity-hedge`.
- **Open follow-up:** full integration into the equity `day_loop` as a
  first-class structure slot (shared `Portfolio` drawdown/exposure accounting)
  rather than the self-contained intraday engine.

### 4.4 Expressions ✅ (shipped with 3.4 in `57bc79a`)

- The Pine options join (`options/backtest.py`) gained the Phase 3.4 intraday
  fields; `referenced_options_fields` picks them up automatically so rolling
  configs can condition on e.g. `iv_change_intraday > 0.05`.

---

## Phase 5 — Validation and guardrails

- **Offline-stub tests throughout** (repo convention, 90% coverage floor):
  synthetic 1m sessions, synthetic chain snapshots with known greeks, fixture
  recorders that replay canned CBOE/NSE payloads.
- **Golden tests**: every phase must keep daily-interval equity backtests and
  the existing EOD options backtests byte-identical. This gate already exists
  for Phase 1 — keep extending it.
- **PIT audits**: property-style tests asserting no lookahead — for every fill
  in an intraday options run, all inputs' timestamps ≤ decision timestamp.
- **Recorder health**: `screener cache status` gains store freshness (last
  snapshot age per market) and gap detection (missing sessions in the 1m
  archive / contract store), so a silently dead cron is noticed within a day.
- **Storage watch**: alert (log line the cron surfaces) when the bar store or
  contract store exceeds a configured size budget.

---

## Sequencing

1. ~~**Now**: install the Phase 1 recorder cron~~ — done; the 1m archive grows
   daily.
2. ~~**Phase 3.2 recorder next**~~ — done; the options snapshot recorder
   (`options record`) captures chains into the contract store, cron installed.
3. ~~Phase 2 (intraday screener), 3.4 (derived panels), 4.1 (EOD realism), and
   4.2/4.3 (intraday options backtesting + mixed hedge)~~ — all shipped
   (`b8bc895`, `57bc79a`, `26879a4`, `b35dbc1`). Every EOD/daily golden path
   stayed byte-identical; the new code is additive and offline-tested.
4. **Remaining open work** (all non-blocking, noted inline above): between-snapshot
   bracket resolution against 1m bars (4.2), full `day_loop` structure-slot
   integration for mixed portfolios (4.3), and Phase 5's storage-size watch.
5. The one open decision: whether a paid US options-history provider
   (Polygon ~$29–199/mo) is on the table. It determines whether US intraday
   options backtests cover the past or only capture-start forward. India is
   capture-forward regardless. The `PolygonOptionsHistoryProvider` /
   `ThetaDataOptionsHistoryProvider` stubs are the drop-in seam when it is.
