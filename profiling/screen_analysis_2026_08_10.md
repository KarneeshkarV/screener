# Screen command - deep bottleneck analysis (2026-08-10)

**Branch:** `analysis/backtest-findings` @ `e63598e` (on main with #127/#128)
**Default command:** `screener screen -m us -c ema -n 50` (sort=`setup_score`)
**Method:** phase timers, cProfile, `-X importtime`, over-fetch rank A/B on live TV data,
Turso on/off comparison.

---

## 0. Punchline

The screen **engine** is already fast. The default shell experience is slow for
three reasons that are almost entirely **outside** `scan()` scoring math:

| # | Bottleneck | Where | Warm CSV impact |
|---|---|---|---:|
| 1 | **Eager CLI imports** of every subcommand | `screener/cli.py` | **~500-650 ms** |
| 2 | **Synchronous Turso usage tracking** (2 remote writes after every command) | `screener/usage.py` via Click wrapper | **~350-500 ms** |
| 3 | **Always-on Plotly HTML report** (non-CSV only) | `screen_workflow` → `render_screen_report` | **~180 ms** + 4.7 MB file |

Only on **cache miss** does the real product work matter:

| # | Bottleneck | Where | Cold impact |
|---|---|---|---:|
| 4 | **TradingView HTTP** for 500 volume-pre-sorted rows | `scanner` → `CachedProvider` | **~500-600 ms** |

Everything else (score, dedupe, Rich table, sqlite history) is **&lt;40 ms**.

```
Warm CSV wall (this machine, .env has TURSO_*):
  with Turso:    ~1.13 s
  without Turso: ~0.66 s   (cwd without .env)
  workflow only: ~0.01 s
```

---

## 1. End-to-end call graph

```
.venv/bin/screener screen ...
  import screener.cli                          # EAGER: all commands
    → factor_tearsheet, optimize, backtest_*,
      conviction, yfinance, options, plotly,
      unusual_volume, earnings_backtest, ...
  cli() group setup + agentio + logging
  UsageTrackedGroup wraps every leaf callback
  screen() Click callback
    ScreenRequest(...)
    run_screen_workflow(request)
      resolve_criteria(("ema",))               # 4 TV filters
      resolve_scorer(("ema",))                 # ScoreSpec + helper cols
      scan(...)
        build_scanner_plan
          columns = DEFAULT(6) + EMA helpers(5) = 11
          fetch_limit = max(50*10, 500) = 500
          query_order_by = "volume"            # NOT setup_score
        TradingViewScannerAdapter.fetch
          Query.set_markets/select/where/order/limit
          cache key = (scanner, market, sorted filters,
                       columns, order_by, fetch_limit)
          CachedProvider.fetch
            HIT  → read parquet + JSON meta     ~2-13 ms
            MISS → rate-limit(2 rps) + retry
                   Query.get_scanner_data()     ~500-600 ms
                   write parquet (~60 KB)
        shape_scan_results
          apply_score (ema weights)             ~4-5 ms
          sort setup_score desc
          drop helper cols
          _dedupe_listings (Python re.sub map)  ~2-5 ms
          head(limit)
      [opt] enrich_days_to_earnings             # can be SECONDS
      if --csv: return
      save_run / previous_run / diff            ~30 ms sqlite
      render_screen_report ALWAYS               ~180 ms, 4.7 MB
        px.histogram + px.bar
        get_plotlyjs() inlined (~4.7 MB)
        table_html
    print_results / print_csv                   ~1-30 ms
  finally: usage.record_feature_usage           ~170-250 ms Turso
           usage.record_feature_invocation      ~200-270 ms Turso
```

---

## 2. Waterfall budgets (measured)

### 2.1 Warm CSV (`--csv`, cache hit)

| Phase | ms | Notes |
|---|---:|---|
| Process start | 10-30 | python / uv |
| `import screener.cli` | **500-650** | see §3 |
| `resolve_*` + plan | &lt;1 | |
| cache parquet read | 2-13 | 500×11 frame, ~60 KB |
| score + sort + dedupe + head | 7-13 | §5 |
| `print_csv` | ~1 | |
| Turso usage ×2 | **350-500** | §4; only if `.env` has creds |
| **Total shell** | **~1100-1400** | |

### 2.2 Warm full table (default, no `--csv`)

| Phase | ms |
|---|---:|
| Import | 500-650 |
| scan warm | ~10 |
| `save_run` + diff | ~30 |
| **`render_screen_report`** | **~180** |
| `print_results` (Rich) | ~30 |
| Turso ×2 | 350-500 |
| **Total shell** | **~1100-1500** |

Report file size: **~4.7 MB** almost entirely from inlined `get_plotlyjs()`.

### 2.3 Cold CSV (`--refresh`)

| Phase | ms |
|---|---:|
| Import | 500-650 |
| **TV HTTP (500 rows)** | **~550** |
| shape | ~8 |
| Turso ×2 | 350-500 |
| **Total shell** | **~1500-1800** |

Network share of **workflow** cold path: **~96%**.
Network share of **shell** cold path: ~30-40% (imports + Turso still dominate).

### 2.4 Live measured shells

| Command | Wall |
|---|---:|
| workflow CSV warm (in-process) | **9 ms** |
| workflow full warm | **205 ms** |
| CLI warm CSV **with** Turso `.env` | **1.13 s** |
| CLI warm CSV **without** Turso (no `.env` in cwd) | **0.66 s** |
| `screener --help` | **0.70 s** |
| import `screener.screen_workflow` alone | **0.59 s** |
| import `screener.cli` | **0.65 s** |

`--help` at 0.70 s proves most of the tax is paid **before any screen logic**.

---

## 3. Import tax - what `screen` pays for

`screener/cli.py` eagerly imports **every** command at module load:

```python
from screener.backtester.factor_tearsheet import factor_tearsheet
from screener.backtester.historical import backtest_historical
from screener.backtester.optimization.cli import optimize
from screener.backtester.rolling import backtest_rolling
from screener.commands.conviction import conviction
...
from screener.options.cli import options
from screener.unusual_volume.cli import unusual_volume
```

`-X importtime` top cumulative modules when loading `screener.cli`:

| Module | cum ms | Why screen does not need it |
|---|---:|---|
| `screener.cli` total | ~528 | |
| `pandas` | ~173 | needed, but pulled early by many cmds |
| `screener.backtester.factor_tearsheet` | ~77 | unrelated |
| `screener.commands.conviction` / yfinance | ~66 | unrelated |
| `screener.screen_workflow` → report → **plotly** | ~55-57 | only for HTML report |
| `screener.usage` | ~42 | Turso client import path |
| `screener.backtester.optimization.cli` | ~26 | unrelated |
| `screener.backtester.historical` | ~28 | unrelated |

### Fix (P0)

Lazy Click registration. Patterns that work with current Click:

1. **`click.Command` with `callback` imported inside a wrapper**, or
2. **string-based lazy loaders** (Click 8.1+ `lazy_loader`), or
3. **import inside `add_command` factory** per subcommand module.

Target: `screener screen` imports only:

- `commands.screen` → `screen_workflow` → `scanner` / `criteria` / `scoring` / `cache` / `providers`
- `display` (Rich)
- `history` only on non-CSV
- `screen_report` + plotly **only if report requested**

Expected: import **~550 ms → ~150-250 ms** for CSV path (pandas + scanner still required).

---

## 4. Turso usage tracking - the hidden 400 ms

Every leaf command is wrapped by `UsageTrackedGroup` → `_instrument_usage_tracking`.
On success **and** failure paths, the `finally` block does:

1. `usage.record_feature_usage(...)` → `_connect()` + `ensure_usage_table` + INSERT
2. `usage.record_feature_invocation(...)` → **another** `_connect()` + `ensure_invocations_table` + INSERT

`_connect()` reads `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` from env **or `.env`**,
then `libsql_client.create_client_sync` over HTTPS.

Measured (this host, real Turso):

| Call | ms |
|---|---:|
| `_connect()` | ~85 |
| `SELECT 1` | ~193 |
| `record_feature_usage` | ~172-243 |
| `record_feature_invocation` | ~208-266 |
| **pair total** | **~420-670** |
| pair with Turso disabled | **0.0** |

There is **no connection reuse** across the two record calls.
There is **no async / background / fire-and-forget**.
There is **no local buffer**.
Each `screener` process pays **two cold remote round-trips after the user already has their results**.

### Fixes (P0)

| Option | Impact | Risk |
|---|---|---|
| Single shared client per process; one combined write | −100-200 ms | low |
| Background thread / `atexit` flush | hide latency from user | medium (lost on kill) |
| Local sqlite queue + periodic ship | hide latency, offline ok | medium |
| Opt-out env `SCREENER_USAGE=0` | instant disable | low |
| Skip usage in agent mode | agents are chatty | low |
| Don't `ensure_*_table` every call | −1 RTT worth of DDL | low |

**Biggest single shell win after lazy imports:** make usage async or batched.

---

## 5. Scan / shape path (already healthy)

### 5.1 Default plan (`-c ema -n 50 --sort setup_score`)

```
filters:  EMA5>EMA20, EMA20>EMA100, EMA100>EMA200, EMA200>0
columns:  name, description, close, change, volume, market_cap_basic,
          EMA5, EMA20, EMA100, EMA200, RSI          (11)
fetch_limit: 500
query_order_by: volume          # TV sorts by volume; we re-rank by score
total_matches (live): 2693
cache parquet: ~60 KB
```

### 5.2 Shape breakdown on 500 rows (warm)

| Step | ms |
|---|---:|
| `add_setup_score` / ema weights | 3.9-5.1 |
| `sort_values` | 0.3-0.5 |
| drop helper columns | 0.3-0.4 |
| `_dedupe_listings` (row-wise `re.sub`) | 2.2-4.6 |
| `head(50)` | &lt;0.1 |
| **shape total** | **~7-11** |

Scoring is **not** a bottleneck. Dedupe is the only shape item worth a tidy-up
(vectorized `str.replace` / `str.lower`), for a couple of milliseconds.

### 5.3 Cache key design

```python
key_parts = (
    "scanner",
    plan.market,
    sorted(repr(f) for f in plan.filters),  # order-independent
    plan.columns,
    plan.order_by,       # client sort name, not query_order_by
    plan.fetch_limit,
)
```

Observations:

- `-n 10` and `-n 50` **share** the same key (both clamp to fetch 500). Good.
- `-n 100` becomes fetch 1000 → **different key** → cold miss. OK.
- `--detail` adds columns → different key → cold miss. Expected.
- `--sort volume` uses fewer columns + fetch 150 → different key.
- `order_by` is in the key even though for `setup_score` the **wire** order is
  always `volume`. Harmless today; would matter if you cached a raw TV payload
  once and re-sorted locally for multiple orderings.

TTL default: **15m**. Stale-serve on provider failure is wired through
`CachedProvider`.

### 5.4 Resilience

TradingView rate limit: **2 rps**. Retry: 3 attempts, 0.5 s base backoff.
On a single cold call this rarely matters; on bursts (agents, scripts) it will
serialize.

---

## 6. Over-fetch and ranking quality

TV returns the top `fetch_limit` rows by **volume** among filter matches
(`total_matches ≈ 2693` for ema). Local score then picks top 50.

Simulated: take volume-top-N from the cached 500, re-score, compare top-50 names
to the "oracle" from full 500:

| Volume pre-fetch N | Top-50 overlap with N=500 oracle |
|---:|---:|
| 50 | **20%** |
| 100 | 38% |
| 150 | 52% |
| 200 | 66% |
| 250 | 76% |
| 300 | 82% |
| 400 | 90% |
| 500 | 100% |

**Conclusion:** the 500 floor is doing real ranking work. Dropping to 100 or
150 would **materially change** results (half the names differ). This is **not**
a free perf win.

Reasonable experiments (need product sign-off):

| Policy | Overlap | Cold payload |
|---|---:|---|
| keep 500 (current) | 100% | baseline |
| 400 | 90% | −20% rows |
| 300 | 82% | −40% rows |
| score-aware TV sort if TV supports expression sort | n/a | ideal |

Cold time is dominated by **RTT + TLS + server**, not JSON parse of 500 vs 300
rows. Cutting rows may save only **tens of ms**, not 2×. Measure before changing.

Better cold wins than shrinking N:

1. **Warm cache proactively** (cron / bot heartbeat every 10-14 min).
2. **HTTP keep-alive / session reuse** if `tradingview-screener` allows.
3. **Don't put `fetch_limit` quality floor at 500 when limit is tiny** only if
   product accepts worse ranks for `-n 10` explorers.

---

## 7. Report path (non-CSV)

Always runs when not `--csv`, even without `--report`:

```python
generated_report = request.report_path or temp_report_path("screen")
render_screen_report(...)
```

### Sub-costs (warm, 50 rows)

| Step | ms | Size |
|---|---:|---:|
| `px.histogram` | ~16 | |
| `figure_html` hist | ~11 | ~9 KB |
| `px.bar` | ~16 | |
| `figure_html` bar | ~11 | |
| **`get_plotlyjs()`** | ~8 | **~4737 KB** |
| `table_html` | ~4 | ~11 KB |
| write file + html_page | rest of ~180 | **~4770 KB total** |

Plotly JS is **inlined into every report**. That is why reports are 4.7 MB and
why import of `screen_report` pulls plotly into the default CLI graph.

### Fixes (P0/P1)

1. **Opt-in report:** only render when `--report` / `--open-report`.
2. Or default to **table-only HTML** (no Plotly) for the auto temp report.
3. Or **CDN link** / shared external `plotly.min.js` instead of inline.
4. Lazy-import plotly inside `render_screen_report` so CSV path never loads it
   (still need lazy CLI for full effect).

---

## 8. History path

`save_run` ~30 ms for 50 rows into `~/.screener/history.db` (sqlite WAL).
Uses `df.iterrows()` to build insert tuples - fine at n=50; would matter at
thousands.

`previous_run` + `diff` are ~1-2 ms.

Not a priority.

---

## 9. Earnings path (`--earnings` / `--earnings-buffer`)

Opt-in. For US:

```python
# fetch_next_earnings_dates → ThreadPoolExecutor
# per symbol: yfinance earnings_dates (cached per ticker)
```

Earlier measurement: **~4.0 s** for 50 symbols on a cold-ish calendar.
That is **400×** the warm scan. When users enable it, it becomes the only
story.

Fixes: stronger multi-day cache, bulk calendar provider, higher worker count
with rate-limit awareness, show partial results without blocking the table.

---

## 10. What the user feels vs what the engine does

```
User types:  screener screen -c ema -n 50

 Reality of a warm run with Turso .env:
 ┌─────────────────────────────────────────────────────────────┐
 │ ████████████████ import cli (~55%)                          │
 │ ████████████ Turso usage x2 (~40%)                          │
 │ █ scan+shape+print (~2%)                                    │
 └─────────────────────────────────────────────────────────────┘
          0.01s of actual screening work
```

```
Same command with --refresh (cold TV):
 ┌─────────────────────────────────────────────────────────────┐
 │ ████████████ import (~35%)                                  │
 │ ████████████ TV network (~35%)                              │
 │ ████████████ Turso (~30%)                                   │
 │ ▏ shape (~1%)                                               │
 └─────────────────────────────────────────────────────────────┘
```

---

## 11. Prioritized speedups

### Tier 0 - do these first (shell UX)

| ID | Change | Expected save | Risk |
|---|---|---:|---|
| S0 | **Async / batched / single-connection Turso usage** | 350-500 ms | low-med |
| S1 | **Lazy CLI command imports** | 300-450 ms | low |
| S2 | **Report opt-in** (or no Plotly by default) | 180 ms on table path | low (UX) |

Combined on warm CSV: **~1.13 s → ~0.15-0.25 s** is realistic.

### Tier 1 - cold / quality

| ID | Change | Expected save | Risk |
|---|---|---:|---|
| S3 | Cache warmer / longer TTL for interactive use | avoid 550 ms | low |
| S4 | Keep-alive TV session | tens of ms | low |
| S5 | Over-fetch A/B (400?) with quality metrics | small cold | **high** (rank) |
| S6 | Earnings bulk cache | seconds when enabled | med |

### Tier 2 - polish

| ID | Change | Expected save |
|---|---|---:|
| S7 | Vectorize `_dedupe_listings` | 2-4 ms |
| S8 | `save_run` without `iterrows` | small |
| S9 | Drop `order_by` from cache key; cache raw TV frame | fewer misses |
| S10 | Lazy plotly import inside report only | import graph cleanup |

### Do not bother yet

- GPU for scoring
- Rewriting ema score math
- Micro-optimizing Rich table
- Cutting fetch to 100 (destroys rank quality - measured)

---

## 12. Suggested implementation order

1. **Usage fire-and-forget** (or process-level client + one write) - pure win, no UX change.
2. **Lazy CLI imports** - pure win for every command, not just screen.
3. **`--report` gate** - confirm product wants default table without HTML.
4. Optional: `SCREENER_USAGE=0` and document Turso cost in README.
5. Re-measure CLI wall; only then touch fetch_limit / TV.

### Acceptance benchmarks (after S0-S2)

| Scenario | Today | Target |
|---|---:|---:|
| Warm CSV shell | ~1.13 s | **&lt;0.25 s** |
| Warm table shell | ~1.2-1.5 s | **&lt;0.35 s** |
| Cold CSV shell | ~1.7 s | **&lt;0.9 s** (import+TV only) |
| Workflow CSV warm | ~10 ms | keep |
| Top-50 rank vs current 500-oracle | 100% | keep ≥90% if fetch changes |

---

## 13. Reproduction

```bash
# Workflow phases
.venv/bin/python profiling/_analysis/screen/  # profiles already under here

# Turso on/off
/usr/bin/time -f '%e' .venv/bin/screener --log-level ERROR screen -m us -c ema -n 50 --csv
cd /tmp && /usr/bin/time -f '%e' /path/to/.venv/bin/screener --log-level ERROR screen -m us -c ema -n 50 --csv

# Import
.venv/bin/python -X importtime -c "import screener.cli" 2>importtime.txt
```

Artifacts:

- `profiling/_analysis/screen/screen_{warm,cold,full}.prof`
- `profiling/_analysis/screen/importtime_cli.txt`
- `profiling/_analysis/screen_codepath_findings.md` (structural companion)

---

## 14. Bottom line

`screen` the **data product** is fine: ~10 ms warm, ~550 ms cold network.

`screener screen` the **CLI product** is slow because every invocation:

1. Imports the entire trading platform (~0.55 s),
2. Optionally writes two remote Turso rows (~0.45 s),
3. And on the default table path, builds a 4.7 MB Plotly report nobody asked for (~0.18 s).

Fix those three and the command will feel instant on warm cache, and cold will
be honest network time only.
