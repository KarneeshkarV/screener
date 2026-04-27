---
description: Run every profitable strategy on today's US + India universes and produce a synthesized buy recommendation
allowed-tools: Bash, Read, Write, Edit, Agent, Monitor, WebSearch, WebFetch
argument-hint: "[--lookback=7] [--limit=500] [--no-research]"
---

You are running the full picks-of-the-day pipeline. The user wants: aggregated buy frequency across all profitable strategies, then sourced research on the top names.

User args (may be empty): `$ARGUMENTS`

## Pipeline

### 1. Ensure the strategy list exists

If `/tmp/good_strategies.json` does not exist OR is older than 1 day, invoke `/analyze_journal` first (or run the same logic inline against `/root/screener/journal.jsonl`). The list is the input contract for the scanner.

### 2. Run the scanner

Run `/root/screener/scan_today.py` (already in the repo). It:
- Loads `/tmp/good_strategies.json`
- Pulls today's OHLCV via TradingView screener + yfinance for both `us` and `india` (cap = `--limit`, default 500)
- For each (strategy, ticker), records entries within the last `--lookback` days (default 7)
- Counts strategy-frequency per ticker
- Writes `picks_today.json` + `picks_today.md` to the repo root

Run with **unbuffered logs** (the python file's own `tail` redirection blocks output) — invoke as:

```
PYTHONUNBUFFERED=1 uv run python scan_today.py > /tmp/scan_stdout.log 2> /tmp/scan_stderr.log &
```

Then arm a `Monitor` on `/tmp/scan_stderr.log` filtering `fetched|usable|strategy [0-9]+/|wrote picks|Traceback|Error` so you get progress events without polling. The full run takes ~6–10 min.

When the monitor emits "wrote picks_today.json", read the top 10 from `/tmp/scan_stdout.log` for each market.

### 3. Launch research subagents (skip if `--no-research`)

Pick the top **6 US** + top **6 India** by frequency. Spawn **4 `general-purpose` Agents in parallel** (single message, multiple Agent tool calls — not sequential):

- **US #1–3** — usually the highest-conviction names; ask for catalysts, recent earnings, analyst PTs, short interest, earnings dates within 30 days, red flags
- **US #4–6** — same template
- **India #1–3** — same template, but bias toward Indian financial sources (Moneycontrol, Economic Times, Livemint, BSE/NSE filings); be aware of fiscal year ending March
- **India #4–6** — same

Each subagent prompt should:
- State the closing price on the latest bar (read from `picks_today.json`)
- State the strategy-frequency count (it's the consensus signal — explain WHY the agent should believe it)
- Ask for under-200-words-per-name verdicts with sources cited
- End each name with a one-line "would you buy at this price" line

Run them with `run_in_background: true` so the user gets per-agent completion notifications.

### 4. Synthesize

When all four reports are back, write `/root/screener/buy_recommendations.md` with three tiers:

1. **Tier 1 — full position (5%)** — clean fundamentals, no near-term binary risk, modest upside to consensus PT, manageable short interest
2. **Tier 2 — half size or wait** — strong setup but has a wrinkle (CEO selling, above PT, binary earnings within ~7 days)
3. **Tier 3 — dividend arb / one-trade** — buying for an event, not a position
4. **AVOID** — the names that the research disqualified

Include for each Tier 1/2 row: ticker, market, close, why-now, stop, target.

End with a single "if forced to pick one" line — the cleanest risk/reward.

### 5. Final report

Tell the user:
- The three artifacts: `picks_today.json`, `picks_today.md`, `buy_recommendations.md`
- The top headline finding (e.g. "US sweep is dominated by oil services — Brent at $X is the catalyst")
- The single best buy

## Hard rules

- **Do not** invent prices or earnings dates. If a subagent can't source it, the verdict says "wait for confirmation."
- **Do not** run the autoresearch mutation loop — this command is forward-only inference, no new strategies are discovered.
- **Do not** kill the scanner before completion. The `tail -120` redirection at the end of any pipeline buffers all output until exit — if you don't see progress, that's the buffer, not a hang. Use `PYTHONUNBUFFERED=1 ... > log 2> err &` and monitor the log files directly.
- If `--no-research` is in `$ARGUMENTS`, stop after step 2 and just print the top-25 tables; don't launch agents.
- Mark each step with TaskCreate / TaskUpdate so the user can see progress.
