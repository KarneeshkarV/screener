---
description: Analyze journal.jsonl from the autoresearch loop — rank strategies by score and show what survived
allowed-tools: Bash, Read, Edit, Write, Grep
argument-hint: "[--top=N] [--market=us|india]"
---

You are analyzing `/root/screener/journal.jsonl` — the per-iteration log of the Karpathy-style autoresearch loop. Each row is one (strategy, iteration) result with IS/OOS metrics, OOS score = `dsr * exp(-|max_drawdown| / 0.25)`.

User args (may be empty): `$ARGUMENTS`

## Steps

1. **Parse the journal** — read every row, group by `strategy` key, keep the row with the highest `score` per strategy. Note unique strategy count and total iterations.

2. **Apply quality filter** — call a strategy "profitable" if its best OOS run satisfies *all* of:
   - `oos.total_return > 0`
   - `oos.sharpe > 0.5`
   - `score > 0.2`
   - `oos.trade_count >= 5`

3. **Print the table** — strategy | best score | OOS return | OOS Sharpe | OOS DD | trade count | iter#. Sort by score desc. If `--top=N` is in `$ARGUMENTS`, cap to N rows; default 50.

4. **Save the profitable list** to `/tmp/good_strategies.json` — array of strategy keys (`stock:foo` / `new:bar`). This is the input contract for `/best_buys`.

5. **Reflect briefly** — call out:
   - Best 3 strategies and why they likely won (read their `hypothesis` fields)
   - Any "template clones" — runs of consecutive iterations with the same recipe (e.g. "oscillator zero-cross + SMA50>SMA200 + EMA20 exit") that all scored similarly. These are local-optimum dead ends.
   - Whether the loop seems to have plateaued (top score not improving over the last ~20 iterations)
   - Any iterations marked with low score / negative return (failed mutations)

6. **Output** the absolute path to `/tmp/good_strategies.json` and the row count saved.

Use a single `python3 <<'EOF' ... EOF` block via Bash for parsing — don't pull pandas if `json` + `Counter` is enough. Don't run the loop, don't refetch OHLCV, don't launch agents — this command is read-only against the journal.
