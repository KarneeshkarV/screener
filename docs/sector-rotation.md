# Sector and industry rotation

`sector-rotation` compares selected Indian sector, industry, and theme baskets with Nifty 500.
It is a research command, not an entry signal or a backtest strategy.
It does not change existing `screen` results or `--sector-neutral` behavior.

## Run it

Install the price extra and list groups without a network request:

```bash
uv sync --extra prices
uv run screener sector-rotation --list-groups
```

Generate a report for every bundled group:

```bash
uv run screener sector-rotation --period 3m --report rotation.html
```

Select groups by their exact IDs, using `--group` more than once:

```bash
uv run screener sector-rotation \
  --group cnc_machines --group shipbuilding --group cables \
  --period 1y --as-of 2026-10-16 \
  --report rotation.html --members-csv rotation-members.csv --csv
```

Supported periods are `1d`, `1w`, `1m`, `3m`, `6m`, `1y`, `3y`, and `5y`.
They are calendar windows measured back from the last benchmark session on or before the requested end.
Each start anchors at the last benchmark session on or before that calculated start date.
For example, a one-day request ending on Sunday compares Friday with Thursday.
A one-day request ending on a trading Monday compares Monday with Friday.
The default end date is yesterday, which avoids incomplete same-day exchange sessions.
Explicit `--as-of` dates must also be before today.
The benchmark defaults to the provider symbol `^CRSLDX`, the Nifty 500 price index.
Use `--benchmark SYMBOL` to replace it explicitly.
The command fails if the benchmark does not cover the start or its endpoint is more than seven calendar days stale.
A successful fetch can still contain unranked groups if member data is incomplete.

`--csv` emits all group metrics on stdout.
Warnings and output-file paths go to stderr.
`--members-csv` writes stock return, coverage, and moving-average readings for each group membership.
Agent mode limits the displayed rows and writes the full metrics to a spill CSV.

## Catalogue and provenance

The bundled catalogue is `screener/sector_rotation/sector_groups.yaml`.
Each group has a stable ID, display name, parent label, kind, video timestamp, selected NSE symbols, and optional exposure notes.
Groups can overlap.
A parent is a classification label, not a declaration that the parent basket contains every child member.
The schema accepts sectors, industries, and themes without forcing themes into exclusive sector labels.
The current release supports India and NSE symbols only.

The label source is Financially Free's [Don't Miss These Sectors Before They Explode](https://www.youtube.com/watch?v=d-I6BG5F-Yg).
The video advertises 160 groups but does not disclose all their names or complete constituent lists.
The bundled catalogue covers identifiable examples, not the full proprietary catalogue.
The transcript was unavailable during implementation, so group references come from visible diagrams and dashboard frames.
Each `video_at` field links a group to an approximate point in that footage.
The video's index formula has not been verified.
Our monthly equal-weight calculation is an explicit independent choice.

The selected stocks are manually curated research examples, not a transcription of the tool's membership.
Some companies have several business activities and appear in more than one group.
Small or diversified baskets are not reliable pure-play industry indices.
All bundled member symbols were checked against the live TradingView NSE stock catalogue on 2026-10-20.
That check verifies symbol availability, not complete industry membership, future provider coverage, or investment suitability.
Private companies shown in the video are not assigned fabricated exchange symbols.
Review membership and business exposure before making investment decisions.

To add groups, copy the bundled YAML and pass a complete replacement:

```bash
uv run screener sector-rotation --catalogue my-sector-groups.yaml --list-groups
uv run screener sector-rotation --catalogue my-sector-groups.yaml --report my-rotation.html
```

Group IDs must be unique lowercase snake_case.
Member lists must contain unique TradingView-style NSE symbols, such as `NSE:JYOTICNC`.
The catalogue records its observation date and states that membership is `curated_research`.
Unknown IDs, duplicate members, invalid symbols, and unsupported configuration fields fail validation.
The catalogue has no point-in-time membership interface in this release.

## Return calculation

1. Fetch adjusted stock closes and benchmark closes, with 400 additional calendar days for moving-average warmup.
2. Use the benchmark's sessions as the common calendar.
3. Rebase every basket and the benchmark to 100 on the same anchor date.
4. Allocate equal capital to quoted members at the anchor.
5. Let holdings drift between month-end rebalances.
6. Reset to equal weights at the prior month's final available benchmark close, for the next month's first session.

No future prices determine rebalance weights.
Newly quoted members can enter only at a subsequent rebalance.
Missing held prices stop the remainder of that basket curve.
The calculation does not forward-fill prices, replace missing returns with zero, or silently sell a missing member.
Invalid, non-finite, and non-positive closes count as missing.
Missing benchmark sessions where a selected stock traded cause an explicit error, rather than silently changing the rebalance calendar.
Where volume is available, zero-volume stock rows do not count as evidence of a session.
This excludes provider holiday placeholders.
This check uses member-session evidence, not an independent exchange holiday calendar.
A session absent from the benchmark and every selected stock cannot be detected by that check.
Invalid benchmark warmup closes outside the return window do not block the report.
`--minimum-coverage` defaults to `0.8` and sets the required fraction of selected members quoted on each session.
A basket that fails initial coverage does not start late and disguise missing history.
An incomplete basket remains visible but unranked.
The report shows both minimum session coverage and the final priced-member count.
Baskets with fewer than three selected members carry a concentration notice.
Transaction costs are not included.

The summary reports separate measures:

- `return_pct`: compounded basket return, in percent.
- `benchmark_return_pct`: benchmark return across the same window.
- `excess_return_pp`: basket return minus benchmark return, in percentage points.
- `relative_strength_pct`: change in the basket-to-benchmark wealth ratio, in percent.
- `breadth_sma50_pct` and `breadth_sma200_pct`: percentage of members above the respective daily moving average, among members with a valid reading.
- `breadth_sma50_members` and `breadth_sma200_members`: the denominators for those breadth readings.

Unknown breadth does not count as below the moving average.
Ranks use descending excess return and leave unavailable groups unranked.
Equal excess returns share a rank.

## Limits

Historical charts use today's selected companies and therefore have survivorship and selection bias.
An observation date is not a historical membership effective date.
Do not connect these results to historical trading gates until dated membership and publication-time rules exist.
Adjusted stock returns include distributions, while the default Nifty 500 price index excludes them.
The report states this mismatch and must not be presented as a total-return benchmark comparison.
Official sector-index history and business-specific metrics such as same-store sales growth are not part of this release.

The offline HTML report has no CDN, external fonts, or chart-library dependency.
It provides group search, parent filtering, benchmark charts, and expandable member tables.
The same calculation result supplies the CSV and HTML report.
