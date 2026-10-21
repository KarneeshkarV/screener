# Sector and industry rotation

`sector-rotation` compares curated sector, industry, and theme baskets for India or the US with a market benchmark.
India uses Nifty 500 and the US uses the S&P 500 by default.
Each market has a separate catalogue and ranks only against its own benchmark.
It is a research command, not an entry signal or a backtest strategy.
It does not change existing `screen` results or `--sector-neutral` behavior.

## Run it

Install the price extra and list groups without a network request:

```bash
uv sync --extra prices
uv run screener sector-rotation --list-groups
uv run screener sector-rotation --market us --list-groups
```

Read business descriptions, drivers, risks, selected members, and source links without fetching prices:

```bash
uv run screener sector-rotation --group cnc_machines --describe-groups
uv run screener sector-rotation --market us --describe-groups
```

Use either `--list-groups` or `--describe-groups`, not both.
The compact `--list-groups` output keeps its original tab-separated format.
In agent mode, `--describe-groups` prints a bounded preview and a `full:` path to a CSV containing every selected group's details, benchmark and membership warning.
`--agent-detail summary` shows only the count and file path; `head` shows the first five groups; `full` prints all details and still writes the CSV.
Use the global `--no-agent` option to get full human-readable descriptions without a spill file.
Legacy catalogues omit missing descriptions, drivers and risks instead of printing placeholders.

Generate a report for every bundled group:

```bash
uv run screener sector-rotation --period 3m --report rotation-india.html
uv run screener sector-rotation --market us --period 3m --report rotation-us.html
```

Select groups by their exact IDs, using `--group` more than once:

```bash
uv run screener sector-rotation \
  --group cnc_machines --group shipbuilding --group cables \
  --period 1y --as-of 2026-10-16 \
  --report rotation.html --members-csv rotation-members.csv --csv
```

`--period` takes a positive whole number and a unit: `d` days, `w` weeks, `m` months, or `y` years.
Examples are `10d`, `2w`, `9m`, `12m`, `18m`, and `5y`.
Month and year periods use calendar months, so `9m` ending on 6 January starts on 6 April.
They are calendar windows measured back from the last benchmark session on or before the requested end.
Each start anchors at the last benchmark session on or before that calculated start date.
For example, a one-day request ending on Sunday compares Friday with Thursday.
A one-day request ending on a trading Monday compares Monday with Friday.
The default end date is yesterday, which avoids incomplete same-day exchange sessions.
Explicit `--as-of` dates must also be before today.
The benchmark defaults to the provider symbol `^CRSLDX`, the Nifty 500 price index, for India.
For the US it defaults to `^GSPC`, the S&P 500 price index.
Returns stay in each market's local currency.
The command does not convert INR and USD or compare cross-market ranks.
Use `--benchmark SYMBOL` to replace it explicitly.
The command fails if the benchmark does not cover the start or its endpoint is more than seven calendar days stale.
A successful fetch can still contain unranked groups if member data is incomplete.
Prices come from yfinance first.
When `FMP_API_KEY` is set, FMP is the fallback, always for the entire ticker history and never spliced into one series.
A ticker that yfinance returns empty is fetched from FMP and named in a warning.
A provider can also omit a benchmark session while returning stock prices for that day.
For example, the yfinance Nifty 500 feed omits 2026-01-01 while the member stocks have prices.
The command then tries the FMP benchmark and uses it for the entire benchmark if it has every missing session.
Otherwise the benchmark's last close is carried forward on those sessions, so the benchmark has zero return on them.
Every fallback and carried-forward session is printed as a CLI warning and shown in the HTML report notice.

`--csv` emits all group metrics on stdout.
It also includes `market`, `description`, `drivers`, `risks`, `notes`, and `sources` so exports keep the business context.
List-valued metadata uses semicolon-separated text in those columns.
Warnings and output-file paths go to stderr.
`--members-csv` writes stock return, coverage, and moving-average readings for each group membership.
Agent mode limits the displayed rows and writes the full metrics to a spill CSV.

## Catalogue and provenance

The India catalogue is `screener/sector_rotation/sector_groups.yaml`.
The US catalogue is `screener/sector_rotation/sector_groups_us.yaml`.
India has 106 groups, including 41 additions to the original 65 video-backed labels.
The US has 46 groups, including all 11 broad sectors, 32 industry baskets, and 3 themes.
The catalogues contain 320 unique India stocks and 152 unique US stocks.
Each group has a stable ID, display name, parent label, kind, selected exchange-qualified symbols, and optional exposure notes.
Both bundled catalogues also provide a business description, drivers, risks, and primary source links for each group.
These qualitative details explain exposure, not forecasts or quantitative trading signals.
A video timestamp is optional and applies only to an original video-backed label.
Groups can overlap.
A parent is a classification label, not a declaration that the parent basket contains every child member.
New India sector baskets share a parent filter label with their related industries, including Banks, Information technology and Automobiles.
Original India parent labels remain unchanged.
The schema accepts sectors, industries, and themes without forcing themes into exclusive sector labels.
India members use `NSE:` symbols.
US members use `NASDAQ:`, `NYSE:`, or `AMEX:` symbols.
An India catalogue cannot contain US members, and a US catalogue cannot contain NSE members.
The command rejects a custom catalogue whose market does not match `--market`.

The original India label source is Financially Free's [Don't Miss These Sectors Before They Explode](https://www.youtube.com/watch?v=d-I6BG5F-Yg).
The video advertises 160 groups but does not disclose all their names or complete constituent lists.
The bundled catalogue covers identifiable examples, not the full proprietary catalogue.
The transcript was unavailable during implementation, so group references come from visible diagrams and dashboard frames.
Each original `video_at` field links a group to an approximate point in that footage.
The expanded India groups and all US groups are independent research additions with no claimed video timestamp.
The video's index formula has not been verified.
Our monthly equal-weight calculation is an explicit independent choice.

The selected stocks are manually curated research examples, not a transcription of the tool's membership.
Some companies have several business activities and appear in more than one group.
Small or diversified baskets are not reliable pure-play industry indices.
The original India release checked its member symbols against the live TradingView NSE stock catalogue on 2026-10-20.
That check does not cover every later addition.
See [India source notes](sector-catalogue-india-sources.md) and [US source notes](sector-catalogue-us-sources.md) for the expansion's evidence and check limits.
Source links document business exposure or classification context, not official index membership, future provider coverage, or investment suitability.
Private companies shown in the video are not assigned fabricated exchange symbols.
Review membership and business exposure before making investment decisions.

To add groups, copy the bundled YAML and pass a complete replacement:

```bash
uv run screener sector-rotation --catalogue my-sector-groups.yaml --list-groups
uv run screener sector-rotation --catalogue my-sector-groups.yaml --report my-rotation.html
uv run screener sector-rotation --market us --catalogue my-us-groups.yaml --list-groups
```

Group IDs must be unique lowercase snake_case.
Member lists must contain unique TradingView-style symbols for the catalogue's market, such as `NSE:JYOTICNC` or `NASDAQ:MSFT`.
Two member spellings that resolve to the same provider ticker cannot occur in one basket.
The catalogue records its observation date and states that membership is `curated_research`.
The catalogue `source` and optional group `sources` must use HTTP or HTTPS URLs with a host and no credentials.
Validation rejects private IP literals and known local or non-public domain endings, including `.lan`, `.home`, `.corp`, `.intranet`, `.private`, `.onion` and `.example`.
This is an offline syntax check, not a DNS lookup or a guarantee that a hostname resolves to a public address.
The report does not fetch these links automatically.
Optional group fields are `description`, `drivers`, `risks`, `sources`, `notes`, and `video_at`.
The original `notes` and `video_at` fields remain compatible with the initial release.
`drivers`, `risks`, and `sources` are lists of strings.
Custom legacy catalogues can omit the new detail fields.
Unknown IDs, duplicate members, cross-market members, invalid symbols, unsafe source URLs, and unsupported configuration fields fail validation.
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
Missing benchmark sessions where a selected stock traded stay in the common calendar, so month-end rebalances do not move.
The benchmark carries its last close on those sessions and a warning names them.
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
Adjusted stock returns include distributions, while the default Nifty 500 and S&P 500 price indices exclude them.
The report states this mismatch and must not be presented as a total-return benchmark comparison.
Official sector-index history and business-specific metrics such as same-store sales growth are not part of this release.

The offline HTML report has no CDN, external fonts, or chart-library dependency.
It provides group search, parent and kind filtering, benchmark charts, and expandable member tables.
Search also matches business descriptions, drivers, risks, notes, member symbols, and source URLs.
Each card shows its description and offers business details and source links next to the member table.
Video links appear only where a video reference exists.
The same calculation result supplies the CSV and HTML report.
