"""Self-contained sector rotation report with ECharts comparisons and member drill-down."""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from screener.html_report import html_page
from screener.report_theme import chart_html, line_chart
from screener.sector_rotation.analysis import SectorRotationResult

_LEGACY_VIDEO_SOURCE = "https://www.youtube.com/watch?v=d-I6BG5F-Yg"

# Layout only: colours, fonts and base typography come from THEME_CSS tokens.
_REPORT_CSS = """
.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:24px 0}
.toolbar label{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:13px}
#search{width:300px;max-width:100%}
#count{color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:var(--gap)}
article{min-width:0;scroll-margin-top:20px}
article h2{margin:6px 0 4px}
article a{font-size:12px}
.meta{margin:6px 0;color:var(--muted);font-size:12px}
.metrics{margin:14px 0 4px}
.positive{color:var(--pos-ink)}
.negative{color:var(--neg-ink)}
.subtle{color:var(--muted)}
.echart{margin-top:10px}
.callout{margin:20px 0}
.callout p{margin:4px 0}
.description,.exposure,.sources,footer{overflow-wrap:anywhere}
details summary{cursor:pointer;padding:10px 0;color:var(--accent);font-size:13px}
details p{font-size:13px}
.detail-heading{font-size:13px;color:var(--muted);margin:16px 0 8px}
.sector-details{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr));gap:16px;margin-top:12px}
.sector-details ul,.sources{padding-left:20px;font-size:13px;line-height:1.6}
.sector-details li,.sources li{margin:6px 0;overflow-wrap:anywhere}
footer{margin-top:32px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
[hidden]{display:none!important}
@media(max-width:600px){article{padding:14px}}
"""


def _percent(value: float, *, points: bool = False) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{value:+.2f}{' pp' if points else '%'}"


def _breadth_percent(value: float) -> str:
    return "Unavailable" if pd.isna(value) else f"{value:.1f}%"


def _chart_id(group_id: str) -> str:
    """DOM-safe chart id: the card's anchor slug plus ``-chart``."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", group_id).strip("-") or "group"
    return f"{slug}-chart"


def _comparison_chart(
    curves: pd.DataFrame, group_id: str, name: str, chart_id: str
) -> str:
    """Basket (series-1) versus benchmark (series-2, dashed), both rebased to 100."""
    if not np.isfinite(curves[group_id].to_numpy(dtype=float)).any():
        return '<p class="empty">No comparable history.</p>'
    option = line_chart(
        list(curves.index),
        [
            {"name": name, "values": curves[group_id].tolist(), "color": "series-1"},
            {
                "name": "Benchmark",
                "values": curves["benchmark"].tolist(),
                "color": "series-2",
                "dashed": True,
            },
        ],
        value_format="num0",
    )
    return chart_html(chart_id, option, height=200)


def _safe_source_link(source: str, label: str | None = None) -> str:
    """Allow only absolute HTTP source links, even for unvalidated catalogue copies."""
    text = html.escape(label if label is not None else source)
    try:
        parts = urlsplit(source)
        safe = (
            parts.scheme in {"http", "https"}
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and not re.search(r"[\s\\\x00-\x1f\x7f]", source)
        )
    except ValueError:
        safe = False
    if not safe:
        return text
    return f'<a href="{html.escape(source, quote=True)}" rel="noreferrer">{text}</a>'


def _sector_detail_list(heading: str, items: tuple[str, ...]) -> str:
    """Render escaped sector drivers or risks without empty headings."""
    if not items:
        return ""
    entries = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<section><h3>{heading}</h3><ul>{entries}</ul></section>"


def _sector_video_reference(source: str, video_at: str | None) -> str:
    """Legacy video timestamps belong only to the original cited video."""
    if (
        source != _LEGACY_VIDEO_SOURCE
        or not video_at
        or not re.fullmatch(r"\d+:[0-5]\d", video_at)
    ):
        return ""
    minutes, seconds = map(int, video_at.split(":"))
    url = f"{source}&t={minutes * 60 + seconds}"
    return f"<p>{_safe_source_link(url, f'Video reference {video_at}')}</p>"


def render_sector_report(result: SectorRotationResult, path: Path) -> None:
    """Write an offline HTML report; all catalogue strings are escaped."""
    catalogue = result.selection.catalogue
    benchmark_name = (
        catalogue.benchmark_name
        if result.benchmark == catalogue.benchmark
        else result.benchmark
    )
    market_name = {"india": "India", "us": "US"}.get(catalogue.market, catalogue.market)
    warnings = "".join(f"<p>{html.escape(warning)}</p>" for warning in result.warnings)
    cards: list[str] = []
    for row in result.summary.to_dict("records"):
        group = next(
            group for group in result.selection.groups if group.id == row["group_id"]
        )
        members = result.members[result.members.group_id == group.id].copy()
        members = members[
            ["ticker", "return_pct", "coverage_pct", "above_sma50", "above_sma200"]
        ]
        members["return_pct"] = members["return_pct"].map(_percent)
        members["coverage_pct"] = members["coverage_pct"].map(
            lambda value: f"{value:.0f}%"
        )
        for column in ("above_sma50", "above_sma200"):
            members[column] = members[column].map(
                lambda value: "Unknown" if value is None else ("Yes" if value else "No")
            )
        members.columns = [
            "Ticker",
            "Return",
            "Coverage",
            "Above SMA50",
            "Above SMA200",
        ]
        return_color = "positive" if row["return_pct"] >= 0 else "negative"
        excess_color = "positive" if row["excess_return_pp"] >= 0 else "negative"
        if row["status"] != "ok":
            return_color = excess_color = "subtle"
        description = getattr(group, "description", "")
        drivers = getattr(group, "drivers", ())
        risks = getattr(group, "risks", ())
        sources = getattr(group, "sources", ())
        description_html = (
            f'<p class="description">{html.escape(description)}</p>'
            if description
            else ""
        )
        detail_lists = _sector_detail_list("Drivers", drivers) + _sector_detail_list(
            "Risks", risks
        )
        sources_html = ""
        if sources:
            links = "".join(
                f"<li>{_safe_source_link(source)}</li>" for source in sources
            )
            sources_html = (
                '<h3 class="detail-heading">Primary sources</h3>'
                f'<ul class="sources">{links}</ul>'
            )
        video_reference = (
            _sector_video_reference(catalogue.source, getattr(group, "video_at", None))
            if catalogue.market == "india"
            else ""
        )
        concentration = " | Concentrated basket" if row["concentrated"] else ""
        status_note = (
            '<p class="callout">Unranked: incomplete basket history. '
            "Any displayed basket line stops at the first data failure.</p>"
            if row["status"] != "ok"
            else ""
        )
        searchable = html.escape(
            " ".join(
                (
                    group.name,
                    group.id,
                    group.parent,
                    group.kind,
                    description,
                    *drivers,
                    *risks,
                    *group.members,
                    *sources,
                    group.notes,
                )
            ).lower(),
            quote=True,
        )
        rank = "Unranked" if pd.isna(row["rank"]) else f"Rank {int(row['rank']):02d}"
        chart_html_block = _comparison_chart(
            result.curves, group.id, group.name, _chart_id(group.id)
        )
        cards.append(f"""<article class="panel" id="{html.escape(group.id, quote=True)}" data-search="{searchable}" data-parent="{html.escape(group.parent, quote=True)}" data-kind="{html.escape(group.kind, quote=True)}">
<div class="meta">{rank} / {html.escape(group.parent)} / {html.escape(group.kind.title())}</div>
<h2>{html.escape(group.name)}</h2>{description_html}{status_note}
<div class="metrics"><div class="metric"><span>Basket return</span><strong class="{return_color}">{_percent(row["return_pct"])}</strong></div>
<div class="metric"><span>Excess vs benchmark</span><strong class="{excess_color}">{_percent(row["excess_return_pp"], points=True)}</strong></div></div>
{chart_html_block}
<p class="meta">{row["priced_members"]}/{row["members"]} members priced at end | Minimum coverage {row["minimum_coverage_pct"]:.0f}%{concentration}</p>
<p class="meta">Above SMA50: {_breadth_percent(row["breadth_sma50_pct"])} ({row["breadth_sma50_members"]} known) | Above SMA200: {_breadth_percent(row["breadth_sma200_pct"])} ({row["breadth_sma200_members"]} known)</p>
<details><summary>Inspect {row["members"]} members and sector details</summary><div class="table-wrap">{members.to_html(index=False, border=0, escape=True, classes="data-table")}</div>
<div class="sector-details">{detail_lists}</div>
<h3 class="detail-heading">Exposure notes</h3><p class="exposure">{html.escape(group.notes or "Selected research basket, not exhaustive industry membership.")}</p>
{sources_html}{video_reference}</details></article>""")
    parents = sorted({group.parent for group in result.selection.groups})
    options = "".join(
        f'<option value="{html.escape(parent, quote=True)}">{html.escape(parent)}</option>'
        for parent in parents
    )
    kind_options = "".join(
        f'<option value="{html.escape(kind, quote=True)}">{html.escape(kind.title())}</option>'
        for kind in sorted({group.kind for group in result.selection.groups})
    )
    notice = (
        f'<section class="callout" aria-label="Research limitations">{warnings}</section>'
        if warnings
        else ""
    )
    body = f"""<main><header><div class="eyebrow">Sector rotation</div><h1>Where strength is moving.</h1>
<div class="subhead"><span>Market: {html.escape(market_name)} / {result.start:%d %b %Y} to {result.end:%d %b %Y} / Benchmark: {html.escape(benchmark_name)} ({html.escape(result.benchmark)})</span></div>
<div class="subhead"><span>Monthly equal-weight research baskets. Both chart lines start at 100. Rank uses excess return, not a buy signal.</span></div></header>
{notice}
<div class="toolbar"><label>Find group <input id="search" type="search" placeholder="Name, ticker, driver, risk..." /></label>
<label>Parent <select id="parent"><option value="">All parents</option>{options}</select></label>
<label>Kind <select id="kind-filter"><option value="">All kinds</option>{kind_options}</select></label><span id="count" role="status">{len(cards)} {"group" if len(cards) == 1 else "groups"}</span></div>
<div class="grid">{"".join(cards)}</div><p id="empty" class="empty" hidden>No groups match these filters.</p>
<footer><p>Weights reset after the last close of each month and drift between rebalances. Missing held prices stop a curve; unavailable groups remain unranked.
No transaction costs are included. Adjusted stock returns and a price-only benchmark are not a total-return comparison.</p>
<p>Catalogue observed {html.escape(str(catalogue.observed_on))}. Catalogue source: {_safe_source_link(catalogue.source)}.</p>
<p>{html.escape(catalogue.membership_note)} Selected stocks are curated research baskets, not official sector or industry indices.</p></footer></main>
<script>
const search=document.getElementById('search'), parent=document.getElementById('parent'), kind=document.getElementById('kind-filter');
function filterGroups(){{let count=0;document.querySelectorAll('article').forEach(card=>{{
card.hidden=!(card.dataset.search.includes(search.value.trim().toLowerCase())&&(!parent.value||card.dataset.parent===parent.value)&&(!kind.value||card.dataset.kind===kind.value));
if(!card.hidden)count++;}});document.getElementById('count').textContent=count+(count===1?' group':' groups');document.getElementById('empty').hidden=count!==0;}}
search.addEventListener('input',filterGroups);parent.addEventListener('change',filterGroups);kind.addEventListener('change',filterGroups);
</script>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        html_page("Sector rotation research", _REPORT_CSS, body, charts=True),
        encoding="utf-8",
    )
