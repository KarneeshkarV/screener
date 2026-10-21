"""Self-contained sector rotation report with inline SVG and member drill-down."""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from screener.html_report import html_page
from screener.sector_rotation.analysis import SectorRotationResult

_LEGACY_VIDEO_SOURCE = "https://www.youtube.com/watch?v=d-I6BG5F-Yg"

_REPORT_CSS = """
:root{color-scheme:dark;--ink:#e5ece9;--muted:#9daea7;--line:#30433b;--green:#8ce6b0;--red:#f0a69e}
*{box-sizing:border-box}body{margin:0;background:#101b16;color:var(--ink);font:15px Georgia,serif}
main{max-width:1500px;margin:auto;padding:48px 32px}header{border-bottom:1px solid var(--line);padding-bottom:28px}
h1{font-size:clamp(32px,5vw,64px);font-weight:normal;margin:12px 0}h2{font-size:24px;font-weight:normal}
p{line-height:1.6}.eyebrow,.tag,th,.metric,summary{font-family:'Courier New',monospace}
.eyebrow{letter-spacing:.15em;color:var(--green);font-size:12px}.subtle,.tag{color:var(--muted)}
.notice{border-left:3px solid #cba66c;background:#1d271f;padding:12px 20px;margin:24px 0}
.notice p{margin:6px 0}.toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:28px 0}
input,select{font:inherit;background:#17271f;color:var(--ink);border:1px solid #587364;border-radius:0;padding:10px;max-width:100%}
input{width:300px}label{display:flex;gap:8px;align-items:center}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:20px}
article{border:1px solid var(--line);padding:22px;background:#15221b;min-width:0;scroll-margin-top:20px}
article h2{margin:8px 0 12px}article a{color:var(--muted);font-size:12px}.stats{display:flex;justify-content:space-between;gap:12px}
.stats>div{min-width:0;flex:1}.metric{font-size:clamp(20px,2.2vw,28px);overflow-wrap:anywhere}.positive{color:var(--green)}.negative{color:var(--red)}.tag{font-size:12px}
svg{width:100%;height:auto;display:block;margin-top:18px}svg text{fill:var(--muted);font:11px 'Courier New',monospace}
.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin:8px 0 18px}.group-key{color:var(--green)}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:9px 8px;border-bottom:1px solid var(--line);white-space:nowrap}td:first-child,th:first-child{text-align:left}
th{font-weight:normal;color:var(--muted);font-size:11px}td a{color:var(--ink);font-size:13px}
summary{cursor:pointer;padding:12px 0;color:var(--green)}details p{font-size:13px}
.description,.exposure,.sources,footer{overflow-wrap:anywhere}.sector-details{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:16px}
.sector-details h3,.detail-heading{font:13px 'Courier New',monospace;color:var(--muted);margin:16px 0 8px}
.sector-details ul,.sources{padding-left:20px;font-size:13px;line-height:1.6}.sector-details li,.sources li{margin:6px 0;overflow-wrap:anywhere}
footer{margin-top:36px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
[hidden]{display:none!important}a:focus-visible,summary:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--green);outline-offset:4px}
@media(max-width:600px){main{padding:24px 14px}article{padding:16px}.stats{flex-wrap:wrap}.stats>div{flex:1 1 140px}.metric{font-size:24px}label{flex-wrap:wrap}}
"""


def _percent(value: float, *, points: bool = False) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{value:+.2f}{' pp' if points else '%'}"


def _breadth_percent(value: float) -> str:
    return "Unavailable" if pd.isna(value) else f"{value:.1f}%"


def _comparison_svg(curves: pd.DataFrame, name: str) -> str:
    values = curves.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return '<p class="subtle">No comparable history.</p>'
    low, high = float(finite.min()), float(finite.max())
    pad = max((high - low) * 0.1, 1.0)
    low, high = low - pad, high + pad
    width, height = 500, 220
    left, right, top, bottom = 44, 488, 12, 188
    svg = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(name, quote=True)} versus benchmark, both rebased to 100">'
    ]
    for level in np.linspace(low, high, 3):
        y = bottom - (level - low) / (high - low) * (bottom - top)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#30433b"/><text x="0" y="{y + 4:.2f}">{level:.0f}</text>'
        )
    for column, color in (("benchmark", "#b8c3bd"), (curves.columns[1], "#8ce6b0")):
        segments: list[str] = []
        connected = False
        for i, value in enumerate(curves[column]):
            if not np.isfinite(value):
                connected = False
                continue
            x = left + i / max(len(curves) - 1, 1) * (right - left)
            y = bottom - (value - low) / (high - low) * (bottom - top)
            segments.append(f"{'L' if connected else 'M'}{x:.2f},{y:.2f}")
            connected = True
        dash = ' stroke-dasharray="4 4"' if column == "benchmark" else ""
        svg.append(
            f'<path d="{" ".join(segments)}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
        )
    svg.extend(
        [
            f'<text x="{left}" y="213">{curves.index[0]:%d %b %Y}</text>',
            f'<text x="{right}" y="213" text-anchor="end">{curves.index[-1]:%d %b %Y}</text>',
            "</svg>",
        ]
    )
    return "".join(svg)


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
            '<p class="negative">Unranked: incomplete basket history. '
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
        cards.append(f"""<article id="{html.escape(group.id, quote=True)}" data-search="{searchable}" data-parent="{html.escape(group.parent, quote=True)}" data-kind="{html.escape(group.kind, quote=True)}">
<div class="tag">{rank} / {html.escape(group.parent)} / {html.escape(group.kind.title())}</div>
<h2>{html.escape(group.name)}</h2>{description_html}{status_note}
<div class="stats"><div><div class="metric {return_color}">{_percent(row["return_pct"])}</div><span class="tag">Basket return</span></div>
<div><div class="metric {excess_color}">{_percent(row["excess_return_pp"], points=True)}</div><span class="tag">Excess vs benchmark</span></div></div>
{_comparison_svg(result.curves[["benchmark", group.id]], group.name)}
<div class="legend"><span class="group-key">Solid: basket</span><span>Dashed: benchmark</span></div>
<p class="tag">{row["priced_members"]}/{row["members"]} members priced at end | Minimum coverage {row["minimum_coverage_pct"]:.0f}%{concentration}</p>
<p class="tag">Above SMA50: {_breadth_percent(row["breadth_sma50_pct"])} ({row["breadth_sma50_members"]} known) | Above SMA200: {_breadth_percent(row["breadth_sma200_pct"])} ({row["breadth_sma200_members"]} known)</p>
<details><summary>Inspect {row["members"]} members and sector details</summary><div class="table-wrap">{members.to_html(index=False, border=0, escape=True)}</div>
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
    body = f"""<main><header><div class="eyebrow">SCREENER / INDUSTRY RESEARCH</div><h1>Where strength is moving.</h1>
<p>Market: {html.escape(market_name)} / {result.start:%d %b %Y} to {result.end:%d %b %Y} / Benchmark: {html.escape(benchmark_name)} ({html.escape(result.benchmark)})</p>
<p class="subtle">Monthly equal-weight research baskets. Both chart lines start at 100. Rank uses excess return, not a buy signal.</p></header>
<section class="notice" aria-label="Research limitations">{warnings}</section>
<div class="toolbar"><label>Find group <input id="search" type="search" placeholder="Name, ticker, driver, risk..." /></label>
<label>Parent <select id="parent"><option value="">All parents</option>{options}</select></label>
<label>Kind <select id="kind-filter"><option value="">All kinds</option>{kind_options}</select></label><span id="count" role="status">{len(cards)} {"group" if len(cards) == 1 else "groups"}</span></div>
<div class="grid">{"".join(cards)}</div><p id="empty" hidden>No groups match these filters.</p>
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
        html_page("Sector rotation research", _REPORT_CSS, body), encoding="utf-8"
    )
