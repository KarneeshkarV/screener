"""Global sector report metadata, provenance, escaping, and filter contracts."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import date
from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from screener.sector_rotation.analysis import SectorRotationResult
from screener.sector_rotation.catalogue import (
    SectorCatalogue,
    SectorGroup,
    SectorSelection,
)
from screener.sector_rotation.report import render_sector_report

_VIDEO_SOURCE = "https://www.youtube.com/watch?v=d-I6BG5F-Yg"


@dataclass
class _ReportElement:
    tag: str
    attrs: dict[str, str | None]
    text: str = ""


class _ReportParser(HTMLParser):
    """Collect decoded report metadata without adding an HTML parser dependency."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.elements: list[_ReportElement] = []
        self.stack: list[_ReportElement] = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        element = _ReportElement(tag, dict(attrs))
        self.elements.append(element)
        if tag not in {"meta", "input", "link", "br", "hr"}:
            self.stack.append(element)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        for element in self.stack:
            element.text += data

    def tagged(self, tag: str) -> list[_ReportElement]:
        return [element for element in self.elements if element.tag == tag]

    def identified(self, identifier: str) -> _ReportElement:
        return next(
            element
            for element in self.elements
            if element.attrs.get("id") == identifier
        )


def _group(**overrides) -> SectorGroup:
    data = {
        "id": "machines",
        "name": "Machines",
        "parent": "Industrials",
        "kind": "industry",
        "video_at": None,
        "members": ("NYSE:AAA", "NASDAQ:BBB"),
        "notes": "Mixed business exposure; not pure-play machine makers.",
        "description": "Precision tools for factory automation.",
        "drivers": ("Replacement demand", "Export orders"),
        "risks": ("Customer concentration", "Commodity costs"),
        "sources": ("https://www.census.gov/manufacturing/",),
    }
    data.update(overrides)
    # Report security must also cover unvalidated model copies.
    return SectorGroup.model_construct(**data)


def _report_result(*groups, **catalogue_overrides) -> SectorRotationResult:
    groups = groups or (_group(),)
    data = {
        "version": 1,
        "market": "us",
        "benchmark": "^GSPC",
        "benchmark_name": "S&P 500 price index",
        "observed_on": date(2026, 1, 6),
        "source": "https://www.census.gov/",
        "membership_basis": "curated_research",
        "membership_note": "Current curated membership; survivorship bias applies.",
        "groups": groups,
    }
    data.update(catalogue_overrides)
    catalogue = SectorCatalogue.model_construct(**data)
    summary = pd.DataFrame(
        [
            {
                "group_id": group.id,
                "rank": index + 1,
                "return_pct": 5.0,
                "excess_return_pp": 3.0,
                "status": "ok",
                "concentrated": True,
                "priced_members": len(group.members),
                "members": len(group.members),
                "minimum_coverage_pct": 100.0,
                "breadth_sma50_pct": 50.0,
                "breadth_sma50_members": 2,
                "breadth_sma200_pct": 100.0,
                "breadth_sma200_members": 1,
            }
            for index, group in enumerate(groups)
        ]
    )
    members = pd.DataFrame(
        [
            {
                "group_id": group.id,
                "ticker": ticker,
                "return_pct": 5.0,
                "coverage_pct": 100.0,
                "above_sma50": True,
                "above_sma200": None,
            }
            for group in groups
            for ticker in group.members
        ]
    )
    curves = pd.DataFrame(
        {"benchmark": [100.0, 102.0], **{group.id: [100.0, 105.0] for group in groups}},
        index=pd.to_datetime(["2026-01-02", "2026-01-06"]),
    )
    return SectorRotationResult(
        selection=SectorSelection(catalogue, groups),
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
        benchmark=catalogue.benchmark,
        summary=summary,
        members=members,
        curves=curves,
        warnings=(catalogue.membership_note,),
    )


def _render_report(tmp_path, result) -> tuple[str, _ReportParser]:
    path = tmp_path / "nested" / "global.html"
    render_sector_report(result, path)
    text = path.read_text(encoding="utf-8")
    return text, _ReportParser(text)


@pytest.mark.parametrize("market", ["india", "us"])
def test_report_accepts_validated_global_catalogue_schema(tmp_path, market):
    group = _group(members=("NSE:AAA",)) if market == "india" else _group()
    result = _report_result(group, market=market)
    catalogue = SectorCatalogue.model_validate(result.selection.catalogue.model_dump())
    result = replace(
        result,
        selection=SectorSelection(catalogue, catalogue.groups),
    )
    _, parsed = _render_report(tmp_path, result)
    assert (
        "Precision tools for factory automation." in parsed.identified("machines").text
    )
    assert not any("Video reference" in link.text for link in parsed.tagged("a"))


def test_us_report_has_market_benchmark_details_and_primary_sources(tmp_path):
    text, parsed = _render_report(tmp_path, _report_result())
    assert "Market: US" in parsed.tagged("header")[0].text
    assert "Benchmark: S&P 500 price index (^GSPC)" in parsed.tagged("header")[0].text
    card = parsed.identified("machines")
    for detail in (
        "Precision tools for factory automation.",
        "Drivers",
        "Replacement demand",
        "Export orders",
        "Risks",
        "Customer concentration",
        "Commodity costs",
        "Exposure notes",
        "Mixed business exposure; not pure-play machine makers.",
        "Primary sources",
        "NYSE:AAA",
        "NASDAQ:BBB",
    ):
        assert detail in card.text
    hrefs = [link.attrs["href"] for link in parsed.tagged("a")]
    assert "https://www.census.gov/manufacturing/" in hrefs
    assert "https://www.census.gov/" in hrefs
    assert "Video reference" not in text
    assert "Group labels come from the cited video" not in text
    assert "survivorship bias applies" in parsed.tagged("footer")[0].text
    assert not any(node.attrs.get("src") for node in parsed.tagged("script"))
    assert not parsed.tagged("link")
    assert parsed.tagged("svg")
    assert "+5.00%" in card.text and "+3.00 pp" in card.text
    assert parsed.identified("count").text == "1 group"
    assert "Unknown" in card.text
    assert "Minimum coverage 100%" in card.text


@pytest.mark.parametrize(
    "market,source,video_at,expected",
    [
        (
            "india",
            _VIDEO_SOURCE,
            "5:03",
            "https://www.youtube.com/watch?v=d-I6BG5F-Yg&t=303",
        ),
        ("india", _VIDEO_SOURCE, None, None),
        ("india", "https://www.mospi.gov.in/", "5:03", None),
        ("india", "https://www.youtube.com/watch?v=other", "5:03", None),
        ("india", _VIDEO_SOURCE + "#other", "5:03", None),
        ("us", _VIDEO_SOURCE, "5:03", None),
        ("india", _VIDEO_SOURCE, "5:99", None),
        ("india", _VIDEO_SOURCE, '<img src=x onerror="alert(1)">', None),
    ],
)
def test_video_timestamps_require_matching_provenance(
    tmp_path, market, source, video_at, expected
):
    text, parsed = _render_report(
        tmp_path,
        _report_result(
            _group(video_at=video_at),
            market=market,
            source=source,
        ),
    )
    references = [link for link in parsed.tagged("a") if "Video reference" in link.text]
    assert [link.attrs["href"] for link in references] == (
        [expected] if expected else []
    )
    assert "Group labels come from the cited video" not in text
    assert "<img" not in text


def test_india_context_and_benchmark_override_do_not_use_us_or_default_label(tmp_path):
    result = _report_result(
        market="india", benchmark="^CRSLDX", benchmark_name="Nifty 500"
    )
    _, parsed = _render_report(tmp_path, replace(result, benchmark="^NSEI"))
    header = parsed.tagged("header")[0].text
    assert "Market: India" in header
    assert "Benchmark: ^NSEI (^NSEI)" in header
    assert "Nifty 500" not in header and "S&P" not in header


def test_legacy_group_without_rich_optional_fields_still_renders(tmp_path):
    group = SimpleNamespace(
        id="legacy",
        name="Legacy tools",
        parent="Industrials",
        kind="industry",
        members=("NSE:AAA",),
        notes="",
        video_at="3:00",
    )
    _, parsed = _render_report(
        tmp_path,
        _report_result(group, market="india", source=_VIDEO_SOURCE),
    )
    card = parsed.identified("legacy")
    assert "Selected research basket, not exhaustive industry membership." in card.text
    assert "Video reference 3:00" in card.text
    assert "Drivers" not in card.text and "Risks" not in card.text
    assert "Primary sources" not in card.text


def test_all_metadata_is_escaped_in_text_attributes_and_links(tmp_path):
    payload = '"><img src=x onerror="alert(1)"><script>alert(2)</script>&'
    source = 'https://example.org/?x="quoted"&y=<tag>'
    group = _group(
        id=payload,
        name=payload,
        parent=payload,
        kind=payload,
        description=payload,
        drivers=(payload,),
        risks=(payload,),
        members=(payload,),
        notes=payload,
        sources=(source, payload),
    )
    result = _report_result(
        group,
        market=payload,
        source=source,
        benchmark=payload,
        benchmark_name=payload,
        membership_note=payload,
        observed_on=payload,
    )
    text, parsed = _render_report(tmp_path, result)
    assert not parsed.tagged("img")
    assert len(parsed.tagged("script")) == 1
    assert not any(
        name.startswith("on") for element in parsed.elements for name in element.attrs
    )
    card = parsed.identified(payload)
    assert card.attrs["data-parent"] == payload
    assert card.attrs["data-kind"] == payload
    assert payload in card.attrs["data-search"]
    assert card.text.count(payload) >= 8
    assert payload in parsed.tagged("header")[0].text
    assert payload in parsed.tagged("footer")[0].text
    assert all(link.attrs["href"] == source for link in parsed.tagged("a"))
    assert "&amp;y=&lt;tag&gt;" in text
    assert "x=&quot;quoted&quot;" in text
    assert parsed.tagged("svg")[0].attrs["aria-label"].startswith(payload)


@pytest.mark.parametrize(
    "unsafe_source",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "//example.org/",
        "/relative/source",
        "https://",
        "https://[broken/",
        "https://user:pass@example.org/",
        "https://example.org/\nmalicious",
        "https://example.org/\\malicious",
    ],
)
def test_unsafe_source_urls_are_text_not_clickable(tmp_path, unsafe_source):
    _, parsed = _render_report(
        tmp_path,
        _report_result(_group(sources=(unsafe_source,)), source=unsafe_source),
    )
    assert not parsed.tagged("a")
    assert unsafe_source in parsed.identified("machines").text
    assert unsafe_source in parsed.tagged("footer")[0].text


def test_search_index_covers_all_details_members_and_notes(tmp_path):
    _, parsed = _render_report(tmp_path, _report_result())
    search_index = parsed.identified("machines").attrs["data-search"]
    for term in (
        "machines",
        "industrials",
        "industry",
        "precision tools",
        "replacement demand",
        "export orders",
        "customer concentration",
        "commodity costs",
        "nyse:aaa",
        "nasdaq:bbb",
        "mixed business exposure",
        "pure-play",
        "census.gov/manufacturing",
    ):
        assert term in search_index
    assert parsed.identified("kind-filter").tag == "select"
    assert parsed.identified("search").attrs["type"] == "search"
    assert parsed.identified("count").attrs["role"] == "status"
    assert "hidden" in parsed.identified("empty").attrs


def test_filters_execute_case_insensitive_search_and_combined_parent_kind(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the inline report filter")
    _, parsed = _render_report(
        tmp_path,
        _report_result(
            _group(),
            _group(id="banking", name="Banking", kind="sector", parent="Financials"),
            _group(id="automation", name="Automation", kind="theme"),
        ),
    )
    cards = [
        {
            "dataset": {
                "search": element.attrs["data-search"],
                "parent": element.attrs["data-parent"],
                "kind": element.attrs["data-kind"],
            },
            "hidden": False,
        }
        for element in parsed.tagged("article")
    ]
    options = {
        element.attrs["value"]: element.text for element in parsed.tagged("option")
    }
    assert {
        "sector": "Sector",
        "industry": "Industry",
        "theme": "Theme",
    }.items() <= options.items()
    script = parsed.tagged("script")[0].text
    harness = f"""
const assert = require('node:assert/strict');
const cards = {json.dumps(cards)};
const elements = Object.fromEntries(['search','parent','kind-filter','count','empty'].map(id =>
  [id, {{value:'',hidden:true,textContent:'',listeners:{{}},
    addEventListener(event, callback) {{ this.listeners[event] = callback; }} }}]));
const document = {{getElementById: id => elements[id], querySelectorAll: () => cards}};
{script}
function check(query, parentValue, kindValue, visible) {{
  elements.search.value=query; elements.parent.value=parentValue; elements['kind-filter'].value=kindValue;
  elements.search.listeners.input();
  assert.deepEqual(cards.map((card,i) => card.hidden ? -1 : i).filter(i => i>=0), visible);
  assert.equal(elements.count.textContent, visible.length + (visible.length===1?' group':' groups'));
  assert.equal(elements.empty.hidden, visible.length!==0);
}}
check('  PRECISION TOOLS  ', '', '', [0,1,2]);
check('NASDAQ:BBB', 'Industrials', 'theme', [2]);
check('commodity costs', 'Financials', 'sector', [1]);
check('mixed business exposure', 'Industrials', 'industry', [0]);
check('replacement demand', 'Financials', 'theme', []);
check('no matching metadata', '', '', []);
check('', '', '', [0,1,2]);
elements.parent.value='Financials'; elements.parent.listeners.change();
assert.equal(elements.count.textContent, '1 group');
elements['kind-filter'].value='theme'; elements['kind-filter'].listeners.change();
assert.equal(elements.count.textContent, '0 groups');
"""
    completed = subprocess.run([node, "-e", harness], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_unavailable_group_keeps_metrics_and_chart_failure_behavior(tmp_path):
    result = _report_result()
    result.summary.loc[0, ["rank", "return_pct", "excess_return_pp"]] = np.nan
    result.summary.loc[0, "status"] = "insufficient_data"
    result.curves.loc[:, "machines"] = np.nan
    _, parsed = _render_report(tmp_path, result)
    card = parsed.identified("machines")
    assert "Unranked: incomplete basket history" in card.text
    assert "Unavailable" in card.text
    assert "Precision tools for factory automation." in card.text
    paths = parsed.tagged("path")
    assert paths[0].attrs["stroke-dasharray"] == "4 4"
    assert paths[1].attrs["d"] == ""
    assert len(parsed.tagged("svg")) == 1
