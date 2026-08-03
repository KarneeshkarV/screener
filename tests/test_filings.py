"""Tests for the FMP US SEC filings reader module and CLI — fully offline."""

from __future__ import annotations

import json
from datetime import date

from click.testing import CliRunner

import screener.commands.filings as filings_cmd
from screener import filings as filings_module
from screener.cli import cli
from screener.filings import (
    Filing,
    FinancialReport,
    ReportSection,
    SectionRow,
    load_filings,
    load_report,
    match_sections,
    parse_date,
    parse_filings,
    parse_report,
)


def _filing_row(
    date_str="2025-10-31 00:00:00",
    accepted="2025-10-31 06:01:26",
    filing_type="10-K",
    symbol="AAPL",
):
    # Note FMP's "fillingDate" misspelling is intentional.
    return {
        "symbol": symbol,
        "fillingDate": date_str,
        "acceptedDate": accepted,
        "cik": "0000320193",
        "type": filing_type,
        "link": "https://sec.gov/index.htm",
        "finalLink": "https://sec.gov/aapl.htm",
    }


# Real-shaped report payload: metadata keys, truncated section names, a "_2"
# de-duplication suffix, a "(Tables)" variant, empty-string and whitespace
# cells, and rows of differing column counts.
_REPORT_PAYLOAD = {
    "symbol": "AAPL",
    "period": "FY",
    "year": 2024,
    "Cover Page": [
        {"Cover Page - USD ($)": ["12 Months Ended"]},
    ],
    "CONSOLIDATED BALANCE SHEETS": [
        {"Balance sheet header": ["Sep. 28, 2024", "Sep. 30, 2023"]},
        {"Current assets:": ["", "  "]},
        {"Cash and cash equivalents": ["29943", "29965"]},
        {"Total assets": ["364980"]},
    ],
    "CONSOLIDATED BALANCE SHEETS (Pa": [
        {"Allowance for credit losses": ["550"]},
    ],
    "Revenue": [
        {"Net sales": ["391035", "383285"]},
    ],
    "Revenue_2": [
        {"Deferred revenue": ["8249"]},
    ],
}


# ── parse_date ───────────────────────────────────────────────────────────────


def test_parse_date_variants():
    assert parse_date("2025-10-31 00:00:00") == date(2025, 10, 31)
    assert parse_date("2025-10-31") == date(2025, 10, 31)
    # fromisoformat rejects the trailing token; the strptime[:10] fallback wins.
    assert parse_date("2025-10-31 weird-suffix") == date(2025, 10, 31)
    assert parse_date(None) is None
    assert parse_date("") is None
    assert parse_date("   ") is None
    assert parse_date("not-a-date") is None


# ── parse_filings ────────────────────────────────────────────────────────────


def test_parse_filings_reads_misspelled_filling_date():
    out = parse_filings([_filing_row()])
    assert out == [
        Filing(
            symbol="AAPL",
            type="10-K",
            filing_date=date(2025, 10, 31),
            accepted_date=date(2025, 10, 31),
            link="https://sec.gov/index.htm",
            final_link="https://sec.gov/aapl.htm",
        )
    ]


def test_parse_filings_skips_non_dicts_and_handles_missing_keys():
    out = parse_filings([{"type": "8-K"}, "junk", 42])
    assert len(out) == 1
    only = out[0]
    assert only.symbol == "" and only.type == "8-K"
    assert only.filing_date is None and only.final_link == ""


def test_parse_filings_non_list_payload_is_empty():
    assert parse_filings({"Error Message": "Invalid API KEY."}) == []
    assert parse_filings(None) == []


# ── parse_report / _parse_section ────────────────────────────────────────────


def test_parse_report_strips_meta_and_parses_sections():
    report = parse_report(_REPORT_PAYLOAD)
    assert report is not None
    assert report.symbol == "AAPL" and report.period == "FY" and report.year == 2024
    names = report.section_names()
    assert "symbol" not in names and "year" not in names
    assert names[0] == "Cover Page"
    assert "Revenue_2" in names
    assert "CONSOLIDATED BALANCE SHEETS (Pa" in names

    balance = next(
        s for s in report.sections if s.name == "CONSOLIDATED BALANCE SHEETS"
    )
    # Differing column counts and empty/whitespace cells survive as-is.
    assert balance.rows[1] == SectionRow("Current assets:", ["", "  "])
    assert balance.rows[3] == SectionRow("Total assets", ["364980"])


def test_parse_report_none_for_non_dict_or_no_sections():
    assert parse_report(None) is None
    assert parse_report([1, 2, 3]) is None
    # Only metadata keys -> no sections -> None.
    assert parse_report({"symbol": "AAPL", "period": "FY", "year": 2024}) is None


def test_parse_report_year_coercion():
    good = parse_report({"symbol": "X", "period": "FY", "year": "2024", "S": []})
    assert good is not None and good.year == 2024
    bad = parse_report({"symbol": "X", "period": "FY", "year": "N/A", "S": []})
    assert bad is not None and bad.year is None
    none_year = parse_report({"symbol": "X", "period": "FY", "S": []})
    assert none_year is not None and none_year.year is None


def test_parse_section_robust_to_messy_shapes():
    report = parse_report(
        {
            "symbol": "X",
            "period": "FY",
            "year": 2024,
            "Scalar cells": [{"row": "not-a-list"}, "junk", 99],
            "Multi key": [{"a": ["1"], "b": ["2"]}],
            "Not a list": "oops",
        }
    )
    assert report is not None
    scalar = next(s for s in report.sections if s.name == "Scalar cells")
    assert scalar.rows == [SectionRow("row", ["not-a-list"])]
    multi = next(s for s in report.sections if s.name == "Multi key")
    assert multi.rows == [SectionRow("a", ["1"]), SectionRow("b", ["2"])]
    not_list = next(s for s in report.sections if s.name == "Not a list")
    assert not_list.rows == []


def test_report_section_raw_roundtrips():
    section = ReportSection("S", [SectionRow("label", ["1", "2"])])
    assert section.raw() == [{"label": ["1", "2"]}]


# ── match_sections ───────────────────────────────────────────────────────────


def test_match_sections_case_insensitive_substring():
    report = parse_report(_REPORT_PAYLOAD)
    assert report is not None
    hits = match_sections(report, "balance sheet")
    # Both the main statement and its truncated "(Pa" parenthetical match.
    assert {s.name for s in hits} == {
        "CONSOLIDATED BALANCE SHEETS",
        "CONSOLIDATED BALANCE SHEETS (Pa",
    }


def test_match_sections_matches_truncated_and_suffixed_names():
    report = parse_report(_REPORT_PAYLOAD)
    assert report is not None
    revenue = {s.name for s in match_sections(report, "revenue")}
    assert revenue == {"Revenue", "Revenue_2"}
    assert [s.name for s in match_sections(report, "cover")] == ["Cover Page"]


def test_match_sections_miss_and_blank():
    report = parse_report(_REPORT_PAYLOAD)
    assert report is not None
    assert match_sections(report, "nonexistent") == []
    assert match_sections(report, "   ") == []


# ── load_filings (stubbed transport, no network) ─────────────────────────────


def test_load_filings_paginates_and_truncates(monkeypatch, fake_provider):
    monkeypatch.setattr(filings_module, "_FMP_FILINGS_PROVIDER", fake_provider())
    seen: list[str] = []

    def fake_request_json(url, *, headers, timeout):
        seen.append(url)
        if "page=0" in url:
            return [_filing_row(filing_type="10-K"), _filing_row(filing_type="10-Q")]
        if "page=1" in url:
            return [_filing_row(filing_type="8-K")]
        return []

    monkeypatch.setattr(filings_module.fmp, "request_json", fake_request_json)

    out = load_filings("AAPL", api_key="key", limit=3, cache_ttl=None, refresh=True)
    # 3 rows requested; page 0 gives 2, page 1 gives 1 -> stop at limit.
    assert len(out) == 3
    assert [f.type for f in out] == ["10-K", "10-Q", "8-K"]
    assert any("sec_filings/AAPL" in u and "apikey=key" in u for u in seen)
    assert len(seen) == 2  # did not walk past the limit


def test_load_filings_passes_type_filter_and_stops_on_empty(monkeypatch, fake_provider):
    monkeypatch.setattr(filings_module, "_FMP_FILINGS_PROVIDER", fake_provider())
    seen: list[str] = []

    def fake_request_json(url, *, headers, timeout):
        seen.append(url)
        if "page=0" in url:
            return [_filing_row()]
        return []  # page 1 empty -> stop

    monkeypatch.setattr(filings_module.fmp, "request_json", fake_request_json)

    out = load_filings(
        "AAPL",
        api_key="key",
        filing_type="10-K",
        limit=20,
        cache_ttl=None,
        refresh=True,
    )
    assert len(out) == 1
    assert any("type=10-K" in u for u in seen)
    assert len(seen) == 2  # page 0 (data) + page 1 (empty)


def test_load_filings_cache_key_includes_type_and_limit(monkeypatch, fake_provider):
    provider = fake_provider()
    monkeypatch.setattr(filings_module, "_FMP_FILINGS_PROVIDER", provider)
    monkeypatch.setattr(
        filings_module.fmp,
        "request_json",
        lambda url, *, headers, timeout: [],
    )

    load_filings("AAPL", api_key="key", filing_type="10-K", limit=5, cache_ttl=None)
    load_filings("AAPL", api_key="key", filing_type="10-Q", limit=5, cache_ttl=None)
    load_filings("AAPL", api_key="key", limit=20, cache_ttl=None)

    keys = [key for key, _refresh in provider.calls]
    # Type filter and limit are both part of the key, so cached pages for one
    # --type/--limit combination can never serve another.
    assert keys[0] == ("sec_filings", "AAPL", "10-K", 5)
    assert keys[1] == ("sec_filings", "AAPL", "10-Q", 5)
    assert keys[2] == ("sec_filings", "AAPL", "", 20)
    assert len(set(keys)) == 3


def test_load_filings_warns_when_page_cap_truncates(monkeypatch, fake_provider, caplog):
    monkeypatch.setattr(filings_module, "_FMP_FILINGS_PROVIDER", fake_provider())
    monkeypatch.setattr(
        filings_module.fmp,
        "request_json",
        lambda url, *, headers, timeout: [_filing_row()],  # every page has data
    )

    with caplog.at_level("WARNING", logger="screener.filings"):
        out = load_filings("AAPL", api_key="key", limit=99, cache_ttl=None)

    assert len(out) == filings_module._MAX_PAGES  # one row per page, cap hit
    assert any("may be truncated" in rec.message for rec in caplog.records)


def test_load_filings_non_list_payload_returns_empty(monkeypatch, fake_provider):
    monkeypatch.setattr(filings_module, "_FMP_FILINGS_PROVIDER", fake_provider())
    monkeypatch.setattr(
        filings_module.fmp,
        "request_json",
        lambda url, *, headers, timeout: {"Error Message": "bad"},
    )
    assert load_filings("AAPL", api_key="key", cache_ttl=None, refresh=True) == []


# ── load_report (stubbed transport) ──────────────────────────────────────────


def test_load_report_parses_v4_payload(monkeypatch, fake_provider):
    monkeypatch.setattr(filings_module, "_FMP_REPORT_PROVIDER", fake_provider())
    seen: dict = {}

    def fake_request_json(url, *, headers, timeout):
        seen["url"] = url
        return _REPORT_PAYLOAD

    monkeypatch.setattr(filings_module.fmp, "request_json", fake_request_json)

    report = load_report(
        "AAPL", api_key="key", year=2024, period="FY", cache_ttl=None, refresh=True
    )
    assert report is not None and report.symbol == "AAPL"
    assert "financial-reports-json" in seen["url"]
    assert "symbol=AAPL" in seen["url"] and "year=2024" in seen["url"]
    assert "period=FY" in seen["url"] and "apikey=key" in seen["url"]


def test_load_report_none_when_missing(monkeypatch, fake_provider):
    monkeypatch.setattr(filings_module, "_FMP_REPORT_PROVIDER", fake_provider())
    monkeypatch.setattr(
        filings_module.fmp, "request_json", lambda url, *, headers, timeout: {}
    )
    assert (
        load_report("ZZZZ", api_key="key", year=2024, cache_ttl=None, refresh=True)
        is None
    )


# ── CLI: filings list ────────────────────────────────────────────────────────


def _sample_filings():
    return [
        Filing(
            symbol="AAPL",
            type="10-K",
            filing_date=date(2025, 10, 31),
            accepted_date=date(2025, 10, 31),
            link="https://sec.gov/index.htm",
            final_link="https://sec.gov/aapl.htm",
        )
    ]


def test_cli_list_requires_api_key(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: None)
    res = CliRunner().invoke(cli, ["filings", "list", "AAPL"])
    assert res.exit_code != 0 and "FMP_API_KEY" in res.output


def test_cli_list_rejects_non_us_market():
    res = CliRunner().invoke(cli, ["filings", "list", "AAPL", "-m", "india"])
    assert res.exit_code != 0


def test_cli_list_rejects_bad_limit(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    res = CliRunner().invoke(cli, ["filings", "list", "AAPL", "--limit", "0"])
    assert res.exit_code != 0 and "--limit" in res.output


def test_cli_list_empty_ticker(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    res = CliRunner().invoke(cli, ["filings", "list", "   "])
    assert res.exit_code != 0 and "ticker" in res.output.lower()


def test_cli_list_no_results(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_filings", lambda *a, **k: [])
    res = CliRunner().invoke(cli, ["filings", "list", "ZZZZ"])
    assert res.exit_code == 0 and "No filings found for ZZZZ." in res.output


def test_cli_list_table(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")

    def fake_load(symbol, *, api_key, filing_type, limit, refresh):
        assert symbol == "AAPL" and api_key == "key"
        assert filing_type == "10-K"
        return _sample_filings()

    monkeypatch.setattr(filings_module, "load_filings", fake_load)
    res = CliRunner().invoke(
        cli, ["filings", "list", "aapl", "--type", "10-K"], catch_exceptions=False
    )
    assert res.exit_code == 0
    assert "SEC filings (US)" in res.output and "10-K" in res.output


def test_cli_list_csv(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(
        filings_module, "load_filings", lambda *a, **k: _sample_filings()
    )
    res = CliRunner().invoke(cli, ["filings", "list", "AAPL", "--csv"])
    assert res.exit_code == 0
    lines = [ln for ln in res.output.splitlines() if ln.strip()]
    assert lines[0].split(",") == [
        "symbol",
        "type",
        "filing_date",
        "accepted_date",
        "final_link",
        "index_link",
    ]
    assert lines[1].startswith("AAPL,10-K,2025-10-31,2025-10-31")


def test_cli_list_csv_blank_dates(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    blank = [
        Filing(
            symbol="AAPL",
            type="8-K",
            filing_date=None,
            accepted_date=None,
            link="i",
            final_link="f",
        )
    ]
    monkeypatch.setattr(filings_module, "load_filings", lambda *a, **k: blank)
    res = CliRunner().invoke(cli, ["filings", "list", "AAPL", "--csv"])
    assert res.exit_code == 0
    assert "AAPL,8-K,-,-,f,i" in res.output


# ── CLI: filings report ──────────────────────────────────────────────────────


def _sample_report():
    return parse_report(_REPORT_PAYLOAD)


def test_cli_report_requires_api_key(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: None)
    res = CliRunner().invoke(cli, ["filings", "report", "AAPL", "--year", "2024"])
    assert res.exit_code != 0 and "FMP_API_KEY" in res.output


def test_cli_report_year_required(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    res = CliRunner().invoke(cli, ["filings", "report", "AAPL"])
    assert res.exit_code != 0 and "year" in res.output.lower()


def test_cli_report_empty_ticker(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    res = CliRunner().invoke(cli, ["filings", "report", "  ", "--year", "2024"])
    assert res.exit_code != 0 and "ticker" in res.output.lower()


def test_cli_report_not_found(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: None)
    res = CliRunner().invoke(
        cli, ["filings", "report", "ZZZZ", "--year", "2024", "--period", "Q3"]
    )
    assert res.exit_code == 0 and "No Q3 report found for ZZZZ 2024." in res.output


def test_cli_report_lists_sections_by_default(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: _sample_report())
    res = CliRunner().invoke(cli, ["filings", "report", "AAPL", "--year", "2024"])
    assert res.exit_code == 0
    assert "73 sections" not in res.output  # only the fixture's sections
    assert "Cover Page" in res.output and "Revenue_2" in res.output


def test_cli_report_list_sections_flag(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: _sample_report())
    res = CliRunner().invoke(
        cli, ["filings", "report", "AAPL", "--year", "2024", "--list-sections"]
    )
    assert res.exit_code == 0 and "AAPL 2024 FY" in res.output


def test_cli_report_renders_matched_section(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: _sample_report())
    res = CliRunner().invoke(
        cli,
        ["filings", "report", "AAPL", "--year", "2024", "--section", "balance sheet"],
    )
    assert res.exit_code == 0
    # Two matching sections rendered (statement + parenthetical).
    assert "CONSOLIDATED BALANCE SHEETS" in res.output
    assert "Cash and cash equivalents" in res.output
    assert "364980" in res.output


def test_cli_report_multiple_patterns_dedup(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: _sample_report())
    res = CliRunner().invoke(
        cli,
        [
            "filings",
            "report",
            "AAPL",
            "--year",
            "2024",
            "--section",
            "revenue",
            "--section",
            "revenue",  # duplicate pattern must not double-render
        ],
    )
    assert res.exit_code == 0
    assert res.output.count("Net sales") == 1


def test_cli_report_unmatched_pattern_lists_sections(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: _sample_report())
    res = CliRunner().invoke(
        cli,
        ["filings", "report", "AAPL", "--year", "2024", "--section", "does-not-exist"],
    )
    assert res.exit_code == 0
    assert "No section matched 'does-not-exist'." in res.output
    # Falls back to listing the available sections.
    assert "Cover Page" in res.output


def test_cli_report_partial_match_and_miss(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: _sample_report())
    res = CliRunner().invoke(
        cli,
        [
            "filings",
            "report",
            "AAPL",
            "--year",
            "2024",
            "--section",
            "cover",
            "--section",
            "nope",
        ],
    )
    assert res.exit_code == 0
    assert "No section matched 'nope'." in res.output
    assert "12 Months Ended" in res.output  # cover page still rendered


def test_cli_report_json_output(monkeypatch, tmp_path):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: _sample_report())
    out_path = tmp_path / "revenue.json"
    res = CliRunner().invoke(
        cli,
        [
            "filings",
            "report",
            "AAPL",
            "--year",
            "2024",
            "--section",
            "revenue",
            "--json",
            str(out_path),
        ],
    )
    assert res.exit_code == 0 and f"Wrote 2 section(s) to {out_path}." in res.output
    data = json.loads(out_path.read_text())
    assert data["Revenue"] == [{"Net sales": ["391035", "383285"]}]
    assert data["Revenue_2"] == [{"Deferred revenue": ["8249"]}]


def test_cli_report_json_with_no_match_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: _sample_report())
    out_path = tmp_path / "missing.json"
    res = CliRunner().invoke(
        cli,
        [
            "filings",
            "report",
            "AAPL",
            "--year",
            "2024",
            "--section",
            "does-not-exist",
            "--json",
            str(out_path),
        ],
    )
    assert res.exit_code != 0
    assert "No sections matched" in res.output
    assert not out_path.exists()


def test_cli_report_renders_empty_section(monkeypatch):
    monkeypatch.setattr(filings_cmd, "resolve_api_key", lambda: "key")
    empty = FinancialReport(
        symbol="AAPL",
        period="FY",
        year=2024,
        sections=[ReportSection("Empty Section", [])],
    )
    monkeypatch.setattr(filings_module, "load_report", lambda *a, **k: empty)
    res = CliRunner().invoke(
        cli, ["filings", "report", "AAPL", "--year", "2024", "--section", "empty"]
    )
    assert res.exit_code == 0 and "No rows in this section." in res.output
