"""Offline contracts for market catalogues, source URLs and provider fetch symbols."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from screener.sector_rotation.analysis import fetch_sector_rotation
from screener.sector_rotation.catalogue import (
    SectorCatalogue,
    SectorGroup,
    load_sector_catalogue,
    select_sector_groups,
)
from screener.symbols import tv_to_yf


def _legacy_catalogue_data(market="india", members=None):
    return {
        "version": 1,
        "market": market,
        "benchmark": "^CRSLDX" if market == "india" else "^GSPC",
        "benchmark_name": "Nifty 500 price index"
        if market == "india"
        else "S&P 500 price index",
        "observed_on": "2026-10-20",
        "source": "https://www.youtube.com/watch?v=d-I6BG5F-Yg",
        "membership_basis": "curated_research",
        "membership_note": "Current curated members, not point-in-time history.",
        "groups": [
            {
                "id": "research_basket",
                "name": "Research basket",
                "parent": "Research",
                "kind": "industry",
                "video_at": "5:00",
                "members": members or ["NSE:RELIANCE", "NSE:TCS"],
            }
        ],
    }


def test_market_defaults_preserve_india_and_load_us():
    india = load_sector_catalogue()
    assert india == load_sector_catalogue(market="india")
    assert india.market == "india"
    assert india.benchmark == "^CRSLDX"
    assert all(m.startswith("NSE:") for g in india.groups for m in g.members)
    us = load_sector_catalogue(market="us")
    assert us.market == "us"
    assert us.version == india.version == 1
    assert us.benchmark == "^GSPC"
    assert us.benchmark_name == "S&P 500 price index"
    assert all(
        m.split(":")[0] in {"NASDAQ", "NYSE", "AMEX"}
        for g in us.groups
        for m in g.members
    )


def test_legacy_schema_one_preserves_defaults_and_video_timestamp(tmp_path):
    catalogue = SectorCatalogue.model_validate(_legacy_catalogue_data())
    group = catalogue.groups[0]
    assert group.video_at == "5:00"
    assert group.description == ""
    assert group.notes == ""
    assert group.drivers == group.risks == group.sources == ()
    path = tmp_path / "legacy.yaml"
    path.write_text(catalogue.model_dump_json(), encoding="utf-8")
    assert load_sector_catalogue(path) == catalogue
    data = _legacy_catalogue_data()
    del data["groups"][0]["video_at"]
    assert SectorCatalogue.model_validate(data).groups[0].video_at is None


def test_custom_catalogue_market_mismatch_is_left_to_caller(tmp_path):
    catalogue = load_sector_catalogue(market="us")
    path = tmp_path / "us.yaml"
    path.write_text(catalogue.model_dump_json(), encoding="utf-8")
    assert load_sector_catalogue(path).market == "us"
    assert load_sector_catalogue(path, market="us") == catalogue


@pytest.mark.parametrize("market", ["", "US", "uk", "bse", None, 1, [], {}])
def test_unsupported_loader_market_is_a_clean_error(market):
    with pytest.raises(ValueError, match="Sector catalogue: unsupported market"):
        load_sector_catalogue(market=market)


@pytest.mark.parametrize("market", ["", "US", "uk", None, 1])
def test_catalogue_rejects_unsupported_market(market):
    with pytest.raises(ValueError, match="market"):
        SectorCatalogue.model_validate(_legacy_catalogue_data(market=market))


def test_catalogue_requires_explicit_market(tmp_path):
    data = _legacy_catalogue_data()
    del data["market"]
    path = tmp_path / "missing-market.yaml"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=r"market\s+Field required"):
        load_sector_catalogue(path)


@pytest.mark.parametrize("text", ["groups: [", "", "[]", "null", "market: us"])
def test_malformed_or_incomplete_replacement_is_a_clean_error(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Sector catalogue: invalid configuration"):
        load_sector_catalogue(path)


def test_missing_replacement_file_is_a_clean_error(tmp_path):
    with pytest.raises(ValueError, match="Sector catalogue: invalid configuration"):
        load_sector_catalogue(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.sec.gov/edgar/search/",
        "http://investor.example.com/reports?year=2026#annual",
        "https://www.msci.com/documents/1296102/11185224/GICS+Methodology+2023.pdf",
        "https://example.com:8443/report",
        "https://example.com/report%20name",
    ],
)
def test_safe_primary_source_urls_are_not_limited_to_youtube(url):
    data = _legacy_catalogue_data()
    data["source"] = url
    data["groups"][0]["sources"] = [url]
    catalogue = SectorCatalogue.model_validate(data)
    assert catalogue.source == url
    assert catalogue.groups[0].sources == (url,)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "javascript:alert(1)",
        "data:text/html,hello",
        "file:///etc/passwd",
        "ftp://example.com/file",
        "//example.com/path",
        "https:///path",
        "https://",
        "https://user@example.com/",
        "https://user:pass@example.com/",
        "https://@example.com/",
        "https://localhost/",
        "https://localhost.localdomain/",
        "https://router.home.arpa/",
        "https://LOCALHOST.LOCALDOMAIN/",
        "https://router.lan/",
        "https://nas.home/",
        "https://intranet.corp/",
        "https://x.intranet/",
        "https://x.private/",
        "https://x.onion/",
        "https://x.example/",
        "https://ROUTER.LAN/",
        "https://router.lan./",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://169.254.169.254/",
        "https://host/",
        "https://foo..com/",
        "https://-foo.com/",
        "https://foo_.com/",
        "https://foo.invalid/",
        "https://foo.local/",
        "https://example.com:bad/",
        "https://example.com:65536/",
        "https://example.com:0/",
        "https://[broken/",
        " https://example.com/",
        "https://example.com/\n",
        "https://exa\tmple.com/",
        "https://example.com/\x00",
        "https://example.com/%0aevil",
        "https://example.com/%7f",
        "https://example.com/\x85",
        "https://example.com\\@evil.com/",
        "https://example%2ecom/",
        "https://example.com/%5cevil",
    ],
)
@pytest.mark.parametrize("field", ["source", "sources"])
def test_unsafe_source_urls_are_rejected_for_catalogue_and_groups(url, field):
    data = _legacy_catalogue_data()
    if field == "source":
        data["source"] = url
    else:
        data["groups"][0]["sources"] = [url]
    with pytest.raises(ValueError, match="Sector catalogue: source"):
        SectorCatalogue.model_validate(data)


@pytest.mark.parametrize(
    ("market", "members"),
    [
        ("india", ["NASDAQ:AAPL"]),
        ("india", ["NYSE:JPM"]),
        ("india", ["BSE:TCS"]),
        ("us", ["NSE:TCS"]),
        ("us", ["NASDAQ:AAPL", "NSE:TCS"]),
        ("us", ["LSE:ULVR"]),
    ],
)
def test_cross_market_exchanges_are_rejected(market, members):
    with pytest.raises(ValueError, match="Sector catalogue: members"):
        SectorCatalogue.model_validate(_legacy_catalogue_data(market, members))


@pytest.mark.parametrize(
    ("market", "members"),
    [
        ("india", ["NSE:TCS", "NSE:TCS"]),
        ("india", ["NSE:BAJAJ_AUTO", "NSE:BAJAJ-AUTO"]),
        ("india", ["NSE:M_M", "NSE:M&M"]),
        ("india", ["NSE:GVT_D", "NSE:GVT&D"]),
        ("us", ["NASDAQ:AAPL", "NYSE:AAPL"]),
        ("us", ["NYSE:BRK-B", "AMEX:BRK-B"]),
    ],
)
def test_provider_normalized_duplicates_are_rejected(market, members):
    with pytest.raises(ValueError, match="duplicate provider-normalized members"):
        SectorCatalogue.model_validate(_legacy_catalogue_data(market, members))


@pytest.mark.parametrize(
    "member",
    [
        "AAPL",
        "aapl",
        "NASDAQ:aapl",
        "NASDAQ: AAPL",
        "NASDAQ:AAPL\n",
        "NASDAQ:AAPL.NS",
        "NYSE:BRK.B",
        "NYSE:BRK_B",
        "NASDAQ:AAPL:EXTRA",
        "NASDAQ:^GSPC",
        "NYSE:BRK--B",
        "NYSE:BRK-",
        "NSE:_TCS",
        "NSE:TCS_",
        "NSE:TCS__X",
        "NSE:TCS/ABC",
    ],
)
def test_member_format_rejects_unmapped_or_ambiguous_symbols(member):
    with pytest.raises(ValueError, match="Sector catalogue: members"):
        SectorCatalogue.model_validate(_legacy_catalogue_data("us", [member]))


def test_valid_india_separator_aliases_still_load_without_duplicate_holdings():
    catalogue = SectorCatalogue.model_validate(
        _legacy_catalogue_data(members=["NSE:M_M", "NSE:BAJAJ_AUTO", "NSE:360ONE"])
    )
    assert [tv_to_yf(m, "india") for m in catalogue.groups[0].members] == [
        "M&M.NS",
        "BAJAJ-AUTO.NS",
        "360ONE.NS",
    ]


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("NASDAQ:AAPL", "AAPL"),
        ("NYSE:JPM", "JPM"),
        ("AMEX:XYZ", "XYZ"),
        ("NYSE:BRK-B", "BRK-B"),
        ("NYSE:BF-B", "BF-B"),
        ("NASDAQ:GOOGL", "GOOGL"),
    ],
)
def test_us_symbols_use_existing_provider_conversion(member, expected):
    catalogue = SectorCatalogue.model_validate(_legacy_catalogue_data("us", [member]))
    assert tv_to_yf(catalogue.groups[0].members[0], catalogue.market) == expected


def test_us_fetch_requests_provider_symbols_and_price_benchmark_once():
    members = ["NASDAQ:AAPL", "NYSE:JPM", "AMEX:XYZ", "NYSE:BRK-B", "NYSE:BF-B"]
    data = _legacy_catalogue_data("us", members)
    # Overlapping groups must not duplicate network requests.
    data["groups"].append(dict(data["groups"][0], id="overlap"))
    catalogue = SectorCatalogue.model_validate(data)

    class RecordingFetcher:
        def fetch(self, tickers, start, end):
            assert tickers == ["AAPL", "BF-B", "BRK-B", "JPM", "XYZ", "^GSPC"]
            assert start == date(2024, 11, 28)
            assert end == date(2026, 1, 6)
            index = pd.bdate_range("2025-01-01", "2026-01-06")
            return {
                ticker: pd.DataFrame({"close": 100.0}, index=index)
                for ticker in tickers
            }

    result = fetch_sector_rotation(
        select_sector_groups(catalogue),
        RecordingFetcher(),
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )
    assert set(result.members.ticker) == set(members)
    assert list(result.summary.status) == ["ok", "ok"]
    assert result.benchmark == "^GSPC"
    assert result.summary.return_pct.tolist() == [0.0, 0.0]


def test_us_catalogue_has_all_broad_sectors_and_useful_detailed_baskets():
    catalogue = load_sector_catalogue(market="us")
    assert {g.name for g in catalogue.groups if g.kind == "sector"} == {
        "Information technology",
        "Financials",
        "Health care",
        "Industrials",
        "Consumer discretionary",
        "Consumer staples",
        "Energy",
        "Materials",
        "Utilities",
        "Communication services",
        "Real estate",
    }
    assert sum(g.kind in {"industry", "theme"} for g in catalogue.groups) >= 25
    for group in catalogue.groups:
        assert 4 <= len(group.members) <= 8, group.id
        assert len(group.description.split()) >= 8, group.id
        assert len(group.drivers) >= 2 and all(
            len(d.split()) >= 3 for d in group.drivers
        ), group.id
        assert len(group.risks) >= 2 and all(
            len(r.split()) >= 3 for r in group.risks
        ), group.id
        assert group.sources, group.id
        assert all(url.startswith("https://") for url in group.sources), group.id
        assert group.video_at is None, group.id
        assert len({tv_to_yf(m, "us") for m in group.members}) == len(group.members)
        assert all("-" not in m and "_" not in m for m in group.members)
    note = catalogue.membership_note.lower()
    assert "not official" in note
    assert "survivorship" in note
    assert "distributions" in note and "excludes" in note
    assert "foreign-domiciled" in note and "reit" in note and "no etfs" in note


def test_overlapping_research_baskets_are_permitted():
    data = _legacy_catalogue_data()
    data["groups"].append(dict(data["groups"][0], id="overlap"))
    assert len(SectorCatalogue.model_validate(data).groups) == 2


def test_sector_group_standalone_still_rejects_duplicate_members():
    group = _legacy_catalogue_data()["groups"][0]
    group["members"] = ["NSE:M_M", "NSE:M&M"]
    with pytest.raises(ValueError, match="duplicate provider-normalized members"):
        SectorGroup.model_validate(group)
