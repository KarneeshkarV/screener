"""Conviction card tests — offline, stubbed providers, no network."""

from __future__ import annotations

import json
from datetime import date

from click.testing import CliRunner

from screener import conviction as conviction_mod
from screener.cli import cli as package_cli
from screener.conviction import (
    PILLAR_WEIGHTS,
    _ok,
    _skipped,
    build_conviction_card,
    compose,
)

from tests.conftest import StubPriceFetcher, make_bars


def _price_env(n: int = 280):
    bars = make_bars(start="2024-01-02", n=n, drift=0.4, seed=7)
    bench = make_bars(start="2024-01-02", n=n, drift=0.05, seed=21, open_base=400.0)
    return bars, bench


def _patch_india_providers(monkeypatch) -> None:
    monkeypatch.setattr(
        conviction_mod,
        "_load_smart_money_india",
        lambda symbol, *, cache_ttl, refresh: {
            "promoter_pct_latest": 51.0,
            "promoter_pct_prev": 50.0,
            "promoter_change": 1.0,
            "latest_quarter": "Mar 2026",
        },
    )
    monkeypatch.setattr(
        conviction_mod,
        "_load_fundamentals",
        lambda symbol, market, *, cache_ttl, refresh: {
            "peg": 1.2,
            "sales_growth_5y": 20.0,
            "operating_profit_growth": 15.0,
            "eps_growth_5y": 18.0,
            "roe_5y": 20.0,
            "roce_or_roic": 22.0,
            "quarterly_profit_growth": 12.0,
        },
    )
    monkeypatch.setattr(conviction_mod, "_load_pledge", lambda symbol, *, refresh: 4.0)
    monkeypatch.setattr(
        conviction_mod, "_load_delivery", lambda symbol, as_of: (52.0, 45.0)
    )


def test_all_pillars_ok_india(monkeypatch):
    bars, bench = _price_env()
    as_of = bars.index[-1].date()
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars, "^NSEI": bench})
    _patch_india_providers(monkeypatch)

    card = build_conviction_card("RELIANCE", "india", as_of, fetcher)

    assert [p.name for p in card.pillars] == [
        "trend",
        "breakout",
        "volume",
        "smart_money",
        "fundamentals",
        "risk",
    ]
    assert all(p.status == "ok" for p in card.pillars)
    by_name = {p.name: p for p in card.pillars}
    assert by_name["smart_money"].score == 75.0  # 50 + 25 * +1.0pp
    assert by_name["fundamentals"].score == 100.0  # all 7 GARP checks pass
    assert by_name["risk"].score == 90.0  # 100 - 2.5 * 4% pledge
    assert "delivery 45.0%→52.0%" in by_name["volume"].evidence
    for pillar in card.pillars:
        assert pillar.evidence, f"{pillar.name} must carry an evidence line"
        assert 0.0 <= pillar.score <= 100.0
    expected = round(
        sum(PILLAR_WEIGHTS[p.name] * p.score for p in card.pillars)
        / sum(PILLAR_WEIGHTS[p.name] for p in card.pillars),
        1,
    )
    assert card.composite == expected


def test_us_skips_unavailable_pillars(monkeypatch):
    bars, bench = _price_env()
    as_of = bars.index[-1].date()
    fetcher = StubPriceFetcher({"AAPL": bars, "SPY": bench})
    monkeypatch.setattr(conviction_mod, "_fmp_api_key", lambda: None)
    monkeypatch.setattr(
        conviction_mod,
        "_load_fundamentals",
        lambda symbol, market, *, cache_ttl, refresh: None,
    )

    card = build_conviction_card("AAPL", "us", as_of, fetcher)

    by_name = {p.name: p for p in card.pillars}
    assert "risk" not in by_name  # India-only pillar
    assert by_name["smart_money"].status == "skipped"
    assert by_name["smart_money"].reason == "FMP_API_KEY not configured"
    assert by_name["smart_money"].label == "skipped(FMP_API_KEY not configured)"
    assert by_name["smart_money"].score is None
    assert by_name["fundamentals"].status == "skipped"
    assert by_name["fundamentals"].reason == "no fundamental data"
    ok = [p for p in card.pillars if p.status == "ok"]
    assert {p.name for p in ok} == {"trend", "breakout", "volume"}
    expected = round(
        sum(PILLAR_WEIGHTS[p.name] * p.score for p in ok)
        / sum(PILLAR_WEIGHTS[p.name] for p in ok),
        1,
    )
    assert card.composite == expected


def test_no_price_data_yields_skipped_card(monkeypatch):
    fetcher = StubPriceFetcher({})
    monkeypatch.setattr(conviction_mod, "_fmp_api_key", lambda: None)
    monkeypatch.setattr(
        conviction_mod,
        "_load_fundamentals",
        lambda symbol, market, *, cache_ttl, refresh: None,
    )

    card = build_conviction_card("AAPL", "us", date(2026, 1, 2), fetcher)

    assert all(p.status == "skipped" for p in card.pillars)
    assert all(p.reason for p in card.pillars)
    assert card.composite is None


def test_compose_renormalizes_weights():
    pillars = [
        _ok("trend", 80.0, "ev"),
        _ok("volume", 40.0, "ev"),
        _skipped("risk", "no data"),
    ]
    expected = round((0.25 * 80.0 + 0.15 * 40.0) / (0.25 + 0.15), 1)
    assert compose(pillars) == expected == 65.0
    assert compose([_skipped("trend", "no data")]) is None
    assert compose([]) is None


def test_cli_json_shape(monkeypatch):
    bars, bench = _price_env()
    as_of = bars.index[-1].date()
    fetcher = StubPriceFetcher({"AAPL": bars, "SPY": bench})
    monkeypatch.setattr(conviction_mod, "_fmp_api_key", lambda: None)
    monkeypatch.setattr(
        conviction_mod,
        "_load_fundamentals",
        lambda symbol, market, *, cache_ttl, refresh: None,
    )

    res = CliRunner().invoke(
        package_cli,
        ["conviction", "AAPL", "-m", "us", "--as-of", as_of.isoformat(), "--json"],
        obj=fetcher,
    )

    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["symbol"] == "AAPL"
    assert payload["market"] == "us"
    assert payload["as_of"] == as_of.isoformat()
    assert isinstance(payload["composite"], float)
    assert payload["weights"] == PILLAR_WEIGHTS
    assert [p["name"] for p in payload["pillars"]] == [
        "trend",
        "breakout",
        "volume",
        "smart_money",
        "fundamentals",
    ]
    for pillar in payload["pillars"]:
        assert {"name", "score", "evidence", "status", "reason"} <= set(pillar)
        assert pillar["status"] in {"ok", "skipped"}
        if pillar["status"] == "skipped":
            assert pillar["score"] is None and pillar["reason"]
        else:
            assert isinstance(pillar["score"], (int, float))


def test_cli_table_output(monkeypatch):
    bars, bench = _price_env()
    as_of = bars.index[-1].date()
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars, "^NSEI": bench})
    _patch_india_providers(monkeypatch)

    res = CliRunner().invoke(
        package_cli,
        ["conviction", "RELIANCE", "-m", "india", "--as-of", as_of.isoformat()],
        obj=fetcher,
    )

    assert res.exit_code == 0, res.output
    assert "Composite conviction" in res.output
    for name in ("trend", "breakout", "volume", "smart_money", "fundamentals", "risk"):
        assert name in res.output
