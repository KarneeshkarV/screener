"""Offline line-coverage tests for insiders / conviction / commands.insiders / pledge.

All tests are deterministic and never touch the network: every provider seam,
HTTP/urlopen, FMP/NSE/openscreener call and scanner fetch is stubbed or
monkeypatched. CLI flows use ``click.testing.CliRunner``.
"""

from __future__ import annotations


import json


from datetime import date


import pandas as pd


from click.testing import CliRunner


from screener import conviction as conviction_mod


from screener import garp as garp_module


from screener import insiders as insiders_mod


from screener.cli import cli as package_cli


from screener.commands import insiders as cmd_insiders


class _FakeYfTicker:
    def __init__(self, purchases):
        self.insider_purchases = purchases


def _purchases_frame():
    return pd.DataFrame(
        {
            "Insider Purchases Last 6m": [
                "Net Shares Purchased (Sold)",
                "% Net Shares Purchased (Sold)",
                "Total Insider Shares Held",
                "Purchases",
                "Sales",
            ],
            "Shares": [1000.0, 5.0, 50000.0, None, None],
            "Trans": [None, None, None, 3.0, 1.0],
        }
    )


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class _FakeStock:
    def __init__(self, rows):
        self._rows = rows

    def __call__(self, name, scraper=None):
        return self

    def shareholding_quarterly(self):
        return self._rows


def _install_openscreener(monkeypatch, rows):
    import sys
    import types

    mod = types.ModuleType("openscreener")
    stock = _FakeStock(rows)
    mod.Stock = lambda name, scraper=None: stock
    monkeypatch.setitem(sys.modules, "openscreener", mod)


class _Ev:
    def __init__(self, symbol):
        self.symbol = symbol
        self.pledge_pct = None


class _VolEvent:
    def __init__(self, rvol, z, direction, strength="strong"):
        self.rvol = rvol
        self.z_score = z
        self.direction = direction
        self.strength = strength


def _stub_scanner(monkeypatch, universe, total=500):
    monkeypatch.setattr(
        cmd_insiders,
        "get_scanner_data_cached",
        lambda *a, **k: (total, universe.copy()),
    )
    monkeypatch.setattr(cmd_insiders, "_dedupe_listings", lambda df: df)


def _base_universe():
    return pd.DataFrame(
        [
            {
                "name": "ACME",
                "description": "Acme Inc",
                "close": 100.0,
                "change": 1.0,
                "volume": 50000,
                "market_cap_basic": 1e9,
            }
        ]
    )


def test_score_smart_money_india_full():
    res = conviction_mod._score_smart_money_india(
        {
            "promoter_change": 1.0,
            "promoter_pct_latest": 51.0,
            "promoter_pct_prev": 50.0,
            "latest_quarter": "Mar 2026",
        }
    )
    assert "promoter 50.00%→51.00%" in res.evidence
    assert "qtr Mar 2026" in res.evidence


def test_promoter_pair_as_of_too_few():
    rows = [{"date": "Mar 2024", "promoters": 50.0}]
    assert conviction_mod._promoter_pair_as_of(rows, date(2026, 1, 1)) is None


def test_promoter_pair_as_of_missing_promoters():
    rows = [
        {"date": "Jun 2023", "promoters": None},
        {"date": "Sep 2023", "promoters": None},
    ]
    assert conviction_mod._promoter_pair_as_of(rows, date(2026, 1, 1)) is None


def test_load_smart_money_india_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "openscreener":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = conviction_mod._load_smart_money_india(
        "RELIANCE", date(2026, 1, 1), cache_ttl=None, refresh=False
    )
    assert out is None


def test_load_smart_money_india_non_list(monkeypatch):
    _install_openscreener(monkeypatch, [])
    monkeypatch.setattr(
        conviction_mod._OPENSCREENER_SHAREHOLDING_PROVIDER,
        "fetch",
        lambda *a, **k: "not-a-list",
    )
    out = conviction_mod._load_smart_money_india(
        "RELIANCE", date(2026, 1, 1), cache_ttl=None, refresh=False
    )
    assert out is None


def test_smart_money_pillar_us_pit_stale():
    res = conviction_mod._smart_money_pillar(
        "AAPL", "us", date(2020, 1, 1), cache_ttl=None, refresh=False
    )
    assert res.status == "skipped"
    assert "point-in-time" in res.reason


def test_smart_money_pillar_us_error(monkeypatch):
    monkeypatch.setattr(conviction_mod, "resolve_api_key", lambda: "k")

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(conviction_mod, "_load_smart_money_us", boom)
    res = conviction_mod._smart_money_pillar(
        "AAPL", "us", date.today(), cache_ttl=None, refresh=False
    )
    assert res.status == "skipped"
    assert "FMP error" in res.reason


def test_smart_money_pillar_us_no_payload(monkeypatch):
    monkeypatch.setattr(conviction_mod, "resolve_api_key", lambda: "k")
    monkeypatch.setattr(conviction_mod, "_load_smart_money_us", lambda *a, **k: None)
    res = conviction_mod._smart_money_pillar(
        "AAPL", "us", date.today(), cache_ttl=None, refresh=False
    )
    assert res.status == "skipped"
    assert "no Form 4" in res.reason


def test_smart_money_pillar_us_ok(monkeypatch):
    monkeypatch.setattr(conviction_mod, "resolve_api_key", lambda: "k")
    monkeypatch.setattr(
        conviction_mod,
        "_load_smart_money_us",
        lambda *a, **k: {
            "fmp_buy_shares_6m": 100.0,
            "fmp_sell_shares_6m": 0.0,
            "fmp_net_shares_6m": 100.0,
            "fmp_buy_trans_6m": 1,
            "fmp_sell_trans_6m": 0,
        },
    )
    res = conviction_mod._smart_money_pillar(
        "AAPL", "us", date.today(), cache_ttl=None, refresh=False
    )
    assert res.status == "ok"


def test_smart_money_pillar_india_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("oops")

    monkeypatch.setattr(conviction_mod, "_load_smart_money_india", boom)
    res = conviction_mod._smart_money_pillar(
        "RELIANCE", "india", date.today(), cache_ttl=None, refresh=False
    )
    assert res.status == "skipped"
    assert "promoter data error" in res.reason


def test_smart_money_pillar_india_no_payload(monkeypatch):
    monkeypatch.setattr(conviction_mod, "_load_smart_money_india", lambda *a, **k: None)
    res = conviction_mod._smart_money_pillar(
        "RELIANCE", "india", date.today(), cache_ttl=None, refresh=False
    )
    assert res.status == "skipped"
    assert "no promoter shareholding" in res.reason


def test_score_fundamentals_insufficient():
    res = conviction_mod.score_fundamentals({"peg": 1.0}, conviction_mod.US_THRESHOLDS)
    assert res.status == "skipped"


def test_score_fundamentals_with_failures():
    row = {
        "peg": 0.5,
        "sales_growth_5y": -5.0,  # fail
        "operating_profit_growth": 100.0,
        "eps_growth_5y": -1.0,  # fail
        "roe_5y": 50.0,
    }
    res = conviction_mod.score_fundamentals(row, conviction_mod.US_THRESHOLDS)
    assert res.status == "ok"
    assert "missed" in res.evidence


def test_load_fundamentals_india(monkeypatch):
    monkeypatch.setattr(garp_module, "cached_json_call", lambda *a, **k: {"x": 1})
    monkeypatch.setattr(
        garp_module, "_india_row", lambda sym, name, payload: {"peg": 1.0}
    )
    out = conviction_mod._load_fundamentals(
        "RELIANCE", "india", cache_ttl=None, refresh=False
    )
    assert out is not None and out["peg"] == 1.0


def test_load_fundamentals_india_non_dict(monkeypatch):
    monkeypatch.setattr(garp_module, "cached_json_call", lambda *a, **k: None)
    out = conviction_mod._load_fundamentals(
        "RELIANCE", "india", cache_ttl=None, refresh=False
    )
    assert out is None


def test_load_fundamentals_us_fmp(monkeypatch):
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: "k")
    monkeypatch.setattr(garp_module, "_fetch_fmp_us_cached", lambda *a, **k: {"raw": 1})
    monkeypatch.setattr(
        garp_module, "_fmp_us_row", lambda sym, name, payload: {"peg": 1.0}
    )
    out = conviction_mod._load_fundamentals("AAPL", "us", cache_ttl=None, refresh=False)
    assert out is not None and out["peg"] == 1.0


def test_load_fundamentals_us_fmp_row_none_falls_back(monkeypatch):
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: "k")
    monkeypatch.setattr(garp_module, "_fetch_fmp_us_cached", lambda *a, **k: {"raw": 1})
    monkeypatch.setattr(garp_module, "_fmp_us_row", lambda sym, name, payload: None)
    monkeypatch.setattr(garp_module, "_us_row", lambda sym, name: {"peg": 2.0})
    out = conviction_mod._load_fundamentals("AAPL", "us", cache_ttl=None, refresh=False)
    assert out is not None and out["peg"] == 2.0


def test_load_fundamentals_us_no_key(monkeypatch):
    monkeypatch.setattr(garp_module, "resolve_api_key", lambda: None)
    monkeypatch.setattr(garp_module, "_us_row", lambda sym, name: {"peg": 3.0})
    out = conviction_mod._load_fundamentals("AAPL", "us", cache_ttl=None, refresh=False)
    assert out is not None and out["peg"] == 3.0


def test_fundamentals_pillar_stale():
    res = conviction_mod._fundamentals_pillar(
        "AAPL", "us", date(2020, 1, 1), cache_ttl=None, refresh=False
    )
    assert res.status == "skipped"
    assert "point-in-time" in res.reason


def test_fundamentals_pillar_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("err")

    monkeypatch.setattr(conviction_mod, "_load_fundamentals", boom)
    res = conviction_mod._fundamentals_pillar(
        "AAPL", "us", date.today(), cache_ttl=None, refresh=False
    )
    assert res.status == "skipped"
    assert "provider error" in res.reason


def test_fundamentals_pillar_no_row(monkeypatch):
    monkeypatch.setattr(conviction_mod, "_load_fundamentals", lambda *a, **k: None)
    res = conviction_mod._fundamentals_pillar(
        "AAPL", "us", date.today(), cache_ttl=None, refresh=False
    )
    assert res.status == "skipped"
    assert "no fundamental data" in res.reason


def test_score_pledge():
    res = conviction_mod.score_pledge(4.0)
    assert res.score == 90.0


def test_load_pledge(monkeypatch):
    monkeypatch.setattr(
        conviction_mod, "resolve_pledge_pct", lambda sym, name, *, refresh: 5.0
    )
    assert conviction_mod._load_pledge("RELIANCE", refresh=False) == 5.0


def test_risk_pillar_stale():
    res = conviction_mod._risk_pillar("RELIANCE", date(2020, 1, 1), refresh=False)
    assert res.status == "skipped"


def test_risk_pillar_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(conviction_mod, "_load_pledge", boom)
    res = conviction_mod._risk_pillar("RELIANCE", date.today(), refresh=False)
    assert res.status == "skipped"
    assert "pledge provider error" in res.reason


def test_risk_pillar_no_pledge(monkeypatch):
    monkeypatch.setattr(conviction_mod, "_load_pledge", lambda *a, **k: None)
    res = conviction_mod._risk_pillar("RELIANCE", date.today(), refresh=False)
    assert res.status == "skipped"
    assert "no promoter pledge" in res.reason


def test_risk_pillar_ok(monkeypatch):
    monkeypatch.setattr(conviction_mod, "_load_pledge", lambda *a, **k: 4.0)
    res = conviction_mod._risk_pillar("RELIANCE", date.today(), refresh=False)
    assert res.status == "ok"
    assert res.score == 90.0


def test_load_delivery(monkeypatch):
    monkeypatch.setattr(
        conviction_mod, "load_delivery_panel", lambda syms, as_of, history_days: "panel"
    )
    monkeypatch.setattr(
        conviction_mod, "delivery_lookup", lambda panel: {"RELIANCE": (50.0, 45.0)}
    )
    out = conviction_mod._load_delivery("RELIANCE", date(2026, 1, 2))
    assert out == (50.0, 45.0)


def test_load_delivery_exception(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("delivery down")

    monkeypatch.setattr(conviction_mod, "load_delivery_panel", boom)
    assert conviction_mod._load_delivery("RELIANCE", date(2026, 1, 2)) is None


def test_render_card_all_skipped():
    from rich.console import Console

    card = conviction_mod.ConvictionCard(
        symbol="X",
        market="us",
        as_of=date(2026, 1, 2),
        composite=None,
        pillars=[conviction_mod._skipped("trend", "no data")],
    )
    console = Console(record=True)
    conviction_mod.render_card(card, console)
    text = console.export_text()
    assert "all pillars skipped" in text


def test_run_promoter_buys_empty_universe(monkeypatch):
    _stub_scanner(monkeypatch, pd.DataFrame(), total=0)
    res = CliRunner().invoke(
        package_cli, ["promoter-buys", "-m", "us", "--universe-size", "5"]
    )
    assert res.exit_code == 0
    assert "No tickers returned" in res.output


def test_run_promoter_buys_us_with_fmp_truncated(monkeypatch):
    universe = _base_universe()
    _stub_scanner(monkeypatch, universe)
    monkeypatch.setattr(
        insiders_mod,
        "fetch_yfinance_insiders",
        lambda u, m, **k: pd.DataFrame(
            [{"name": "ACME", "yf_net_shares_6m": 10.0, "yf_net_pct_6m": 5.0}]
        ),
    )
    monkeypatch.setattr(
        insiders_mod,
        "fetch_fmp_insiders",
        lambda u, m, **k: pd.DataFrame(
            [
                {
                    "name": "ACME",
                    "fmp_symbol": "ACME",
                    "fmp_truncated": True,
                    "fmp_net_shares_6m": 500.0,
                }
            ]
        ),
    )
    res = CliRunner().invoke(
        package_cli,
        ["promoter-buys", "-m", "us", "--min-market-cap", "1000"],
    )
    assert res.exit_code == 0, res.output
    assert "hit the page cap" in res.output
    assert "ACME" in res.output


def test_run_promoter_buys_us_csv_fmp_only(monkeypatch):
    universe = _base_universe()
    _stub_scanner(monkeypatch, universe)
    monkeypatch.setattr(
        insiders_mod, "fetch_yfinance_insiders", lambda u, m, **k: pd.DataFrame()
    )
    monkeypatch.setattr(
        insiders_mod,
        "fetch_fmp_insiders",
        lambda u, m, **k: pd.DataFrame(
            [
                {
                    "name": "ACME",
                    "fmp_symbol": "ACME",
                    "fmp_truncated": False,
                    "fmp_net_shares_6m": 500.0,
                }
            ]
        ),
    )
    res = CliRunner().invoke(package_cli, ["promoter-buys", "-m", "us", "--csv"])
    assert res.exit_code == 0, res.output
    assert "ACME" in res.output


def test_run_promoter_buys_us_no_insider_data(monkeypatch):
    universe = _base_universe()
    _stub_scanner(monkeypatch, universe)
    monkeypatch.setattr(
        insiders_mod, "fetch_yfinance_insiders", lambda u, m, **k: pd.DataFrame()
    )
    monkeypatch.setattr(
        insiders_mod, "fetch_fmp_insiders", lambda u, m, **k: pd.DataFrame()
    )
    res = CliRunner().invoke(package_cli, ["promoter-buys", "-m", "us"])
    assert res.exit_code == 0
    assert "No insider data returned" in res.output


def test_run_promoter_buys_us_no_matches(monkeypatch):
    universe = _base_universe()
    _stub_scanner(monkeypatch, universe)
    monkeypatch.setattr(
        insiders_mod,
        "fetch_yfinance_insiders",
        lambda u, m, **k: pd.DataFrame([{"name": "ACME", "yf_net_shares_6m": -10.0}]),
    )
    monkeypatch.setattr(
        insiders_mod, "fetch_fmp_insiders", lambda u, m, **k: pd.DataFrame()
    )
    res = CliRunner().invoke(package_cli, ["promoter-buys", "-m", "us"])
    assert res.exit_code == 0
    assert "No tickers passed" in res.output


def test_run_promoter_buys_india_merge(monkeypatch):
    universe = _base_universe()
    _stub_scanner(monkeypatch, universe)
    monkeypatch.setattr(
        insiders_mod,
        "fetch_yfinance_insiders",
        lambda u, m, **k: pd.DataFrame(
            [{"name": "ACME", "yf_net_shares_6m": 5.0, "yf_net_pct_6m": 2.0}]
        ),
    )
    monkeypatch.setattr(
        insiders_mod,
        "fetch_openscreener_promoters",
        lambda u, **k: pd.DataFrame([{"name": "ACME", "promoter_change": 1.0}]),
    )
    res = CliRunner().invoke(package_cli, ["promoter-buys", "-m", "india"])
    assert res.exit_code == 0, res.output
    assert "ACME" in res.output


def test_run_promoter_buys_india_openscreener_empty_fallback(monkeypatch):
    universe = _base_universe()
    _stub_scanner(monkeypatch, universe)
    # The yf frame carries a promoter_change column so the India filter builds
    # a real Series mask on the fallback path (here all below threshold).
    monkeypatch.setattr(
        insiders_mod,
        "fetch_yfinance_insiders",
        lambda u, m, **k: pd.DataFrame(
            [{"name": "ACME", "yf_net_shares_6m": 5.0, "promoter_change": -1.0}]
        ),
    )
    monkeypatch.setattr(
        insiders_mod, "fetch_openscreener_promoters", lambda u, **k: pd.DataFrame()
    )
    res = CliRunner().invoke(package_cli, ["promoter-buys", "-m", "india"])
    assert res.exit_code == 0, res.output
    # yfinance-only fallback path; no India promoter_change so filter drops all.
    assert "Falling back to yfinance only" in res.output
    assert "No tickers passed" in res.output


def test_run_promoter_buys_us_outer_merge(monkeypatch):
    universe = _base_universe()
    _stub_scanner(monkeypatch, universe)
    monkeypatch.setattr(
        insiders_mod,
        "fetch_yfinance_insiders",
        lambda u, m, **k: pd.DataFrame(
            [{"name": "ACME", "yf_net_shares_6m": 5.0, "yf_net_pct_6m": 2.0}]
        ),
    )
    monkeypatch.setattr(
        insiders_mod,
        "fetch_fmp_insiders",
        lambda u, m, **k: pd.DataFrame(
            [
                {
                    "name": "ACME",
                    "fmp_symbol": "ACME",
                    "fmp_truncated": False,
                    "fmp_net_shares_6m": 200.0,
                }
            ]
        ),
    )
    res = CliRunner().invoke(package_cli, ["promoter-buys", "-m", "us"])
    assert res.exit_code == 0, res.output
    assert "ACME" in res.output


def test_resolve_api_key_is_fmp_resolver():
    import screener.fmp as fmp_mod

    assert insiders_mod.resolve_api_key is fmp_mod.resolve_api_key


def test_load_insider_aggregate_keys_name_on_symbol(monkeypatch):
    captured: dict = {}

    def fake(name, symbol, *, api_key, cache_ttl, refresh):
        captured.update(name=name, symbol=symbol, api_key=api_key, refresh=refresh)
        return {"name": name, "fmp_net_shares_6m": 5.0}

    monkeypatch.setattr(insiders_mod, "_fetch_fmp_insider_one", fake)
    out = insiders_mod.load_insider_aggregate(
        "AAPL", api_key="k", cache_ttl=None, refresh=True
    )
    assert out == {"name": "AAPL", "fmp_net_shares_6m": 5.0}
    # Both the display name and the FMP symbol are keyed on the single argument.
    assert captured == {
        "name": "AAPL",
        "symbol": "AAPL",
        "api_key": "k",
        "refresh": True,
    }
