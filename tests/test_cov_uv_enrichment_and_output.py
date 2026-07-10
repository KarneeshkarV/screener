"""Offline coverage tests for the unusual_volume build-up / output / enrich /
detector / filters / classify modules.

All tests are deterministic and never touch the network: every NSE / provider /
openscreener seam is monkeypatched.
"""

from __future__ import annotations


import json


import sys


import types


from datetime import date


import numpy as np


import pandas as pd


from rich.console import Console


from screener.unusual_volume import enrich as E


from screener.unusual_volume import output as O


from screener.unusual_volume.detector import Event


def _event(**overrides) -> Event:
    base = dict(
        symbol="AAA",
        date=date(2026, 4, 24),
        close=100.0,
        pct_change=1.0,
        volume=1_000.0,
        avg_volume_20d=500.0,
        rvol=2.0,
        rvol_5d=2.0,
        rvol_50d=2.0,
        rvol_90d=2.0,
        z_score=2.0,
        pct_rank_252d=0.9,
        direction="BUYING",
        strength="MODERATE",
    )
    base.update(overrides)
    return Event(**base)


def _compression_bars(n: int = 80, seed: int = 11) -> pd.DataFrame:
    """A long, tightly-compressed, gently rising panel that lights every
    build-up fingerprint."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    # Gently rising price with shrinking range toward the end.
    base = 100.0 + np.linspace(0, 4.0, n)
    rng = np.linspace(2.0, 0.2, n)
    openp = base - rng * 0.1
    close = base + rng * 0.1  # close near high, up days
    high = base + rng * 0.5
    low = base - rng * 0.5
    # Up-day volume heavy.
    volume = np.full(n, 20_000.0)
    df = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    return df


def test_deep_enrich_india_extract_raises_typeerror_skipped(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, **kwargs):
            pass

        def fetch(self, sections):
            return {"shareholding": [{"promoters": object()}]}  # float() -> TypeError

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    import screener.insiders as ins

    monkeypatch.setattr(ins, "_HttpScraper", lambda *a, **k: object(), raising=False)
    ev = _event(notes="")
    E.deep_enrich_india([ev])
    assert ev.notes == ""


def test_deep_enrich_india_extract_exception_continues(monkeypatch):
    class FakeStock:
        def __init__(self, symbol, **kwargs):
            pass

        def fetch(self, sections):
            return {"shareholding": [{"promoters": "10"}]}

    monkeypatch.setitem(
        sys.modules, "openscreener", types.SimpleNamespace(Stock=FakeStock)
    )
    import screener.insiders as ins

    monkeypatch.setattr(ins, "_HttpScraper", lambda *a, **k: object(), raising=False)

    def boom(_df):
        raise ValueError("explode")

    monkeypatch.setattr(E, "_extract_promoter_pct", boom)
    ev = _event(notes="orig")
    E.deep_enrich_india([ev])
    assert ev.notes == "orig"  # exception swallowed, note untouched


def test_fetch_shareholding_fetch_typeerror_falls_back_no_arg():
    class Stock:
        def fetch(self, sections=None):
            if sections is not None:
                raise TypeError("no positional")
            return {"shareholding": [{"promoters": "10"}]}

    out = E._fetch_shareholding_quarterly(Stock())
    assert out == [{"promoters": "10"}]


def test_fetch_shareholding_payload_without_shareholding_uses_attr():
    class Stock:
        def fetch(self, sections):
            return {"something_else": 1}  # no "shareholding" key

        def shareholding_quarterly(self):
            return ["data"]

    assert E._fetch_shareholding_quarterly(Stock()) == ["data"]


def test_fetch_shareholding_non_callable_attr():
    class Stock:
        fetch = None  # not callable
        shareholding_quarterly = ["frozen"]  # not callable -> returned directly

    assert E._fetch_shareholding_quarterly(Stock()) == ["frozen"]


def test_extract_promoter_pct_none():
    assert E._extract_promoter_pct(None) is None


def test_extract_promoter_pct_empty_list():
    assert E._extract_promoter_pct([]) is None


def test_extract_promoter_pct_list_not_dict():
    assert E._extract_promoter_pct(["not a dict"]) is None


def test_extract_promoter_pct_list_no_promoter_key():
    assert E._extract_promoter_pct([{"foo": "1"}]) is None


def test_extract_promoter_pct_list_value():
    assert E._extract_promoter_pct([{"promoters": "55.5%"}]) == 55.5


def test_extract_promoter_pct_dataframe_index():
    df = pd.DataFrame({"Mar 2026": ["48.2%"]}, index=["Promoters"])
    assert E._extract_promoter_pct(df) == 48.2


def test_extract_promoter_pct_dataframe_empty_row():
    df = pd.DataFrame(index=["Promoters"])  # zero columns -> len(row)==0
    assert E._extract_promoter_pct(df) is None


def test_extract_promoter_pct_dataframe_nan_value():
    df = pd.DataFrame({"q1": [float("nan")]}, index=["Promoters"])
    assert E._extract_promoter_pct(df) is None


def test_extract_promoter_pct_no_promoter_row():
    df = pd.DataFrame({"q1": ["10%"]}, index=["Public"])
    assert E._extract_promoter_pct(df) is None


def test_extract_promoter_pct_value_error_swallowed():
    df = pd.DataFrame({"q1": ["not-a-number"]}, index=["Promoters"])
    assert E._extract_promoter_pct(df) is None


def test_sort_events_nan_rvol():
    a = _event(symbol="A", strength="HIGH", rvol=float("nan"))
    b = _event(symbol="B", strength="HIGH", rvol=5.0)
    out = O.sort_events([a, b])
    assert out[0].symbol == "B"  # nan rvol sorts low


def test_render_rich_empty():
    console = Console(record=True)
    O.render_rich([], "us", date(2026, 1, 1), console)
    assert "No unusual-volume events" in console.export_text()


def test_render_rich_us_table():
    console = Console(record=True)
    ev = _event(symbol="AAA", sector="Tech", notes="hi", buildup_score=0.5)
    O.render_rich([ev], "us", date(2026, 1, 1), console)
    txt = console.export_text()
    assert "AAA" in txt and "Unusual Volume" in txt


def test_render_rich_india_with_fii_footer():
    console = Console(record=True)
    ev = _event(
        symbol="REL",
        direction="QUIET_ACCUMULATION",
        strength="EXTREME",
        delivery_pct=60.0,
        delivery_rvol=2.0,
        conviction_score=1.5,
        pcr=0.8,
        call_put_oi_ratio=1.2,
        pledge_pct=5.0,
        fii_5d_net=1000.0,
        dii_5d_net=-500.0,
        fii_trend=0.3,
    )
    O.render_rich([ev], "india", date(2026, 1, 1), console)
    txt = console.export_text()
    assert "REL" in txt
    assert "FII" in txt


def test_render_rich_india_no_fii_footer():
    console = Console(record=True)
    ev = _event(symbol="REL")  # all FII/DII None -> footer empty
    O.render_rich([ev], "india", date(2026, 1, 1), console)
    assert "REL" in console.export_text()


def test_color_helpers_unknown_passthrough():
    assert O._color_direction("WEIRD") == "WEIRD"
    assert O._color_strength("WEIRD") == "WEIRD"
    assert "green" in O._color_direction("BUYING")
    assert "red" in O._color_strength("EXTREME")


def test_json_safe_variants():
    assert O._json_safe(None) is None
    assert O._json_safe({"a": 1, "b": float("nan")}) == {"a": 1, "b": None}
    assert O._json_safe([1, 2.0, float("inf")]) == [1, 2.0, None]
    assert O._json_safe((1,)) == [1]
    assert O._json_safe(True) is True  # bool not coerced to int
    assert O._json_safe(np.int64(5)) == 5
    assert O._json_safe(3.5) == 3.5
    assert O._json_safe("text") == "text"


def test_json_safe_isna_raises_falls_through():
    # pd.isna on a multi-element array raises -> except branch (lines 158-159),
    # then the value (a non-Real container) is returned unchanged.
    arr = np.array([1, 2, 3])
    out = O._json_safe(arr)
    assert isinstance(out, np.ndarray)


def test_write_json_roundtrip(tmp_path):
    ev = _event(symbol="AAA")
    path = tmp_path / "events.json"
    O.write_json([ev], path)
    payload = json.loads(path.read_text())
    assert payload[0]["symbol"] == "AAA"


def test_write_markdown_us(tmp_path):
    evs = [
        _event(symbol="A", direction="BUYING"),
        _event(symbol="B", direction="SELLING"),
        _event(symbol="R", direction="REVERSAL"),
        _event(symbol="C", direction="CHURN"),
        _event(
            symbol="BU",
            direction="BUILDUP",
            buildup_score=0.7,
            buildup_flags=["compression"],
        ),
    ]
    path = tmp_path / "out.md"
    O.write_markdown(evs, path, "us", date(2026, 1, 1))
    text = path.read_text()
    assert "# Unusual Volume — US" in text
    assert "## BUYING" in text
    assert "## BUILDUP" in text
    assert "compression" in text


def test_write_markdown_india_with_quiet_and_fii(tmp_path):
    evs = [
        _event(
            symbol="Q",
            direction="QUIET_ACCUMULATION",
            delivery_pct=60.0,
            fii_5d_net=1.0,
            dii_5d_net=2.0,
            fii_trend=0.1,
        ),
    ]
    path = tmp_path / "out_india.md"
    O.write_markdown(evs, path, "india", date(2026, 1, 1))
    text = path.read_text()
    assert "# Unusual Volume — INDIA" in text
    assert "## QUIET_ACCUMULATION" in text
    assert "Market-wide FII/DII" in text


def test_write_markdown_buildup_none_score(tmp_path):
    ev = _event(symbol="BU", direction="BUILDUP", buildup_score=None, buildup_flags=[])
    path = tmp_path / "bu.md"
    O.write_markdown([ev], path, "us", date(2026, 1, 1))
    text = path.read_text()
    assert "## BUILDUP" in text


def test_sort_by_buildup_handles_none():
    a = _event(symbol="A", buildup_score=None)
    b = _event(symbol="B", buildup_score=0.9)
    out = O._sort_by_buildup([a, b])
    assert out[0].symbol == "B"
