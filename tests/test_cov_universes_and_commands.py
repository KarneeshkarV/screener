"""Offline coverage tests for core utility/CLI modules.

Drives several small modules to (near) 100% line coverage without any
network access. All external seams — Turso/libSQL client, price fetchers,
HTTP/Wikipedia/NSE calls, the FMP provider — are stubbed or monkeypatched.
"""

from __future__ import annotations


import json


from datetime import date


from pathlib import Path


import pandas as pd


import pytest


from click.testing import CliRunner


from screener import history as history_mod


from screener import regime as regime_mod


from screener import universes as universes_mod


from screener import usage


from screener.cli import cli


class _FakeRows:
    def __init__(self, rows):
        self.rows = rows


class _FakeClient:
    def __init__(self, select_rows=None):
        self.statements: list[tuple[str, list[object] | None]] = []
        self.closed = False
        self._select_rows = select_rows or []

    def execute(self, stmt: str, args: list[object] | None = None):
        self.statements.append((stmt, args))
        if stmt.lstrip().upper().startswith("SELECT"):
            return _FakeRows(self._select_rows)
        return _FakeRows([])

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def history_db(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    monkeypatch.setattr(history_mod, "DB_PATH", db)
    return db


class _StubQuery:
    """Mimics tradingview_screener.Query's fluent chaining."""

    def __init__(self, result):
        self._result = result

    def set_markets(self, *a):
        return self

    def select(self, *a):
        return self

    def where(self, *a):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a):
        return self

    def get_scanner_data(self):
        return self._result


def _scanner_frame():
    return pd.DataFrame(
        {
            "name": ["AAA", "BBB"],
            "description": ["Alpha", "Beta"],
            "close": [100.0, 50.0],
            "change": [1.0, -1.0],
            "volume": [10000.0, 20000.0],
            "market_cap_basic": [1e9, 2e9],
            "EMA5": [101, 51],
            "EMA20": [100, 50],
            "EMA100": [98, 48],
            "EMA200": [95, 45],
            "RSI": [60, 40],
        }
    )


@pytest.fixture
def universes_dir(tmp_path, monkeypatch):
    d = tmp_path / "universes"
    monkeypatch.setattr(universes_mod, "CACHE_DIR", d)
    return d


class _Resp:
    def __init__(self, text=None, status=None):
        self.text = text or ""
        self._status = status

    def raise_for_status(self):
        if self._status:
            raise RuntimeError("bad status")


def test_vol_regime_short_series_unknown():
    out = regime_mod.vol_regime(pd.Series([100.0]))
    assert (out == "unknown").all()


def test_universe_validators():
    u = universes_mod.Universe(
        name="sp500",
        symbols=(" AAPL ", "", "MSFT"),
        source=" wiki ",
        cached_path=Path("/x"),
    )
    assert u.symbols == ("AAPL", "MSFT")
    assert u.source == "wiki"
    with pytest.raises(Exception):
        universes_mod.Universe(
            name="sp500", symbols=("",), source="s", cached_path=Path("/x")
        )
    with pytest.raises(Exception):
        universes_mod.Universe(
            name="sp500", symbols=("AAPL",), source="  ", cached_path=Path("/x")
        )


def test_write_and_read_cache_roundtrip(universes_dir):
    path = universes_mod._write_cache(
        "sp500",
        date(2024, 1, 1),
        ["AAPL", "MSFT"],
        "wiki",
        point_in_time=True,
        metadata={"k": "v"},
    )
    assert path.exists()
    result = universes_mod._read_cache("sp500", date(2024, 1, 1))
    assert result is not None
    universe, pit, metadata = result
    assert universe.symbols == ("AAPL", "MSFT")
    assert pit is True
    assert metadata["k"] == "v"


def test_read_cache_missing_file(universes_dir):
    assert universes_mod._read_cache("sp500", date(2024, 1, 1)) is None


def test_read_cache_skips_blank_lines(universes_dir):
    universes_dir.mkdir(parents=True, exist_ok=True)
    path = universes_mod._cache_path("sp500", date(2024, 1, 1))
    path.write_text("# point_in_time=true\n\n# source=wiki\nAAPL\n\nMSFT\n")
    result = universes_mod._read_cache("sp500", date(2024, 1, 1))
    assert result is not None
    universe, pit, _ = result
    assert universe.symbols == ("AAPL", "MSFT")
    assert pit is True


def test_read_cache_without_pit_header_is_miss(universes_dir):
    universes_dir.mkdir(parents=True, exist_ok=True)
    path = universes_mod._cache_path("sp500", date(2024, 1, 1))
    path.write_text("# source=wiki\nAAPL\nMSFT\n")
    assert universes_mod._read_cache("sp500", date(2024, 1, 1)) is None


def test_dedupe():
    assert universes_mod._dedupe(["A", "A", "", "B"]) == ["A", "B"]


def test_flatten_and_clean_symbol():
    assert universes_mod._flatten_columns([("Added", "Ticker"), "Date"]) == [
        "added ticker",
        "date",
    ]
    assert universes_mod._clean_change_symbol(None) == ""
    assert universes_mod._clean_change_symbol(float("nan")) == ""
    assert universes_mod._clean_change_symbol("nan") == ""
    assert universes_mod._clean_change_symbol("brk.b") == "BRK-B"


def test_normalize_sp500_symbols():
    out = universes_mod._normalize_sp500_symbols(pd.Series([" brk.b ", "aapl"]))
    assert out.tolist() == ["BRK-B", "AAPL"]


def test_warn_not_point_in_time_emits():
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        universes_mod._warn_not_point_in_time("nifty50", date(2020, 1, 1))


def test_load_current_universe_cache_hit(universes_dir, monkeypatch):
    universes_mod._write_cache(
        "sp500", date.today(), ["AAPL"], "wiki", point_in_time=True
    )

    # No fetch should occur.
    def boom(*a, **k):
        raise AssertionError("must not fetch")

    monkeypatch.setattr(universes_mod, "_fetch_sp500_pit", boom)
    u = universes_mod.load_current_universe("sp500", as_of=date.today())
    assert u.symbols == ("AAPL",)


def test_load_current_universe_unknown_name(universes_dir):
    with pytest.raises(ValueError, match="unknown universe"):
        universes_mod.load_current_universe("bogus", use_cache=False)  # type: ignore[arg-type]


def test_load_current_universe_sp500_fetch(universes_dir, monkeypatch):
    monkeypatch.setattr(
        universes_mod,
        "_fetch_sp500_pit",
        lambda as_of, use_cache: (["AAPL", "MSFT"], "wiki", True),
    )
    u = universes_mod.load_current_universe(
        "sp500", as_of=date.today(), use_cache=False
    )
    assert u.symbols == ("AAPL", "MSFT")
    assert u.source == "wiki"


def test_load_current_universe_nifty_past_warns(universes_dir, monkeypatch):
    monkeypatch.setattr(universes_mod, "_fetch_nifty50", lambda: (["RELIANCE"], "nse"))
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        u = universes_mod.load_current_universe(
            "nifty50", as_of=date(2000, 1, 1), use_cache=False
        )
    assert u.symbols == ("RELIANCE",)


def test_load_current_universe_sp500_past_stale_cache(universes_dir, monkeypatch):
    past = date(2020, 1, 1)
    universes_mod._write_cache("sp500", past, ["OLD"], "wiki", point_in_time=True)
    monkeypatch.setattr(
        universes_mod, "_sp500_pit_cache_matches_change_log", lambda metadata: False
    )
    monkeypatch.setattr(
        universes_mod,
        "_fetch_sp500_pit",
        lambda as_of, use_cache: (["NEW"], "wiki", True),
    )
    monkeypatch.setattr(
        universes_mod,
        "_sp500_changes_cache_metadata",
        lambda: {"sp500_changes_mtime_ns": "1"},
    )
    u = universes_mod.load_current_universe("sp500", as_of=past)
    assert u.symbols == ("NEW",)


def test_load_current_universe_cache_hit_warns_when_not_pit(universes_dir):
    past = date(2000, 1, 1)
    universes_mod._write_cache(
        "nifty50", past, ["RELIANCE"], "nse", point_in_time=False
    )
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        u = universes_mod.load_current_universe("nifty50", as_of=past)
    assert u.symbols == ("RELIANCE",)


def test_sp500_changes_cache_metadata_missing(universes_dir):
    assert universes_mod._sp500_changes_cache_metadata() is None


def test_sp500_changes_cache_metadata_present(universes_dir):
    universes_dir.mkdir(parents=True, exist_ok=True)
    universes_mod._changes_cache_path().write_text("[]")
    meta = universes_mod._sp500_changes_cache_metadata()
    assert meta is not None and "sp500_changes_mtime_ns" in meta


def test_sp500_pit_cache_matches_change_log_stale(universes_dir, monkeypatch):
    monkeypatch.setattr(universes_mod, "is_fresh", lambda *a, **k: False)
    assert universes_mod._sp500_pit_cache_matches_change_log({}) is False


def test_sp500_pit_cache_matches_change_log_no_expected(universes_dir, monkeypatch):
    monkeypatch.setattr(universes_mod, "is_fresh", lambda *a, **k: True)
    monkeypatch.setattr(universes_mod, "_sp500_changes_cache_metadata", lambda: None)
    assert universes_mod._sp500_pit_cache_matches_change_log({}) is False


def test_sp500_pit_cache_matches_change_log_true(universes_dir, monkeypatch):
    monkeypatch.setattr(universes_mod, "is_fresh", lambda *a, **k: True)
    monkeypatch.setattr(
        universes_mod,
        "_sp500_changes_cache_metadata",
        lambda: {"sp500_changes_mtime_ns": "5"},
    )
    assert (
        universes_mod._sp500_pit_cache_matches_change_log(
            {"sp500_changes_mtime_ns": "5"}
        )
        is True
    )


def test_read_sp500_html_resilience_none(monkeypatch):
    monkeypatch.setattr(universes_mod, "call_with_resilience", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        universes_mod._read_sp500_html()


def test_read_sp500_html_parses_tables(monkeypatch):
    expected = [pd.DataFrame({"Symbol": ["AAPL"]})]
    monkeypatch.setattr(
        universes_mod, "call_with_resilience", lambda *a, **k: _Resp(text="<html/>")
    )
    monkeypatch.setattr(universes_mod.pd, "read_html", lambda *a, **k: expected)
    tables = universes_mod._read_sp500_html()
    assert tables and "Symbol" in tables[0].columns


def test_read_sp500_html_no_tables(monkeypatch):
    monkeypatch.setattr(
        universes_mod, "call_with_resilience", lambda *a, **k: _Resp(text="<html/>")
    )
    monkeypatch.setattr(universes_mod.pd, "read_html", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="table not found"):
        universes_mod._read_sp500_html()


def test_fetch_sp500_table_missing_symbol(monkeypatch):
    df = pd.DataFrame({"Other": ["x"]})
    monkeypatch.setattr(universes_mod, "_read_sp500_html", lambda: [df])
    with pytest.raises(RuntimeError, match="missing Symbol"):
        universes_mod._fetch_sp500_table()


def test_fetch_sp500(monkeypatch):
    df = pd.DataFrame({"Symbol": ["aapl", "brk.b", "aapl"]})
    monkeypatch.setattr(universes_mod, "_read_sp500_html", lambda: [df])
    symbols, source = universes_mod._fetch_sp500()
    assert symbols == ["AAPL", "BRK-B"]
    assert "wikipedia" in source


def test_fetch_sp500_changes_parses(monkeypatch):
    constituents = pd.DataFrame({"Symbol": ["AAPL"]})
    changes = pd.DataFrame(
        {
            ("Date", "Date"): ["January 1, 2023", "bad-date", "February 1, 2023"],
            ("Added", "Ticker"): ["NEW", "", ""],
            ("Removed", "Ticker"): ["OLD", "GONE", ""],
        }
    )
    monkeypatch.setattr(
        universes_mod, "_read_sp500_html", lambda: [constituents, changes]
    )
    rows = universes_mod._fetch_sp500_changes()
    assert (date(2023, 1, 1), "NEW", "OLD") in rows
    # Unparseable date is dropped; the all-blank Feb row is dropped too.
    assert len(rows) == 1


def test_fetch_sp500_changes_no_changes_table(monkeypatch):
    constituents = pd.DataFrame({"Symbol": ["AAPL"]})
    monkeypatch.setattr(universes_mod, "_read_sp500_html", lambda: [constituents])
    assert universes_mod._fetch_sp500_changes() == []


def test_fetch_sp500_changes_missing_columns(monkeypatch):
    constituents = pd.DataFrame({"Symbol": ["AAPL"]})
    # Has 'date' and 'added' header words but no ticker columns.
    changes = pd.DataFrame({"date added foo": ["x"], "added foo": ["y"]})
    monkeypatch.setattr(
        universes_mod, "_read_sp500_html", lambda: [constituents, changes]
    )
    assert universes_mod._fetch_sp500_changes() == []


def test_load_sp500_changes_cache_read(universes_dir):
    universes_dir.mkdir(parents=True, exist_ok=True)
    path = universes_mod._changes_cache_path()
    path.write_text(json.dumps([["2023-01-01", "NEW", "OLD"]]))
    changes = universes_mod._load_sp500_changes(use_cache=True)
    assert changes == [(date(2023, 1, 1), "NEW", "OLD")]


def test_load_sp500_changes_corrupt_cache_refetches(universes_dir, monkeypatch):
    universes_dir.mkdir(parents=True, exist_ok=True)
    path = universes_mod._changes_cache_path()
    path.write_text("not json")
    monkeypatch.setattr(universes_mod, "is_fresh", lambda *a, **k: True)
    monkeypatch.setattr(
        universes_mod, "_fetch_sp500_changes", lambda: [(date(2023, 1, 1), "N", "O")]
    )
    changes = universes_mod._load_sp500_changes(use_cache=True)
    assert changes == [(date(2023, 1, 1), "N", "O")]


def test_load_sp500_changes_stale_cache(universes_dir, monkeypatch):
    universes_dir.mkdir(parents=True, exist_ok=True)
    path = universes_mod._changes_cache_path()
    path.write_text(json.dumps([]))
    monkeypatch.setattr(universes_mod, "is_fresh", lambda *a, **k: False)
    monkeypatch.setattr(universes_mod, "_fetch_sp500_changes", lambda: [])
    assert universes_mod._load_sp500_changes(use_cache=True) == []


def test_fetch_sp500_pit_reconstructs(monkeypatch):
    monkeypatch.setattr(
        universes_mod, "_fetch_sp500", lambda: (["AAPL", "NEW"], "wiki")
    )
    monkeypatch.setattr(
        universes_mod,
        "_load_sp500_changes",
        lambda use_cache: [(date(2025, 1, 1), "NEW", "OLD")],
    )
    # as_of before the only change: undo it (remove NEW, add back OLD).
    symbols, source, pit = universes_mod._fetch_sp500_pit(
        date(2024, 1, 1), use_cache=False
    )
    assert "OLD" in symbols
    assert "NEW" not in symbols
    # The log's earliest change is after as_of so the set is incomplete.
    assert pit is False


def test_fetch_sp500_pit_log_reaches_back_is_pit(monkeypatch):
    monkeypatch.setattr(
        universes_mod, "_fetch_sp500", lambda: (["AAPL", "NEW"], "wiki")
    )
    monkeypatch.setattr(
        universes_mod,
        "_load_sp500_changes",
        lambda use_cache: [
            (date(2025, 1, 1), "NEW", "OLD"),
            # A change on/before as_of is left in place (exercises the skip).
            (date(2023, 1, 1), "KEEP", "DROP"),
        ],
    )
    symbols, source, pit = universes_mod._fetch_sp500_pit(
        date(2024, 1, 1), use_cache=False
    )
    assert "OLD" in symbols
    assert "NEW" not in symbols
    # The earliest logged change (2023) predates as_of, so the set is complete.
    assert pit is True


def test_fetch_sp500_pit_no_changes(monkeypatch):
    monkeypatch.setattr(universes_mod, "_fetch_sp500", lambda: (["AAPL"], "wiki"))
    monkeypatch.setattr(universes_mod, "_load_sp500_changes", lambda use_cache: [])
    symbols, source, pit = universes_mod._fetch_sp500_pit(
        date(2020, 1, 1), use_cache=False
    )
    assert symbols == ["AAPL"]
    assert pit is False


def test_fetch_sp500_pit_today_is_pit(monkeypatch):
    monkeypatch.setattr(universes_mod, "_fetch_sp500", lambda: (["AAPL"], "wiki"))
    monkeypatch.setattr(universes_mod, "_load_sp500_changes", lambda use_cache: [])
    symbols, source, pit = universes_mod._fetch_sp500_pit(date.today(), use_cache=False)
    assert pit is True


def test_fetch_nifty50(monkeypatch):
    csv = "Symbol\nRELIANCE\nTCS\n"
    monkeypatch.setattr(
        universes_mod, "call_with_resilience", lambda *a, **k: _Resp(text=csv)
    )
    symbols, source = universes_mod._fetch_nifty50()
    assert symbols == ["RELIANCE", "TCS"]


def test_fetch_nifty50_lowercase_col(monkeypatch):
    csv = "SYMBOL\nreliance\n"
    monkeypatch.setattr(
        universes_mod, "call_with_resilience", lambda *a, **k: _Resp(text=csv)
    )
    symbols, _ = universes_mod._fetch_nifty50()
    assert symbols == ["RELIANCE"]


def test_fetch_nifty50_resilience_none(monkeypatch):
    monkeypatch.setattr(universes_mod, "call_with_resilience", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        universes_mod._fetch_nifty50()


def test_fetch_nifty50_missing_symbol_col(monkeypatch):
    csv = "Foo\nbar\n"
    monkeypatch.setattr(
        universes_mod, "call_with_resilience", lambda *a, **k: _Resp(text=csv)
    )
    with pytest.raises(RuntimeError, match="missing Symbol"):
        universes_mod._fetch_nifty50()


def test_fetch_nifty500(monkeypatch):
    csv = "Symbol\nRELIANCE\nTCS\n"
    monkeypatch.setattr(
        universes_mod, "call_with_resilience", lambda *a, **k: _Resp(text=csv)
    )
    symbols, source = universes_mod._fetch_nifty500()
    assert symbols == ["RELIANCE.NS", "TCS.NS"]
    assert "nifty500" in source


def test_fetch_nifty500_lowercase_col(monkeypatch):
    csv = "SYMBOL\nreliance\n"
    monkeypatch.setattr(
        universes_mod, "call_with_resilience", lambda *a, **k: _Resp(text=csv)
    )
    symbols, _ = universes_mod._fetch_nifty500()
    assert symbols == ["RELIANCE.NS"]


def test_fetch_nifty500_resilience_none(monkeypatch):
    monkeypatch.setattr(universes_mod, "call_with_resilience", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        universes_mod._fetch_nifty500()


def test_fetch_nifty500_missing_symbol_col(monkeypatch):
    csv = "Foo\nbar\n"
    monkeypatch.setattr(
        universes_mod, "call_with_resilience", lambda *a, **k: _Resp(text=csv)
    )
    with pytest.raises(RuntimeError, match="missing Symbol"):
        universes_mod._fetch_nifty500()


def test_load_current_universe_nifty500_fetch(universes_dir, monkeypatch):
    monkeypatch.setattr(
        universes_mod,
        "_fetch_nifty500",
        lambda: (["RELIANCE.NS", "TCS.NS"], "nse"),
    )
    u = universes_mod.load_current_universe(
        "nifty500", as_of=date.today(), use_cache=False
    )
    assert u.symbols == ("RELIANCE.NS", "TCS.NS")
    assert u.source == "nse"


def test_load_current_universe_nifty500_past_warns(universes_dir, monkeypatch):
    monkeypatch.setattr(
        universes_mod, "_fetch_nifty500", lambda: (["RELIANCE.NS"], "nse")
    )
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        u = universes_mod.load_current_universe(
            "nifty500", as_of=date(2000, 1, 1), use_cache=False
        )
    assert u.symbols == ("RELIANCE.NS",)


def test_load_sp500_membership_cache_read(universes_dir):
    path = universes_mod._membership_cache_path("sp500", date.today())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"AAPL": "2020-01-01", "MSFT": None}))
    mem = universes_mod.load_sp500_membership(as_of=date.today())
    assert mem == {"AAPL": date(2020, 1, 1), "MSFT": None}


def test_load_sp500_membership_corrupt_cache_refetches(universes_dir, monkeypatch):
    path = universes_mod._membership_cache_path("sp500", date.today())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json")
    df = pd.DataFrame({"Symbol": ["AAPL", "AAPL"], "Date added": ["2020-01-01", "x"]})
    monkeypatch.setattr(universes_mod, "_fetch_sp500_table", lambda: df)
    mem = universes_mod.load_sp500_membership(as_of=date.today())
    assert mem == {"AAPL": date(2020, 1, 1)}


def test_load_sp500_membership_missing_date_col(universes_dir, monkeypatch):
    df = pd.DataFrame({"Symbol": ["AAPL"]})
    monkeypatch.setattr(universes_mod, "_fetch_sp500_table", lambda: df)
    with pytest.raises(RuntimeError, match="Date added"):
        universes_mod.load_sp500_membership(as_of=date.today(), use_cache=False)


def test_load_sp500_membership_fetch_writes_cache(universes_dir, monkeypatch):
    df = pd.DataFrame(
        {"Symbol": ["AAPL", "MSFT", ""], "Date added": ["2020-01-01", None, "x"]}
    )
    monkeypatch.setattr(universes_mod, "_fetch_sp500_table", lambda: df)
    mem = universes_mod.load_sp500_membership(as_of=date.today(), use_cache=False)
    assert mem["AAPL"] == date(2020, 1, 1)
    assert mem["MSFT"] is None


def test_cache_status_with_resolve_failure(monkeypatch, tmp_path):
    """_iter_files skips files whose resolve() raises OSError."""
    from screener.commands import cache as cache_cmd

    real_resolve = Path.resolve

    target = tmp_path / "root"
    target.mkdir()
    (target / "a.txt").write_text("x")

    def fake_resolve(self, *a, **k):
        if self.name == "a.txt":
            raise OSError("boom")
        return real_resolve(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    files = list(cache_cmd._iter_files(target))
    assert files == []


def test_human_size_units():
    from screener.commands import cache as cache_cmd

    assert cache_cmd._human_size(10) == "10 B"
    assert "KB" in cache_cmd._human_size(2048)
    assert "TB" in cache_cmd._human_size(5 * 1024**4)


def test_cache_clean_handles_unlink_failure(monkeypatch, tmp_path):
    from screener.commands import cache as cache_cmd
    import screener.cache as screener_cache

    root = tmp_path / "scanner"
    root.mkdir()
    old_file = root / "old.parquet"
    old_file.write_text("x")
    import os as _os

    old = __import__("time").time() - 100 * 86400
    _os.utime(old_file, (old, old))

    monkeypatch.setattr(screener_cache, "CACHE_ROOT", root)
    monkeypatch.setattr(
        cache_cmd,
        "known_cache_dirs",
        lambda: {"scanner": root},
    )

    def fake_unlink(self, *a, **k):
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", fake_unlink)
    res = CliRunner().invoke(cli, ["cache", "clean", "--older-than", "1"])
    assert res.exit_code == 0
    assert "Failed to remove" in res.output


def test_cache_clean_stat_failure(monkeypatch, tmp_path):
    """A file vanishing between listing and stat is skipped, not fatal."""
    from screener.commands import cache as cache_cmd

    root = tmp_path / "scanner"
    root.mkdir()
    f = root / "x.parquet"
    f.write_text("x")

    monkeypatch.setattr(cache_cmd, "known_cache_dirs", lambda: {"scanner": root})
    monkeypatch.setattr(cache_cmd, "_iter_files", lambda r: [root / "ghost.parquet"])

    res = CliRunner().invoke(cli, ["cache", "clean", "--older-than", "0"])
    assert res.exit_code == 0


def test_index_inclusion_no_events(monkeypatch):
    import screener.commands.index_inclusion as ii_mod
    from screener.index_inclusion import InclusionStudy

    monkeypatch.setattr(ii_mod, "load_sp500_membership", lambda **k: {})
    monkeypatch.setattr(ii_mod, "build_price_fetcher", lambda *a, **k: object())
    monkeypatch.setattr(
        ii_mod,
        "run_inclusion_study",
        lambda *a, **k: InclusionStudy(
            events=[], skipped=3, horizons=(5,), summaries=[]
        ),
    )
    res = CliRunner().invoke(cli, ["index-inclusion", "-m", "us"])
    assert res.exit_code == 0
    assert "No S&P 500 additions" in res.output
    assert "Skipped 3 event(s)" in res.output


def test_cli_config_overrides_log_level_and_json(tmp_path, monkeypatch):
    import screener.cli as cli_mod

    captured = {}
    monkeypatch.setattr(
        cli_mod,
        "configure_logging",
        lambda level, json: captured.update(level=level, json=json),
    )
    # Force re-evaluation by passing a config that supplies both.
    path = tmp_path / "c.yaml"
    path.write_text("log_level: DEBUG\nlog_json: true\n")
    res = CliRunner().invoke(cli, ["--config", str(path), "usage-report"])
    assert res.exit_code == 0
    assert captured == {"level": "DEBUG", "json": True}


def test_usage_report_with_invocations(monkeypatch):
    monkeypatch.setattr(
        usage,
        "feature_usage_counts",
        lambda: [
            usage.UsageCount(feature="screen", count=3, last_used_at="2026-05-10")
        ],
    )
    monkeypatch.setattr(
        usage,
        "invocation_rollup",
        lambda limit: [
            usage.InvocationRollup(
                feature="screen",
                market="us",
                criteria="garp",
                status="success",
                count=3,
                last_used_at="2026-05-10",
                top_extras="top=10",
            )
        ],
    )
    res = CliRunner().invoke(cli, ["usage-report"], env={"COLUMNS": "250"})
    assert res.exit_code == 0
    assert "Recent invocations" in res.output
    assert "garp" in res.output


def test_usage_report_no_invocations(monkeypatch):
    monkeypatch.setattr(usage, "feature_usage_counts", lambda: [])
    monkeypatch.setattr(usage, "invocation_rollup", lambda limit: [])
    res = CliRunner().invoke(cli, ["usage-report"])
    assert res.exit_code == 0
    assert "No invocations recorded yet." in res.output


def test_institutional_no_symbols():
    res = CliRunner().invoke(cli, ["institutional", "--tickers", " , "])
    assert res.exit_code != 0
    assert "at least one symbol" in res.output
