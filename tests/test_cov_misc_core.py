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

from screener import cache as cache_mod
from screener import config as config_mod
from screener import history as history_mod
from screener import logging_config
from screener import regime as regime_mod
from screener import universes as universes_mod
from screener import usage
from screener import scanner as scanner_mod
from screener._registry import Registry, autodiscover
from screener.cli import cli


# ───────────────────────── usage.py ─────────────────────────


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


def test_usage_count_normalizes_and_rejects_empty_feature():
    uc = usage.UsageCount(feature="  screen  ", count=1, last_used_at=None)
    assert uc.feature == "screen"
    with pytest.raises(ValueError):
        usage.UsageCount(feature="   ", count=1, last_used_at=None)


def test_load_env_file_parses_and_skips_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "\n"
        "NO_EQUALS_LINE\n"
        'TURSO_DATABASE_URL="libsql://db.example"\n'
        "TURSO_AUTH_TOKEN='secret'\n"
    )
    values = usage._load_env_file(env)
    assert values == {
        "TURSO_DATABASE_URL": "libsql://db.example",
        "TURSO_AUTH_TOKEN": "secret",
    }


def test_load_env_file_missing_returns_empty(tmp_path):
    assert usage._load_env_file(tmp_path / "nope.env") == {}


def test_env_value_prefers_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SOME_KEY", "from-env")
    assert usage._env_value("SOME_KEY") == "from-env"


def test_database_url_rewrites_libsql_scheme(monkeypatch):
    monkeypatch.setattr(usage, "_env_value", lambda name: "libsql://db.example")
    assert usage._database_url() == "https://db.example"


def test_database_url_passthrough_non_libsql(monkeypatch):
    monkeypatch.setattr(usage, "_env_value", lambda name: "https://db.example")
    assert usage._database_url() == "https://db.example"


def test_database_url_none(monkeypatch):
    monkeypatch.setattr(usage, "_env_value", lambda name: None)
    assert usage._database_url() is None


def test_connect_returns_none_without_credentials(monkeypatch):
    monkeypatch.setattr(usage, "_database_url", lambda: None)
    monkeypatch.setattr(usage, "_env_value", lambda name: None)
    assert usage._connect() is None


def test_connect_builds_client(monkeypatch):
    import sys
    import types

    captured = {}

    def fake_create_client_sync(url, auth_token):
        captured["url"] = url
        captured["token"] = auth_token
        return "CLIENT"

    fake_mod = types.ModuleType("libsql_client")
    fake_mod.create_client_sync = fake_create_client_sync
    monkeypatch.setitem(sys.modules, "libsql_client", fake_mod)
    monkeypatch.setattr(usage, "_database_url", lambda: "https://db")
    monkeypatch.setattr(usage, "_env_value", lambda name: "tok")

    client = usage._connect()
    assert client == "CLIENT"
    assert captured == {"url": "https://db", "token": "tok"}


def test_ensure_tables_execute_ddl():
    client = _FakeClient()
    usage.ensure_usage_table(client)
    usage.ensure_invocations_table(client)
    joined = " ".join(stmt for stmt, _ in client.statements)
    assert "CREATE TABLE IF NOT EXISTS feature_usage" in joined
    assert "CREATE TABLE IF NOT EXISTS feature_usage_invocations" in joined


def test_coerce_bool_to_int():
    assert usage._coerce_bool_to_int(True) == 1
    assert usage._coerce_bool_to_int(False) == 0
    assert usage._coerce_bool_to_int(5) == 5


def test_normalize_criteria_variants():
    assert usage._normalize_criteria(None) is None
    assert usage._normalize_criteria(["a", None, "b"]) == "a,b"
    assert usage._normalize_criteria([None]) is None
    assert usage._normalize_criteria("x") == "x"


def test_record_feature_invocation_skips_under_pytest(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")
    called = {"connect": False}
    monkeypatch.setattr(usage, "_connect", lambda: called.__setitem__("connect", True))
    usage.record_feature_invocation("screen")
    assert called["connect"] is False


def test_record_feature_invocation_no_client(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: None)
    # No exception, no crash.
    usage.record_feature_invocation("screen")


def test_record_feature_invocation_inserts_with_params(monkeypatch):
    client = _FakeClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: client)
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "u")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")

    usage.record_feature_invocation(
        "screen",
        command_path="screener screen",
        duration_ms=12,
        status="success",
        params={
            "market": "us",
            "criteria_names": ["garp", "value"],
            "limit": 10,
            "refresh": True,
            "output_csv": False,
            "cache_ttl": "15m",
            "detail": True,
            "ignored_none": None,
        },
    )
    insert = [
        s for s in client.statements if "INSERT INTO feature_usage_invocations" in s[0]
    ]
    assert insert
    args = insert[0][1]
    # project, feature, market, criteria, limit_n, refresh, output_csv,
    # cache_ttl, extras_json, duration_ms, status, username, hostname
    assert args[0] == "screener"
    assert args[1] == "screen"
    assert args[2] == "us"
    assert args[3] == "garp,value"
    assert args[4] == 10
    assert args[5] == 1  # refresh True -> 1
    assert args[6] == "False"  # output_csv coerced to str
    assert args[7] == "15m"
    extras = json.loads(args[8])
    assert extras == {"detail": "True"}  # None dropped, flattened keys excluded
    assert client.closed


def test_record_feature_invocation_no_extras(monkeypatch):
    client = _FakeClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: client)
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "u")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")

    usage.record_feature_invocation("screen", params={"market": None})
    insert = [
        s for s in client.statements if "INSERT INTO feature_usage_invocations" in s[0]
    ]
    # extras_json column should be None
    assert insert[0][1][8] is None


def test_record_feature_invocation_swallows_exceptions(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", boom)
    # Must not raise.
    usage.record_feature_invocation("screen")


def test_invocation_rollup_no_client(monkeypatch):
    monkeypatch.setattr(usage, "_connect", lambda: None)
    assert usage.invocation_rollup() == []


def test_invocation_rollup_aggregates_rows(monkeypatch):
    rows = [
        (
            "screen",
            "us",
            "garp",
            "success",
            "2026-05-10T10:00:00Z",
            json.dumps({"top": "10"}),
        ),
        (
            "screen",
            "us",
            "garp",
            "success",
            "2026-05-11T10:00:00Z",
            json.dumps({"top": "10"}),
        ),
        ("screen", "us", "garp", "success", "2026-05-09T10:00:00Z", "not-json"),
        ("garp", "india", "", "Error", None, json.dumps(["list-not-dict"])),
        ("garp", "india", "", "Error", "2026-05-12T10:00:00Z", None),
    ]
    client = _FakeClient(select_rows=rows)
    monkeypatch.setattr(usage, "_connect", lambda: client)

    result = usage.invocation_rollup(limit=10)
    by_key = {(r.feature, r.status): r for r in result}
    screen = by_key[("screen", "success")]
    assert screen.count == 3
    assert screen.last_used_at == "2026-05-11T10:00:00Z"
    assert "top=10" in screen.top_extras
    garp = by_key[("garp", "Error")]
    assert garp.count == 2
    assert garp.top_extras == ""
    assert client.closed


def test_record_feature_usage_no_client(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: None)
    usage.record_feature_usage("screen")


def test_feature_usage_counts_no_client(monkeypatch):
    monkeypatch.setattr(usage, "_connect", lambda: None)
    assert usage.feature_usage_counts() == []


def test_elapsed_ms_is_non_negative():
    import time as _time

    assert usage.elapsed_ms(_time.perf_counter()) >= 0


# ───────────────────────── history.py ─────────────────────────


@pytest.fixture
def history_db(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    monkeypatch.setattr(history_mod, "DB_PATH", db)
    return db


def test_to_float_variants():
    assert history_mod._to_float(None) is None
    assert history_mod._to_float("3.5") == 3.5
    assert history_mod._to_float("abc") is None
    assert history_mod._to_float(float("nan")) is None
    assert history_mod._to_float(7) == 7.0


def test_save_run_and_previous_run_and_diff(history_db, monkeypatch):
    # Distinct run_ts per save so the (run_ts, market, criteria) UNIQUE
    # constraint never collides within the same wall-clock second.
    from datetime import datetime as _dt, timezone as _tz

    times = iter(
        [
            _dt(2026, 1, 1, 0, 0, 0, tzinfo=_tz.utc),
            _dt(2026, 1, 1, 0, 0, 1, tzinfo=_tz.utc),
        ]
    )

    class _FixedDateTime:
        @staticmethod
        def now(tz=None):
            return next(times)

    monkeypatch.setattr(history_mod, "datetime", _FixedDateTime)

    df1 = pd.DataFrame(
        {
            "name": ["AAA", "BBB", ""],  # blank ticker is skipped
            "description": ["Alpha Co", None, "Skip"],
            "close": [10.0, 20.0, 1.0],
            "change": [1.0, -2.0, 0.0],
            "volume": [1000, 2000, 5],
            "market_cap_basic": [1e9, 2e9, 1.0],
            "setup_score": [50.0, 60.0, 0.0],
        }
    )
    run_id = history_mod.save_run("us", "garp", 2, df1)
    assert run_id == 1

    df2 = pd.DataFrame(
        {
            "name": ["AAA", "CCC"],
            "description": ["Alpha Co", "Gamma"],
            "close": [11.0, 30.0],
            "change": [1.0, 3.0],
            "volume": [1100, 3000],
            "market_cap_basic": [1.1e9, 3e9],
            "setup_score": [55.0, 70.0],
        }
    )
    run_id2 = history_mod.save_run("us", "garp", 2, df2)
    assert run_id2 == 2

    prev = history_mod.previous_run("us", "garp", before_id=run_id2)
    assert prev is not None
    assert sorted(prev["ticker"].tolist()) == ["AAA", "BBB"]

    added, removed = history_mod.diff(df2, prev)
    assert added == ["CCC"]
    assert removed == ["BBB"]


def test_save_run_with_no_valid_rows(history_db):
    df = pd.DataFrame({"name": ["", None]})
    run_id = history_mod.save_run("us", "garp", 0, df)
    assert run_id == 1
    # No previous run before the first one.
    assert history_mod.previous_run("us", "garp", before_id=run_id) is None


def test_diff_handles_empty_and_none():
    added, removed = history_mod.diff(pd.DataFrame(), pd.DataFrame())
    assert added == [] and removed == []
    cur = pd.DataFrame({"name": ["AAA"]})
    added, removed = history_mod.diff(cur, None)
    assert added == ["AAA"] and removed == []


# ───────────────────────── _registry.py ─────────────────────────


def test_registry_full_api():
    reg: Registry[int] = Registry("widget")

    @reg.register("a", color="red")
    def _val():  # the decorator returns the value unchanged
        return 1

    reg.add("b", 2)
    assert "a" in reg
    assert len(reg) == 2
    assert reg.get_optional("a") is not None
    assert reg.get_optional(None) is None
    assert reg.get_optional("missing") is None
    assert sorted(reg.names()) == ["a", "b"]
    assert dict(reg.items()) == reg.as_dict()
    assert set(iter(reg)) == {"a", "b"}
    assert reg.meta("a") == {"color": "red"}
    assert reg.meta("b") == {}
    with pytest.raises(ValueError):
        reg.add("a", 99)
    with pytest.raises(KeyError):
        reg.get("missing")
    assert reg.get("b") == 2


def test_autodiscover_rejects_non_package():
    import types

    mod = types.ModuleType("notapkg")
    with pytest.raises(TypeError):
        autodiscover(mod)


def test_autodiscover_imports_submodules():
    import screener.commands as commands_pkg

    # Should import every submodule without error (side-effect registration).
    autodiscover(commands_pkg)


# ───────────────────────── config.py ─────────────────────────


def test_config_log_level_validation():
    cfg = config_mod.CliConfig(log_level="  DEBUG  ")
    assert cfg.log_level == "DEBUG"
    assert config_mod.CliConfig(log_level=None).log_level is None
    with pytest.raises(Exception):
        config_mod.CliConfig(log_level="   ")


def test_config_rejects_non_string_keys():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        config_mod.CliConfig.model_validate({1: "x"})


def test_load_config_yaml_and_json(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("log_level: DEBUG\nlog_json: true\nextra: 1\n")
    out = config_mod.load_config(y)
    assert out["log_level"] == "DEBUG"
    assert out["log_json"] is True
    assert out["extra"] == 1

    j = tmp_path / "c.json"
    j.write_text(json.dumps({"log_level": "INFO"}))
    assert config_mod.load_config(j)["log_level"] == "INFO"


def test_load_config_empty_yaml(tmp_path):
    y = tmp_path / "empty.yaml"
    y.write_text("")
    assert config_mod.load_config(y) == {}


def test_load_config_missing_file(tmp_path):
    import click

    with pytest.raises(click.UsageError, match="Config file not found"):
        config_mod.load_config(tmp_path / "missing.yaml")


def test_load_config_not_a_file(tmp_path):
    import click

    d = tmp_path / "adir.yaml"
    d.mkdir()
    with pytest.raises(click.UsageError, match="not a file"):
        config_mod.load_config(d)


def test_load_config_unsupported_extension(tmp_path):
    import click

    p = tmp_path / "c.txt"
    p.write_text("{}")
    with pytest.raises(click.UsageError, match="Unsupported"):
        config_mod.load_config(p)


def test_load_config_bad_yaml(tmp_path):
    import click

    p = tmp_path / "c.yaml"
    p.write_text("key: : : bad")
    with pytest.raises(click.UsageError, match="Could not load"):
        config_mod.load_config(p)


def test_load_config_validation_error(tmp_path):
    import click

    p = tmp_path / "c.yaml"
    p.write_text("log_level: ''")
    with pytest.raises(click.UsageError, match="must not be empty"):
        config_mod.load_config(p)


# ───────────────────────── cache.py ─────────────────────────


def test_parse_ttl_numeric_and_default():
    assert cache_mod.parse_ttl(None) is None
    assert cache_mod.parse_ttl(None, default=5.0) == 5.0
    assert cache_mod.parse_ttl(30) == 30.0
    assert cache_mod.parse_ttl("120") == 120.0


def test_is_fresh_negative_ttl_always_fresh(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x")
    assert cache_mod.is_fresh(p, -1) is True
    assert cache_mod.is_fresh(tmp_path / "missing", 10) is False
    assert cache_mod.is_fresh(p, None) is False


def test_read_json_default_on_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert cache_mod.read_json(p, default={"x": 1}) == {"x": 1}


def test_read_frame_default_on_error(tmp_path):
    p = tmp_path / "bad.parquet"
    p.write_text("garbage")
    assert cache_mod.read_frame(p) is None


def test_panel_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "PANEL_ROOT", tmp_path)
    assert cache_mod.panel_path("fii_dii") == tmp_path / "fii_dii.parquet"


def test_append_panel_snapshot_roundtrip_and_dedupe(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "PANEL_ROOT", tmp_path)

    # Empty rows on a non-existent panel returns an empty frame.
    out = cache_mod.append_panel_snapshot("p", pd.DataFrame(), dedupe_keys=["date"])
    assert out.empty

    first = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "v": [1, 2]})
    cache_mod.append_panel_snapshot("p", first, dedupe_keys=["date"])

    # Re-run same key overwrites (keep last) and date column normalized.
    second = pd.DataFrame({"date": ["2026-01-02", "2026-01-03"], "v": [99, 3]})
    merged = cache_mod.append_panel_snapshot("p", second, dedupe_keys=["date"])
    assert len(merged) == 3
    row = merged[merged["date"] == pd.Timestamp("2026-01-02")]
    assert row["v"].iloc[0] == 99

    # Empty rows on an existing panel returns the existing frame.
    existing = cache_mod.append_panel_snapshot(
        "p", pd.DataFrame(), dedupe_keys=["date"]
    )
    assert len(existing) == 3


def test_append_panel_snapshot_date_key_all_nan_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "PANEL_ROOT", tmp_path)
    # A date-named key whose values are all NaN -> empty parse sample ->
    # normalization is skipped (continue), but the row still persists.
    rows = pd.DataFrame({"as_of_date": [pd.NaT, pd.NaT], "v": [1, 2]})
    merged = cache_mod.append_panel_snapshot("d", rows, dedupe_keys=["as_of_date", "v"])
    assert "as_of_date" in merged.columns
    assert sorted(merged["v"].tolist()) == [1, 2]


def test_append_panel_snapshot_non_date_key(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "PANEL_ROOT", tmp_path)
    rows = pd.DataFrame({"sym": ["AAA"], "v": [1]})
    merged = cache_mod.append_panel_snapshot("q", rows, dedupe_keys=["sym"])
    assert merged["sym"].tolist() == ["AAA"]


def test_cached_json_call_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"v": 1}

    a = cache_mod.cached_json_call(
        "ns", ("k",), ttl_seconds=60, refresh=False, fetch=fetch
    )
    b = cache_mod.cached_json_call(
        "ns", ("k",), ttl_seconds=60, refresh=False, fetch=fetch
    )
    assert a == b == {"v": 1}
    assert calls["n"] == 1
    # refresh forces a re-fetch
    cache_mod.cached_json_call("ns", ("k",), ttl_seconds=60, refresh=True, fetch=fetch)
    assert calls["n"] == 2


# ───────────────────────── scanner.py ─────────────────────────


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


def test_get_scanner_data_cached_fetch_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path)
    df = _scanner_frame()
    query = _StubQuery((2, df))

    count, out = scanner_mod.get_scanner_data_cached(
        query, key_parts=("k",), columns=list(df.columns), cache_ttl=60, refresh=False
    )
    assert count == 2
    assert len(out) == 2

    # Second call hits the cache (use a query that would error if fetched).
    class _Boom(_StubQuery):
        def get_scanner_data(self):
            raise AssertionError("should not fetch")

    count2, out2 = scanner_mod.get_scanner_data_cached(
        _Boom(None),
        key_parts=("k",),
        columns=list(df.columns),
        cache_ttl=60,
        refresh=False,
    )
    assert count2 == 2
    assert len(out2) == 2


def test_get_scanner_data_cached_resilience_none(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(scanner_mod, "call_with_resilience", lambda *a, **k: None)
    count, out = scanner_mod.get_scanner_data_cached(
        _StubQuery(None), key_parts=("z",), columns=["name", "close"], refresh=True
    )
    assert count == 0
    assert out.empty
    assert list(out.columns) == ["name", "close"]


def test_scan_setup_score_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path)
    df = _scanner_frame()
    monkeypatch.setattr(scanner_mod, "Query", lambda: _StubQuery((2, df)))

    count, out = scanner_mod.scan(
        "us", filters=[], limit=10, order_by="setup_score", detail=False, refresh=True
    )
    assert count == 2
    assert "setup_score" in out.columns
    # setup-score helper columns are hidden in the output.
    assert "EMA5" not in out.columns


def test_scan_default_order_and_detail(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path)
    df = _scanner_frame()
    monkeypatch.setattr(scanner_mod, "Query", lambda: _StubQuery((2, df)))

    count, out = scanner_mod.scan(
        "us", filters=[], limit=10, order_by="volume", detail=True, refresh=True
    )
    assert count == 2
    assert not out.empty


def test_dedupe_listings_uses_ticker_fallback():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA-DUP"],
            "description": ["", ""],
        }
    )
    out = scanner_mod._dedupe_listings(df)
    # Both empty descriptions fall back to ticker, which differs -> both kept.
    assert len(out) == 2


def test_dedupe_listings_empty_returns_input():
    df = pd.DataFrame({"name": ["AAA"]})  # no description column
    assert scanner_mod._dedupe_listings(df).equals(df)


# ───────────────────────── logging_config.py ─────────────────────────


def test_configure_logging_json(monkeypatch):
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    logging_config.configure_logging(level="debug", json=True)
    assert logging_config._CONFIGURED is True
    # Second call is a no-op (idempotent).
    logging_config.configure_logging(level="INFO", json=False)


def test_get_logger_autoconfigures(monkeypatch):
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    log = logging_config.get_logger("x")
    assert log is not None
    assert logging_config._CONFIGURED is True


# ───────────────────────── regime.py ─────────────────────────


def test_vol_regime_short_series_unknown():
    out = regime_mod.vol_regime(pd.Series([100.0]))
    assert (out == "unknown").all()


# ───────────────────────── universes.py ─────────────────────────


@pytest.fixture
def universes_dir(tmp_path, monkeypatch):
    d = tmp_path / "universes"
    monkeypatch.setattr(universes_mod, "CACHE_DIR", d)
    return d


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


class _Resp:
    def __init__(self, text=None, status=None):
        self.text = text or ""
        self._status = status

    def raise_for_status(self):
        if self._status:
            raise RuntimeError("bad status")


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


# ───────────────────────── commands/cache.py ─────────────────────────


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


# ───────────────────────── commands/index_inclusion.py ─────────────────────────


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


# ───────────────────────── cli.py ─────────────────────────


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


# ───────────────────────── commands/institutional.py ─────────────────────────


def test_institutional_no_symbols():
    res = CliRunner().invoke(cli, ["institutional", "--tickers", " , "])
    assert res.exit_code != 0
    assert "at least one symbol" in res.output


def test_institutional_no_api_key(monkeypatch):
    import screener.insiders as insiders_mod

    monkeypatch.setattr(insiders_mod, "_fmp_api_key", lambda: None)
    res = CliRunner().invoke(cli, ["institutional", "--tickers", "AAPL"])
    assert res.exit_code != 0
    assert "FMP_API_KEY is not set" in res.output


def test_institutional_renders_results(monkeypatch):
    import screener.insiders as insiders_mod
    import screener.institutional as inst_mod

    monkeypatch.setattr(insiders_mod, "_fmp_api_key", lambda: "key")
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "holders": [10],
            "total_shares": [1000.0],
            "qoq_change_shares": [50.0],
            "qoq_change_pct": [5.0],
        }
    )
    monkeypatch.setattr(inst_mod, "fetch_fmp_institutional", lambda *a, **k: df)
    res = CliRunner().invoke(
        cli, ["institutional", "--tickers", "AAPL,MSFT"], env={"COLUMNS": "250"}
    )
    assert res.exit_code == 0
    # MSFT missing -> reported on stderr (mixed into output by CliRunner).
    assert "AAPL" in res.output


# ───────────────────────── institutional.py line 101 ─────────────────────────


def test_fetch_fmp_institutional_one_empty_rows(monkeypatch):
    import screener.institutional as inst_mod

    captured = {}

    def fake_provider_fetch(key, fn, **kwargs):
        captured["result"] = fn()
        return captured["result"]

    monkeypatch.setattr(
        inst_mod._FMP_INSTITUTIONAL_PROVIDER, "fetch", fake_provider_fetch
    )

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps([]).encode()  # empty list -> not rows -> None

    monkeypatch.setattr(inst_mod.urllib.request, "urlopen", lambda *a, **k: _R())
    out = inst_mod._fetch_fmp_institutional_one(
        "AAPL", api_key="k", cache_ttl=10, refresh=False
    )
    assert out is None


def test_fetch_fmp_institutional_one_aggregation_none(monkeypatch):
    """Non-empty rows that aggregate to None (no numeric shares) -> None."""
    import screener.institutional as inst_mod

    monkeypatch.setattr(
        inst_mod._FMP_INSTITUTIONAL_PROVIDER, "fetch", lambda key, fn, **kw: fn()
    )

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            # One row, but its 'shares' is non-numeric so aggregation yields None.
            return json.dumps([{"holder": "F", "shares": "not-a-number"}]).encode()

    monkeypatch.setattr(inst_mod.urllib.request, "urlopen", lambda *a, **k: _R())
    out = inst_mod._fetch_fmp_institutional_one(
        "AAPL", api_key="k", cache_ttl=10, refresh=False
    )
    assert out is None
