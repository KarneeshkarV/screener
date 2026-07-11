"""Offline coverage tests for core utility/CLI modules.

Drives several small modules to (near) 100% line coverage without any
network access. All external seams — Turso/libSQL client, price fetchers,
HTTP/Wikipedia/NSE calls, the FMP provider — are stubbed or monkeypatched.
"""

from __future__ import annotations


import json


import pandas as pd


import pytest


from screener import cache as cache_mod


from screener import config as config_mod


from screener import history as history_mod


from screener import logging_config


from screener import universes as universes_mod


from screener import usage


from screener import scanner as scanner_mod


from screener._registry import Registry, autodiscover


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


def test_cached_json_call_reuses_cached_none(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return None

    first = cache_mod.cached_json_call(
        "ns", ("missing",), ttl_seconds=60, refresh=False, fetch=fetch
    )
    second = cache_mod.cached_json_call(
        "ns", ("missing",), ttl_seconds=60, refresh=False, fetch=fetch
    )

    assert first is None and second is None
    assert calls["n"] == 1


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
