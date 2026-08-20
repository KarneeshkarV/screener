from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from screener import usage
from screener.cli import cli


class FakeResult:
    rows = [("screen", 2, "2026-05-10T12:00:00.000Z"), ("garp", 1, None)]


class FakeClient:
    def __init__(self, rows=None) -> None:
        self.statements: list[tuple[str, list[object] | None]] = []
        self.closed = False
        self.rows = rows if rows is not None else FakeResult.rows

    def execute(self, stmt: str, args: list[object] | None = None):
        self.statements.append((stmt, args))
        if stmt.lstrip().upper().startswith("SELECT"):
            return types.SimpleNamespace(rows=self.rows)
        return FakeResult()

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_usage_client_state():
    usage._reset_client_state()
    yield
    usage._reset_client_state()


def test_record_feature_usage_inserts_success(monkeypatch):
    client = FakeClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: client)
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "karneeshkar")
    monkeypatch.setattr(usage.platform, "node", lambda: "workstation")

    usage.record_feature_usage("screen", command_path="screener screen", duration_ms=42)
    # Usage alone is staged; flush ships it (CLI normally pairs with invocation).
    usage.flush_usage(timeout_s=1.0)

    insert = [
        item for item in client.statements if "INSERT INTO feature_usage" in item[0]
    ]
    assert insert
    assert insert[0][1] == [
        "screener",
        "screen",
        "screener screen",
        "success",
        42,
        "karneeshkar",
        "workstation",
    ]


def test_usage_models_and_env_helpers(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="feature must not be empty"):
        usage.UsageCount(feature=" ", count=1, last_used_at=None)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n# comment\nTURSO_DATABASE_URL='libsql://db.example'\n"
        'TURSO_AUTH_TOKEN="token"\nIGNORED\n'
    )
    assert usage._load_env_file(tmp_path / "missing.env") == {}
    assert usage._load_env_file(env_file) == {
        "TURSO_DATABASE_URL": "libsql://db.example",
        "TURSO_AUTH_TOKEN": "token",
    }

    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.setattr(
        usage, "_load_env_file", lambda: {"TURSO_DATABASE_URL": "file-url"}
    )
    assert usage._env_value("TURSO_DATABASE_URL") == "file-url"

    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://remote")
    assert usage._database_url() == "https://remote"
    monkeypatch.setenv("TURSO_DATABASE_URL", "https://remote")
    assert usage._database_url() == "https://remote"


def test_connect_uses_libsql_client_when_configured(monkeypatch):
    created = {}
    create_calls = {"n": 0}

    def create_client_sync(url, auth_token):
        create_calls["n"] += 1
        created["url"] = url
        created["auth_token"] = auth_token
        return "client"

    monkeypatch.setattr(usage, "_database_url", lambda: None)
    assert usage._connect() is None

    monkeypatch.setattr(usage, "_database_url", lambda: "https://remote")
    monkeypatch.setattr(usage, "_env_value", lambda name: "token")
    monkeypatch.setitem(
        sys.modules,
        "libsql_client",
        types.SimpleNamespace(create_client_sync=create_client_sync),
    )

    assert usage._connect() == "client"
    assert usage._connect() == "client"
    assert create_calls["n"] == 1
    assert created == {"url": "https://remote", "auth_token": "token"}


def test_record_pair_reuses_one_connect(monkeypatch):
    """Usage + invocation for one command must not open two clients."""
    client = FakeClient()
    creates = {"n": 0}

    def create_client_sync(url, auth_token):
        creates["n"] += 1
        return client

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("SCREENER_USAGE", raising=False)
    monkeypatch.setattr(usage, "_database_url", lambda: "https://remote")
    monkeypatch.setattr(usage, "_env_value", lambda name: "token")
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "user")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")
    monkeypatch.setitem(
        sys.modules,
        "libsql_client",
        types.SimpleNamespace(create_client_sync=create_client_sync),
    )

    monkeypatch.setenv("SCREENER_USAGE_FLUSH_MS", "2000")
    usage.record_feature_usage("screen", command_path="screener screen", duration_ms=1)
    usage.record_feature_invocation(
        "screen",
        command_path="screener screen",
        duration_ms=1,
        status="success",
        params={"market": "us"},
    )
    usage.flush_usage(timeout_s=2.0)

    # One connect for the whole pair; the writer does not close the shared
    # client under other callers (the atexit handler owns it).
    assert creates["n"] == 1
    assert client.closed is False
    inserts = [s for s in client.statements if "INSERT INTO" in s[0]]
    assert len(inserts) == 2
    # DDL runs once on the shared client; a second pair must not re-run it.
    create_stmts_before = sum(1 for s in client.statements if "CREATE TABLE" in s[0])
    usage.record_feature_usage("screen", command_path="screener screen", duration_ms=1)
    usage.record_feature_invocation(
        "screen",
        command_path="screener screen",
        duration_ms=1,
        status="success",
        params={"market": "us"},
    )
    usage.flush_usage(timeout_s=2.0)
    create_stmts_second = sum(1 for s in client.statements if "CREATE TABLE" in s[0])
    assert create_stmts_second == create_stmts_before
    assert creates["n"] == 1
    assert create_stmts_before >= 1


def test_second_pair_is_not_dropped_while_flush_is_running(monkeypatch):
    """A second usage/invocation pair must not overwrite the first."""
    import time as time_mod

    class SlowClient(FakeClient):
        def execute(self, stmt: str, args: list[object] | None = None):
            time_mod.sleep(0.04)
            return super().execute(stmt, args)

    client = SlowClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("SCREENER_USAGE", raising=False)
    monkeypatch.setenv("SCREENER_USAGE_FLUSH_MS", "20")
    monkeypatch.setattr(usage, "_connect", lambda: client)
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "user")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")

    def record_pair(feature: str) -> None:
        usage.record_feature_usage(feature, command_path=feature, duration_ms=1)
        usage.record_feature_invocation(
            feature,
            command_path=feature,
            duration_ms=1,
            status="success",
            params={"market": "us"},
        )

    record_pair("screen")
    record_pair("garp")
    usage.flush_usage(timeout_s=5.0)

    inserts = [item for item in client.statements if "INSERT INTO" in item[0]]
    assert len(inserts) == 4
    usage_features = [
        item[1][1]
        for item in inserts
        if item[1] is not None and "feature_usage_invocations" not in item[0]
    ]
    inv_features = [
        item[1][1]
        for item in inserts
        if item[1] is not None and "feature_usage_invocations" in item[0]
    ]
    assert usage_features == ["screen", "garp"]
    assert inv_features == ["screen", "garp"]


def test_screener_usage_env_opt_out(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("SCREENER_USAGE", "0")
    monkeypatch.setattr(
        usage,
        "_connect",
        lambda: (_ for _ in ()).throw(AssertionError("should not connect")),
    )
    usage.record_feature_usage("screen")
    usage.record_feature_invocation("screen")


def test_record_feature_invocation_flattens_params(monkeypatch):
    client = FakeClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("SCREENER_USAGE_FLUSH_MS", "2000")
    monkeypatch.setattr(usage, "_connect", lambda: client)
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "user")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")

    usage.record_feature_invocation(
        "screen",
        command_path="ignored",
        duration_ms=12,
        status="error",
        params={
            "market": "india",
            "criteria_names": ["ema", "breakout"],
            "limit": "5",
            "refresh": True,
            "output_csv": False,
            "cache_ttl": "1h",
            "extra": 7,
            "none": None,
        },
    )
    usage.flush_usage(timeout_s=2.0)

    insert = [
        item
        for item in client.statements
        if "INSERT INTO feature_usage_invocations" in item[0]
    ][0]
    assert insert[1] == [
        "screener",
        "screen",
        "india",
        "ema,breakout",
        5,
        1,
        "False",
        "1h",
        '{"extra": "7"}',
        12,
        "error",
        "user",
        "host",
    ]


def test_record_pair_is_non_blocking_for_slow_client(monkeypatch):
    """Flush join budget must not wait for the full remote RTT."""
    import time as time_mod

    class SlowClient(FakeClient):
        def execute(self, stmt: str, args: list[object] | None = None):
            time_mod.sleep(0.4)
            return super().execute(stmt, args)

    client = SlowClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("SCREENER_USAGE_FLUSH_MS", "50")
    monkeypatch.setattr(usage, "_connect", lambda: client)
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "user")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")

    t0 = time_mod.perf_counter()
    usage.record_feature_usage("screen", command_path="screener screen", duration_ms=1)
    usage.record_feature_invocation(
        "screen",
        command_path="screener screen",
        duration_ms=1,
        status="success",
        params={},
    )
    elapsed = time_mod.perf_counter() - t0
    # Default flush budget is ~50 ms; slow client takes 400 ms+ per execute.
    assert elapsed < 0.25
    leftover = usage._flush_thread
    if leftover is not None and leftover.is_alive():
        leftover.join()


def test_rows_staged_while_worker_mid_flush_are_written(monkeypatch):
    """Rows staged while a flush worker is mid-write must not be stranded."""
    import threading as threading_mod
    import time as time_mod

    inserts: list[str] = []
    release = threading_mod.Event()

    class BlockingClient(FakeClient):
        def execute(self, stmt: str, args: list[object] | None = None):
            if stmt.lstrip().upper().startswith("INSERT"):
                inserts.append(stmt)
            if "INSERT INTO feature_usage_invocations" in stmt:
                release.wait(timeout=5.0)
            return super().execute(stmt, args)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("SCREENER_USAGE_FLUSH_MS", "10")
    monkeypatch.setattr(usage, "_connect", lambda: BlockingClient())
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "user")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")

    usage.record_feature_usage("screen", command_path="c0", duration_ms=1)
    usage.record_feature_invocation("screen", command_path="c0", duration_ms=1)
    try:
        # Wait until the worker is mid-flush (blocked on pair 0's invocation).
        deadline = time_mod.perf_counter() + 5.0
        while (
            not any("feature_usage_invocations" in s for s in inserts)
            and time_mod.perf_counter() < deadline
        ):
            time_mod.sleep(0.005)
        assert any("feature_usage_invocations" in s for s in inserts), (
            "worker did not start flushing"
        )

        # Stage a second pair while the worker is still busy with the first.
        usage.record_feature_usage("screen", command_path="c1", duration_ms=1)
        usage.record_feature_invocation("screen", command_path="c1", duration_ms=1)
    finally:
        release.set()
    usage.flush_usage(timeout_s=5.0)

    # Both pairs written: 2 usage + 2 invocation inserts.
    assert len(inserts) == 4


def test_flush_usage_respects_zero_budget(monkeypatch):
    """flush_usage must not block past its budget on a stuck worker."""
    import threading as threading_mod
    import time as time_mod

    release = threading_mod.Event()

    class BlockingClient(FakeClient):
        def execute(self, stmt: str, args: list[object] | None = None):
            if "INSERT INTO feature_usage_invocations" in stmt:
                release.wait(timeout=5.0)
            return super().execute(stmt, args)

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("SCREENER_USAGE_FLUSH_MS", "10")
    monkeypatch.setattr(usage, "_connect", lambda: BlockingClient())
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "user")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")

    usage.record_feature_usage("screen", duration_ms=1)
    usage.record_feature_invocation("screen", duration_ms=1)
    try:
        t0 = time_mod.perf_counter()
        usage.flush_usage(timeout_s=0.0)
        assert time_mod.perf_counter() - t0 < 0.25
    finally:
        release.set()


def test_flush_pending_on_exit_writes_pending_rows(monkeypatch):
    """The atexit handler must write staged rows instead of dropping them."""
    client = FakeClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: client)
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "user")
    monkeypatch.setattr(usage.platform, "node", lambda: "host")

    usage.record_feature_usage("screen", command_path="screener screen", duration_ms=1)
    assert not [s for s in client.statements if "INSERT INTO" in s[0]]

    usage._flush_pending_on_exit()

    inserts = [s for s in client.statements if "INSERT INTO feature_usage" in s[0]]
    assert len(inserts) == 1
    assert inserts[0][1][1] == "screen"


def test_atexit_handler_writes_pending_rows_on_process_exit(tmp_path):
    """Registered atexit handler writes pending rows at interpreter exit."""
    import subprocess

    log = tmp_path / "written.jsonl"
    script = tmp_path / "atexit_probe.py"
    script.write_text(
        f"""
import json
import os
import sys
import types

LOG = {str(log)!r}

class Client:
    def execute(self, stmt, args=None):
        if stmt.lstrip().upper().startswith("INSERT"):
            with open(LOG, "a") as fh:
                fh.write(json.dumps(args) + "\\n")
    def close(self):
        pass

sys.modules["libsql_client"] = types.SimpleNamespace(
    create_client_sync=lambda url, auth_token=None: Client()
)
os.environ["TURSO_DATABASE_URL"] = "libsql://probe.invalid"
os.environ["TURSO_AUTH_TOKEN"] = "probe-token"

from screener.usage import record_feature_usage

record_feature_usage("screen", command_path="screener screen", duration_ms=1)
"""
    )
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("SCREENER_USAGE", None)
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        check=True,
        timeout=30,
    )
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])[1] == "screen"


def test_usage_invocation_normalizers_cover_scalar_and_empty_values():
    assert usage._coerce_bool_to_int("yes") == "yes"
    assert usage._normalize_criteria(None) is None
    assert usage._normalize_criteria([None]) is None
    assert usage._normalize_criteria("ema") == "ema"


def test_record_feature_invocation_early_returns(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_usage.py::x")
    monkeypatch.setattr(
        usage,
        "_connect",
        lambda: (_ for _ in ()).throw(AssertionError("should not connect")),
    )
    usage.record_feature_invocation("screen")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: None)
    usage.record_feature_invocation("screen")


def test_feature_usage_counts_maps_rows(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(usage, "_connect", lambda: client)

    rows = usage.feature_usage_counts()

    assert [(row.feature, row.count, row.last_used_at) for row in rows] == [
        ("screen", 2, "2026-05-10T12:00:00.000Z"),
        ("garp", 1, None),
    ]
    assert client.closed is False


def test_invocation_rollup_groups_extras_and_limits(monkeypatch):
    rows = [
        ("screen", "us", "ema", "success", "2026-01-01T00:00:00Z", '{"foo": "a"}'),
        ("screen", "us", "ema", "success", "2026-01-02T00:00:00Z", '{"foo": "a"}'),
        ("screen", "us", "ema", "success", "2026-01-03T00:00:00Z", '{"foo": "b"}'),
        ("screen", "us", "ema", "success", "2026-01-03T00:00:01Z", None),
        ("garp", None, None, "error", None, "not-json"),
        ("other", "india", "", "success", "2026-01-04T00:00:00Z", "[1]"),
    ]
    client = FakeClient(rows=rows)
    monkeypatch.setattr(usage, "_connect", lambda: client)

    rollup = usage.invocation_rollup(limit=2)

    assert [(r.feature, r.market, r.criteria, r.status, r.count) for r in rollup] == [
        ("screen", "us", "ema", "success", 4),
        ("other", "india", "", "success", 1),
    ]
    assert rollup[0].last_used_at == "2026-01-03T00:00:01Z"
    assert rollup[0].top_extras == "foo=a"
    assert client.closed is False


def test_invocation_rollup_no_client(monkeypatch):
    monkeypatch.setattr(usage, "_connect", lambda: None)
    assert usage.invocation_rollup() == []


def test_record_feature_usage_early_returns(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_usage.py::x")
    monkeypatch.setattr(
        usage,
        "_connect",
        lambda: (_ for _ in ()).throw(AssertionError("should not connect")),
    )
    usage.record_feature_usage("screen")

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: None)
    usage.record_feature_usage("screen")


def test_feature_usage_counts_no_client(monkeypatch):
    monkeypatch.setattr(usage, "_connect", lambda: None)
    assert usage.feature_usage_counts() == []


def test_elapsed_ms_is_non_negative(monkeypatch):
    ticks = iter([10.0, 9.0])
    monkeypatch.setattr(usage.time, "perf_counter", lambda: next(ticks))
    assert usage.elapsed_ms(10.0) == 0


def test_successful_command_records_usage(monkeypatch):
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        usage,
        "record_feature_usage",
        lambda feature, **kwargs: calls.append((feature, kwargs.get("command_path"))),
    )

    result = CliRunner().invoke(cli, ["screen", "--help"])

    assert result.exit_code == 0
    assert calls == []


def test_failed_command_does_not_record_usage(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        usage,
        "record_feature_usage",
        lambda feature, **kwargs: calls.append(feature),
    )

    result = CliRunner().invoke(cli, ["backtest-rolling", "--csv", "--dashboard"])

    assert result.exit_code != 0
    assert calls == []


def test_usage_report_renders_zero_state(monkeypatch):
    monkeypatch.setattr(usage, "feature_usage_counts", list)

    result = CliRunner().invoke(cli, ["usage-report"])

    assert result.exit_code == 0
    assert "No feature usage recorded" in result.output


def test_usage_report_renders_counts(monkeypatch):
    monkeypatch.setattr(
        usage,
        "feature_usage_counts",
        lambda: [
            usage.UsageCount(
                feature="screen", count=2, last_used_at="2026-05-10T12:00:00.000Z"
            ),
            usage.UsageCount(feature="garp", count=1, last_used_at=None),
        ],
    )

    result = CliRunner().invoke(cli, ["usage-report"])

    assert result.exit_code == 0
    assert "screen" in result.output
    assert "garp" in result.output
