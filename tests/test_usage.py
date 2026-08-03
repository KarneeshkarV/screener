from __future__ import annotations

import sys
import types

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


def test_record_feature_usage_inserts_success(monkeypatch):
    client = FakeClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(usage, "_connect", lambda: client)
    monkeypatch.setattr(usage.getpass, "getuser", lambda: "karneeshkar")
    monkeypatch.setattr(usage.platform, "node", lambda: "workstation")

    usage.record_feature_usage("screen", command_path="screener screen", duration_ms=42)

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
    assert client.closed


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

    def create_client_sync(url, auth_token):
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
    assert created == {"url": "https://remote", "auth_token": "token"}


def test_record_feature_invocation_flattens_params(monkeypatch):
    client = FakeClient()
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
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
    assert client.closed


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
    assert client.closed


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
    assert client.closed


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
