from __future__ import annotations

from click.testing import CliRunner

from screener import usage
from screener.cli import cli


class FakeResult:
    rows = [("screen", 2, "2026-05-10T12:00:00.000Z"), ("garp", 1, None)]


class FakeClient:
    def __init__(self) -> None:
        self.statements: list[tuple[str, list[object] | None]] = []
        self.closed = False

    def execute(self, stmt: str, args: list[object] | None = None):
        self.statements.append((stmt, args))
        if stmt.lstrip().upper().startswith("SELECT"):
            return FakeResult()
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


def test_feature_usage_counts_maps_rows(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(usage, "_connect", lambda: client)

    rows = usage.feature_usage_counts()

    assert [(row.feature, row.count, row.last_used_at) for row in rows] == [
        ("screen", 2, "2026-05-10T12:00:00.000Z"),
        ("garp", 1, None),
    ]
    assert client.closed


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
    monkeypatch.setattr(usage, "feature_usage_counts", lambda: [])

    result = CliRunner().invoke(cli, ["usage-report"])

    assert result.exit_code == 0
    assert "No feature usage recorded" in result.output


def test_usage_report_renders_counts(monkeypatch):
    monkeypatch.setattr(
        usage,
        "feature_usage_counts",
        lambda: [
            usage.UsageCount("screen", 2, "2026-05-10T12:00:00.000Z"),
            usage.UsageCount("garp", 1, None),
        ],
    )

    result = CliRunner().invoke(cli, ["usage-report"])

    assert result.exit_code == 0
    assert "screen" in result.output
    assert "garp" in result.output
