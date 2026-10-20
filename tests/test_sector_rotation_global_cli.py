"""Both market catalogues must reach the same offline CLI and pricing path."""

from __future__ import annotations

import io
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli
from screener.sector_rotation.analysis import fetch_sector_rotation
from screener.sector_rotation.catalogue import (
    load_sector_catalogue,
    select_sector_groups,
)
from screener.symbols import tv_to_yf


class GlobalSectorFetcher:
    """Return deterministic completed-session prices and record requested symbols."""

    def __init__(self):
        self.calls = []

    def fetch(self, tickers, start, end):
        self.calls.append((tickers, start, end))
        sessions = pd.bdate_range(start, end)
        return {
            ticker: pd.DataFrame(
                {
                    "close": 100
                    + np.arange(len(sessions))
                    * (0.1 if ticker.startswith("^") else 0.2)
                },
                index=sessions,
            )
            for ticker in tickers
        }


@pytest.mark.parametrize("market", ["india", "us"])
def test_global_catalogue_lists_and_details_need_no_prices(market):
    catalogue = load_sector_catalogue(market=market)
    group = catalogue.groups[0]
    listed = CliRunner().invoke(
        cli,
        ["sector-rotation", "--market", market, "--group", group.id, "--list-groups"],
    )
    assert listed.exit_code == 0, listed.output
    assert listed.stdout.strip() == "\t".join(
        [group.id, group.name, group.parent, ",".join(group.members)]
    )
    described = CliRunner().invoke(
        cli,
        [
            "sector-rotation",
            "--market",
            market,
            "--group",
            group.id,
            "--describe-groups",
        ],
    )
    assert described.exit_code == 0, described.output
    for value in (
        group.description,
        group.drivers[0],
        group.risks[0],
        group.sources[0],
    ):
        assert value in described.stdout
    assert catalogue.benchmark in described.stdout
    assert "Membership:" in described.stdout


def test_legacy_catalogue_can_describe_groups_without_new_fields(tmp_path):
    catalogue = load_sector_catalogue()
    group = catalogue.groups[0].model_dump(
        exclude={"description", "drivers", "risks", "sources"}
    )
    data = catalogue.model_dump(mode="json")
    data["groups"] = [group]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    result = CliRunner().invoke(
        cli, ["sector-rotation", "--catalogue", str(path), "--describe-groups"]
    )
    assert result.exit_code == 0, result.output
    assert "No description supplied." in result.stdout
    assert "Drivers: Not supplied." in result.stdout
    assert "Risks: Not supplied." in result.stdout
    assert catalogue.source in result.stdout


def test_global_cli_rejects_conflicting_discovery_options():
    result = CliRunner().invoke(
        cli, ["sector-rotation", "--list-groups", "--describe-groups"]
    )
    assert result.exit_code != 0
    assert "use either --list-groups or --describe-groups" in result.output


def test_custom_catalogue_must_match_cli_market_before_price_fetch(tmp_path):
    path = tmp_path / "india.json"
    path.write_text(load_sector_catalogue().model_dump_json())
    fetcher = GlobalSectorFetcher()
    result = CliRunner().invoke(
        cli,
        [
            "sector-rotation",
            "--market",
            "us",
            "--catalogue",
            str(path),
            "--list-groups",
        ],
        obj=fetcher,
    )
    assert result.exit_code != 0
    assert "catalogue market does not match --market" in result.output
    assert not fetcher.calls


@pytest.mark.parametrize("market", ["india", "us"])
def test_global_cli_csv_report_and_members_use_correct_market(market, tmp_path):
    catalogue = load_sector_catalogue(market=market)
    group = catalogue.groups[0]
    fetcher = GlobalSectorFetcher()
    report = tmp_path / f"{market}.html"
    members = tmp_path / f"{market}-members.csv"
    result = CliRunner().invoke(
        cli,
        [
            "sector-rotation",
            "--market",
            market,
            "--group",
            group.id,
            "--as-of",
            "2026-01-06",
            "--period",
            "1m",
            "--csv",
            "--report",
            str(report),
            "--members-csv",
            str(members),
        ],
        obj=fetcher,
    )
    assert result.exit_code == 0, result.output
    summary = pd.read_csv(io.StringIO(result.stdout))
    assert summary.market.tolist() == [market]
    assert summary.description.tolist() == [group.description]
    assert summary.drivers.iloc[0] == "; ".join(group.drivers)
    assert summary.risks.iloc[0] == "; ".join(group.risks)
    assert summary.sources.iloc[0] == "; ".join(group.sources)
    assert summary.status.tolist() == ["ok"]
    assert summary.breadth_sma200_members.tolist() == [len(group.members)]
    tickers = fetcher.calls[0][0]
    assert set(tickers) == {catalogue.benchmark} | {
        tv_to_yf(ticker, market) for ticker in group.members
    }
    if market == "us":
        assert not any(ticker.endswith(".NS") for ticker in tickers)
    assert set(pd.read_csv(members).ticker) == set(group.members)
    assert group.description in report.read_text()


@pytest.mark.parametrize("market", ["india", "us"])
def test_all_global_groups_produce_complete_consistent_metrics(market):
    catalogue = load_sector_catalogue(market=market)
    selection = select_sector_groups(catalogue)
    fetcher = GlobalSectorFetcher()
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        result = fetch_sector_rotation(
            selection, fetcher, start=date(2026, 1, 2), end=date(2026, 1, 6)
        )
    assert len(result.summary) == len(catalogue.groups)
    assert result.summary.status.eq("ok").all()
    assert result.summary.market.eq(market).all()
    assert len(result.members) == sum(len(group.members) for group in catalogue.groups)
    assert result.curves.iloc[0].eq(100).all()
    assert result.summary.minimum_coverage_pct.eq(100).all()
    assert len(fetcher.calls) == 1
    assert len(fetcher.calls[0][0]) == len(set(fetcher.calls[0][0]))


def test_us_agent_digest_remains_bounded_and_spills_details(tmp_path, monkeypatch):
    from screener import agentio

    monkeypatch.setenv("SCREENER_AGENT_DIR", str(tmp_path))
    try:
        result = CliRunner().invoke(
            cli,
            ["--agent", "sector-rotation", "--market", "us", "--as-of", "2026-01-06"],
            obj=GlobalSectorFetcher(),
        )
        assert result.exit_code == 0, result.output
        assert len(result.stdout.splitlines()) < 20
        files = list(tmp_path.glob("sector-rotation-*.csv"))
        assert len(files) == 1
        rows = pd.read_csv(files[0])
        assert len(rows) == len(load_sector_catalogue(market="us").groups)
        assert rows.market.eq("us").all()
        assert rows.description.notna().all()
        assert rows.sources.notna().all()
    finally:
        agentio.reset()


def test_us_benchmark_override_is_fetched_and_reported(tmp_path):
    catalogue = load_sector_catalogue(market="us")
    fetcher = GlobalSectorFetcher()
    report = tmp_path / "override.html"
    result = CliRunner().invoke(
        cli,
        [
            "sector-rotation",
            "--market",
            "us",
            "--group",
            catalogue.groups[0].id,
            "--as-of",
            "2026-01-06",
            "--benchmark",
            "^RUA",
            "--csv",
            "--report",
            str(report),
        ],
        obj=fetcher,
    )
    assert result.exit_code == 0, result.output
    assert "^RUA" in fetcher.calls[0][0]
    assert "^GSPC" not in fetcher.calls[0][0]
    assert "Benchmark: ^RUA" in report.read_text()
