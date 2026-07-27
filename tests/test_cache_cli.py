"""Tests for `screener cache status` / `screener cache clean` — offline."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

import screener.cache as screener_cache
from screener.cli import cli
from screener.commands.cache import known_cache_dirs


@pytest.fixture
def cache_dirs(tmp_path) -> dict[str, Path]:
    screener_cache.reset_cache_area_paths()
    dirs = {
        "prices": tmp_path / "prices",
        "fmp_prices": tmp_path / "fmp_prices",
        "bars": tmp_path / "bars",
        "contracts": tmp_path / "contracts",
        "universes": tmp_path / "universes",
        "scanner": tmp_path / "cache",
        "panels": tmp_path / "panels",
        "bhavcopy": tmp_path / "bhavcopy",
        "nse_bhavcopy": tmp_path / "nse_bhavcopy",
    }
    for name, path in dirs.items():
        screener_cache.set_cache_area_path(name, path)
    try:
        yield dirs
    finally:
        screener_cache.reset_cache_area_paths()


def _write(path: Path, content: bytes = b"x" * 10, age_days: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))


def test_known_cache_dirs_reflect_monkeypatched_modules(cache_dirs):
    assert known_cache_dirs() == cache_dirs


def test_storage_status_flags_over_budget(cache_dirs):
    from screener.commands.cache import storage_status

    _write(cache_dirs["bars"] / "AAPL.parquet", b"x" * 2_000_000)  # ~2 MB
    _write(cache_dirs["contracts"] / "us" / "SPY.parquet", b"y" * 1_000)
    # 1 MB budget for bars → over; contracts left unbudgeted → never over.
    statuses = {s.name: s for s in storage_status({"bars": 1.0, "contracts": None})}
    assert statuses["bars"].over_budget is True
    assert statuses["contracts"].over_budget is False
    assert statuses["contracts"].budget_bytes is None


def test_storage_status_reads_env_budget(cache_dirs, monkeypatch):
    from screener.commands.cache import storage_status

    _write(cache_dirs["contracts"] / "us" / "SPY.parquet", b"z" * 500_000)
    monkeypatch.setenv("SCREENER_CONTRACTS_BUDGET_MB", "0.1")  # 100 KB < 500 KB
    statuses = {s.name: s for s in storage_status()}
    assert statuses["contracts"].over_budget is True


def test_storage_watch_command_exits_nonzero_when_over(cache_dirs):
    _write(cache_dirs["bars"] / "AAPL.parquet", b"x" * 2_000_000)
    res = CliRunner().invoke(cli, ["cache", "storage-watch", "--bars-budget-mb", "1"])
    assert res.exit_code != 0
    assert "storage budget exceeded" in res.output
    assert "bars" in res.output


def test_storage_watch_command_ok_when_within_budget(cache_dirs):
    _write(cache_dirs["bars"] / "AAPL.parquet", b"x" * 1_000)
    res = CliRunner().invoke(cli, ["cache", "storage-watch", "--bars-budget-mb", "100"])
    assert res.exit_code == 0
    assert "ok" in res.output


def test_cache_status_lists_every_dir_with_counts(cache_dirs):
    _write(cache_dirs["prices"] / "AAPL.parquet", b"a" * 100)
    _write(cache_dirs["prices"] / "nested" / "MSFT.parquet", b"b" * 50)
    _write(cache_dirs["panels"] / "fii_dii.parquet", b"c" * 25)
    res = CliRunner().invoke(cli, ["cache", "status"], env={"COLUMNS": "250"})
    assert res.exit_code == 0, res.output
    assert "Cache status" in res.output
    for name in cache_dirs:
        assert name in res.output
    prices_row = next(line for line in res.output.splitlines() if " prices " in line)
    assert " 2 " in prices_row
    assert "150 B" in prices_row
    panels_row = next(line for line in res.output.splitlines() if " panels " in line)
    assert " 1 " in panels_row
    # Empty dirs are reported, not skipped.
    scanner_row = next(line for line in res.output.splitlines() if " scanner " in line)
    assert " 0 " in scanner_row


def test_cache_clean_dry_run_deletes_nothing(cache_dirs):
    old = cache_dirs["prices"] / "old.parquet"
    fresh = cache_dirs["prices"] / "fresh.parquet"
    _write(old, b"o" * 10, age_days=40)
    _write(fresh, b"f" * 10)
    res = CliRunner().invoke(cli, ["cache", "clean", "--older-than", "30", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert old.exists()
    assert fresh.exists()
    assert f"Would remove [prices] {old}" in res.output
    assert "fresh.parquet" not in res.output
    assert "Would reclaim 10 B from 1 file(s)" in res.output


def test_cache_clean_removes_only_old_files(cache_dirs):
    old = cache_dirs["panels"] / "option_chain.parquet"
    fresh = cache_dirs["panels"] / "fii_dii.parquet"
    _write(old, b"o" * 10, age_days=40)
    _write(fresh, b"f" * 10)
    res = CliRunner().invoke(cli, ["cache", "clean", "--older-than", "30"])
    assert res.exit_code == 0, res.output
    assert not old.exists()
    assert fresh.exists()
    assert f"Removed [panels] {old}" in res.output
    assert "Reclaimed 10 B from 1 file(s)" in res.output


def test_cache_clean_preserves_lock_sidecars(cache_dirs):
    """``.lock`` files must never be deleted — they back cache mutual exclusion (M14)."""
    data = cache_dirs["prices"] / "AAPL.parquet"
    lock = cache_dirs["prices"] / "AAPL.parquet.lock"
    stale_lock = cache_dirs["prices"] / "orphan.lock"
    _write(data, b"d" * 10, age_days=40)
    _write(lock, b"", age_days=40)
    _write(stale_lock, b"", age_days=90)
    res = CliRunner().invoke(cli, ["cache", "clean", "--older-than", "30"])
    assert res.exit_code == 0, res.output
    assert not data.exists()
    assert lock.exists(), "active data.lock sidecar must survive clean"
    assert stale_lock.exists(), "any *.lock must survive clean"
    assert ".lock" not in res.output
    assert "Reclaimed 10 B from 1 file(s)" in res.output


def test_cache_clean_dir_option_scopes_to_one_dir(cache_dirs):
    panels_old = cache_dirs["panels"] / "old_panel.parquet"
    prices_old = cache_dirs["prices"] / "old_price.parquet"
    _write(panels_old, age_days=40)
    _write(prices_old, age_days=40)
    res = CliRunner().invoke(
        cli, ["cache", "clean", "--older-than", "30", "--dir", "panels"]
    )
    assert res.exit_code == 0, res.output
    assert not panels_old.exists()
    assert prices_old.exists()


def test_cache_clean_refuses_unknown_dir(cache_dirs, tmp_path):
    outside = tmp_path / "outside"
    _write(outside / "victim.txt", age_days=40)
    res = CliRunner().invoke(
        cli, ["cache", "clean", "--older-than", "0", "--dir", str(outside)]
    )
    assert res.exit_code != 0
    assert "unknown cache dir" in res.output
    assert (outside / "victim.txt").exists()
