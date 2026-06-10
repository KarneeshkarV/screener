"""Tests for `screener cache status` / `screener cache clean` — offline."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

import screener.backtester.data as backtester_data
import screener.cache as screener_cache
import screener.operator.fetch as operator_fetch
import screener.universes as universes
import screener.unusual_volume.delivery as delivery
from screener.cli import cli
from screener.commands.cache import known_cache_dirs


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch) -> dict[str, Path]:
    dirs = {
        "prices": tmp_path / "prices",
        "fmp_prices": tmp_path / "fmp_prices",
        "universes": tmp_path / "universes",
        "scanner": tmp_path / "cache",
        "panels": tmp_path / "panels",
        "bhavcopy": tmp_path / "bhavcopy",
        "nse_bhavcopy": tmp_path / "nse_bhavcopy",
    }
    monkeypatch.setattr(backtester_data, "CACHE_DIR", dirs["prices"])
    monkeypatch.setattr(backtester_data, "FMP_CACHE_DIR", dirs["fmp_prices"])
    monkeypatch.setattr(universes, "CACHE_DIR", dirs["universes"])
    monkeypatch.setattr(screener_cache, "CACHE_ROOT", dirs["scanner"])
    monkeypatch.setattr(screener_cache, "PANEL_ROOT", dirs["panels"])
    monkeypatch.setattr(delivery, "CACHE_DIR", dirs["bhavcopy"])
    monkeypatch.setattr(operator_fetch, "CACHE_ROOT", dirs["nse_bhavcopy"])
    return dirs


def _write(path: Path, content: bytes = b"x" * 10, age_days: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))


def test_known_cache_dirs_reflect_monkeypatched_modules(cache_dirs):
    assert known_cache_dirs() == cache_dirs


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
