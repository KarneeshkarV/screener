"""Screen → backtest replay bridge tests — offline, no network."""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner

from screener import history as history_mod
from screener.cli import cli

from tests.conftest import StubPriceFetcher, make_bars


@pytest.fixture
def history_db(tmp_path, monkeypatch):
    db = tmp_path / "history.db"
    monkeypatch.setattr(history_mod, "DB_PATH", db)
    return db


def _screen_df(tickers):
    return pd.DataFrame(
        [
            {
                "name": t,
                "description": f"{t} Corp",
                "close": 10.0 + i,
                "change": 1.0,
                "volume": 1_000_000,
                "market_cap_basic": 1_000_000_000,
                "setup_score": 50.0 + i,
            }
            for i, t in enumerate(tickers)
        ]
    )


def _seed_run(market, criteria, tickers, run_date):
    run_id = history_mod.save_run(market, criteria, len(tickers), _screen_df(tickers))
    conn = sqlite3.connect(history_mod.DB_PATH)
    conn.execute(
        "UPDATE runs SET run_ts = ? WHERE id = ?",
        (f"{run_date.isoformat()}T04:00:00+00:00", run_id),
    )
    conn.commit()
    conn.close()
    return run_id


def test_load_run_roundtrip(history_db):
    run_id = _seed_run("us", "ema", ["AAA", "BBB", "CCC"], date(2024, 3, 1))

    snap = history_mod.load_run(run_id)

    assert snap is not None
    assert snap.run_id == run_id
    assert snap.market == "us"
    assert snap.criteria == "ema"
    assert snap.total_matches == 3
    assert snap.run_date == date(2024, 3, 1)
    assert snap.tickers == ["AAA", "BBB", "CCC"]
    assert snap.rows["rank"].tolist() == [1, 2, 3]


def test_load_run_missing_returns_none(history_db):
    assert history_mod.load_run(999) is None


def test_find_run_respects_on_or_before(history_db):
    old_id = _seed_run("us", "ema", ["AAA"], date(2024, 2, 1))
    new_id = _seed_run("us", "ema", ["BBB"], date(2024, 3, 1))

    assert history_mod.find_run("us", "ema") == new_id
    assert history_mod.find_run("us", "ema", on_or_before=date(2024, 2, 15)) == old_id
    assert history_mod.find_run("us", "ema", on_or_before=date(2024, 1, 1)) is None
    assert history_mod.find_run("india", "ema") is None


def test_list_runs_filters_and_counts_rows(history_db):
    _seed_run("us", "ema", ["AAA", "BBB"], date(2024, 2, 1))
    _seed_run("india", "breakout", ["XXX"], date(2024, 3, 1))

    all_runs = history_mod.list_runs()
    assert len(all_runs) == 2
    assert all_runs.iloc[0]["criteria"] == "breakout"  # newest first
    assert all_runs.iloc[1]["saved_rows"] == 2

    us_only = history_mod.list_runs(market="us")
    assert us_only["market"].tolist() == ["us"]


def test_resolve_replay_run_by_id_and_selector(history_db):
    run_id = _seed_run("us", "ema", ["AAA", "BBB"], date(2024, 3, 1))

    by_id = history_mod.resolve_replay_run(str(run_id))
    assert by_id.run_id == run_id

    by_selector = history_mod.resolve_replay_run("us:ema")
    assert by_selector.run_id == run_id


def test_resolve_replay_run_errors(history_db):
    with pytest.raises(ValueError, match="no screen run with id"):
        history_mod.resolve_replay_run("42")
    with pytest.raises(ValueError, match="expected a run id or MARKET:CRITERIA"):
        history_mod.resolve_replay_run("ema")
    with pytest.raises(ValueError, match="no persisted us/ema screen run"):
        history_mod.resolve_replay_run("us:ema")

    _seed_run("us", "ema", ["AAA"], date(2024, 3, 1))
    with pytest.raises(ValueError, match="at least 99999 days old"):
        history_mod.resolve_replay_run("us:ema", min_age_days=99999)


def test_resolve_replay_run_rejects_empty_snapshot(history_db):
    run_id = history_mod.save_run("us", "ema", 0, pd.DataFrame())
    with pytest.raises(ValueError, match="no saved rows"):
        history_mod.resolve_replay_run(str(run_id))


def test_history_command_lists_runs(history_db):
    run_id = _seed_run("us", "ema", ["AAA", "BBB"], date(2024, 3, 1))

    res = CliRunner().invoke(cli, ["history"])
    assert res.exit_code == 0, res.output
    assert str(run_id) in res.output
    assert "ema" in res.output

    res_csv = CliRunner().invoke(cli, ["history", "--csv"])
    assert res_csv.exit_code == 0, res_csv.output
    assert "id,run_ts,market,criteria,total_matches,saved_rows" in res_csv.output


def test_history_command_empty_db(history_db):
    res = CliRunner().invoke(cli, ["history"])
    assert res.exit_code == 0, res.output
    assert "No screen runs recorded yet" in res.output


def _stub_env():
    bars_a = make_bars(n=60, seed=11, open_base=100.0)
    bars_b = make_bars(n=60, seed=12, open_base=50.0)
    spy = make_bars(n=60, seed=99, open_base=400.0)
    return StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy}), bars_a


def test_backtest_from_run_id_replays_snapshot(history_db):
    fetcher, bars = _stub_env()
    as_of = bars.index[39].date()
    run_id = _seed_run("us", "ema", ["AAA", "BBB"], as_of)

    res = CliRunner().invoke(
        cli,
        ["backtest-historical", "--from-run", str(run_id), "--hold", "5"],
        obj=fetcher,
    )

    assert res.exit_code == 0, res.output
    assert f"Replaying screen run #{run_id} (us/ema @ {as_of.isoformat()}" in res.output
    assert "Total Return" in res.output


def test_backtest_from_run_selector_with_age(history_db):
    fetcher, bars = _stub_env()
    old_date = bars.index[30].date()
    new_date = bars.index[45].date()
    old_id = _seed_run("us", "ema", ["AAA", "BBB"], old_date)
    _seed_run("us", "ema", ["AAA", "BBB"], new_date)

    age = (date.today() - old_date).days
    res = CliRunner().invoke(
        cli,
        [
            "backtest-historical",
            "--from-run",
            "us:ema",
            "--run-age-days",
            str(age),
            "--hold",
            "5",
        ],
        obj=fetcher,
    )

    assert res.exit_code == 0, res.output
    assert f"Replaying screen run #{old_id}" in res.output


def test_from_run_usage_errors(history_db):
    fetcher, bars = _stub_env()
    as_of = bars.index[39].date()
    run_id = _seed_run("us", "ema", ["AAA", "BBB"], as_of)
    runner = CliRunner()

    res = runner.invoke(
        cli,
        ["backtest-historical", "--from-run", str(run_id), "--tickers", "AAA"],
        obj=fetcher,
    )
    assert res.exit_code != 0
    assert "--from-run supplies the universe" in res.output

    res = runner.invoke(
        cli,
        [
            "backtest-historical",
            "--from-run",
            str(run_id),
            "--as-of",
            as_of.isoformat(),
        ],
        obj=fetcher,
    )
    assert res.exit_code != 0
    assert "derives --as-of" in res.output

    res = runner.invoke(
        cli,
        ["backtest-historical", "--from-run", str(run_id), "-m", "india"],
        obj=fetcher,
    )
    assert res.exit_code != 0
    assert "is for market 'us'" in res.output

    res = runner.invoke(
        cli,
        ["backtest-historical", "--tickers", "AAA", "--entry", "close > 0"],
        obj=fetcher,
    )
    assert res.exit_code != 0
    assert "--as-of is required unless --from-run" in res.output
