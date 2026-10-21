"""Canonical history keys must reach replay and backup without a migration."""

from datetime import date

import pandas as pd
from click.testing import CliRunner

from screener import history, history_sync
from screener.cli import cli
from tests.test_history_storage import StubTursoClient
from tests.conftest import StubPriceFetcher, make_bars


def test_canonical_history_keys_reach_historical_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    bars_nse = make_bars(n=60, seed=11, open_base=100.0)
    bars_bse = make_bars(n=60, seed=12, open_base=200.0)
    fetcher = StubPriceFetcher(
        {
            "TCS.NS": bars_nse,
            "TCS.BO": bars_bse,
            "^NSEI": make_bars(n=60, seed=99, open_base=400.0),
        }
    )
    run_id = history.save_run(
        "india",
        "ema",
        2,
        pd.DataFrame(
            {
                "ticker": ["NSE:TCS", "BSE:TCS"],
                "name": ["TCS", "TCS"],
                "close": [100.0, 200.0],
            }
        ),
    )
    conn = history._connect()
    try:
        conn.execute(
            "UPDATE runs SET run_ts = ? WHERE id = ?",
            (
                bars_nse.index[39].date().isoformat() + "T12:00:00+00:00",
                run_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    requested = set()
    original_fetch = fetcher.fetch

    def record_fetch(tickers, start, end):
        tickers = list(tickers)
        requested.update(tickers)
        return original_fetch(tickers, start, end)

    monkeypatch.setattr(fetcher, "fetch", record_fetch)
    result = CliRunner().invoke(
        cli,
        ["backtest-historical", "--from-run", str(run_id), "--hold", "5"],
        obj=fetcher,
    )
    assert result.exit_code == 0, result.output
    assert "2 tickers" in result.output
    assert {"TCS.NS", "TCS.BO"} <= requested


def test_history_backup_restore_keeps_old_and_canonical_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "source.db")
    history.save_run("india", "legacy", 1, pd.DataFrame({"name": ["TCS"]}))
    history.save_run(
        "india",
        "canonical",
        2,
        pd.DataFrame(
            {
                "ticker": ["NSE:TCS", "BSE:TCS"],
                "name": ["TCS", "TCS"],
                "close": [100.0, 200.0],
            }
        ),
    )
    client = StubTursoClient()
    summary = history_sync.backup_history(client)
    assert summary.rows_pushed == 3
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "restored.db")
    restored = history_sync.restore_history(client)
    assert restored.rows_restored == 3
    old = history.resolve_replay_run("india:legacy")
    new = history.resolve_replay_run("india:canonical")
    assert old.tickers == ["TCS"]
    assert new.tickers == ["NSE:TCS", "BSE:TCS"]
    assert new.rows.close.tolist() == [100.0, 200.0]
    assert old.run_date <= date.today()
