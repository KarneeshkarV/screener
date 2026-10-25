"""Connors signal and simulation integration with timestamped market evidence."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.connors_crash_research import (
    load_connors_pit_bundle,
    simulate_connors_crash,
)
from tests.test_connors_crash_research import connors_test_market
from tests.test_pit_market_data import change_csv, write_pit_bundle


def load_bundle(root, calendar):
    return load_connors_pit_bundle(root, "us", calendar[-1], calendar[110])


def test_connors_pit_late_bar_blocks_decision_and_cannot_fill(tmp_path):
    calendar = write_pit_bundle(tmp_path)
    change_csv(
        tmp_path,
        "bars.csv",
        lambda d: d.loc.__setitem__((100, "available_at"), "2030-01-01T00:00:00Z"),
    )
    data = load_bundle(tmp_path, calendar)
    data.arrays["crsi"][:] = 95
    data.arrays["volatility"][:] = 110
    result = simulate_connors_crash(data, calendar[110], calendar[-1])
    assert result.summary["entries_filled"] == 0
    assert all(
        row["reason"] == "bar_not_available_at_decision" for row in result.decisions
    )


def test_connors_pit_missing_bar_resets_daily_indicators(tmp_path):
    calendar = write_pit_bundle(tmp_path)
    change_csv(tmp_path, "bars.csv", lambda d: d.drop(index=110, inplace=True))
    data = load_bundle(tmp_path, calendar)
    assert np.isnan(data.arrays["crsi"][111:, 0]).all()
    assert not data.arrays["complete_window"][111:, 0].any()
    assert data.missing_sessions.iloc[0]["date"] == str(calendar[110].date())


def test_connors_pit_held_gap_queues_cover_without_waiting_for_indicator_warmup(
    tmp_path,
):
    calendar = write_pit_bundle(tmp_path)
    change_csv(tmp_path, "bars.csv", lambda d: d.drop(index=[112, 113], inplace=True))
    data = load_bundle(tmp_path, calendar)
    data.arrays["crsi"][109, 0] = 95
    data.arrays["volatility"][109, 0] = 110
    result = simulate_connors_crash(data, calendar[110], calendar[-1])
    assert result.trades[0]["exit_date"] == str(calendar[114].date())
    assert result.trades[0]["exit_reason"] == "exit_indicator_unavailable"
    assert result.summary["exit_indicator_unavailable_days"] == 2
    assert result.summary["unfilled_exit_days"] == 1
    assert result.summary["open_positions_at_end"] == 0


def test_connors_pit_unavailable_indicator_preserves_existing_exit_reason():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    data.arrays["crsi"][1, 0] = 20
    data.arrays["crsi"][2:, 0] = np.nan
    for key in ("open", "high", "low", "close"):
        data.arrays[key][2:4, 0] = np.nan
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[-1])
    assert result.trades[0]["exit_reason"] == "crsi_below_30"
    assert result.trades[0]["exit_date"] == str(data.calendar[4].date())
    assert result.summary["exit_indicator_unavailable_days"] == 2


def test_connors_pit_all_listed_interval_counts_include_empty_sessions(tmp_path):
    calendar = write_pit_bundle(tmp_path)
    cutoff = str(calendar[110].date())
    change_csv(tmp_path, "securities.csv", lambda d: d.__setitem__("listed_on", cutoff))
    for name in ("universe.csv", "bars.csv"):
        change_csv(
            tmp_path,
            name,
            lambda d: d.drop(index=d.index[d.session < cutoff], inplace=True),
        )
    change_csv(
        tmp_path,
        "universe_sessions.csv",
        lambda d: d.__setitem__("member_count", (d.session >= cutoff).astype(int)),
    )
    data = load_bundle(tmp_path, calendar)
    assert not data.membership[:110].any()
    assert data.membership[110:].all()
    assert np.isnan(data.arrays["crsi"]).all()
    change_csv(
        tmp_path, "universe_sessions.csv", lambda d: d.__setitem__("member_count", 2)
    )
    with pytest.raises(ValueError, match="member count mismatch"):
        load_bundle(tmp_path, calendar)


def test_connors_pit_future_prices_and_actions_do_not_change_past_signals(tmp_path):
    calendar = write_pit_bundle(tmp_path)
    before = load_bundle(tmp_path, calendar)
    change_csv(
        tmp_path,
        "bars.csv",
        lambda d: d.loc.__setitem__(
            (124, ["open", "high", "low", "close"]), [50.0, 52.0, 49.0, 50.0]
        ),
    )
    pd.DataFrame(
        [
            {
                "session": str(calendar[-1].date()),
                "security_id": "SEC001",
                "available_at": str(calendar[-1].date()) + "T07:00:00Z",
                "split": 2.0,
                "dividend": 0.0,
                "unsupported": 0,
            }
        ]
    ).to_csv(tmp_path / "actions.csv", index=False)
    after = load_bundle(tmp_path, calendar)
    for key in ("crsi", "volatility"):
        np.testing.assert_allclose(
            before.arrays[key][:-1], after.arrays[key][:-1], equal_nan=True
        )
    assert after.arrays["volatility"][-1, 0] == pytest.approx(0)


def test_connors_pit_refuses_held_delisting_without_action_row(tmp_path):
    calendar = write_pit_bundle(tmp_path)
    cutoff = str(calendar[115].date())
    change_csv(
        tmp_path, "securities.csv", lambda d: d.__setitem__("delisted_on", cutoff)
    )
    for name in ("universe.csv", "bars.csv"):
        change_csv(
            tmp_path,
            name,
            lambda d: d.drop(index=d.index[d.session > cutoff], inplace=True),
        )
    change_csv(
        tmp_path,
        "universe_sessions.csv",
        lambda d: d.__setitem__("member_count", (d.session <= cutoff).astype(int)),
    )
    data = load_bundle(tmp_path, calendar)
    data.arrays["crsi"][:] = 95
    data.arrays["volatility"][:] = 110
    with pytest.raises(ValueError, match="held delisting requires"):
        simulate_connors_crash(data, calendar[110], calendar[-1])
    with pytest.raises(ValueError, match="held delisting requires"):
        simulate_connors_crash(data, calendar[110], calendar[115])


def test_connors_pit_unknown_borrow_blocks_executable_claims():
    data = connors_test_market()
    with pytest.raises(ValueError, match="permissions required"):
        simulate_connors_crash(
            data, data.calendar[1], data.calendar[-1], require_short_permissions=True
        )


def test_connors_pit_borrow_capacity_rechecked_and_held_permission_loss_aborts():
    data = connors_test_market()
    data.arrays["short_capacity"] = np.full_like(data.membership, 10, dtype=float)
    data.arrays["borrow_rate"] = np.full_like(data.membership, 0.1, dtype=float)
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    result = simulate_connors_crash(
        data, data.calendar[1], data.calendar[-1], require_short_permissions=True
    )
    assert result.open_positions[0]["quantity"] == 10
    assert result.summary["borrow_cost"] > 0
    assert result.summary["executable"] is False
    data.arrays["short_capacity"][2, 0] = 0
    with pytest.raises(ValueError, match="held short lacks permission"):
        simulate_connors_crash(
            data, data.calendar[1], data.calendar[-1], require_short_permissions=True
        )


def test_connors_pit_late_held_quote_aborts_instead_of_using_future_mark():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    data.arrays["prior_quote_available"] = np.ones_like(data.membership)
    data.arrays["prior_quote_available"][2, 0] = False
    with pytest.raises(ValueError, match="held valuation unavailable"):
        simulate_connors_crash(data, data.calendar[1], data.calendar[-1])


def test_connors_pit_missing_held_quote_remains_stale_not_future_data():
    data = connors_test_market()
    data.arrays["crsi"][0, 0] = 95
    data.arrays["high"][1, 0] = 104
    data.arrays["prior_quote_available"] = np.ones_like(data.membership)
    data.arrays["prior_quote_available"][3, 0] = False
    for key in ("open", "high", "low", "close"):
        data.arrays[key][2, 0] = np.nan
    result = simulate_connors_crash(data, data.calendar[1], data.calendar[-1])
    assert result.summary["stale_position_days"] == 1


def test_connors_pit_pending_order_cancels_when_capacity_falls():
    data = connors_test_market()
    data.arrays["short_capacity"] = np.full_like(data.membership, 25, dtype=float)
    data.arrays["borrow_rate"] = np.zeros_like(data.membership, dtype=float)
    data.arrays["crsi"][0, 0] = 95
    data.arrays["short_capacity"][2:, 0] = 5
    data.arrays["high"][2:, 0] = 104
    result = simulate_connors_crash(
        data, data.calendar[1], data.calendar[-1], require_short_permissions=True
    )
    assert result.summary["entries_filled"] == 0
    assert result.orders[-1]["event"] == "cancel_short_capacity"


def test_connors_pit_cli_writes_evidence_and_blocked_decisions(tmp_path):
    root = tmp_path / "bundles"
    calendar = write_pit_bundle(root / "us", count=1450)
    write_pit_bundle(root / "india", "india", count=1450)
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.connors_crash_research",
            "--pit-bundles",
            str(root),
            "--end",
            str(calendar[-1].date()),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 24
    assert not summary.executable.any()
    assert set(summary.evidence_classification) == {"timestamped_signal_research"}
    assert (output / "us_5y_gross" / "decisions.csv").exists()
    assert (
        json.loads((output / "manifest.json").read_text())["classification"]
        == "timestamped_signal_research"
    )
