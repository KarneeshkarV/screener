"""Timestamped evidence contracts, future-data rejection, and short permission tests."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.connors_crash_research import simulate_connors_crash
from scripts.connors_pit_data import load_connors_pit_bundle
from tests.test_connors_crash_research import connors_test_market


def write_connors_bundle(root, market="us", count=125):
    """Write nominal synthetic evidence with permanent IDs and an independent calendar."""
    root.mkdir(parents=True, exist_ok=True)
    calendar = pd.bdate_range("2020-01-01", periods=count)
    session = calendar.strftime("%Y-%m-%d")
    metadata = {
        "schema_version": 1,
        "market": market,
        "universe_kind": "all_listed",
        "actions_complete_through": session[-1],
    }
    metadata.update(
        {
            key: "synthetic test evidence"
            for key in (
                "calendar_source",
                "universe_source",
                "security_source",
                "price_source",
                "action_source",
                "borrow_source",
            )
        }
    )
    (root / "metadata.json").write_text(json.dumps(metadata))
    pd.DataFrame(
        {
            "session": session,
            "decision_at": session + "T08:00:00Z",
            "open_at": session + "T09:00:00Z",
            "close_at": session + "T16:00:00Z",
            "available_at": "2019-01-01T00:00:00Z",
        }
    ).to_csv(root / "sessions.csv", index=False)
    pd.DataFrame(
        {
            "security_id": ["SEC001"],
            "available_at": ["2019-01-01T00:00:00Z"],
            "listed_on": ["2019-01-01"],
            "delisted_on": [""],
        }
    ).to_csv(root / "securities.csv", index=False)
    pd.DataFrame(
        {"session": session, "available_at": session + "T07:00:00Z", "member_count": 1}
    ).to_csv(root / "universe_sessions.csv", index=False)
    pd.DataFrame(
        {
            "session": session,
            "security_id": "SEC001",
            "ticker": "OLD",
            "available_at": session + "T07:00:00Z",
        }
    ).to_csv(root / "universe.csv", index=False)
    pd.DataFrame(
        {
            "session": session,
            "security_id": "SEC001",
            "available_at": session + "T16:01:00Z",
            "open": 100.0,
            "high": 104.0,
            "low": 99.0,
            "close": 100.0,
            "turnover": 2e6,
        }
    ).to_csv(root / "bars.csv", index=False)
    pd.DataFrame(
        columns=[
            "session",
            "security_id",
            "available_at",
            "split",
            "dividend",
            "unsupported",
        ]
    ).to_csv(root / "actions.csv", index=False)
    return calendar


def load_bundle(root, calendar):
    return load_connors_pit_bundle(root, "us", calendar[-1], calendar[110])


def change_csv(root, name, change):
    frame = pd.read_csv(root / name)
    change(frame)
    frame.to_csv(root / name, index=False)


def test_connors_pit_nominal_bundle_and_ticker_rename(tmp_path):
    calendar = write_connors_bundle(tmp_path)
    change_csv(
        tmp_path,
        "universe.csv",
        lambda d: d.loc.__setitem__((d.index >= 110, "ticker"), "NEW"),
    )
    data = load_bundle(tmp_path, calendar)
    assert data.symbols == ["SEC001"]
    assert data.arrays["prior_bar_available"][111, 0]
    assert data.arrays["complete_window"][110, 0]
    assert data.coverage.missing_member_sessions.sum() == 0
    assert data.evidence["executable"] is False
    assert data.evidence["source_assertions_verified"] is False


@pytest.mark.parametrize(
    ("name", "column", "value", "message"),
    [
        ("universe.csv", "available_at", "2030-01-01T00:00:00Z", "future universe"),
        (
            "universe_sessions.csv",
            "available_at",
            "2030-01-01T00:00:00Z",
            "manifest unavailable",
        ),
        ("securities.csv", "available_at", "2030-01-01T00:00:00Z", "future universe"),
        (
            "sessions.csv",
            "available_at",
            "2030-01-01T00:00:00Z",
            "calendar availability",
        ),
        ("sessions.csv", "decision_at", "2020-01-01 08:00:00", "needs timezone"),
        ("bars.csv", "available_at", "2019-01-01T00:00:00Z", "before close"),
        ("universe.csv", "ticker", "DUMMYRAYMN", "placeholder"),
        ("securities.csv", "listed_on", "2025-01-01", "outside listing"),
    ],
)
def test_connors_pit_rejects_bad_evidence(tmp_path, name, column, value, message):
    calendar = write_connors_bundle(tmp_path)
    change_csv(tmp_path, name, lambda d: d.__setitem__(column, value))
    with pytest.raises(ValueError, match=message):
        load_bundle(tmp_path, calendar)


def test_connors_pit_universe_gap_and_count_mismatch(tmp_path):
    calendar = write_connors_bundle(tmp_path)
    change_csv(
        tmp_path, "universe_sessions.csv", lambda d: d.drop(index=4, inplace=True)
    )
    with pytest.raises(ValueError, match="every calendar session"):
        load_bundle(tmp_path, calendar)
    write_connors_bundle(tmp_path)
    change_csv(tmp_path, "universe.csv", lambda d: d.drop(index=4, inplace=True))
    with pytest.raises(ValueError, match="omits a listed security"):
        load_bundle(tmp_path, calendar)


def test_connors_pit_rejects_duplicate_identity_and_unknown_security(tmp_path):
    calendar = write_connors_bundle(tmp_path)
    change_csv(tmp_path, "securities.csv", lambda d: d.loc.__setitem__(1, d.iloc[0]))
    with pytest.raises(ValueError, match="duplicate permanent"):
        load_bundle(tmp_path, calendar)
    write_connors_bundle(tmp_path)
    change_csv(tmp_path, "bars.csv", lambda d: d.__setitem__("security_id", "UNKNOWN"))
    with pytest.raises(ValueError, match="unknown security"):
        load_bundle(tmp_path, calendar)


def test_connors_pit_late_bar_blocks_decision_and_cannot_fill(tmp_path):
    calendar = write_connors_bundle(tmp_path)
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
    calendar = write_connors_bundle(tmp_path)
    change_csv(tmp_path, "bars.csv", lambda d: d.drop(index=110, inplace=True))
    data = load_bundle(tmp_path, calendar)
    assert np.isnan(data.arrays["crsi"][111:, 0]).all()
    assert not data.arrays["complete_window"][111:, 0].any()
    assert data.missing_sessions.iloc[0]["date"] == str(calendar[110].date())


def test_connors_pit_held_gap_queues_cover_without_waiting_for_indicator_warmup(
    tmp_path,
):
    calendar = write_connors_bundle(tmp_path)
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


def test_connors_pit_rejects_insufficient_calendar_warmup(tmp_path):
    calendar = write_connors_bundle(tmp_path)
    with pytest.raises(ValueError, match="requires 101 pre-start sessions; found 1"):
        load_connors_pit_bundle(tmp_path, "us", calendar[-1], calendar[1])


def test_connors_pit_accepts_exactly_101_calendar_warmup_sessions(tmp_path):
    calendar = write_connors_bundle(tmp_path)
    data = load_connors_pit_bundle(tmp_path, "us", calendar[-1], calendar[101])
    assert data.arrays["complete_window"][100, 0]


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
    calendar = write_connors_bundle(tmp_path)
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


def test_connors_pit_availability_max_resets_at_missing_quote(tmp_path):
    calendar = write_connors_bundle(tmp_path)
    change_csv(
        tmp_path,
        "bars.csv",
        lambda d: d.loc.__setitem__((100, "available_at"), "2030-01-01T00:00:00Z"),
    )
    change_csv(tmp_path, "bars.csv", lambda d: d.drop(index=111, inplace=True))
    data = load_bundle(tmp_path, calendar)
    assert not data.arrays["prior_bar_available"][111, 0]
    assert not data.arrays["prior_quote_available"][112, 0]
    assert data.arrays["prior_bar_available"][113, 0]
    assert not data.arrays["complete_window"][113, 0]


@pytest.mark.parametrize("arrival", ["07:59:59.999999999Z", "08:00:00.000000001Z"])
def test_connors_pit_availability_preserves_nanosecond_decision_boundary(
    tmp_path, arrival
):
    calendar = write_connors_bundle(tmp_path)
    change_csv(
        tmp_path,
        "bars.csv",
        lambda d: d.loc.__setitem__(
            (109, "available_at"), str(calendar[110].date()) + "T" + arrival
        ),
    )
    data = load_bundle(tmp_path, calendar)
    assert bool(data.arrays["prior_quote_available"][110, 0]) == arrival.startswith(
        "07"
    )
    assert bool(data.arrays["prior_bar_available"][110, 0]) == arrival.startswith("07")


def test_connors_pit_future_prices_and_actions_do_not_change_past_signals(tmp_path):
    calendar = write_connors_bundle(tmp_path)
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


@pytest.mark.parametrize(
    ("unsupported", "available", "message"),
    [
        (1, "07:00:00Z", "unsupported corporate"),
        (0, "12:00:00Z", "unknown at decision"),
    ],
)
def test_connors_pit_refuses_unsupported_or_late_actions(
    tmp_path, unsupported, available, message
):
    calendar = write_connors_bundle(tmp_path)
    pd.DataFrame(
        [
            {
                "session": str(calendar[-1].date()),
                "security_id": "SEC001",
                "available_at": str(calendar[-1].date()) + "T" + available,
                "split": 1.0,
                "dividend": 0.0,
                "unsupported": unsupported,
            }
        ]
    ).to_csv(tmp_path / "actions.csv", index=False)
    with pytest.raises(ValueError, match=message):
        load_bundle(tmp_path, calendar)


@pytest.mark.parametrize("before_listing", [True, False])
def test_connors_pit_rejects_quotes_outside_listing_bounds(tmp_path, before_listing):
    calendar = write_connors_bundle(tmp_path)
    column, value = (
        ("listed_on", str(calendar[10].date()))
        if before_listing
        else ("delisted_on", str(calendar[-10].date()))
    )
    # Use index mode to test quote bounds independently of all-listed membership.
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["universe_kind"] = "index"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    change_csv(tmp_path, "securities.csv", lambda d: d.__setitem__(column, value))
    universe = pd.read_csv(tmp_path / "universe.csv")
    universe = (
        universe[universe.session >= value]
        if before_listing
        else universe[universe.session <= value]
    )
    universe.to_csv(tmp_path / "universe.csv", index=False)
    change_csv(
        tmp_path,
        "universe_sessions.csv",
        lambda d: d.__setitem__(
            "member_count", d.session.isin(universe.session).astype(int)
        ),
    )
    with pytest.raises(ValueError, match="quote outside listing"):
        load_bundle(tmp_path, calendar)


def test_connors_pit_refuses_held_delisting_without_action_row(tmp_path):
    calendar = write_connors_bundle(tmp_path)
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


def test_connors_pit_delisting_after_horizon_does_not_block(tmp_path):
    calendar = write_connors_bundle(tmp_path)
    change_csv(
        tmp_path, "securities.csv", lambda d: d.__setitem__("delisted_on", "2030-01-01")
    )
    data = load_bundle(tmp_path, calendar)
    assert not data.arrays.get("delisted", np.zeros(1)).any()


@pytest.mark.parametrize("market", ["us", "india"])
def test_connors_pit_exchange_date_and_valid_offsets(tmp_path, market):
    calendar = write_connors_bundle(tmp_path, market)
    # Existing UTC timestamps are on the same local session date in both markets.
    data = load_connors_pit_bundle(tmp_path, market, calendar[-1], calendar[110])
    assert len(data.calendar) == len(calendar)
    frame = pd.read_csv(tmp_path / "sessions.csv")
    for column in ("decision_at", "open_at", "close_at"):
        frame[column] = (
            pd.to_datetime(frame[column], utc=True) + pd.DateOffset(years=10)
        ).astype(str)
    frame.to_csv(tmp_path / "sessions.csv", index=False)
    with pytest.raises(ValueError, match="exchange-local timestamp"):
        load_connors_pit_bundle(tmp_path, market, calendar[-1], calendar[110])


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


def test_connors_pit_india_requires_slb_contract(tmp_path):
    calendar = write_connors_bundle(tmp_path, "india")
    pd.DataFrame(
        [
            {
                "session": str(calendar[-1].date()),
                "security_id": "SEC001",
                "available_at": str(calendar[-1].date()) + "T07:00:00Z",
                "max_quantity": 10.0,
                "annual_rate": 0.1,
                "contract_id": "",
                "valid_until": str(calendar[-1].date()) + "T17:00:00Z",
            }
        ]
    ).to_csv(tmp_path / "short_permissions.csv", index=False)
    with pytest.raises(ValueError, match="SLB contract"):
        load_connors_pit_bundle(tmp_path, "india", calendar[-1], calendar[110])


def test_connors_pit_cli_writes_evidence_and_blocked_decisions(tmp_path):
    root = tmp_path / "bundles"
    calendar = write_connors_bundle(root / "us", count=1450)
    write_connors_bundle(root / "india", "india", count=1450)
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
