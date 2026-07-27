"""Phase 3.4: daily panel as a reduction over the contract store.

Offline-stub tests that build a synthetic multi-snapshot contract store with
:func:`contract_store.append_snapshot` and assert the store-derived panel row,
the three new intraday columns, and that the no-store path is unchanged.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from screener import cache
from screener.options import contract_store
from screener.options.backtest import OPTION_EXPRESSION_FIELDS, merge_options_into_bars
from screener.options.models import OptionChain, OptionContract
from screener.options.panels import (
    INTRADAY_PANEL_FIELDS,
    build_india_panel,
    metrics_row,
    store_panel_rows,
)

# 2026-07-10 in Asia/Kolkata (+5:30): both timestamps land in the same session.
OPEN_TS = datetime(2026, 7, 10, 4, 0, tzinfo=timezone.utc)  # 09:30 IST
CLOSE_TS = datetime(2026, 7, 10, 9, 30, tzinfo=timezone.utc)  # 15:00 IST
SESSION_DAY = date(2026, 7, 10)
EXPIRY = date(2026, 7, 31)


def _contract(
    right: str, *, strike: float, oi: float, volume: float, iv: float, as_of: datetime
) -> OptionContract:
    return OptionContract(
        symbol=f"NIFTY-{strike}-{right}",
        underlying="NIFTY",
        expiry=EXPIRY,
        strike=strike,
        right=right,  # type: ignore[arg-type]
        oi=oi,
        volume=volume,
        iv=iv,
        last=5.0,
        previous_close=4.0,
        lot_size=50.0,
        as_of=as_of,
        source="fixture",
    )


def _chain(
    *, call_oi: float, put_oi: float, call_vol: float, put_vol: float, iv: float, ts
) -> OptionChain:
    return OptionChain(
        underlying="NIFTY",
        market="india",
        spot=100.0,
        as_of=ts,
        source="fixture",
        contracts=(
            _contract("call", strike=100, oi=call_oi, volume=call_vol, iv=iv, as_of=ts),
            _contract("put", strike=100, oi=put_oi, volume=put_vol, iv=iv, as_of=ts),
        ),
    )


@pytest.fixture
def store_root(tmp_path: Path):
    cache.set_cache_area_path("contracts", tmp_path / "contracts")
    cache.set_cache_area_path("panels", tmp_path / "panels")
    try:
        yield tmp_path / "contracts"
    finally:
        cache.reset_cache_area_paths()


def _seed_two_snapshots() -> None:
    # ``observed_at`` drives the session-date partition (PIT wall-clock); pin it
    # to the fixture session so rows land under SESSION_DAY, not "today".
    contract_store.append_snapshot(
        _chain(call_oi=100, put_oi=90, call_vol=10, put_vol=20, iv=0.20, ts=OPEN_TS),
        market="india",
        enrich=False,
        observed_at=OPEN_TS,
    )
    contract_store.append_snapshot(
        _chain(call_oi=150, put_oi=140, call_vol=30, put_vol=30, iv=0.30, ts=CLOSE_TS),
        market="india",
        enrich=False,
        observed_at=CLOSE_TS,
    )


def test_metrics_row_carries_null_intraday_fields_on_eod_path():
    row = metrics_row(
        _chain(call_oi=100, put_oi=90, call_vol=10, put_vol=20, iv=0.2, ts=CLOSE_TS)
    )
    for field in INTRADAY_PANEL_FIELDS:
        assert field in row
        assert row[field] is None


def test_intraday_fields_registered_as_expression_fields():
    assert set(INTRADAY_PANEL_FIELDS) <= OPTION_EXPRESSION_FIELDS


def test_store_panel_rows_reduces_snapshots_with_intraday_fields(store_root: Path):
    _seed_two_snapshots()
    rows = store_panel_rows("india", SESSION_DAY)
    assert len(rows) == 1
    row = rows[0]
    # Base row is the last snapshot's metrics, dated to the session day.
    assert row["SYMBOL"] == "NIFTY"
    assert row["as_of"] == pd.Timestamp(SESSION_DAY)
    assert row["call_oi"] == 150
    assert row["put_oi"] == 140
    # Intraday-derived columns from the two snapshots.
    assert row["oi_change_intraday"] == pytest.approx((150 + 140) - (100 + 90))
    assert row["iv_change_intraday"] == pytest.approx(0.30 - 0.20)
    # Mean of per-snapshot put/call volume ratio: mean(20/10, 30/30) = 1.5.
    assert row["pcr_volume_intraday"] == pytest.approx(1.5)


def test_store_panel_rows_symbol_filter_and_missing(store_root: Path):
    _seed_two_snapshots()
    assert store_panel_rows("india", SESSION_DAY, symbols={"banknifty"}) == []
    assert store_panel_rows("india", SESSION_DAY, symbols={"nifty"})  # matched
    # A day with nothing recorded yields no rows (legacy fallback signal).
    assert store_panel_rows("india", date(2026, 7, 9)) == []


def test_single_snapshot_leaves_intraday_fields_null(store_root: Path):
    contract_store.append_snapshot(
        _chain(call_oi=100, put_oi=90, call_vol=10, put_vol=20, iv=0.20, ts=CLOSE_TS),
        market="india",
        enrich=False,
        observed_at=CLOSE_TS,
    )
    rows = store_panel_rows("india", SESSION_DAY)
    assert len(rows) == 1
    for field in INTRADAY_PANEL_FIELDS:
        assert rows[0][field] is None


def test_build_india_panel_prefers_store_over_bhavcopy(store_root: Path):
    _seed_two_snapshots()

    def _fetch_should_not_run(day: date) -> pd.DataFrame:  # pragma: no cover
        raise AssertionError("bhavcopy must not be fetched when store covers request")

    panel = build_india_panel(
        SESSION_DAY,
        SESSION_DAY,
        symbols={"NIFTY"},
        fetcher=_fetch_should_not_run,
        trading_day=lambda _day: True,
    )
    assert len(panel) == 1
    stored = panel.iloc[0]
    assert stored["SYMBOL"] == "NIFTY"
    assert stored["oi_change_intraday"] == pytest.approx(100)
    assert stored["iv_change_intraday"] == pytest.approx(0.10)


def test_build_india_panel_merges_store_with_bhavcopy_for_missing(store_root: Path):
    """Store rows win on conflict; bhavcopy fills underlyings the store missed."""
    _seed_two_snapshots()  # only NIFTY in the contract store
    calls: list[date] = []

    def _fetch(day: date, **_kwargs: object) -> dict[str, OptionChain]:
        calls.append(day)
        return {
            # Same underlying as the store — store row must win (keep intraday fields).
            "NIFTY": _chain(
                call_oi=1, put_oi=1, call_vol=1, put_vol=1, iv=0.99, ts=CLOSE_TS
            ),
            # Extra F&O name only present on the bhavcopy.
            "RELIANCE": OptionChain(
                underlying="RELIANCE",
                market="india",
                spot=2500.0,
                as_of=CLOSE_TS,
                source="bhavcopy",
                contracts=(
                    OptionContract(
                        symbol="RELIANCE-2500-call",
                        underlying="RELIANCE",
                        expiry=EXPIRY,
                        strike=2500,
                        right="call",
                        oi=10,
                        volume=5,
                        iv=0.25,
                        last=5.0,
                        previous_close=4.0,
                        lot_size=250.0,
                        as_of=CLOSE_TS,
                        source="bhavcopy",
                    ),
                    OptionContract(
                        symbol="RELIANCE-2500-put",
                        underlying="RELIANCE",
                        expiry=EXPIRY,
                        strike=2500,
                        right="put",
                        oi=12,
                        volume=6,
                        iv=0.25,
                        last=5.0,
                        previous_close=4.0,
                        lot_size=250.0,
                        as_of=CLOSE_TS,
                        source="bhavcopy",
                    ),
                ),
            ),
        }

    import screener.options.panels as panels_module

    original = panels_module.load_bhavcopy_chains
    panels_module.load_bhavcopy_chains = _fetch  # type: ignore[assignment]
    try:
        # symbols=None → store does not cover the full F&O universe; merge both.
        panel = build_india_panel(
            SESSION_DAY,
            SESSION_DAY,
            symbols=None,
            trading_day=lambda _day: True,
        )
    finally:
        panels_module.load_bhavcopy_chains = original  # type: ignore[assignment]

    assert calls == [SESSION_DAY]
    by_symbol = {str(row["SYMBOL"]).upper(): row for _, row in panel.iterrows()}
    assert set(by_symbol) == {"NIFTY", "RELIANCE"}
    # Store row preferred for NIFTY (intraday fields populated, not bhavcopy's 0.99 IV).
    assert by_symbol["NIFTY"]["oi_change_intraday"] == pytest.approx(100)
    assert by_symbol["NIFTY"]["call_oi"] == 150
    assert by_symbol["RELIANCE"]["call_oi"] == 10
    assert pd.isna(by_symbol["RELIANCE"]["oi_change_intraday"])


def test_build_india_panel_falls_back_when_store_empty(store_root: Path):
    calls: list[date] = []

    def _fetch(day: date, **_kwargs: object) -> dict[str, OptionChain]:
        calls.append(day)
        return {
            "NIFTY": _chain(
                call_oi=100, put_oi=90, call_vol=10, put_vol=20, iv=0.2, ts=CLOSE_TS
            )
        }

    # No snapshots seeded → the legacy bhavcopy path runs and its rows carry the
    # additive intraday columns as NaN (single EOD observation).
    import screener.options.panels as panels_module

    original = panels_module.load_bhavcopy_chains
    panels_module.load_bhavcopy_chains = _fetch  # type: ignore[assignment]
    try:
        panel = build_india_panel(
            SESSION_DAY,
            SESSION_DAY,
            symbols={"NIFTY"},
            trading_day=lambda _day: True,
        )
    finally:
        panels_module.load_bhavcopy_chains = original  # type: ignore[assignment]
    assert calls == [SESSION_DAY]
    assert len(panel) == 1
    for field in INTRADAY_PANEL_FIELDS:
        assert pd.isna(panel.iloc[0][field])


# --------------------------------------------------------------------------- #
# H5: daily option rows only join onto intraday bars at/after session close
# --------------------------------------------------------------------------- #
def test_merge_options_intraday_available_only_after_session_close():
    """US July-8 panel row is available only from 16:00 ET (20:00 UTC) onward."""
    # Naive-UTC intraday stamps: 09:30 ET = 13:30 UTC, 16:00 ET = 20:00 UTC (EDT).
    index = pd.DatetimeIndex(
        [
            "2026-07-08 13:30:00",  # open — must NOT see today's close metrics
            "2026-07-08 19:59:00",  # one minute before close
            "2026-07-08 20:00:00",  # session close — first available
            "2026-07-09 13:30:00",  # next open — still carries July-8 via ffill
        ]
    )
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        },
        index=index,
    )
    panel = pd.DataFrame(
        [{"as_of": "2026-07-08", "SYMBOL": "AAPL", "pcr": 0.7, "iv_rank": 25.0}]
    )
    joined = merge_options_into_bars(
        {"NASDAQ:AAPL": bars},
        market="us",
        fields={"pcr", "iv_rank"},
        panel=panel,
    ).bars_by_tv["NASDAQ:AAPL"]
    assert pd.isna(joined.loc[index[0], "pcr"])
    assert pd.isna(joined.loc[index[1], "pcr"])
    assert joined.loc[index[2], "pcr"] == pytest.approx(0.7)
    assert joined.loc[index[3], "pcr"] == pytest.approx(0.7)
    assert joined.loc[index[2], "iv_rank"] == pytest.approx(25.0)


def test_merge_options_daily_bars_unchanged_at_midnight():
    """Daily (midnight-normalized) targets keep same-day availability."""
    index = pd.bdate_range("2026-07-06", periods=4)  # Mon–Thu, all midnight
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        },
        index=index,
    )
    panel = pd.DataFrame(
        [
            {"as_of": index[1], "SYMBOL": "AAPL", "pcr": 0.7},
            {"as_of": index[3], "SYMBOL": "AAPL", "pcr": 1.2},
        ]
    )
    joined = merge_options_into_bars(
        {"NASDAQ:AAPL": bars},
        market="us",
        fields={"pcr"},
        panel=panel,
    ).bars_by_tv["NASDAQ:AAPL"]
    assert pd.isna(joined.loc[index[0], "pcr"])
    assert joined.loc[index[1], "pcr"] == pytest.approx(0.7)
    assert joined.loc[index[2], "pcr"] == pytest.approx(0.7)
    assert joined.loc[index[3], "pcr"] == pytest.approx(1.2)
