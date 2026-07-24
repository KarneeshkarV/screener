from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from screener import cache
from screener.options import contract_store
from screener.options.contract_store import (
    append_snapshot,
    chain_to_frame,
    contract_path,
    enrich_contracts,
    frame_to_chains,
    load_contracts,
    load_range,
    store_health,
)
from screener.options.greeks import black_scholes_price
from screener.options.models import OptionChain, OptionContract

AS_OF = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)


def _contract(**overrides) -> OptionContract:
    values = {
        "symbol": "ABC260731C00100000",
        "underlying": "ABC",
        "expiry": date(2026, 7, 31),
        "strike": 100.0,
        "right": "call",
        "oi": 100.0,
        "oi_change": 10.0,
        "volume": 20.0,
        "iv": 0.25,
        "bid": 4.0,
        "ask": 6.0,
        "last": 5.0,
        "previous_close": 4.0,
        "delta": 0.25,
        "gamma": 0.01,
        "theta": -0.5,
        "vega": 0.1,
        "rho": 0.05,
        "lot_size": 10.0,
        "as_of": AS_OF,
        "source": "fixture",
    }
    values.update(overrides)
    return OptionContract(**values)


def _chain(*contracts: OptionContract, underlying: str = "ABC", spot: float = 100.0):
    return OptionChain(
        underlying=underlying,
        market="us",
        spot=spot,
        as_of=AS_OF,
        source="fixture",
        contracts=contracts or (_contract(),),
    )


def test_chain_to_frame_schema_and_values():
    frame = chain_to_frame(_chain())
    assert list(frame.columns) == list(contract_store.CONTRACT_COLUMNS)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["underlying"] == "ABC"
    assert row["contract_symbol"] == "ABC260731C00100000"
    assert row["snapshot_ts"] == pd.Timestamp("2026-07-10 15:00:00")
    assert row["spot"] == 100.0


def test_append_and_load_round_trip(tmp_path: Path):
    merged = append_snapshot(_chain(), market="us", root=tmp_path, enrich=False)
    assert len(merged) == 1
    day = date(2026, 7, 10)
    loaded = load_contracts("ABC", market="us", day=day, root=tmp_path)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.iloc[0]["strike"] == 100.0


def test_append_is_idempotent_and_preserves_mtime(tmp_path: Path):
    append_snapshot(_chain(), market="us", root=tmp_path, enrich=False)
    path = contract_path("ABC", market="us", day=date(2026, 7, 10), root=tmp_path)
    mtime = path.stat().st_mtime_ns
    merged = append_snapshot(_chain(), market="us", root=tmp_path, enrich=False)
    assert len(merged) == 1
    assert path.stat().st_mtime_ns == mtime  # no-op left the file untouched


def test_dedupe_keeps_distinct_snapshots_same_day(tmp_path: Path):
    append_snapshot(_chain(), market="us", root=tmp_path, enrich=False)
    later = OptionChain(
        underlying="ABC",
        market="us",
        spot=101.0,
        as_of=AS_OF + timedelta(minutes=15),
        source="fixture",
        contracts=(_contract(last=5.5),),
    )
    merged = append_snapshot(later, market="us", root=tmp_path, enrich=False)
    # Same contract, two snapshot timestamps -> two rows in one day partition.
    assert len(merged) == 2
    assert merged["snapshot_ts"].nunique() == 2


def test_enrich_fills_missing_iv_and_greeks(tmp_path: Path):
    # A mark priced at a known IV should invert back to it, and greeks fill.
    time_years = (date(2027, 7, 10) - date(2026, 7, 10)).days / 365.25
    mark = black_scholes_price(100.0, 100.0, time_years, 0.045, 0.2, "call")
    assert mark is not None
    contract = _contract(
        expiry=date(2027, 7, 10),
        iv=None,
        bid=None,
        ask=None,
        last=mark,
        previous_close=None,
        delta=None,
        gamma=None,
        theta=None,
        vega=None,
        rho=None,
    )
    frame = enrich_contracts(chain_to_frame(_chain(contract)), market="us")
    row = frame.iloc[0]
    assert row["iv"] == pytest.approx(0.2, abs=1e-3)
    assert 0.4 < row["delta"] < 0.75  # sensible for a ~1y ATM call
    assert not pd.isna(row["gamma"])
    assert not pd.isna(row["vega"])


def test_enrich_leaves_unidentifiable_rows_untouched(tmp_path: Path):
    contract = _contract(
        expiry=date(2020, 1, 1),  # already expired relative to snapshot
        iv=None,
        bid=None,
        ask=None,
        last=None,
        previous_close=None,
        delta=None,
    )
    frame = enrich_contracts(chain_to_frame(_chain(contract)), market="us")
    assert pd.isna(frame.iloc[0]["iv"])
    assert pd.isna(frame.iloc[0]["delta"])


def test_contract_path_sanitizes_underlying(tmp_path: Path):
    path = contract_path("BRK/B", market="us", day=date(2026, 7, 10), root=tmp_path)
    assert path.name == "BRK_B.parquet"
    assert "2026-07-10" in str(path)


def test_frame_to_chains_reconstructs(tmp_path: Path):
    append_snapshot(_chain(), market="us", root=tmp_path, enrich=False)
    frame = load_contracts("ABC", market="us", day=date(2026, 7, 10), root=tmp_path)
    chains = frame_to_chains(frame, market="us")
    assert len(chains) == 1
    chain = chains[0]
    assert chain.underlying == "ABC"
    assert chain.contracts[0].strike == 100.0
    assert chain.contracts[0].right == "call"


def test_load_range_spans_days(tmp_path: Path):
    append_snapshot(_chain(), market="us", root=tmp_path, enrich=False)
    next_day = OptionChain(
        underlying="ABC",
        market="us",
        spot=100.0,
        as_of=datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc),
        source="fixture",
        contracts=(_contract(as_of=datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)),),
    )
    append_snapshot(next_day, market="us", root=tmp_path, enrich=False)
    rows = load_range(
        "ABC",
        market="us",
        start=date(2026, 7, 10),
        end=date(2026, 7, 13),
        root=tmp_path,
    )
    assert len(rows) == 2
    windowed = load_range(
        "ABC",
        market="us",
        start=date(2026, 7, 13),
        end=date(2026, 7, 13),
        root=tmp_path,
    )
    assert len(windowed) == 1


def test_store_health_detects_gap(tmp_path: Path):
    # Two sessions with a business-day gap between them (Fri 10th, Wed 15th).
    append_snapshot(_chain(), market="us", root=tmp_path, enrich=False)
    later = OptionChain(
        underlying="ABC",
        market="us",
        spot=100.0,
        as_of=datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc),
        source="fixture",
        contracts=(_contract(as_of=datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc)),),
    )
    append_snapshot(later, market="us", root=tmp_path, enrich=False)
    health = store_health(
        "us", root=tmp_path, now=datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)
    )
    assert health.last_snapshot is not None
    assert set(health.sessions_present) == {date(2026, 7, 10), date(2026, 7, 15)}
    # Mon 13th and Tue 14th are business days with no partition.
    assert date(2026, 7, 13) in health.missing_sessions
    assert date(2026, 7, 14) in health.missing_sessions
    assert "missing" in health.summary()


def test_store_health_empty_and_stale(tmp_path: Path):
    empty = store_health("us", root=tmp_path)
    assert empty.last_snapshot is None
    assert empty.is_stale
    assert "no snapshots" in empty.summary()


def test_default_root_honours_cache_override(tmp_path: Path):
    cache.set_cache_area_path("contracts", tmp_path / "contracts")
    try:
        append_snapshot(_chain(), market="us", enrich=False)
        loaded = load_contracts("ABC", market="us", day=date(2026, 7, 10))
        assert loaded is not None
        assert (tmp_path / "contracts" / "us" / "2026-07-10" / "ABC.parquet").exists()
    finally:
        cache.reset_cache_area_paths()
