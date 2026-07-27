from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from screener.cli import cli as _root_cli  # noqa: F401 - import-order guard
from screener.options import cli as options_cli
from screener.options import contract_store, recorder
from screener.options.cboe import US_OPTION_LOT_SIZE, parse_cboe_chain
from screener.options.models import OptionChain, OptionContract
from screener.options.nse_live import DEFAULT_INDIA_LOT_SIZES, parse_nse_chain

AS_OF = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures"


def _chain(underlying: str, *, lot_size: float | None = 10.0) -> OptionChain:
    contract = OptionContract(
        symbol=f"{underlying}C100",
        underlying=underlying,
        expiry=date(2026, 7, 31),
        strike=100.0,
        right="call",
        oi=100.0,
        volume=10.0,
        iv=0.25,
        bid=4.0,
        ask=6.0,
        last=5.0,
        lot_size=lot_size,
        as_of=AS_OF,
        source="stub",
    )
    return OptionChain(
        underlying=underlying,
        market="us",
        spot=100.0,
        as_of=AS_OF,
        source="stub",
        contracts=(contract,),
    )


class _StubProvider:
    """Returns a chain for known symbols, None otherwise; counts calls."""

    def __init__(self, known: set[str], *, lot_size: float | None = 10.0) -> None:
        self.known = known
        self.calls = 0
        self.lot_size = lot_size

    def fetch_chain(self, symbol, market, expiry=None, *, refresh=False):
        self.calls += 1
        if symbol in self.known:
            return _chain(symbol, lot_size=self.lot_size)
        return None


def test_resolve_watchlist_default_explicit_file_and_cap(tmp_path: Path):
    assert recorder.resolve_watchlist("us") == list(recorder.DEFAULT_WATCHLISTS["us"])
    assert recorder.resolve_watchlist("us", watchlist="spy, qqq ,spy") == [
        "SPY",
        "QQQ",
    ]
    listing = tmp_path / "wl.txt"
    listing.write_text("# indices\nNIFTY, BANKNIFTY\nFINNIFTY\n")
    assert recorder.resolve_watchlist("india", watchlist_file=listing) == [
        "NIFTY",
        "BANKNIFTY",
        "FINNIFTY",
    ]
    assert recorder.resolve_watchlist("us", watchlist="a,b,c,d", max_underlyings=2) == [
        "A",
        "B",
    ]


def test_within_session_boundaries():
    # 2026-07-10 is a Friday; 14:00 UTC == 10:00 ET (inside US session).
    assert recorder.within_session(
        "us", now=datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)
    )
    # 21:00 UTC == 17:00 ET (after close).
    assert not recorder.within_session(
        "us", now=datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc)
    )
    # 2026-07-11 is a Saturday.
    assert not recorder.within_session(
        "us", now=datetime(2026, 7, 11, 14, 0, tzinfo=timezone.utc)
    )


def test_run_pass_records_and_reports_missing(tmp_path: Path):
    provider = _StubProvider(known={"SPY"})
    result = recorder.run_pass(
        "us",
        ["SPY", "NOPE"],
        provider=provider,
        root=tmp_path,
        enrich=False,
        observed_at=AS_OF,
    )
    assert result.recorded == [("SPY", 1)]
    assert result.missing == ["NOPE"]
    assert result.contract_count == 1
    stored = contract_store.load_contracts(
        "SPY", market="us", day=date(2026, 7, 10), root=tmp_path
    )
    assert stored is not None and len(stored) == 1
    # Observed time stamped; venue quote preserved separately.
    assert stored.iloc[0]["snapshot_ts"] == pd_timestamp(AS_OF)
    assert stored.iloc[0]["quote_ts"] == pd_timestamp(AS_OF)


def pd_timestamp(value: datetime):
    import pandas as pd

    return (
        pd.Timestamp(value).tz_localize(None) if value.tzinfo else pd.Timestamp(value)
    )


def test_run_pass_is_idempotent(tmp_path: Path):
    provider = _StubProvider(known={"SPY"})
    recorder.run_pass(
        "us",
        ["SPY"],
        provider=provider,
        root=tmp_path,
        enrich=False,
        observed_at=AS_OF,
    )
    recorder.run_pass(
        "us",
        ["SPY"],
        provider=provider,
        root=tmp_path,
        enrich=False,
        observed_at=AS_OF,
    )
    stored = contract_store.load_contracts(
        "SPY", market="us", day=date(2026, 7, 10), root=tmp_path
    )
    assert stored is not None and len(stored) == 1


def test_run_pass_degrades_when_provider_raises(tmp_path: Path):
    class _Boom:
        def fetch_chain(self, symbol, market, expiry=None, *, refresh=False):
            raise RuntimeError("boom")

    result = recorder.run_pass("us", ["SPY"], provider=_Boom(), root=tmp_path)
    assert result.missing == ["SPY"]
    assert result.recorded == []


def test_run_pass_warns_when_lot_size_missing(tmp_path: Path, caplog):
    provider = _StubProvider(known={"SPY"}, lot_size=None)
    with caplog.at_level(logging.WARNING, logger="screener.options.recorder"):
        result = recorder.run_pass(
            "us",
            ["SPY"],
            provider=provider,
            root=tmp_path,
            enrich=False,
            observed_at=AS_OF,
        )
    assert result.recorded == [("SPY", 1)]
    assert any("lot_size missing" in record.message for record in caplog.records)


def test_cboe_parser_sets_standard_us_lot_size():
    """C2: US equity/index options carry the 100-share multiplier at ingest."""
    raw = json.loads((FIXTURES / "cboe_aapl_delayed_sample.json").read_text())
    chain = parse_cboe_chain(raw, requested_symbol="AAPL")
    assert chain is not None
    assert chain.contracts
    assert all(c.lot_size == US_OPTION_LOT_SIZE for c in chain.contracts)
    assert US_OPTION_LOT_SIZE == 100.0


def test_nse_parser_sets_index_lot_size_defaults():
    """C2: India index options resolve lot_size (defaults when history absent)."""
    raw = json.loads((FIXTURES / "nse_live_option_chain_sample.json").read_text())
    # Sample fixture is RELIANCE equity — not in default table; inject map.
    chain = parse_nse_chain(raw, symbol="RELIANCE", lot_sizes={"RELIANCE": 500.0})
    assert chain is not None
    assert chain.contracts
    assert all(c.lot_size == 500.0 for c in chain.contracts)

    nifty_raw = {
        "records": {
            "timestamp": "10-Jul-2026 15:30:00",
            "expiryDates": ["31-Jul-2026"],
            "data": [
                {
                    "strikePrice": 25000,
                    "expiryDate": "31-Jul-2026",
                    "CE": {
                        "identifier": "NIFTY31JUL2525000CE",
                        "openInterest": 100,
                        "changeinOpenInterest": 1,
                        "totalTradedVolume": 10,
                        "impliedVolatility": 12.5,
                        "bidprice": 100,
                        "askPrice": 101,
                        "lastPrice": 100.5,
                        "underlyingValue": 25000,
                    },
                    "PE": {
                        "identifier": "NIFTY31JUL2525000PE",
                        "openInterest": 90,
                        "changeinOpenInterest": -1,
                        "totalTradedVolume": 8,
                        "impliedVolatility": 13.0,
                        "bidprice": 90,
                        "askPrice": 91,
                        "lastPrice": 90.5,
                        "underlyingValue": 25000,
                    },
                }
            ],
        }
    }
    nifty = parse_nse_chain(nifty_raw, symbol="NIFTY")
    assert nifty is not None
    assert all(c.lot_size == DEFAULT_INDIA_LOT_SIZES["NIFTY"] for c in nifty.contracts)
    assert DEFAULT_INDIA_LOT_SIZES["NIFTY"] == 75.0
    assert DEFAULT_INDIA_LOT_SIZES["BANKNIFTY"] == 35.0
    assert DEFAULT_INDIA_LOT_SIZES["FINNIFTY"] == 65.0


def test_run_pass_records_lot_size_for_us_market(tmp_path: Path):
    """C2 end-to-end: lot_size lands on stored contract rows."""
    raw = json.loads((FIXTURES / "cboe_aapl_delayed_sample.json").read_text())
    chain = parse_cboe_chain(raw, requested_symbol="AAPL")
    assert chain is not None

    class _CboeStub:
        def fetch_chain(self, symbol, market, expiry=None, *, refresh=False):
            return chain

    result = recorder.run_pass(
        "us",
        ["AAPL"],
        provider=_CboeStub(),
        root=tmp_path,
        enrich=False,
        observed_at=AS_OF,
    )
    assert result.recorded == [("AAPL", len(chain.contracts))]
    stored = contract_store.load_contracts(
        "AAPL", market="us", day=date(2026, 7, 10), root=tmp_path
    )
    assert stored is not None
    assert (stored["lot_size"] == US_OPTION_LOT_SIZE).all()


def test_run_pass_records_lot_size_for_india_market(tmp_path: Path):
    """C2 end-to-end: India default watchlist lots land on stored rows."""
    nifty_raw = {
        "records": {
            "timestamp": "10-Jul-2026 15:30:00",
            "expiryDates": ["31-Jul-2026"],
            "data": [
                {
                    "strikePrice": 25000,
                    "expiryDate": "31-Jul-2026",
                    "CE": {
                        "identifier": "NIFTY31JUL2525000CE",
                        "openInterest": 100,
                        "totalTradedVolume": 10,
                        "impliedVolatility": 12.5,
                        "bidprice": 100,
                        "askPrice": 101,
                        "lastPrice": 100.5,
                        "underlyingValue": 25000,
                    },
                }
            ],
        }
    }
    chain = parse_nse_chain(nifty_raw, symbol="NIFTY")
    assert chain is not None

    class _NseStub:
        def fetch_chain(self, symbol, market, expiry=None, *, refresh=False):
            return chain

    result = recorder.run_pass(
        "india",
        ["NIFTY"],
        provider=_NseStub(),
        root=tmp_path,
        enrich=False,
        observed_at=AS_OF,
    )
    assert result.recorded == [("NIFTY", 1)]
    # Partition by observed_at in market TZ (Asia/Kolkata).
    stored = contract_store.load_contracts(
        "NIFTY", market="india", day=date(2026, 7, 10), root=tmp_path
    )
    assert stored is not None
    assert (stored["lot_size"] == DEFAULT_INDIA_LOT_SIZES["NIFTY"]).all()


def test_record_loop_runs_in_session_then_stops(tmp_path: Path):
    provider = _StubProvider(known={"SPY"})
    times = iter(
        [
            datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc),  # inside -> one pass
            datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc),  # after close -> break
        ]
    )
    lines: list[str] = []
    results = recorder.record_loop(
        "us",
        ["SPY"],
        provider=provider,
        every_seconds=1.0,
        root=tmp_path,
        echo=lines.append,
        sleep=lambda _s: None,
        clock=lambda: next(times),
    )
    assert len(results) == 1
    assert results[0].recorded == [("SPY", 1)]
    assert any("outside session" in line for line in lines)


def test_cli_record_once_with_injected_provider(tmp_path: Path):
    provider = _StubProvider(known={"SPY"})
    runner = CliRunner()
    # CLI uses wall-clock observed_at; pin store day via monkeypatched now is
    # heavy — instead assert via store_health / present partitions.
    result = runner.invoke(
        options_cli.options,
        ["record", "-m", "us", "--once", "--watchlist", "SPY,NOPE"],
        obj={"provider": provider, "root": tmp_path},
    )
    assert result.exit_code == 0, result.output
    assert "1/2 underlyings" in result.output
    # Partition day follows wall-clock observation; locate the written file.
    us_root = tmp_path / "us"
    assert us_root.is_dir()
    parquets = list(us_root.glob("*/*.parquet"))
    assert len(parquets) == 1
    assert parquets[0].stem == "SPY"


def test_cli_record_rejects_empty_watchlist(tmp_path: Path):
    provider = _StubProvider(known=set())
    runner = CliRunner()
    result = runner.invoke(
        options_cli.options,
        ["record", "-m", "us", "--once", "--watchlist", " , "],
        obj={"provider": provider, "root": tmp_path},
    )
    assert result.exit_code != 0
    assert "no underlyings" in result.output
