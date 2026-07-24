from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from screener.cli import cli as _root_cli  # noqa: F401 - import-order guard
from screener.options import cli as options_cli
from screener.options import contract_store, recorder
from screener.options.models import OptionChain, OptionContract

AS_OF = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)


def _chain(underlying: str) -> OptionChain:
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
        lot_size=10.0,
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

    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.calls = 0

    def fetch_chain(self, symbol, market, expiry=None, *, refresh=False):
        self.calls += 1
        if symbol in self.known:
            return _chain(symbol)
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
        "us", ["SPY", "NOPE"], provider=provider, root=tmp_path, enrich=False
    )
    assert result.recorded == [("SPY", 1)]
    assert result.missing == ["NOPE"]
    assert result.contract_count == 1
    stored = contract_store.load_contracts(
        "SPY", market="us", day=date(2026, 7, 10), root=tmp_path
    )
    assert stored is not None and len(stored) == 1


def test_run_pass_is_idempotent(tmp_path: Path):
    provider = _StubProvider(known={"SPY"})
    recorder.run_pass("us", ["SPY"], provider=provider, root=tmp_path, enrich=False)
    recorder.run_pass("us", ["SPY"], provider=provider, root=tmp_path, enrich=False)
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
    result = runner.invoke(
        options_cli.options,
        ["record", "-m", "us", "--once", "--watchlist", "SPY,NOPE"],
        obj={"provider": provider, "root": tmp_path},
    )
    assert result.exit_code == 0, result.output
    assert "1/2 underlyings" in result.output
    stored = contract_store.load_contracts(
        "SPY", market="us", day=date(2026, 7, 10), root=tmp_path
    )
    assert stored is not None and len(stored) == 1


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
