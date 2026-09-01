from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from screener import universes
from screener.backtester.core import _resolve_universe
from screener.backtester.rolling_candidates import _build_rolling_candidate_matrices


def test_builtin_registry_includes_richer_indices() -> None:
    assert set(universes.available_universes()) >= {
        "sp500",
        "nifty50",
        "nifty500",
        "sensex",
    }
    assert universes.get_universe_definition("sensex").benchmark == "^BSESN"


def test_sensex_loader_preserves_bse_symbols(monkeypatch) -> None:
    rows = "".join(
        f"<tr><td>Company {i}</td><td>STOCK{i}.BO</td><td>1 January 2020</td></tr>"
        for i in range(30)
    )
    rows += "<tr><td>Future</td><td>FUTURE.BO</td><td>1 January 2099</td></tr>"
    html = (
        "<table><tr><th>Company</th><th>Symbol</th><th>Entry date</th></tr>"
        f"{rows}</table>"
    )
    response = SimpleNamespace(text=html, raise_for_status=lambda: None)
    monkeypatch.setattr(universes.requests, "get", lambda *args, **kwargs: response)

    symbols, source = universes._fetch_sensex()

    assert len(symbols) == 30
    assert all(symbol.endswith(".BO") for symbol in symbols)
    assert "FUTURE.BO" not in symbols
    assert "wikipedia.org" in source


def test_custom_static_universe_from_toml(tmp_path) -> None:
    config = tmp_path / "universes.toml"
    config.write_text(
        """
[universes.my_stocks]
type = "static"
market = "india"
benchmark = "^NSEI"
symbols = ["NSE:RELIANCE", "NSE:TCS", "NSE:RELIANCE"]
""".strip()
    )

    selection = universes.load_universe_selection(
        "my_stocks",
        market="india",
        as_of=date(2025, 1, 1),
        config_path=config,
    )

    assert selection.symbols == ("NSE:RELIANCE", "NSE:TCS")
    assert selection.benchmark == "^NSEI"
    assert "sha256:" in selection.source


def test_custom_snapshot_csv_builds_half_open_membership_windows(tmp_path) -> None:
    snapshots = tmp_path / "members.csv"
    pd.DataFrame(
        {
            "effective_date": [
                "2024-01-01",
                "2024-01-01",
                "2024-07-01",
                "2024-07-01",
            ],
            "symbol": ["AAA", "BBB", "BBB", "CCC"],
        }
    ).to_csv(snapshots, index=False)
    config = tmp_path / "universes.yaml"
    config.write_text(
        """
universes:
  my_index:
    type: snapshots
    market: us
    path: members.csv
""".strip()
    )

    selection = universes.load_universe_selection(
        "my_index",
        market="us",
        as_of=date(2024, 12, 31),
        config_path=config,
    )

    assert selection.symbols == ("AAA", "BBB", "CCC")
    assert ("AAA", date(2024, 1, 1), date(2024, 7, 1)) in selection.membership_windows
    assert ("BBB", date(2024, 7, 1), None) in selection.membership_windows
    assert ("CCC", date(2024, 7, 1), None) in selection.membership_windows


def test_custom_dynamic_universe_resolves_base(monkeypatch, tmp_path) -> None:
    config = tmp_path / "universes.json"
    config.write_text(
        '{"universes":{"liquid":{"type":"dynamic","market":"us",'
        '"base":"sp500","size":2,"lookback":20,"rebalance":"weekly"}}}'
    )
    fake = SimpleNamespace(symbols=("AAA", "BBB", "CCC"))
    monkeypatch.setattr(universes, "load_current_universe", lambda *a, **k: fake)

    selection = universes.load_universe_selection(
        "liquid",
        market="us",
        as_of=date(2025, 1, 1),
        config_path=config,
    )

    assert selection.symbols == ("AAA", "BBB", "CCC")
    assert selection.dynamic_size == 2
    assert selection.dynamic_lookback == 20
    assert selection.dynamic_rebalance == "weekly"


def test_dynamic_universe_ranks_with_prior_bars_only() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D")

    def bars(volume: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": volume,
            },
            index=index,
        )

    matrices = _build_rolling_candidate_matrices(
        {
            "AAA": bars([100, 100, 100, 100, 100, 100]),
            "BBB": bars([1, 1, 10_000, 10_000, 10_000, 10_000]),
        },
        {
            "AAA": pd.Series(True, index=index),
            "BBB": pd.Series(True, index=index),
        },
        {},
        list(index),
        lookback_required=0,
        dynamic_universe_size=1,
        dynamic_universe_lookback=2,
        dynamic_universe_rebalance="daily",
    )

    # BBB's day-2 volume spike cannot affect membership until day 3 because
    # dynamic ADV is shifted by one bar before rolling.
    assert bool(matrices.signal_mat.loc[index[2], "AAA"])
    assert not bool(matrices.signal_mat.loc[index[2], "BBB"])
    assert not bool(matrices.signal_mat.loc[index[3], "AAA"])
    assert bool(matrices.signal_mat.loc[index[3], "BBB"])


def test_snapshot_sync_is_idempotent_and_appends_changes(monkeypatch, tmp_path) -> None:
    output = tmp_path / "snapshots.csv"
    state = {"symbols": ("AAA", "BBB")}

    def load(*args, **kwargs):
        return SimpleNamespace(symbols=state["symbols"])

    monkeypatch.setattr(universes, "load_current_universe", load)
    path, changed, count = universes.sync_universe_snapshot(
        "sp500", output=output, as_of=date(2024, 1, 1)
    )
    assert path == output
    assert changed and count == 2

    _, changed, _ = universes.sync_universe_snapshot(
        "sp500", output=output, as_of=date(2024, 2, 1)
    )
    assert not changed

    state["symbols"] = ("BBB", "CCC")
    _, changed, _ = universes.sync_universe_snapshot(
        "sp500", output=output, as_of=date(2024, 3, 1)
    )
    assert changed
    frame = pd.read_csv(output, dtype=str)
    assert set(frame["effective_date"]) == {"2024-01-01", "2024-03-01"}
    latest = frame[frame["effective_date"] == "2024-03-01"]
    assert set(latest["symbol"]) == {"BBB", "CCC"}


def test_resolve_universe_does_not_cap_dynamic_or_snapshot_pools() -> None:
    """Dynamic/snapshot selection runs downstream; max_universe must not
    silently truncate the candidate pool first (regression)."""
    tickers = tuple(f"NSE:T{i}" for i in range(500))

    dynamic_cfg = SimpleNamespace(
        tickers=tickers,
        universe_file=None,
        membership_windows=(),
        dynamic_universe_size=100,
        max_universe=200,
    )
    symbols, warnings = _resolve_universe(dynamic_cfg)
    assert len(symbols) == 500
    assert warnings == []

    snapshot_cfg = SimpleNamespace(
        tickers=tickers,
        universe_file=None,
        membership_windows=(("NSE:T0", date(2024, 1, 1), None),),
        dynamic_universe_size=None,
        max_universe=200,
    )
    symbols, warnings = _resolve_universe(snapshot_cfg)
    assert len(symbols) == 500
    assert warnings == []

    # Plain universes keep the existing cap + warning behavior.
    plain_cfg = SimpleNamespace(
        tickers=tickers,
        universe_file=None,
        membership_windows=(),
        dynamic_universe_size=None,
        max_universe=200,
    )
    symbols, warnings = _resolve_universe(plain_cfg)
    assert len(symbols) == 200
    assert warnings == ["capped universe from 500 to 200 tickers"]


def _register_probe_universe(monkeypatch, tmp_path, loader) -> str:
    """Register a throwaway universe backed by ``loader`` in an isolated cache."""
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path)
    name = "probe_universe"
    definition = universes.UniverseDefinition(
        name=name, market="us", benchmark="SPY", loader=loader
    )
    registry = dict(universes._UNIVERSE_REGISTRY)
    registry[name] = definition
    monkeypatch.setattr(universes, "_UNIVERSE_REGISTRY", registry)
    return name


def test_universe_cache_for_today_expires_and_refetches(monkeypatch, tmp_path) -> None:
    calls = {"n": 0}

    def loader() -> tuple[list[str], str]:
        calls["n"] += 1
        return [f"AAA{calls['n']}"], "probe"

    name = _register_probe_universe(monkeypatch, tmp_path, loader)
    today = date.today()

    first = universes.load_current_universe(name, as_of=today)
    assert calls["n"] == 1
    # Age the entry past the today-dated TTL: the snapshot is live data, so it
    # must be refetched rather than served for the rest of the process's life.
    stale_mtime = time.time() - universes._UNIVERSE_CACHE_TTL_SECONDS - 60
    os.utime(first.cached_path, (stale_mtime, stale_mtime))

    second = universes.load_current_universe(name, as_of=today)
    assert calls["n"] == 2
    assert second.symbols == ("AAA2",)


def test_universe_cache_for_a_past_date_is_pinned(monkeypatch, tmp_path) -> None:
    calls = {"n": 0}

    def loader() -> tuple[list[str], str]:
        calls["n"] += 1
        return ["AAA"], "probe"

    name = _register_probe_universe(monkeypatch, tmp_path, loader)
    as_of = date.today() - timedelta(days=30)

    with pytest.warns(UserWarning):
        first = universes.load_current_universe(name, as_of=as_of)
    ancient = time.time() - 400 * 86400
    os.utime(first.cached_path, (ancient, ancient))

    # A finished day cannot be re-derived more accurately, so age never expires
    # the entry.
    with pytest.warns(UserWarning):
        universes.load_current_universe(name, as_of=as_of)
    assert calls["n"] == 1


def test_universe_fetch_failure_serves_the_stale_cache(
    monkeypatch, tmp_path, caplog
) -> None:
    state = {"fail": False}

    def loader() -> tuple[list[str], str]:
        if state["fail"]:
            raise RuntimeError("constituents unavailable")
        return ["AAA", "BBB"], "probe"

    name = _register_probe_universe(monkeypatch, tmp_path, loader)
    today = date.today()

    first = universes.load_current_universe(name, as_of=today)
    stale_mtime = time.time() - universes._UNIVERSE_CACHE_TTL_SECONDS - 60
    os.utime(first.cached_path, (stale_mtime, stale_mtime))
    state["fail"] = True

    with caplog.at_level(logging.WARNING, logger=universes.LOG.name):
        served = universes.load_current_universe(name, as_of=today)

    assert served.symbols == ("AAA", "BBB")
    assert "Serving stale probe_universe universe cache" in caplog.text


def test_universe_fetch_failure_without_cache_still_raises(
    monkeypatch, tmp_path
) -> None:
    def loader() -> tuple[list[str], str]:
        raise RuntimeError("constituents unavailable")

    name = _register_probe_universe(monkeypatch, tmp_path, loader)

    with pytest.raises(RuntimeError, match="constituents unavailable"):
        universes.load_current_universe(name, as_of=date.today())


def test_shipped_nifty500_pit_config_yields_delisted_members() -> None:
    """The committed snapshot history must still resolve into real PIT windows.

    ``nifty500_pit`` exists to undo the survivorship bias of the built-in
    ``nifty500`` loader, so the guard that matters is that names NSE has since
    dropped are eligible on the dates they were in the index and closed out
    afterwards. A regenerated or truncated CSV that quietly collapsed to
    today's membership would still load; it would just stop being point-in-time.
    """
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "universes.yaml"

    selection = universes.load_universe_selection(
        "nifty500_pit",
        market="india",
        as_of=date(2026, 9, 1),
        config_path=config,
    )

    assert selection.benchmark == "^NSEI"
    assert all(symbol.endswith(".NS") for symbol in selection.symbols)
    boundaries = sorted({window[1] for window in selection.membership_windows})
    assert len(boundaries) >= 10
    assert boundaries[0].year <= 2019

    open_members = {w[0] for w in selection.membership_windows if w[2] is None}
    # Ever-members far exceed current members only if delisted and demoted
    # names survived the reconstruction.
    assert len(selection.symbols) > len(open_members) + 100

    # Allahabad Bank was merged away in 2020; it must be selectable in 2019 and
    # never afterwards.
    albk = [w for w in selection.membership_windows if w[0] == "ALBK.NS"]
    assert albk, "expected a delisted 2019-era constituent in the history"
    assert all(window[2] is not None for window in albk)
    assert min(window[1] for window in albk).year <= 2019
