from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

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
