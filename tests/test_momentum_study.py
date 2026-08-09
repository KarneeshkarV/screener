"""Unit tests for the momentum study runner and its site builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_momentum_site import (
    _benchmark_cagr,
    build,
    drawdown_profile,
    is_baseline,
)
from scripts.run_momentum_study import (
    HOLD_GRID,
    LEVER_BY_KEY,
    LEVERS,
    MARKETS,
    PERIODS,
    REGIME_BY_KEY,
    REGIME_FILTERS,
    STRATEGIES,
    _cli_defaults,
    build_request,
    run_key,
)


def test_cli_defaults_cover_every_request_field() -> None:
    # The runner builds its request from the CLI's own declared defaults, so a
    # new option on backtest-rolling must not leave the request half-built.
    defaults = _cli_defaults()
    for name in ("hold", "top", "strategy_name", "universe", "point_in_time"):
        assert name in defaults


def test_repeatable_options_default_to_empty_tuples() -> None:
    # Click reports an unset repeatable option as a sentinel, which is not
    # iterable; the runner must hand the request the empty tuple instead.
    defaults = _cli_defaults()
    for name in ("regime_filter_args", "breadth_filter_args", "fundamental_field_args"):
        assert defaults[name] == ()


def test_build_request_applies_the_study_parameters() -> None:
    strategy = next(s for s in STRATEGIES if s.name == "momentum_12_1")
    request = build_request(strategy, MARKETS["india"], 5, fetcher=None)
    assert request.mode == "rolling"
    assert request.market == "india"
    assert request.years == 5
    assert request.strategy_name == "momentum_12_1"
    assert request.hold == strategy.hold
    assert request.point_in_time is True
    assert request.universe == "nifty500_extended_pit"
    assert request.cost_model == "india"


def test_every_strategy_has_a_family_and_holding_period() -> None:
    for strategy in STRATEGIES:
        assert strategy.family in {"A", "B", "C", "D"}
        assert strategy.hold > 0
        assert strategy.paper
        assert strategy.note


def test_run_key_is_unique_per_cell() -> None:
    keys = {
        run_key(strategy, spec, years, regime)
        for strategy in STRATEGIES
        for spec in MARKETS.values()
        for years in PERIODS
        for regime in REGIME_FILTERS
    }
    expected = len(STRATEGIES) * len(MARKETS) * len(PERIODS) * len(REGIME_FILTERS)
    assert len(keys) == expected


def test_unfiltered_run_key_has_no_suffix() -> None:
    # Keys written before the regime sweep existed must keep resolving.
    strategy = STRATEGIES[0]
    assert run_key(strategy, MARKETS["us"], 5) == f"us__{strategy.name}__5y"
    assert run_key(strategy, MARKETS["us"], 5, REGIME_BY_KEY["bull"]).endswith("__bull")


def test_regime_filter_reaches_the_request() -> None:
    strategy = STRATEGIES[0]
    plain = build_request(strategy, MARKETS["us"], 5, fetcher=None)
    assert plain.regime_filter_args == ()
    assert plain.breadth_filter_args == ()

    trend = build_request(
        strategy, MARKETS["us"], 5, fetcher=None, regime=REGIME_BY_KEY["bull"]
    )
    assert trend.regime_filter_args == ("bull",)
    assert trend.breadth_filter_args == ()

    breadth = build_request(
        strategy, MARKETS["us"], 5, fetcher=None, regime=REGIME_BY_KEY["breadth"]
    )
    assert breadth.regime_filter_args == ()
    assert breadth.breadth_filter_args


def test_hold_override_reaches_the_request_and_the_key() -> None:
    strategy = next(s for s in STRATEGIES if s.name == "momentum_12_1")
    swept = build_request(strategy, MARKETS["us"], 10, fetcher=None, hold=21)
    assert swept.hold == 21
    assert run_key(strategy, MARKETS["us"], 10, hold=21).endswith("__h21")
    # A sweep value equal to the strategy's own hold is the baseline run, and
    # must not claim a second key for the identical simulation.
    assert run_key(strategy, MARKETS["us"], 10, hold=strategy.hold) == run_key(
        strategy, MARKETS["us"], 10
    )


@pytest.mark.parametrize("lever", LEVERS)
def test_every_lever_overrides_a_real_request_field(lever) -> None:
    defaults = _cli_defaults()
    for field in lever.overrides:
        assert field in defaults, f"{lever.key} overrides unknown option {field}"
    strategy = STRATEGIES[0]
    request = build_request(strategy, MARKETS["us"], 10, fetcher=None, lever=lever)
    for field, value in lever.overrides.items():
        assert getattr(request, field) == value
    assert lever.why, lever.key


def test_lever_keys_are_distinct_in_run_keys() -> None:
    strategy = STRATEGIES[0]
    keys = {run_key(strategy, MARKETS["us"], 10, lever=lever) for lever in LEVERS}
    assert len(keys) == len(LEVERS)
    assert run_key(strategy, MARKETS["us"], 10) not in keys


def test_hold_grid_covers_every_strategy_default() -> None:
    # Otherwise a strategy's baseline would sit outside its own sweep, and the
    # variant table would have nothing to compare against.
    assert {s.hold for s in STRATEGIES} <= set(HOLD_GRID)


def test_lever_choices_are_addressable_by_key() -> None:
    assert set(LEVER_BY_KEY) == {lever.key for lever in LEVERS}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"hold": 63, "default_hold": 63, "regime": "", "lever": ""}, True),
        ({"hold": 21, "default_hold": 63, "regime": "", "lever": ""}, False),
        ({"hold": 63, "default_hold": 63, "regime": "bull", "lever": ""}, False),
        ({"hold": 63, "default_hold": 63, "regime": "", "lever": "invvol"}, False),
        # Runs written before the sweep existed carry none of these fields.
        ({"hold": 63}, True),
    ],
)
def test_is_baseline(payload: dict, expected: bool) -> None:
    assert is_baseline(payload) is expected


def _curve(values: list[float], start: str = "2024-01-01") -> list[dict]:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return [
        {"date": d.date().isoformat(), "value": v}
        for d, v in zip(dates, values, strict=True)
    ]


def test_drawdown_profile_finds_the_worst_decline_and_its_recovery() -> None:
    # Up to 120, down to 60 (-50%), back above 120.
    values = [100.0, 120.0, 90.0, 60.0, 80.0, 121.0, 130.0]
    profile = drawdown_profile(_curve(values))
    assert profile["max_drawdown_daily"] == pytest.approx(-0.5)
    assert profile["peak_date"] == "2024-01-02"
    assert profile["trough_date"] == "2024-01-04"
    assert profile["recovery_date"] == "2024-01-06"
    assert profile["decline_days"] == 2
    assert profile["recovery_days"] == 2
    assert profile["recovered"] is True


def test_drawdown_profile_reports_an_unrecovered_decline() -> None:
    # Troughs at 110 on day 4, climbs to 115 but never regains the 150 peak.
    profile = drawdown_profile(_curve([100.0, 150.0, 120.0, 110.0, 115.0]))
    assert profile["recovered"] is False
    assert profile["recovery_date"] is None
    assert profile["trough_date"] == "2024-01-04"
    # Days elapsed since the trough, not a completed recovery.
    assert profile["recovery_days"] == 1


def test_month_end_drawdown_is_shallower_than_the_daily_one() -> None:
    # A crash and full recovery inside one month is invisible to month-end marks.
    values = [100.0] * 10 + [50.0] + [100.0] * 50
    profile = drawdown_profile(_curve(values))
    assert profile["max_drawdown_daily"] == pytest.approx(-0.5)
    assert profile["max_drawdown_monthly"] == pytest.approx(0.0)


def test_drawdown_profile_handles_an_empty_curve() -> None:
    profile = drawdown_profile([])
    assert profile["max_drawdown_daily"] is None
    assert profile["recovered"] is None


@pytest.mark.parametrize(
    ("total_return", "years", "expected"),
    [
        (0.0, 5, 0.0),
        (1.0, 1, 1.0),
        (3.0, 2, 1.0),  # 4x over two years is 100% a year
        (None, 5, None),
        (-1.0, 5, -1.0),  # a total wipeout stays -100%, it does not go complex
    ],
)
def test_benchmark_cagr(
    total_return: float | None, years: int, expected: float | None
) -> None:
    result = _benchmark_cagr(total_return, years)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def _fake_run(strategy: str, market: str, years: int) -> dict:
    return {
        "strategy": strategy,
        "family": "A",
        "label": "12-1 cross-sectional",
        "paper": "Jegadeesh & Titman (1993)",
        "note": "note",
        "market": market,
        "market_label": "India - Nifty 500 (point-in-time)",
        "benchmark": "^NSEI",
        "years": years,
        "start": "2021-01-01",
        "end": "2026-01-01",
        "top": 20,
        "hold": 63,
        "cost_model": "india",
        "slippage_bps": 10.0,
        "commission_bps": 0.0,
        "universe_note": None,
        "elapsed_seconds": 1.0,
        "metrics": {"sharpe": 1.0, "cagr": 0.2, "benchmark_return": 1.0},
        "equity_curve": [{"date": "2021-01-01", "value": 100.0}],
        "benchmark_curve": [{"date": "2021-01-01", "value": 50.0}],
        "trades": [],
        "warnings": [],
        "generated": "2026-01-01",
    }


def test_build_writes_an_index_and_copies_the_page(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "india__momentum_12_1__5y.json").write_text(
        json.dumps(_fake_run("momentum_12_1", "india", 5)), encoding="utf-8"
    )

    assert build(tmp_path) == 0

    site = tmp_path / "site"
    assert (site / "index.html").exists()
    index = json.loads((site / "data" / "index.json").read_text(encoding="utf-8"))
    assert index["periods"] == [5]
    assert [m["id"] for m in index["markets"]] == ["india"]
    assert index["families"].keys() == {"A", "B", "C", "D"}
    assert [r["id"] for r in index["regimes"]] == [""]
    (entry,) = index["runs"]
    assert entry["key"] == "india__momentum_12_1__5y"
    assert entry["order"] == 0
    # 100% total return over five years annualizes to about 14.9%.
    assert entry["metrics"]["benchmark_cagr"] == pytest.approx(0.1487, abs=1e-4)
    # The per-run payload is republished with the derived metric attached.
    payload = json.loads(
        (site / "data" / "runs" / "india__momentum_12_1__5y.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["metrics"]["benchmark_cagr"] == pytest.approx(0.1487, abs=1e-4)


def test_build_reports_an_empty_run_directory(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    assert build(tmp_path) == 1


def _serve(tmp_path: Path):
    """Start the site server on an ephemeral port and yield its base URL."""
    import threading
    from functools import partial

    from scripts.serve_momentum_site import _GzipHandler, _ReusableServer

    handler = partial(_GzipHandler, directory=str(tmp_path))
    server = _ReusableServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(url: str, *, gzip_ok: bool) -> tuple[bytes, str | None]:
    import urllib.request

    request = urllib.request.Request(url)
    if gzip_ok:
        request.add_header("Accept-Encoding", "gzip")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read(), response.headers.get("Content-Encoding")


def test_the_server_gzips_json_for_clients_that_accept_it(tmp_path: Path) -> None:
    # The study index grows a row per run, so by the end of a sweep it is
    # several megabytes that every page load pulls before anything renders.
    payload = json.dumps([{"key": f"run-{i}", "sharpe": 1.0} for i in range(5000)])
    (tmp_path / "index.json").write_text(payload)
    server, base = _serve(tmp_path)
    try:
        body, encoding = _get(f"{base}/index.json", gzip_ok=True)
    finally:
        server.shutdown()
        server.server_close()
    import gzip as gzip_module

    assert encoding == "gzip"
    assert gzip_module.decompress(body).decode() == payload


def test_a_client_without_gzip_still_gets_the_plain_body(tmp_path: Path) -> None:
    payload = json.dumps([{"key": f"run-{i}"} for i in range(5000)])
    (tmp_path / "index.json").write_text(payload)
    server, base = _serve(tmp_path)
    try:
        body, encoding = _get(f"{base}/index.json", gzip_ok=False)
    finally:
        server.shutdown()
        server.server_close()
    assert encoding is None
    assert body.decode() == payload


def test_a_small_file_is_not_worth_compressing(tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text('{"ok": true}')
    server, base = _serve(tmp_path)
    try:
        body, encoding = _get(f"{base}/index.json", gzip_ok=True)
    finally:
        server.shutdown()
        server.server_close()
    assert encoding is None
    assert body.decode() == '{"ok": true}'
