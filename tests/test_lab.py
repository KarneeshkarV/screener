"""Backtest lab comparison tests."""

from __future__ import annotations

import http.client
import json
import threading
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path

from screener.backtester import lab
from screener.backtester.lab import LabHandler
from screener.universes import Universe

from tests.conftest import StubPriceFetcher, make_bars


# ---------------------------------------------------------------------------
# Handler-level tests (real ThreadingHTTPServer on ephemeral port 0)
# ---------------------------------------------------------------------------


def _start_server() -> tuple[ThreadingHTTPServer, int]:
    """Start a LabHandler server on an ephemeral port and return (server, port)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), LabHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def test_handler_get_strategies_same_origin_returns_200() -> None:
    """GET /api/strategies with valid Host header returns 200."""
    server, port = _start_server()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/api/strategies", headers={"Host": f"127.0.0.1:{port}"})
        resp = conn.getresponse()
        assert resp.status == 200
        body = json.loads(resp.read())
        assert "strategies" in body
    finally:
        server.shutdown()
        server.server_close()


def test_handler_post_cross_origin_returns_403() -> None:
    """POST /api/run with an evil Origin is rejected with 403."""
    server, port = _start_server()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        payload = b"{}"
        conn.request(
            "POST",
            "/api/run",
            body=payload,
            headers={
                "Host": f"127.0.0.1:{port}",
                "Origin": "http://evil.example",
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        resp = conn.getresponse()
        body_text = resp.read()
        assert resp.status == 403
        # Must not contain backtest data fields
        assert b"results" not in body_text
        assert b"metrics" not in body_text
    finally:
        server.shutdown()
        server.server_close()


def test_handler_bad_host_returns_403() -> None:
    """Request with Host: evil.example is rejected with 403."""
    server, port = _start_server()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/api/strategies", headers={"Host": "evil.example"})
        resp = conn.getresponse()
        assert resp.status == 403
    finally:
        server.shutdown()
        server.server_close()


def test_handler_same_origin_invalid_payload_returns_400() -> None:
    """POST /api/run same-origin but invalid payload returns 400 (guard passed through)."""
    server, port = _start_server()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        # Intentionally omit required "start" key so existing validation raises
        payload = json.dumps({"market": "us", "strategies": ["ema_trend"]}).encode()
        conn.request(
            "POST",
            "/api/run",
            body=payload,
            headers={
                "Host": f"127.0.0.1:{port}",
                "Origin": f"http://127.0.0.1:{port}",
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
        )
        resp = conn.getresponse()
        assert resp.status == 400
        # No backtest data was returned
        body = json.loads(resp.read())
        assert "error" in body
    finally:
        server.shutdown()
        server.server_close()


def test_compare_payload_runs_multiple_named_strategies(monkeypatch):
    bars_a = make_bars(n=80, seed=21, open_base=100.0)
    bars_b = make_bars(n=80, seed=22, open_base=50.0)
    spy = make_bars(n=80, seed=23, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})
    monkeypatch.setattr(lab, "build_price_fetcher", lambda auto_adjust=True: fetcher)

    payload = lab.compare_payload(
        market="us",
        strategies=["ema_trend", "breakout"],
        tickers=("AAA", "BBB"),
        start_date=bars_a.index[20].date(),
        end_date=bars_a.index[60].date(),
        hold=5,
        top=2,
        initial_capital=100_000,
    )

    assert [item["strategy"] for item in payload["results"]] == [
        "ema_trend · tickers",
        "breakout · tickers",
    ]
    assert payload["request"]["tickers"] == ("AAA", "BBB")
    assert all("metrics" in item for item in payload["results"])
    assert all("curves" in item for item in payload["results"])


def test_compare_payload_requires_strategy_and_ticker():
    try:
        lab.compare_payload(
            market="us",
            strategies=[],
            tickers=("AAA",),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),
            hold=5,
            top=1,
            initial_capital=100_000,
        )
    except ValueError as exc:
        assert "strategy" in str(exc)
    else:
        raise AssertionError("expected missing strategy error")


def test_compare_payload_can_load_named_universe(monkeypatch):
    bars_a = make_bars(n=80, seed=31, open_base=100.0)
    bars_b = make_bars(n=80, seed=32, open_base=50.0)
    spy = make_bars(n=80, seed=33, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})
    monkeypatch.setattr(lab, "build_price_fetcher", lambda auto_adjust=True: fetcher)
    monkeypatch.setattr(
        lab,
        "load_current_universe",
        lambda name, as_of, use_cache=True: Universe(
            name=name,
            symbols=("AAA", "BBB"),
            source="test",
            cached_path=Path("/tmp/test-universe.txt"),
        ),
    )

    payload = lab.compare_payload(
        market="us",
        strategies=["ema_trend"],
        tickers=(),
        start_date=bars_a.index[20].date(),
        end_date=bars_a.index[60].date(),
        hold=5,
        top=2,
        initial_capital=100_000,
        universe="sp500",
    )

    assert payload["request"]["tickers"] == ("AAA", "BBB")
    assert payload["request"]["universe"] == "sp500"
    assert payload["request"]["universe_note"]["symbol_count"] == 2


def test_compare_payload_can_compare_tickers_against_universe(monkeypatch):
    bars_a = make_bars(n=80, seed=41, open_base=100.0)
    bars_b = make_bars(n=80, seed=42, open_base=50.0)
    spy = make_bars(n=80, seed=43, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "BBB": bars_b, "SPY": spy})
    monkeypatch.setattr(lab, "build_price_fetcher", lambda auto_adjust=True: fetcher)
    monkeypatch.setattr(
        lab,
        "load_current_universe",
        lambda name, as_of, use_cache=True: Universe(
            name=name,
            symbols=("AAA", "BBB"),
            source="test",
            cached_path=Path("/tmp/test-universe.txt"),
        ),
    )

    payload = lab.compare_payload(
        market="us",
        strategies=["ema_trend"],
        tickers=("AAA",),
        start_date=bars_a.index[20].date(),
        end_date=bars_a.index[60].date(),
        hold=5,
        top=2,
        initial_capital=100_000,
        compare_universe="sp500",
    )

    assert [item["strategy"] for item in payload["results"]] == [
        "ema_trend · tickers",
        "ema_trend · sp500",
    ]
    assert payload["request"]["compare_universe"] == "sp500"
    assert payload["request"]["compare_universe_note"]["symbol_count"] == 2
    assert all("trades" in item for item in payload["results"])
