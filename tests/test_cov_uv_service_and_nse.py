"""Offline line-coverage tests for the unusual-volume service stack.

Covers ``service``, ``cli``, ``nse_client``, ``delivery``, ``fii_dii`` and
``option_chain``. Everything is stubbed/monkeypatched — no network, no disk
caches beyond ``tmp_path``, fully deterministic.
"""

from __future__ import annotations


import io


import sys


import types


from datetime import date


import pandas as pd


import pytest


from click.testing import CliRunner


from rich.console import Console


from screener import cache


from screener.unusual_volume import (
    cli as uv_cli,
    fii_dii,
    nse_client,
    option_chain,
    service,
)


from screener.unusual_volume.buildup import BuildupScore


from screener.unusual_volume.detector import Event


def _console() -> Console:
    return Console(file=io.StringIO())


def _event(symbol: str = "RELIANCE", d: date = date(2026, 5, 15), **over) -> Event:
    base = dict(
        symbol=symbol,
        date=d,
        close=2500.0,
        pct_change=1.0,
        volume=150_000.0,
        avg_volume_20d=50_000.0,
        rvol=3.0,
        rvol_5d=3.0,
        rvol_50d=3.0,
        rvol_90d=3.0,
        z_score=2.5,
        pct_rank_252d=0.9,
        direction="BUYING",
        strength="HIGH",
    )
    base.update(over)
    return Event(**base)


def _bars(n: int = 30, as_of: date = date(2026, 5, 15)) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(as_of), periods=n)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [100_000.0] * n,
        },
        index=idx,
    )


class _Fetcher:
    def __init__(self, frames=None, exc=None):
        self._frames = frames or {}
        self._exc = exc

    def fetch(self, syms, start, end):
        if self._exc is not None:
            raise self._exc
        return self._frames


def _score(sym="AAA"):
    return BuildupScore(
        symbol=sym,
        as_of=date(2026, 5, 15),
        window=20,
        range_compression=0.7,
        updown_volume=0.6,
        higher_lows=0.6,
        sustained_delivery=None,
        close_near_high=0.7,
        composite=0.65,
        flags=["compression"],
    )


def _req(**over):
    base = dict(
        market="india",
        as_of=date(2026, 5, 15),
        universe=["NSE:RELIANCE"],
        min_rvol=0.0,
        min_z=0.0,
        strength_floor="MODERATE",
        min_avg_volume=0.0,
        min_market_cap=0.0,
        include_fno_ban=True,
        deep_india=False,
        buildup_enabled=False,
        buildup_window=20,
        buildup_min_score=0.0,
        option_chain=False,
        fii_dii=False,
        pledge=False,
    )
    base.update(over)
    return service.UnusualVolumeRequest(**base)


def _patch_run(monkeypatch, result):
    monkeypatch.setattr(uv_cli, "_resolve_universe", lambda m, t, f: ["AAA"])
    monkeypatch.setattr(uv_cli, "run_unusual_volume_scan", lambda req, console: result)


def _result(events, fetched=1, liquid=1):
    return service.UnusualVolumeResult(
        events=events, fetched_count=fetched, liquid_count=liquid
    )


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


def _reset_tls():
    for name in ("session", "primed", "primed_pages"):
        if hasattr(nse_client._tls, name):
            delattr(nse_client._tls, name)


def test_fetch_bars_maps_yf_back_to_tv(monkeypatch):
    bars = _bars()
    monkeypatch.setattr(service, "tv_to_yf", lambda t, m: t + ".NS")
    monkeypatch.setattr(
        service,
        "build_price_fetcher",
        lambda refresh=False: _Fetcher({"RELIANCE.NS": bars}),
    )
    out = service.fetch_bars(["RELIANCE"], "india", date(2026, 5, 15), _console())
    assert "RELIANCE" in out
    assert not out["RELIANCE"].empty


def test_fetch_bars_swallows_fetch_exception(monkeypatch):
    monkeypatch.setattr(service, "tv_to_yf", lambda t, m: t)
    monkeypatch.setattr(
        service,
        "build_price_fetcher",
        lambda refresh=False: _Fetcher(exc=ValueError("boom")),
    )
    out = service.fetch_bars(["AAA"], "us", date(2026, 5, 15), _console())
    assert out == {}


def test_fetch_bars_drops_empty_and_missing(monkeypatch):
    monkeypatch.setattr(service, "tv_to_yf", lambda t, m: t)
    frames = {"AAA": _bars(), "BBB": pd.DataFrame()}
    monkeypatch.setattr(
        service, "build_price_fetcher", lambda refresh=False: _Fetcher(frames)
    )
    out = service.fetch_bars(["AAA", "BBB"], "us", date(2026, 5, 15), _console())
    assert set(out) == {"AAA"}


def test_india_symbol_variants():
    assert service.india_symbol("NSE:reliance") == "RELIANCE"
    assert service.india_symbol("tcs") == "TCS"


def test_bars_on_or_before_as_of_paths():
    assert service.bars_on_or_before_as_of(None, date(2026, 5, 15)).empty
    assert service.bars_on_or_before_as_of(pd.DataFrame(), date(2026, 5, 15)).empty
    # frame with a date column instead of DatetimeIndex
    df = pd.DataFrame(
        {
            "date": ["2026-05-13", "2026-05-14"],
            "close": [1.0, 2.0],
            "volume": [1.0, 2.0],
        }
    )
    out = service.bars_on_or_before_as_of(df, date(2026, 5, 13))
    assert len(out) == 1
    # frame without DatetimeIndex and no date column → empty
    no_date = pd.DataFrame({"close": [1.0]})
    assert service.bars_on_or_before_as_of(no_date, date(2026, 5, 15)).empty


def test_standalone_buildup_event_with_and_without_flags():
    bars = _bars()
    ev = service.standalone_buildup_event(_score(), bars, date(2026, 5, 15))
    assert ev is not None and ev.direction == "BUILDUP"
    assert "multi-week build-up: compression" in ev.notes
    # empty bars → None
    assert (
        service.standalone_buildup_event(_score(), pd.DataFrame(), date(2026, 5, 15))
        is None
    )
    # no flags → generic note; single-bar prev_close fallback and prev_close<=0
    flat = BuildupScore(
        symbol="BBB",
        as_of=date(2026, 5, 15),
        window=20,
        range_compression=0.0,
        updown_volume=0.0,
        higher_lows=0.0,
        sustained_delivery=None,
        close_near_high=0.0,
        composite=0.0,
        flags=[],
    )
    one = pd.DataFrame(
        {"close": [0.0], "volume": [5.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-15")]),
    )
    ev2 = service.standalone_buildup_event(flat, one, date(2026, 5, 15))
    assert ev2 is not None and ev2.notes == "multi-week build-up"
    assert ev2.pct_change == 0.0  # prev_close == close (single bar) → 0


def test_human_mcap_tiers():
    assert service._human_mcap(2e9) == "$2.0B"
    assert service._human_mcap(5e6) == "$5M"
    assert service._human_mcap(1234.0) == "$1,234"


def test_run_scan_no_bars_returns_empty(monkeypatch):
    monkeypatch.setattr(service, "fetch_bars", lambda *a, **k: {})
    res = service.run_unusual_volume_scan(_req(), _console())
    assert res.events == [] and res.fetched_count == 0 and res.liquid_count == 0


def test_run_scan_fno_ban_filters(monkeypatch):
    bars = {"NSE:RELIANCE": _bars(), "NSE:SAIL": _bars()}
    monkeypatch.setattr(service, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(service, "fetch_fno_ban_list", lambda: {"SAIL"})
    monkeypatch.setattr(service, "passes_volume_floor", lambda *a, **k: True)
    monkeypatch.setattr(service, "detect_market", lambda *a, **k: [])
    monkeypatch.setattr(
        service, "_overlay_india_delivery", lambda *a, **k: pd.DataFrame()
    )
    monkeypatch.setattr(service, "fetch_sector_map", lambda *a, **k: {})
    res = service.run_unusual_volume_scan(
        _req(include_fno_ban=False, universe=["NSE:RELIANCE", "NSE:SAIL"]), _console()
    )
    # both dropped from bars by ban, but no events anyway
    assert res.liquid_count == 1  # only RELIANCE survives ban


def test_run_scan_no_liquid_returns_empty(monkeypatch):
    monkeypatch.setattr(
        service, "fetch_bars", lambda *a, **k: {"NSE:RELIANCE": _bars()}
    )
    monkeypatch.setattr(service, "passes_volume_floor", lambda *a, **k: False)
    res = service.run_unusual_volume_scan(_req(), _console())
    assert res.events == [] and res.liquid_count == 0 and res.fetched_count == 1


def test_run_scan_full_path_with_mcap_and_deep(monkeypatch):
    bars = {"NSE:RELIANCE": _bars()}
    monkeypatch.setattr(service, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(service, "passes_volume_floor", lambda *a, **k: True)
    ev = _event("RELIANCE", date(2026, 5, 15), market_cap=1e10)
    monkeypatch.setattr(service, "detect_market", lambda *a, **k: [ev])
    monkeypatch.setattr(
        service, "_overlay_india_delivery", lambda *a, **k: pd.DataFrame()
    )
    monkeypatch.setattr(service, "_overlay_india_microstructure", lambda *a, **k: None)
    monkeypatch.setattr(
        service, "fetch_sector_map", lambda *a, **k: {"RELIANCE": "Energy"}
    )
    captured = {}
    monkeypatch.setattr(
        service, "attach_sector", lambda evs, m: captured.setdefault("sec", True)
    )
    monkeypatch.setattr(service, "passes_market_cap", lambda mc, floor: True)
    monkeypatch.setattr(
        service, "deep_enrich_india", lambda evs: captured.setdefault("deep", True)
    )
    res = service.run_unusual_volume_scan(
        _req(min_market_cap=1e9, deep_india=True), _console()
    )
    assert len(res.events) == 1
    assert captured.get("sec") and captured.get("deep")


def test_run_scan_mcap_default_and_buildup(monkeypatch):
    bars = {"NSE:RELIANCE": _bars()}
    monkeypatch.setattr(service, "fetch_bars", lambda *a, **k: bars)
    monkeypatch.setattr(service, "passes_volume_floor", lambda *a, **k: True)
    ev = _event("RELIANCE", date(2026, 5, 15), market_cap=1.0)
    monkeypatch.setattr(service, "detect_market", lambda *a, **k: [ev])
    monkeypatch.setattr(
        service, "_overlay_india_delivery", lambda *a, **k: pd.DataFrame()
    )
    monkeypatch.setattr(service, "_overlay_india_microstructure", lambda *a, **k: None)
    monkeypatch.setattr(service, "fetch_sector_map", lambda *a, **k: {})
    monkeypatch.setattr(service, "_apply_buildup_overlay", lambda *a, **k: None)
    # default mcap floor for india (5e9) drops the small-cap event
    monkeypatch.setattr(service, "passes_market_cap", lambda mc, floor: False)
    res = service.run_unusual_volume_scan(
        _req(min_market_cap=None, buildup_enabled=True), _console()
    )
    assert res.events == []


def test_overlay_india_delivery_non_india_returns_empty():
    out = service._overlay_india_delivery(
        _req(market="us"), {"AAA": _bars()}, [], _console()
    )
    assert out.empty


def test_overlay_india_delivery_load_failure(monkeypatch):
    monkeypatch.setattr(
        service,
        "load_delivery_panel",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    ev = _event("NSE:RELIANCE")
    out = service._overlay_india_delivery(
        _req(), {"NSE:RELIANCE": _bars()}, [ev], _console()
    )
    assert out.empty
    assert ev.symbol == "RELIANCE"  # normalized before the failure


def test_overlay_india_delivery_empty_panel(monkeypatch):
    monkeypatch.setattr(service, "load_delivery_panel", lambda *a, **k: pd.DataFrame())
    out = service._overlay_india_delivery(
        _req(), {"NSE:RELIANCE": _bars()}, [_event("RELIANCE")], _console()
    )
    assert out.empty


def test_overlay_india_delivery_with_quiet(monkeypatch):
    panel = pd.DataFrame(
        [
            {
                "SYMBOL": "RELIANCE",
                "date": date(2026, 5, 15),
                "TTL_TRD_QNTY": 1.0,
                "DELIV_QTY": 1.0,
                "DELIV_PER": 50.0,
            }
        ]
    )
    monkeypatch.setattr(service, "load_delivery_panel", lambda *a, **k: panel)
    monkeypatch.setattr(service, "overlay_events", lambda evs, p: evs)
    quiet_ev = _event("QUIETSYM")
    monkeypatch.setattr(
        service, "quiet_accumulation_events", lambda *a, **k: [quiet_ev]
    )
    events = [_event("RELIANCE")]
    out = service._overlay_india_delivery(
        _req(), {"NSE:RELIANCE": _bars()}, events, _console()
    )
    assert not out.empty
    assert quiet_ev in events


def test_microstructure_noop_when_no_overlays():
    # no events
    service._overlay_india_microstructure(_req(option_chain=True), [], _console())
    # events but all overlay flags false
    service._overlay_india_microstructure(_req(), [_event()], _console())


def test_microstructure_historical_uses_copies(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    monkeypatch.setattr(service, "_live_nse_snapshot_date", lambda: date(2026, 5, 20))
    calls = {}

    def overlay_oc(events, refresh=False):
        for ev in events:
            ev.pcr = 9.9
        return {
            "RELIANCE": {
                "ce_oi": 1.0,
                "pe_oi": 2.0,
                "call_put_oi_ratio": 0.5,
                "pcr": 2.0,
            }
        }

    monkeypatch.setattr(option_chain, "overlay_option_chain", overlay_oc)

    def overlay_fd(events, snap, refresh=False):
        calls["fd"] = True
        return {"fii_5d_net": 1.0, "dii_5d_net": 2.0, "fii_trend": 1.1}

    monkeypatch.setattr(fii_dii, "overlay_fii_dii", overlay_fd)
    ev = _event("RELIANCE", date(2026, 5, 1))
    service._overlay_india_microstructure(
        _req(as_of=date(2026, 5, 1), option_chain=True, fii_dii=True, pledge=True),
        [ev],
        _console(),
    )
    # historical: real event untouched, copy mutated; pledge skipped (returns)
    assert ev.pcr is None
    assert calls.get("fd")
    snap = cache.read_frame(cache.panel_path("option_chain"))
    assert snap is not None


def test_microstructure_live_attaches_and_pledge(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    monkeypatch.setattr(service, "_live_nse_snapshot_date", lambda: date(2026, 5, 15))
    monkeypatch.setattr(
        option_chain, "overlay_option_chain", lambda events, refresh=False: {}
    )
    monkeypatch.setattr(
        fii_dii, "overlay_fii_dii", lambda events, snap, refresh=False: None
    )
    pledge_mod = types.SimpleNamespace(overlay_pledge=lambda evs, refresh=False: evs)
    monkeypatch.setitem(sys.modules, "screener.pledge", pledge_mod)
    ev = _event("RELIANCE", date(2026, 5, 15))
    service._overlay_india_microstructure(
        _req(option_chain=True, fii_dii=True, pledge=True), [ev], _console()
    )


def test_microstructure_overlay_exceptions(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "PANEL_ROOT", tmp_path)
    monkeypatch.setattr(service, "_live_nse_snapshot_date", lambda: date(2026, 5, 15))

    def boom(*a, **k):
        raise RuntimeError("nse down")

    monkeypatch.setattr(option_chain, "overlay_option_chain", boom)
    monkeypatch.setattr(fii_dii, "overlay_fii_dii", boom)
    pledge_mod = types.SimpleNamespace(overlay_pledge=boom)
    monkeypatch.setitem(sys.modules, "screener.pledge", pledge_mod)
    console = _console()
    service._overlay_india_microstructure(
        _req(option_chain=True, fii_dii=True, pledge=True), [_event()], console
    )
    out = console.file.getvalue()
    assert "Option-chain overlay failed" in out
    assert "FII/DII overlay failed" in out
    assert "Pledge overlay failed" in out


def test_apply_buildup_overlay_annotate_and_standalone(monkeypatch):
    bars = _bars()
    liquid = {"NSE:RELIANCE": bars, "NSE:NEWCO": bars}
    panel = pd.DataFrame(
        [
            {
                "SYMBOL": "RELIANCE",
                "date": date(2026, 5, 15),
                "DELIV_QTY": 1.0,
                "DELIV_PER": 50.0,
                "TTL_TRD_QNTY": 1.0,
            }
        ]
    )
    monkeypatch.setattr(
        service,
        "compute_buildup_score",
        lambda sym, b, as_of, delivery_panel=None, window=20: (
            _score(sym) if sym == "RELIANCE" else None
        ),
    )
    monkeypatch.setattr(
        service,
        "scan_buildups",
        lambda *a, **k: [_score("RELIANCE"), _score("NEWCO"), _score("GHOST")],
    )
    events = [_event("RELIANCE", date(2026, 5, 15))]
    service._apply_buildup_overlay(_req(), liquid, panel, events, _console())
    # RELIANCE annotated; NEWCO added standalone; GHOST has no bars → skipped
    syms = {e.symbol for e in events}
    assert "NEWCO" in syms and "GHOST" not in syms
    assert events[0].buildup_flags == ["compression"]


def test_apply_buildup_overlay_us_and_empty_bars(monkeypatch):
    bars = _bars()
    liquid = {"AAA": bars, "EMPTYCO": pd.DataFrame()}
    monkeypatch.setattr(service, "compute_buildup_score", lambda *a, **k: None)
    monkeypatch.setattr(
        service,
        "scan_buildups",
        lambda *a, **k: [_score("EMPTYCO"), _score("MISSING")],
    )
    events = [_event("AAA")]
    service._apply_buildup_overlay(
        _req(market="us"), liquid, pd.DataFrame(), events, _console()
    )
    # EMPTYCO bars empty → skipped; MISSING not in bars → skipped
    assert [e.symbol for e in events] == ["AAA"]


def test_apply_buildup_standalone_none(monkeypatch):
    liquid = {"AAA": _bars()}
    monkeypatch.setattr(service, "compute_buildup_score", lambda *a, **k: None)
    monkeypatch.setattr(service, "scan_buildups", lambda *a, **k: [_score("AAA")])
    monkeypatch.setattr(service, "standalone_buildup_event", lambda *a, **k: None)
    events = []
    service._apply_buildup_overlay(
        _req(market="us"), liquid, pd.DataFrame(), events, _console()
    )
    assert events == []


def test_live_snapshot_date_uses_operator(monkeypatch):
    fake = types.SimpleNamespace(latest_trading_day=lambda today: date(2026, 5, 14))
    monkeypatch.setitem(sys.modules, "screener.operator.fetch", fake)
    assert service._live_nse_snapshot_date() == date(2026, 5, 14)


def test_live_snapshot_date_falls_back(monkeypatch):
    def boom(today):
        raise RuntimeError("x")

    fake = types.SimpleNamespace(latest_trading_day=boom)
    monkeypatch.setitem(sys.modules, "screener.operator.fetch", fake)
    assert service._live_nse_snapshot_date() == date.today()


def test_request_rejects_empty_market():
    with pytest.raises(ValueError):
        _req(market="  ")


def test_request_rejects_empty_universe():
    with pytest.raises(ValueError):
        _req(universe=["  ", ""])


def test_resolve_universe_tickers():
    assert uv_cli._resolve_universe("us", "AAA, BBB ,", None) == ["AAA", "BBB"]


def test_resolve_universe_file(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("NSE:AAA\n\nNSE:BBB\n")
    assert uv_cli._resolve_universe("india", None, str(f)) == ["NSE:AAA", "NSE:BBB"]


def test_resolve_universe_file_missing():
    import click

    with pytest.raises(click.UsageError):
        uv_cli._resolve_universe("us", None, "/no/such/file.txt")


def test_resolve_universe_loader(monkeypatch):
    fake = types.SimpleNamespace(load_universe=lambda m: ["X:Y"])
    monkeypatch.setitem(sys.modules, "screener.backtester.pine_runner", fake)
    assert uv_cli._resolve_universe("us", None, None) == ["X:Y"]


def test_cli_command_no_data_aborts(monkeypatch):
    _patch_run(monkeypatch, _result([], fetched=0, liquid=0))
    res = CliRunner().invoke(
        uv_cli.unusual_volume, ["--tickers", "AAA", "--no-output-files"]
    )
    assert res.exit_code == 1
    assert "No OHLCV data fetched" in res.output


def test_cli_command_no_liquid(monkeypatch):
    _patch_run(monkeypatch, _result([], fetched=5, liquid=0))
    res = CliRunner().invoke(uv_cli.unusual_volume, ["--tickers", "AAA"])
    assert res.exit_code == 0
    assert "No tickers passed the volume floor" in res.output


def test_cli_command_no_events(monkeypatch):
    _patch_run(monkeypatch, _result([], fetched=5, liquid=3))
    res = CliRunner().invoke(
        uv_cli.unusual_volume, ["--tickers", "AAA", "--as-of", "2026-05-15"]
    )
    assert res.exit_code == 0
    assert "No unusual-volume events" in res.output


def test_cli_command_renders_and_writes(monkeypatch, tmp_path):
    ev = _event("AAA", date(2026, 5, 15))
    _patch_run(monkeypatch, _result([ev]))
    json_p = tmp_path / "o.json"
    md_p = tmp_path / "o.md"
    res = CliRunner().invoke(
        uv_cli.unusual_volume,
        [
            "--tickers",
            "AAA",
            "--as-of",
            "2026-05-15",
            "--json",
            str(json_p),
            "--md",
            str(md_p),
        ],
    )
    assert res.exit_code == 0, res.output
    assert json_p.exists() and md_p.exists()
    assert "Wrote" in res.output


def test_cli_command_no_output_files(monkeypatch):
    ev = _event("AAA", date(2026, 5, 15))
    _patch_run(monkeypatch, _result([ev]))
    res = CliRunner().invoke(
        uv_cli.unusual_volume,
        ["--tickers", "AAA", "--as-of", "2026-05-15", "--no-output-files"],
    )
    assert res.exit_code == 0, res.output
    assert "Wrote" not in res.output


def test_cli_command_empty_universe(monkeypatch):
    monkeypatch.setattr(uv_cli, "_resolve_universe", lambda m, t, f: [])
    res = CliRunner().invoke(uv_cli.unusual_volume, ["--tickers", "AAA"])
    assert res.exit_code != 0
    assert "Empty universe" in res.output


def test_cli_command_default_today(monkeypatch):
    # exercises the resolve_as_of(None) -> date.today() branch (no --as-of)
    captured = {}

    def fake_run(request, **kwargs):
        captured["request"] = request
        return True

    monkeypatch.setattr(uv_cli, "_resolve_universe", lambda m, t, f: ["AAA"])
    monkeypatch.setattr(uv_cli, "run_unusual_volume", fake_run)
    res = CliRunner().invoke(uv_cli.unusual_volume, ["--tickers", "AAA"])
    assert res.exit_code == 0
    assert captured["request"].as_of == date.today()


def test_new_session_uses_jugaad(monkeypatch):
    class _S:
        def __init__(self):
            self.headers = {}

    class _Arch:
        def __init__(self):
            self.s = _S()

    fake = types.SimpleNamespace(NSEArchives=_Arch)
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", fake)
    sess = nse_client._new_session()
    assert "User-Agent" in sess.headers


def test_prime_page_success_then_cached(monkeypatch):
    _reset_tls()

    class _S:
        headers: dict = {}

        def __init__(self):
            self.hits = 0

        def get(self, url, timeout=10):
            self.hits += 1
            return _Resp(200)

    s = _S()
    nse_client._prime_page(s, "https://x/page")
    nse_client._prime_page(s, "https://x/page")  # cached → no second hit
    assert s.hits == 1


def test_prime_page_failure_not_cached(monkeypatch):
    _reset_tls()

    class _S:
        headers: dict = {}

        def get(self, url, timeout=10):
            raise RuntimeError("boom")

    # should swallow the exception (no raise)
    nse_client._prime_page(_S(), "https://x/page")


def test_fetch_nse_text_success(monkeypatch):
    _reset_tls()
    sess = types.SimpleNamespace(
        headers={}, get=lambda url, timeout=8.0: _Resp(200, text="hello")
    )
    monkeypatch.setattr(nse_client, "_new_session", lambda: sess)
    monkeypatch.setattr(
        nse_client, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    assert nse_client.fetch_nse_text("https://x", "op") == "hello"


def test_fetch_nse_text_non_200_is_none(monkeypatch):
    _reset_tls()
    sess = types.SimpleNamespace(
        headers={}, get=lambda url, timeout=8.0: _Resp(500, text="err")
    )
    monkeypatch.setattr(nse_client, "_new_session", lambda: sess)
    monkeypatch.setattr(
        nse_client, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    assert nse_client.fetch_nse_text("https://x", "op") is None


def test_fetch_nse_text_soft_block_then_reprime(monkeypatch):
    _reset_tls()

    class _S:
        def __init__(self, name):
            self.name = name
            self.headers = {}

        def get(self, url, timeout=8.0):
            if url.endswith("/"):
                return _Resp(200)
            return _Resp(403) if self.name == "old" else _Resp(200, text="ok")

    sessions = iter([_S("old"), _S("new")])
    monkeypatch.setattr(nse_client, "_new_session", lambda: next(sessions))
    monkeypatch.setattr(
        nse_client, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    assert nse_client.fetch_nse_text("https://x/api", "op") == "ok"


def test_fetch_nse_text_soft_block_survives_reprime(monkeypatch):
    _reset_tls()

    class _S:
        headers: dict = {}

        def get(self, url, timeout=8.0):
            if url.endswith("/"):
                return _Resp(200)
            return _Resp(403)

    monkeypatch.setattr(nse_client, "_new_session", lambda: _S())
    monkeypatch.setattr(
        nse_client, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )
    assert nse_client.fetch_nse_text("https://x/api", "op") is None
