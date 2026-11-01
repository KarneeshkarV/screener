"""Regression tests for screen field, rank, identity, and policy correctness."""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener import history, scanner
from screener.api import screen
from screener.backtester.sessions import drop_incomplete_sessions
from screener.commands.screen import screen as screen_command
from screener.screen_candidates import resolve_screen_strategy, resolve_universe_field
from screener.screen_workflow import ScreenRequest, run_screen_workflow
from screener.strategies.spec import PrepareCtx, registry
from screener.symbols import tv_to_yf

AS_OF = date(2025, 6, 20)


class FrozenDate(date):
    @classmethod
    def today(cls):
        return AS_OF


def price_frame(values, dates):
    values = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "open": values,
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


class Prices:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def fetch(self, tickers, start, end):
        self.calls.append(list(tickers))
        return {
            t: self.data[t].loc[str(start) : str(end)].copy()
            for t in tickers
            if t in self.data
        }


class Snapshot:
    """Apply the actual query filters to offline live snapshot columns."""

    def __init__(self, data):
        self.data = data
        self.plans = []

    def fetch(self, plan, **kwargs):
        self.plans.append(plan)
        rows = []
        for ticker, bars in self.data.items():
            if ticker == "SPY":
                continue
            close = bars.close
            row = {
                "ticker": ticker,
                "name": ticker.split(":")[-1],
                "description": ticker + " Inc",
                "close": close.iloc[-1],
                "volume": bars.volume.iloc[-1],
                "change": close.pct_change().iloc[-1] * 100,
                "market_cap_basic": 1e9,
                "price_52_week_low": bars.low.iloc[-252:].min(),
                "price_52_week_high": bars.high.iloc[-252:].max(),
                "average_volume_10d_calc": bars.volume.iloc[-10:].mean(),
                **{f"SMA{n}": close.iloc[-n:].mean() for n in (50, 150, 200)},
            }

            def operand(value):
                return row[value] if isinstance(value, str) else value

            def passes(raw):
                rule = dict(raw)
                left = operand(rule["left"])
                right = rule["right"]
                op = rule["operation"]
                if op == "greater":
                    return left > operand(right)
                if op == "egreater":
                    return left >= operand(right)
                if op == "above%":
                    return left >= operand(right[0]) * right[1]
                raise AssertionError(rule)

            if all(passes(f) for f in plan.filters):
                rows.append(row)
        rows.sort(key=lambda r: r[plan.query_order_by], reverse=True)
        return (
            len(rows),
            pd.DataFrame(rows[plan.offset : plan.offset + plan.fetch_limit]),
            datetime(2025, 6, 20, 22, tzinfo=UTC),
        )


@pytest.fixture
def offline_screen(monkeypatch):
    monkeypatch.setattr("screener.screen_workflow.date", FrozenDate)
    monkeypatch.setattr("screener.screen_candidates.date", FrozenDate)

    def install(data, snapshot_data=None):
        provider = Prices(data)
        snapshot = Snapshot(data if snapshot_data is None else snapshot_data)
        monkeypatch.setattr(
            "screener.backtester.data.build_price_fetcher", lambda **kw: provider
        )
        monkeypatch.setattr(scanner, "TRADINGVIEW_SCANNER", snapshot)
        return provider, snapshot

    return install


def exact_universe(tmp_path, tickers):
    path = tmp_path / "universe.txt"
    path.write_text("\n".join(tickers))
    return str(path)


def test_minervini_live_screen_preserves_full_rs_reference(offline_screen, tmp_path):
    dates = pd.bdate_range(end=AS_OF, periods=1000)
    data = {"AAA": price_frame(50 * np.exp(0.0018 * np.arange(1000)), dates)}
    for i in range(9):
        values = (10 + i) * np.exp(0.003 * np.arange(1000))
        values[-20:] = np.linspace(values[-21], values[-21] * 0.9, 20)
        data[f"P{i}"] = price_frame(values, dates)
    data["SPY"] = price_frame(np.linspace(50, 100, 1000), dates)
    provider, _ = offline_screen(data)
    exact = screen(
        criteria="mark_minervini",
        universe=exact_universe(tmp_path, [t for t in data if t != "SPY"]),
    )
    default = screen(criteria="mark_minervini")
    assert exact.df.empty
    assert default.df.empty
    assert set(provider.calls[-1]) == set(data)


def test_live_volume_cannot_cut_completed_daily_breakout(offline_screen, tmp_path):
    dates = pd.bdate_range(end=AS_OF, periods=1000)
    bars = price_frame(50 * np.exp(0.0018 * np.arange(1000)), dates)
    bars.loc[dates[-2], "volume"] = 2_000_000.0
    bars.loc[dates[-1], "volume"] = 100_000.0
    live = {"AAA": bars, "SPY": bars}
    completed = {
        t: drop_incomplete_sessions(b, t, now=datetime(2025, 6, 20, 14, tzinfo=UTC))
        for t, b in live.items()
    }
    _, snapshot = offline_screen(completed, live)
    exact = screen(criteria="breakout", universe=exact_universe(tmp_path, ["AAA"]))
    default = screen(criteria="breakout")
    cli = CliRunner().invoke(screen_command, ["-c", "breakout", "--csv"])
    assert exact.df.name.tolist() == ["AAA"]
    assert default.df.name.tolist() == ["AAA"]
    assert cli.exit_code == 0
    assert "AAA" in cli.output
    assert not snapshot.plans[-1].filters


@pytest.mark.parametrize("membership_kind", ["windows", "added"])
def test_mark_minervini_masks_rank_reference(membership_kind):
    dates = pd.bdate_range(end="2024-01-05", periods=1000)
    bars = {
        t: price_frame(100 * np.exp(rate * np.arange(1000)), dates)
        for t, rate in (("AAA", 0.002), ("BBB", 0.004))
    }
    ctx = PrepareCtx(
        market="us",
        benchmark="SPY",
        bars_by_tv=bars,
        price_panel=bars,
        tv_symbols=list(bars),
        start=date(2024, 1, 2),
        end=date(2024, 1, 5),
        fetcher=Prices(bars),
        warnings=[],
        membership_windows=(
            ("AAA", date(2010, 1, 1), None),
            ("BBB", date(2025, 1, 1), None),
        )
        if membership_kind == "windows"
        else (),
        membership_added=(("BBB", date(2025, 1, 1)),)
        if membership_kind == "added"
        else (),
    )
    spec = registry.get("mark_minervini")
    prepared = spec.prepare_bars(ctx)
    assert prepared["AAA"].rs_rank.iloc[-1] == 100.0
    assert pd.isna(prepared["BBB"].rs_rank.iloc[-1])
    assert len(prepared["BBB"]) == len(bars["BBB"])
    assert prepared["BBB"].sma200_up_1m.iloc[-1] == 1.0
    static_ctx = ctx.model_copy(
        update={"membership_windows": (), "membership_added": ()}
    )
    static = spec.prepare_bars(static_ctx)
    assert static["AAA"].rs_rank.iloc[-1] == 50.0
    assert static["BBB"].rs_rank.iloc[-1] == 100.0


@pytest.mark.parametrize("symbol", ["NYSE:BRK.B", "BRK.B", "BRK-B"])
def test_share_class_translation(symbol):
    assert tv_to_yf(symbol, "us") == "BRK-B"
    assert tv_to_yf("BSE:TCS", "india") == "TCS.BO"
    assert tv_to_yf("TCS.NS", "india") == "TCS.NS"


def test_share_class_default_screen_loads_bars(offline_screen, tmp_path):
    dates = pd.bdate_range(end=AS_OF, periods=1000)
    bars = price_frame(np.linspace(200, 500, 1000), dates)
    offline_screen({"BRK-B": bars, "SPY": bars}, {"NYSE:BRK.B": bars})
    default = screen(criteria="ema")
    exact = screen(criteria="ema", universe=exact_universe(tmp_path, ["BRK-B"]))
    assert default.df.ticker.tolist() == ["NYSE:BRK.B"]
    assert exact.df.ticker.tolist() == ["BRK-B"]


def test_screen_history_preserves_exchange_identity(
    offline_screen, tmp_path, monkeypatch
):
    dates = pd.bdate_range(end=AS_OF, periods=1000)
    offline_screen(
        {
            "TCS.NS": price_frame(np.linspace(100, 200, 1000), dates),
            "TCS.BO": price_frame(np.linspace(100, 210, 1000), dates),
            "^NSEI": price_frame(np.linspace(100, 200, 1000), dates),
        }
    )
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    result = screen(
        market="india",
        criteria="ema",
        universe=exact_universe(tmp_path, ["NSE:TCS", "BSE:TCS"]),
        persist=True,
        report_path=tmp_path / "report.html",
    )
    run_id = int(history.list_runs().iloc[0]["id"])
    saved = history.resolve_replay_run(str(run_id))
    assert saved.tickers == [row.ticker for row in result.signals()]
    assert saved.rows.close.tolist() == [210.0, 200.0]
    assert saved.rows["rank"].tolist() == [1, 2]
    assert history.diff(result.df, saved.rows) == ([], [])
    assert history.diff(result.df.iloc[:1], saved.rows) == ([], ["NSE:TCS"])


def test_legacy_history_rows_remain_readable_and_diffable(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    old = pd.DataFrame({"name": ["TCS"], "description": ["Tata"], "close": [200.0]})
    run_id = history.save_run("india", "ema", 1, old)
    saved = history.resolve_replay_run(str(run_id))
    assert saved.tickers == ["TCS"]
    current = pd.DataFrame({"ticker": ["NSE:TCS"], "name": ["TCS"]})
    assert history.diff(current, saved.rows) == ([], [])
    # A legacy key cannot identify which of two exchanges it represented.
    ambiguous = pd.DataFrame({"ticker": ["NSE:TCS", "BSE:TCS"], "name": ["TCS", "TCS"]})
    assert history.diff(ambiguous, saved.rows) == (["BSE:TCS", "NSE:TCS"], ["TCS"])


def test_exact_screen_timestamp_is_utc(offline_screen, tmp_path):
    dates = pd.bdate_range(end=AS_OF, periods=1000)
    bars = price_frame(np.linspace(100, 200, 1000), dates)
    offline_screen({"AAA": bars, "SPY": bars})
    result = screen(criteria="ema", universe=exact_universe(tmp_path, ["AAA"]))
    assert result.as_of.tzinfo == UTC
    assert (datetime.now(UTC) - result.as_of).total_seconds() >= 0


def test_dynamic_policy_fingerprint_uses_resolved_definition(offline_screen, tmp_path):
    import json

    dates = pd.bdate_range(end=AS_OF, periods=1000)
    data = {
        t: price_frame(np.linspace(50, end, 1000), dates)
        for t, end in (("AAA", 100), ("BBB", 200), ("SPY", 100))
    }
    offline_screen(data)
    config = tmp_path / "universes.json"
    request = ScreenRequest(
        market="us",
        criteria_names=("ema",),
        limit=50,
        order_by="setup_score",
        output_csv=True,
        detail=False,
        refresh=False,
        cache_ttl="15m",
        report_path=None,
        universe="book",
        universe_config=str(config),
    )

    def run(**changes):
        config.write_text(
            json.dumps(
                {
                    "universes": {
                        "book": {
                            "type": "dynamic",
                            "market": "us",
                            "symbols": ["AAA", "BBB"],
                            "size": 1,
                            **changes,
                        }
                    }
                }
            )
        )
        return run_screen_workflow(request)

    first = run()
    second = run(size=2)
    assert len(first.df) == 1
    assert len(second.df) == 2
    assert first.label != second.label
    for policy in ({"lookback": 30}, {"rebalance": "weekly"}, {"benchmark": "QQQ"}):
        assert run(**policy).label != first.label
    assert run().label == first.label
    assert run(symbols=["BBB", "AAA"]).label == first.label
    assert run(symbols=["AAA"]).label != first.label
    # Market data and selected members change, but the definition does not.
    data["AAA"].loc[:, "volume"] = 10_000_000.0
    assert run().label == first.label


def test_dynamic_base_policy_does_not_hash_current_members(monkeypatch):
    from types import SimpleNamespace

    members = ["AAA", "BBB"]
    monkeypatch.setattr(
        "screener.universes.load_current_universe",
        lambda *args, **kwargs: SimpleNamespace(
            symbols=tuple(members), source="offline"
        ),
    )
    first = resolve_universe_field("dynamic", "us")
    members[:] = ["CCC", "DDD"]
    changed_members = resolve_universe_field("dynamic", "us", dynamic_base="sp500")
    assert first.policy == changed_members.policy
    assert first.tickers != changed_members.tickers
    changed_base = resolve_universe_field("dynamic", "us", dynamic_base="nifty50")
    assert first.policy != changed_base.policy
    assert first.policy["definition"] == {"base": "sp500"}


def test_snapshot_timestamp_normalizes_legacy_naive_utc(monkeypatch):
    stamp = datetime(2025, 6, 20, 22)
    monkeypatch.setattr(
        "screener.screen_workflow.scan",
        lambda **kwargs: (0, pd.DataFrame(), stamp),
    )
    result = screen(criteria="above_avg_volume")
    assert result.as_of == stamp.replace(tzinfo=UTC)


def test_minervini_declares_no_reference_changing_prefilter():
    strategy = resolve_screen_strategy(("mark_minervini",))
    assert strategy is not None
    assert strategy.tv_prefilter is None
