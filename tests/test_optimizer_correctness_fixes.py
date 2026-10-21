"""Offline regressions for validated optimization and public research reports."""

from datetime import date
import json

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.backtester.metrics import compute_metrics
from screener.backtester.optimization.cli import optimize
from screener.backtester.optimization.grid import grid_search
from screener.backtester.optimization.walk_forward import walk_forward_optimize
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.slippage import (
    CompositeSlippage,
    EstimatedHalfSpreadSlippage,
    FixedBpsSlippage,
    HalfSpreadSlippage,
    VolumeImpactSlippage,
)
from tests.conftest import StubPriceFetcher, make_bars
from tests.test_optimizer_regressions import _config


@pytest.fixture
def prices():
    bars = make_bars(n=80, drift=0.2, seed=7)
    return bars, StubPriceFetcher({t: bars for t in ("AAA", "OLD", "NEW", "SPY")})


def test_validated_config_update_preserves_slippage_collaborators():
    impact = VolumeImpactSlippage(k=0.3)
    spread = HalfSpreadSlippage(half_spread_bps=5)
    estimated = EstimatedHalfSpreadSlippage()
    model = CompositeSlippage(
        models=(
            CompositeSlippage(models=(FixedBpsSlippage(), spread)),
            impact,
            estimated,
        )
    )
    cfg = _config(slippage_model=model, spread_proxy=True)
    updated = cfg.validated_update({"slippage_bps": 100})
    assert updated.slippage_model.models[0].models[0].bps == 100
    assert updated.slippage_model.models[0].models[1] is spread
    assert updated.slippage_model.models[1] is impact
    assert updated.slippage_model.models[2] is estimated
    assert cfg.slippage_model.models[0].models[0].bps == 0
    assert cfg.validated_update({"hold": 2}).slippage_model is model


@pytest.mark.parametrize(
    "model",
    [
        VolumeImpactSlippage(),
        HalfSpreadSlippage(half_spread_bps=2),
        CompositeSlippage(models=(FixedBpsSlippage(), FixedBpsSlippage())),
    ],
)
def test_slippage_update_rejects_ambiguous_fixed_component(model):
    cfg = _config(slippage_model=model)
    with pytest.raises(ValueError, match="slippage_bps.*exactly one"):
        cfg.validated_update({"slippage_bps": 20})
    replacement = FixedBpsSlippage(bps=20)
    assert (
        cfg.validated_update(
            {"slippage_bps": 20, "slippage_model": replacement}
        ).slippage_model
        is replacement
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_grid_slippage_matches_fresh_execution(prices, tmp_path, workers):
    bars, fetcher = prices
    cfg = _config(entry_expr="close > 0", hold=3)
    start, end = bars.index[5].date(), bars.index[-1].date()
    rows = grid_search(
        cfg,
        fetcher,
        {"slippage_bps": [0, 100]},
        metric="total_return",
        top_n=2,
        runner="rolling",
        start_date=start,
        end_date=end,
        max_workers=workers,
        trial_db_path=tmp_path / "trials.db",
    )
    assert rows[0].score != rows[1].score
    for row in rows:
        fresh = _config(entry_expr="close > 0", hold=3, **row.params)
        expected = run_rolling_backtest(fresh, fetcher, start_date=start, end_date=end)
        assert row.score == pytest.approx(expected.metrics["total_return"])


@pytest.mark.parametrize(
    "params",
    [
        {"stop_los": [0.1]},
        {"hold": [1, 0]},
        {"top": [0]},
        {"slippage_bps": [-1]},
        {"risk_free_rate": [float("nan")]},
        {"rank_exit_every": [1], "top": [51]},
    ],
)
def test_invalid_grid_rejected_before_fetch_or_trial_register(tmp_path, params):
    class NoFetch:
        def fetch(self, *args, **kwargs):
            pytest.fail("invalid grid fetched bars")

    trial_db = tmp_path / "invalid.db"
    with pytest.raises(ValueError):
        grid_search(_config(), NoFetch(), params, trial_db_path=trial_db)
    assert not trial_db.exists()


def test_optimizer_rejects_fetcher_configured_for_another_market(tmp_path):
    from screener.backtester.fundamentals import FMPFundamentalFetcher

    class NoFetch:
        def fetch(self, *args, **kwargs):
            pytest.fail("incompatible collaborator fetched bars")

    trial_db = tmp_path / "wrong-market.db"
    fetcher = FMPFundamentalFetcher(market="us", api_key="offline")
    with pytest.raises(ValueError, match="configured market"):
        grid_search(
            _config(market="india", fundamentals_provider="fmp"),
            NoFetch(),
            {"hold": [2]},
            runner="rolling",
            fundamental_fetcher=fetcher,
            trial_db_path=trial_db,
        )
    assert not trial_db.exists()


class FundamentalStub:
    markets = frozenset({"us"})
    lag_days = 1
    effective_date_kind = "filing_timestamp"

    def fetch(self, tickers, start, end):
        return {
            t: pd.DataFrame({"roe_ttm": [20.0]}, index=pd.DatetimeIndex(["2023-12-01"]))
            for t in tickers
        }


@pytest.mark.parametrize("workers", [1, 2])
def test_fundamental_grid_supports_api_collaborator(prices, tmp_path, workers):
    bars, fetcher = prices
    cfg = _config(entry_expr="roe_ttm > 10", fundamentals_provider="fmp")
    rows = grid_search(
        cfg,
        fetcher,
        {"hold": [3, 4]},
        runner="rolling",
        start_date=bars.index[5].date(),
        end_date=bars.index[-1].date(),
        fundamental_fetcher=FundamentalStub(),
        max_workers=workers,
        trial_db_path=tmp_path / "trials.db",
    )
    assert all(row.error is None and row.trade_count > 0 for row in rows)


def test_fundamental_grid_missing_collaborator_fails_before_trials(prices, tmp_path):
    bars, fetcher = prices
    with pytest.raises(ValueError, match="resolved FundamentalFetcher"):
        grid_search(
            _config(entry_expr="roe_ttm > 10"),
            fetcher,
            {"hold": [3]},
            runner="rolling",
            start_date=bars.index[5].date(),
            end_date=bars.index[-1].date(),
            trial_db_path=tmp_path / "trials.db",
        )
    assert not (tmp_path / "trials.db").exists()


def report_args(bars, tmp_path):
    return [
        "research-report",
        "--entry",
        "close > 0",
        "--start",
        str(bars.index[5].date()),
        "--end",
        str(bars.index[-1].date()),
        "--hold",
        "3",
        "--top",
        "1",
        "--train-days",
        "30",
        "--test-days",
        "15",
        "--step-days",
        "15",
        "--mc-iterations",
        "10",
        "--mc-block",
        "2",
        "--min-price",
        "0",
        "--min-avg-dollar-volume",
        "0",
        "--metric",
        "total_return",
        "--trial-db",
        str(tmp_path / "trials.db"),
        "--out",
        str(tmp_path / "report"),
    ]


def test_public_report_slippage_and_selected_hurdle(prices, tmp_path, monkeypatch):
    bars, fetcher = prices
    monkeypatch.setattr(
        "screener.backtester.optimization.cli._fetcher", lambda: fetcher
    )
    result = CliRunner().invoke(
        optimize,
        report_args(bars, tmp_path)
        + [
            "--tickers",
            "AAA",
            "--param",
            "slippage_bps=0,100",
            "--param",
            "risk_free_rate=0.2",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "report.json").read_text())
    assert len({r["score"] for r in payload["grid"]["results"]}) == 2
    assert payload["walk_forward"]["aggregate_metrics"]["risk_free_rate"] == 0.2


@pytest.mark.parametrize(
    "spec", ["stop_los=0.01,0.99", "hold=3,0", "top=1,0", "slippage_bps=0,-1"]
)
def test_public_report_invalid_params_no_artifacts(prices, tmp_path, monkeypatch, spec):
    bars, _ = prices

    def no_fetcher():
        pytest.fail("invalid parameters reached price fetcher")

    monkeypatch.setattr("screener.backtester.optimization.cli._fetcher", no_fetcher)
    result = CliRunner().invoke(
        optimize, report_args(bars, tmp_path) + ["--tickers", "AAA", "--param", spec]
    )
    assert result.exit_code != 0
    assert spec.split("=")[0] in result.output
    assert not (tmp_path / "trials.db").exists()
    assert not (tmp_path / "report.json").exists()


def test_public_report_period_wide_membership(prices, tmp_path, monkeypatch):
    import screener.universes as universes

    bars, fetcher = prices
    monkeypatch.setattr(
        "screener.backtester.optimization.cli._fetcher", lambda: fetcher
    )
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path / "universe-cache")
    monkeypatch.setattr(
        universes,
        "_fetch_sp500_revision_snapshot",
        lambda as_of, **kwargs: (
            123,
            ["NEW"] if as_of >= date(2024, 4, 1) else ["OLD"],
        ),
    )
    from screener.backtester.optimization import research_report as report_module

    real_walk_forward = report_module.walk_forward_optimize
    captured = []

    def capture_walk_forward(*args, **kwargs):
        summary = real_walk_forward(*args, **kwargs)
        captured.append(summary)
        return summary

    monkeypatch.setattr(report_module, "walk_forward_optimize", capture_walk_forward)
    result = CliRunner().invoke(
        optimize,
        report_args(bars, tmp_path)
        + ["--universe", "sp500", "--max-universe", "0", "--param", "hold=3"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "report.json").read_text())
    assert set(payload["config"]["tickers"]) == {"OLD", "NEW"}
    assert payload["config"]["membership_windows"]
    assert captured[0].oos_trades
    assert all(
        t.ticker != "NEW" or t.signal_date >= date(2024, 4, 1)
        for t in captured[0].oos_trades
    )
    assert any(t.ticker == "OLD" for t in captured[0].oos_trades)


def test_selected_risk_free_aggregate_matches_equity(prices, tmp_path):
    bars, fetcher = prices
    cfg = _config(entry_expr="close > 0", hold=3)
    summary = walk_forward_optimize(
        cfg,
        fetcher,
        {"hold": [3], "risk_free_rate": [0.2]},
        start_date=bars.index[5].date(),
        end_date=bars.index[-1].date(),
        train_days=30,
        test_days=15,
        step_days=15,
        trial_db_path=tmp_path / "trials.db",
    )
    expected = compute_metrics(
        summary.oos_equity,
        bars["close"].reindex(summary.oos_equity.index, method="ffill"),
        list(summary.oos_trades),
        cfg.top,
        risk_free_rate=0.2,
    )
    assert summary.aggregate_metrics["sharpe"] == pytest.approx(expected["sharpe"])
    assert summary.aggregate_metrics["risk_free_rate"] == 0.2


@pytest.mark.parametrize("command", ["grid", "walk-forward", "research-report"])
def test_fundamental_cli_requires_explicit_provider(
    prices, tmp_path, monkeypatch, command
):
    bars, _ = prices
    monkeypatch.setattr(
        "screener.backtester.optimization.cli._fetcher",
        lambda: pytest.fail("missing provider reached fetcher"),
    )
    args = (
        report_args(bars, tmp_path)
        if command == "research-report"
        else [command, "--tickers", "AAA", "--hold", "3"]
    )
    result = CliRunner().invoke(
        optimize, args + ["--tickers", "AAA", "--entry", "roe_ttm > 10"]
    )
    assert result.exit_code != 0
    assert "--fundamentals-provider" in result.output
    assert not (tmp_path / "trials.db").exists()


def test_public_report_resolves_fundamental_provider(prices, tmp_path, monkeypatch):
    bars, fetcher = prices
    monkeypatch.setattr(
        "screener.backtester.optimization.cli._fetcher", lambda: fetcher
    )
    monkeypatch.setattr(
        "screener.backtester.fundamentals.build_fundamental_fetcher",
        lambda *args, **kwargs: FundamentalStub(),
    )
    result = CliRunner().invoke(
        optimize,
        report_args(bars, tmp_path)
        + [
            "--tickers",
            "AAA",
            "--entry",
            "roe_ttm > 10",
            "--fundamentals-provider",
            "fmp",
            "--param",
            "hold=3",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["config"]["fundamentals_provider"] == "fmp"
    assert payload["grid"]["results"][0]["trade_count"] > 0
    assert payload["walk_forward"]["windows"]


def test_walk_forward_rejects_different_selected_hurdles(prices, tmp_path, monkeypatch):
    from screener.backtester.optimization import walk_forward as wf_module
    from screener.backtester.optimization.grid import GridSearchResult

    bars, fetcher = prices
    selections = iter([0.1, 0.2])

    def select_hurdle(*args, **kwargs):
        return [
            GridSearchResult(
                params={"risk_free_rate": next(selections)},
                score=1.0,
                metrics={},
                trade_count=10,
            )
        ]

    monkeypatch.setattr(wf_module, "grid_search", select_hurdle)
    with pytest.raises(
        ValueError, match="risk_free_rate is ambiguous.*vary across folds"
    ):
        walk_forward_optimize(
            _config(entry_expr="close > 0"),
            fetcher,
            {"risk_free_rate": [0.1, 0.2]},
            start_date=bars.index[5].date(),
            end_date=bars.index[-1].date(),
            train_days=30,
            test_days=15,
            trial_db_path=tmp_path / "trials.db",
        )


def test_public_report_rejects_varying_hurdles_clearly(prices, tmp_path, monkeypatch):
    from screener.backtester.optimization import walk_forward as wf_module
    from screener.backtester.optimization.grid import GridSearchResult

    bars, fetcher = prices
    selections = iter([0.1, 0.2])

    def select_hurdle(*args, **kwargs):
        return [
            GridSearchResult(
                params={"risk_free_rate": next(selections)},
                score=1.0,
                metrics={},
                trade_count=10,
            )
        ]

    monkeypatch.setattr(wf_module, "grid_search", select_hurdle)
    monkeypatch.setattr(
        "screener.backtester.optimization.cli._fetcher", lambda: fetcher
    )
    result = CliRunner().invoke(
        optimize,
        report_args(bars, tmp_path)
        + ["--tickers", "AAA", "--param", "risk_free_rate=0.1,0.2"],
    )
    assert result.exit_code == 1
    assert "risk_free_rate is ambiguous" in result.output
    assert not (tmp_path / "report.json").exists()


def test_parameter_parser_rejects_infinite_range_before_iteration():
    from screener.backtester.optimization.cli import _parse_param_specs
    import click

    with pytest.raises(click.UsageError, match="range bounds and step must be finite"):
        _parse_param_specs(("slippage_bps=0:inf",))


def test_slippage_protocol_object_is_not_serialized_on_update():
    class CustomSlippage:
        def adverse_fraction(self, side, shares, adv, sigma_daily, half_spread=0.0):
            return 0.001

    custom = CustomSlippage()
    cfg = _config(slippage_model=CompositeSlippage(models=(FixedBpsSlippage(), custom)))
    updated = cfg.validated_update({"slippage_bps": 10})
    assert updated.slippage_model.models[1] is custom
    assert updated.slippage_model.adverse_fraction("buy", 1, 1, 1) == pytest.approx(
        0.002
    )


def test_trial_and_cache_identity_use_executed_slippage_config():
    from screener.backtester.optimization.cache import (
        FrozenInputIdentity,
        make_cache_key,
    )
    from screener.backtester.optimization.grid import _trial_key_for
    from screener.backtester.optimization.trials import trial_config_identity

    base = _config()
    fresh = _config(slippage_bps=100)
    args = dict(
        runner="rolling",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 4, 1),
        metric="sharpe",
        min_trades=1,
    )
    assert _trial_key_for(base, {"slippage_bps": 100}, **args) == trial_config_identity(
        fresh, **args
    )
    identity = FrozenInputIdentity(
        price_snapshot_hash="prices",
        universe_membership_identity="membership",
        source="test",
    )
    assert make_cache_key(
        base, {"slippage_bps": 100}, frozen_input_identity=identity, **args
    ) == make_cache_key(
        fresh, {"slippage_bps": 100}, frozen_input_identity=identity, **args
    )


@pytest.mark.parametrize("command", ["grid", "walk-forward"])
def test_other_optimizer_commands_resolve_fundamental_provider(
    prices, tmp_path, monkeypatch, command
):
    bars, fetcher = prices
    monkeypatch.setattr(
        "screener.backtester.optimization.cli._fetcher", lambda: fetcher
    )
    monkeypatch.setattr(
        "screener.backtester.fundamentals.build_fundamental_fetcher",
        lambda *args, **kwargs: FundamentalStub(),
    )
    args = [
        command,
        "--tickers",
        "AAA",
        "--entry",
        "roe_ttm > 10",
        "--fundamentals-provider",
        "fmp",
        "--start",
        str(bars.index[5].date()),
        "--end",
        str(bars.index[-1].date()),
        "--hold",
        "3",
        "--top",
        "1",
        "--stop-loss",
        "none",
        "--take-profit",
        "none",
        "--trailing-stop",
        "none",
        "--min-price",
        "0",
        "--min-avg-dollar-volume",
        "0",
        "--metric",
        "total_return",
        "--trial-db",
        str(tmp_path / "trials.db"),
        "--json",
        str(tmp_path / "output.json"),
    ]
    if command == "walk-forward":
        args += ["--train-days", "30", "--test-days", "15"]
    result = CliRunner().invoke(optimize, args)
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "output.json").read_text())
    if command == "grid":
        assert payload["results"][0]["trade_count"] > 0
    else:
        assert payload["windows"]


def test_fundamental_cache_requires_fundamental_data_identity(prices, tmp_path):
    from screener.backtester.optimization.cache import (
        build_frozen_input_identity,
        sqlite_cache_path,
    )

    bars, fetcher = prices
    cfg = _config(entry_expr="roe_ttm > 10", fundamentals_provider="fmp")
    cache_path = tmp_path / "cache.db"
    rows = grid_search(
        cfg,
        fetcher,
        {"hold": [3]},
        fundamental_fetcher=FundamentalStub(),
        runner="rolling",
        start_date=bars.index[5].date(),
        end_date=bars.index[-1].date(),
        trial_db_path=tmp_path / "trials.db",
        cache_path=cache_path,
        frozen_input_identity=build_frozen_input_identity(
            cfg, {"AAA": bars, "SPY": bars}
        ),
    )
    assert rows[0].trade_count > 0
    assert not rows[0].cached
    assert not sqlite_cache_path(cache_path).exists()


def test_fundamental_lag_sweep_rejected_before_trial_register(prices, tmp_path):
    bars, fetcher = prices
    with pytest.raises(ValueError, match="fundamental_lag_days must match"):
        grid_search(
            _config(entry_expr="roe_ttm > 10", fundamentals_provider="fmp"),
            fetcher,
            {"fundamental_lag_days": [1, 20]},
            fundamental_fetcher=FundamentalStub(),
            runner="rolling",
            start_date=bars.index[5].date(),
            end_date=bars.index[-1].date(),
            trial_db_path=tmp_path / "trials.db",
        )
    assert not (tmp_path / "trials.db").exists()


def _report_without_membership_history(prices, tmp_path, monkeypatch, *extra):
    import screener.universes as universes

    bars, fetcher = prices
    monkeypatch.setattr(
        "screener.backtester.optimization.cli._fetcher", lambda: fetcher
    )
    monkeypatch.setattr(universes, "CACHE_DIR", tmp_path / "universe-cache")
    monkeypatch.setattr(
        universes, "_fetch_sp500_revision_snapshot", lambda as_of, **kwargs: None
    )
    monkeypatch.setattr(universes, "_fetch_sp500", lambda **kwargs: (["AAA"], "live"))
    monkeypatch.setattr(
        universes,
        "load_sp500_membership",
        lambda **kwargs: {"AAA": date(2000, 1, 3)},
    )
    with pytest.warns(UserWarning, match="NOT point-in-time"):
        return CliRunner().invoke(
            optimize,
            report_args(bars, tmp_path)
            + [
                "--universe",
                "sp500",
                "--max-universe",
                "0",
                "--param",
                "hold=3",
                *extra,
            ],
        )


def test_public_report_explicit_point_in_time_fails_without_history(
    prices, tmp_path, monkeypatch
):
    """A typed --point-in-time must not silently run survivorship-biased."""
    result = _report_without_membership_history(
        prices, tmp_path, monkeypatch, "--point-in-time"
    )
    assert result.exit_code != 0
    assert "--point-in-time is unavailable" in result.output
    assert not (tmp_path / "report.json").exists()


def test_public_report_default_point_in_time_falls_back_to_date_added(
    prices, tmp_path, monkeypatch
):
    result = _report_without_membership_history(prices, tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["config"]["membership_added"]
    assert any("'date added'" in w for w in payload["config"]["universe_warnings"])
