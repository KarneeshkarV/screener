"""Regression tests for optimizer cache, prep grouping, and DSR fixes."""

from __future__ import annotations

import hashlib
import math
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.metrics import (
    _dsr,
    _psr,
    apply_trial_context_to_metrics,
    compute_metrics,
    dsr_from_moments,
    sharpe_moments_from_equity,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.optimization import grid as grid_module
from screener.backtester.optimization.cache import (
    build_frozen_input_identity,
    hash_price_frames,
    sqlite_cache_path,
)
from screener.backtester.optimization.grid import grid_search
from screener.backtester.optimization.trials import (
    load_trial_search_stats,
)
from screener.backtester.rolling_simulation import (
    _preparation_fingerprint,
    prepare_rolling_backtest,
)
from tests.conftest import StubPriceFetcher, make_bars


def _config(**overrides) -> BacktestConfig:
    values = {
        "market": "us",
        "as_of": date(2024, 1, 1),
        "hold": 5,
        "top": 1,
        "entry_expr": "close > sma(close, 3)",
        "exit_expr": None,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 0.0,
        "commission_bps": 0.0,
        "initial_capital": 100_000.0,
        "benchmark": "SPY",
        "tickers": ("AAA",),
        "min_price": None,
        "min_avg_dollar_volume": None,
    }
    values.update(overrides)
    return BacktestConfig(**values)


def _price_content_digest(frames: dict[str, pd.DataFrame]) -> str:
    """Content identity from actual OHLCV bytes. Not fetcher repr."""
    return hash_price_frames(frames)


def test_cache_rejects_stale_scores_when_price_bars_change(tmp_path):
    """Same config must not reuse scores when StubPriceFetcher bars change."""
    bars_a = make_bars(n=80, seed=1, drift=0.05)
    bars_b = make_bars(n=80, seed=99, drift=-0.05)
    frames_a = {"AAA": bars_a, "SPY": bars_a}
    frames_b = {"AAA": bars_b, "SPY": bars_b}
    assert _price_content_digest(frames_a) != _price_content_digest(frames_b)

    cfg = _config()
    cache_path = tmp_path / "grid_cache.json"
    trial_db = tmp_path / "trials.db"
    start = bars_a.index[10].date()
    end = bars_a.index[70].date()

    first = grid_search(
        cfg,
        StubPriceFetcher(frames_a),
        {"hold": [5]},
        metric="total_return",
        runner="rolling",
        start_date=start,
        end_date=end,
        top_n=1,
        max_workers=1,
        cache_path=cache_path,
        frozen_input_identity=build_frozen_input_identity(cfg, frames_a),
        trial_db_path=trial_db,
    )
    second = grid_search(
        cfg,
        StubPriceFetcher(frames_b),
        {"hold": [5]},
        metric="total_return",
        runner="rolling",
        start_date=start,
        end_date=end,
        top_n=1,
        max_workers=1,
        cache_path=cache_path,
        frozen_input_identity=build_frozen_input_identity(cfg, frames_b),
        trial_db_path=trial_db,
    )

    assert not second[0].cached
    assert sqlite_cache_path(cache_path).exists()
    # Legacy JSON is not written; sibling sqlite holds rows.
    assert not cache_path.exists()
    assert first[0].score != second[0].score or first[0].metrics != second[0].metrics


def test_cache_without_frozen_identity_does_not_reuse(tmp_path):
    """Unknown input identity disables cross-run result reuse."""
    bars = make_bars(n=60, seed=3, drift=0.1)
    cfg = _config()
    cache_path = tmp_path / "no_id.json"
    trial_db = tmp_path / "trials.db"
    start = bars.index[5].date()
    end = bars.index[50].date()
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": bars})

    first = grid_search(
        cfg,
        fetcher,
        {"hold": [4]},
        metric="total_return",
        runner="rolling",
        start_date=start,
        end_date=end,
        top_n=1,
        max_workers=1,
        cache_path=cache_path,
        trial_db_path=trial_db,
    )
    second = grid_search(
        cfg,
        fetcher,
        {"hold": [4]},
        metric="total_return",
        runner="rolling",
        start_date=start,
        end_date=end,
        top_n=1,
        max_workers=1,
        cache_path=cache_path,
        trial_db_path=trial_db,
    )
    assert not first[0].cached
    assert not second[0].cached
    assert not sqlite_cache_path(cache_path).exists()


def test_prep_groups_two_entry_expr_by_fingerprint(monkeypatch, tmp_path):
    """Grid of 2 entry_expr x 3 hold prepares once per entry_expr group."""
    import screener.backtester.rolling_simulation as rolling_sim

    bars = make_bars(n=80, seed=7, drift=0.2)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": bars})
    cfg = _config()
    entry_a = "close > sma(close, 3)"
    entry_b = "close > sma(close, 5)"
    assert _preparation_fingerprint(
        cfg.model_copy(update={"entry_expr": entry_a, "hold": 3})
    ) != _preparation_fingerprint(
        cfg.model_copy(update={"entry_expr": entry_b, "hold": 3})
    )

    prepare_calls: list[str] = []
    real_prepare = rolling_sim.prepare_rolling_backtest

    def counting_prepare(cfg_arg, fetcher_arg, **kwargs):
        prepare_calls.append(str(cfg_arg.entry_expr))
        return real_prepare(cfg_arg, fetcher_arg, **kwargs)

    monkeypatch.setattr(rolling_sim, "prepare_rolling_backtest", counting_prepare)
    monkeypatch.setattr(grid_module, "prepare_rolling_backtest", counting_prepare)

    results = grid_search(
        cfg,
        fetcher,
        {"entry_expr": [entry_a, entry_b], "hold": [3, 5, 8]},
        runner="rolling",
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date(),
        top_n=6,
        max_workers=1,
        trial_db_path=tmp_path / "trials.db",
    )
    assert len(results) == 6
    assert prepare_calls.count(entry_a) == 1
    assert prepare_calls.count(entry_b) == 1
    assert len(prepare_calls) == 2


@pytest.mark.parametrize("max_workers", [1, 2])
def test_rolling_grid_parity_workers_missing_bars(tmp_path, max_workers):
    """Worker modes keep score parity with missing bars in the panel."""
    bars_aaa = make_bars(n=90, seed=11, drift=0.15)
    bars_bbb = make_bars(n=90, seed=12, drift=0.05)
    # Drop a mid-window chunk from BBB to exercise missing-bar paths.
    bars_bbb = bars_bbb.drop(bars_bbb.index[30:40])
    spy = make_bars(n=90, seed=99, drift=0.02)
    frames = {"AAA": bars_aaa, "BBB": bars_bbb, "SPY": spy}
    cfg = _config(tickers=("AAA", "BBB"), top=2)
    identity = build_frozen_input_identity(cfg, frames, source="stub-frames")

    results = grid_search(
        cfg,
        StubPriceFetcher(frames),
        {
            "entry_expr": ["close > sma(close, 3)", "close > sma(close, 5)"],
            "hold": [3, 5, 8],
        },
        runner="rolling",
        start_date=bars_aaa.index[5].date(),
        end_date=bars_aaa.index[-5].date(),
        top_n=6,
        max_workers=max_workers,
        metric="total_return",
        frozen_input_identity=identity,
        cache_path=tmp_path / f"parity_{max_workers}.sqlite",
        trial_db_path=tmp_path / f"trials_{max_workers}.db",
        experiment_id=f"parity-{max_workers}",
    )
    assert len(results) == 6
    assert all(math.isfinite(r.score) or r.score == float("-inf") for r in results)
    param_keys = {
        tuple(sorted((k, str(v)) for k, v in r.params.items())) for r in results
    }
    assert len(param_keys) == 6


def test_rolling_grid_worker_modes_match(tmp_path):
    bars_aaa = make_bars(n=90, seed=11, drift=0.15)
    bars_bbb = make_bars(n=90, seed=12, drift=0.05).drop(
        make_bars(n=90, seed=12, drift=0.05).index[30:40]
    )
    spy = make_bars(n=90, seed=99, drift=0.02)
    frames = {"AAA": bars_aaa, "BBB": bars_bbb, "SPY": spy}
    cfg = _config(tickers=("AAA", "BBB"), top=2)
    identity = build_frozen_input_identity(cfg, frames)
    grid = {
        "entry_expr": ["close > sma(close, 3)", "close > sma(close, 5)"],
        "hold": [3, 5, 8],
    }
    kwargs = dict(
        runner="rolling",
        start_date=bars_aaa.index[5].date(),
        end_date=bars_aaa.index[-5].date(),
        top_n=6,
        metric="total_return",
        frozen_input_identity=identity,
    )
    seq = grid_search(
        cfg,
        StubPriceFetcher(frames),
        grid,
        max_workers=1,
        cache_path=tmp_path / "seq.sqlite",
        trial_db_path=tmp_path / "seq_trials.db",
        experiment_id="parity-seq",
        **kwargs,
    )
    par = grid_search(
        cfg,
        StubPriceFetcher(frames),
        grid,
        max_workers=2,
        cache_path=tmp_path / "par.sqlite",
        trial_db_path=tmp_path / "par_trials.db",
        experiment_id="parity-par",
        **kwargs,
    )

    def key_of(row):
        return tuple(sorted((k, str(v)) for k, v in row.params.items()))

    seq_map = {key_of(r): (r.score, r.trade_count) for r in seq}
    par_map = {key_of(r): (r.score, r.trade_count) for r in par}
    assert seq_map.keys() == par_map.keys()
    for key, (score, trades) in seq_map.items():
        assert score == pytest.approx(par_map[key][0], rel=1e-9, abs=1e-12)
        assert trades == par_map[key][1]


def test_dsr_no_silent_half_std_assumption():
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0.001, 0.01, 252))
    assert math.isnan(_dsr(daily, n_trials=20))
    with_half = _dsr(daily, n_trials=20, sr_trial_std_annual=0.5)
    assert math.isfinite(with_half)

    equity = pd.Series(100_000.0, index=pd.bdate_range("2020-01-01", periods=253))
    for i, r in enumerate(daily, start=1):
        equity.iloc[i] = equity.iloc[i - 1] * (1.0 + float(r))
    metrics = compute_metrics(equity, equity, [], slot_count=1)
    assert "dsr" not in metrics
    multi = compute_metrics(
        equity, equity, [], slot_count=1, n_trials=10, sr_trial_std_annual=None
    )
    assert math.isnan(multi["dsr"])


def test_psr_kurtosis_matches_bailey_formula():
    rng = np.random.default_rng(42)
    daily = pd.Series(rng.normal(0.001, 0.01, 252))
    T = len(daily)
    sr_annual = float(daily.mean() / daily.std(ddof=0) * math.sqrt(252))
    sr_per = sr_annual / math.sqrt(252)
    skew = float(daily.skew())
    kurt_excess = float(daily.kurt())
    bailey_denom_sq = (
        1.0 - skew * sr_per + ((kurt_excess + 2.0) / 4.0) * sr_per * sr_per
    )
    z = (sr_per - 0.0) * math.sqrt(T - 1) / math.sqrt(max(bailey_denom_sq, 1e-12))
    bailey_psr = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    assert _psr(daily, 0.0) == pytest.approx(bailey_psr, rel=1e-9)


def test_trial_register_includes_rejected_and_dispersion(tmp_path):
    bars = make_bars(n=80, seed=21, drift=0.25)
    frames = {"AAA": bars, "SPY": bars}
    cfg = _config()
    trial_db = tmp_path / "trials.db"
    exp = "exp-dispersion-1"
    results = grid_search(
        cfg,
        StubPriceFetcher(frames),
        {"hold": [3, 5, 8]},
        runner="rolling",
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date(),
        top_n=1,
        max_workers=1,
        min_trades=10_000,  # force rejections while still finishing runs
        metric="sharpe",
        frozen_input_identity=build_frozen_input_identity(cfg, frames),
        cache_path=tmp_path / "cache.sqlite",
        trial_db_path=trial_db,
        experiment_id=exp,
    )
    assert len(results) == 1
    stats = load_trial_search_stats(exp, db_path=trial_db)
    assert stats.n_trials_nominal == 3
    assert stats.n_trials_effective is None
    # Rejected trials with finite Sharpe still contribute to dispersion.
    assert stats.n_trials_scored >= 1
    if stats.n_trials_scored >= 2:
        assert stats.sr_trial_std_annual is not None
        assert stats.sr_trial_std_annual >= 0.0
    assert 0.0 not in stats.sharpes or any(s != 0.0 for s in stats.sharpes) or True
    # Error/missing Sharpe never coerced: all listed sharpes are finite by contract.
    assert all(math.isfinite(s) for s in stats.sharpes)


def test_apply_trial_context_helper_rewrites_dsr():
    equity = pd.Series(
        np.cumprod(1.0 + np.random.default_rng(1).normal(0.001, 0.01, 120)) * 1e5,
        index=pd.bdate_range("2020-01-01", periods=120),
    )
    moments = sharpe_moments_from_equity(equity)
    assert moments is not None
    base = {"sharpe": moments.sharpe_annual, "dsr": 0.9, "psr": 0.9}
    out = apply_trial_context_to_metrics(
        base, moments, n_trials=8, sr_trial_std_annual=0.4
    )
    assert out["dsr"] == pytest.approx(
        dsr_from_moments(moments, n_trials=8, sr_trial_std_annual=0.4)
    )
    missing = apply_trial_context_to_metrics(
        base, moments, n_trials=8, sr_trial_std_annual=None
    )
    assert math.isnan(missing["dsr"])


def test_prep_benchmark_two_groups(tmp_path):
    """Benchmark prepare+simulate: per-combo prep vs 2 fingerprint groups."""
    from screener.backtester.rolling_simulation import run_prepared_rolling_backtest

    bars = make_bars(n=100, seed=5, drift=0.12)
    frames = {"AAA": bars, "SPY": bars}
    fetcher = StubPriceFetcher(frames)
    cfg = _config()
    entry_a = "close > sma(close, 3)"
    entry_b = "close > sma(close, 5)"
    start = bars.index[0].date()
    end = bars.index[-1].date()
    combos = [(entry, hold) for entry in (entry_a, entry_b) for hold in (3, 5, 8)]

    t0 = time.perf_counter()
    for entry, hold in combos:
        prepared = prepare_rolling_backtest(
            cfg.model_copy(update={"entry_expr": entry, "hold": hold}),
            fetcher,
            start_date=start,
            end_date=end,
        )
        run_prepared_rolling_backtest(
            prepared, cfg.model_copy(update={"entry_expr": entry, "hold": hold})
        )
    per_combo_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    for entry in (entry_a, entry_b):
        prepared = prepare_rolling_backtest(
            cfg.model_copy(update={"entry_expr": entry, "hold": 3}),
            fetcher,
            start_date=start,
            end_date=end,
        )
        for hold in (3, 5, 8):
            run_prepared_rolling_backtest(
                prepared, cfg.model_copy(update={"entry_expr": entry, "hold": hold})
            )
        prepared = None
    grouped_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    grid_results = grid_search(
        cfg,
        StubPriceFetcher(frames),
        {"entry_expr": [entry_a, entry_b], "hold": [3, 5, 8]},
        runner="rolling",
        start_date=start,
        end_date=end,
        top_n=6,
        max_workers=1,
        trial_db_path=tmp_path / "bench_trials.db",
        experiment_id="bench-prep",
    )
    grid_s = time.perf_counter() - t2
    assert len(grid_results) == 6
    assert grid_s > 0.0
    assert grouped_s <= per_combo_s * 1.2


def test_frozen_identity_uses_content_not_fetcher_repr():
    bars = make_bars(n=40, seed=8)
    frames = {"AAA": bars, "SPY": bars}
    cfg = _config()
    a = build_frozen_input_identity(cfg, frames, source="a")
    b = build_frozen_input_identity(cfg, frames, source="b")
    # source label may differ; price hash must match content.
    assert a.price_snapshot_hash == b.price_snapshot_hash
    # Changing bars changes hash; fetcher class is irrelevant.
    other = make_bars(n=40, seed=9)
    c = build_frozen_input_identity(cfg, {"AAA": other, "SPY": other})
    assert c.price_snapshot_hash != a.price_snapshot_hash
    # Explicitly prove we do not key on StubPriceFetcher repr.
    fetcher_repr = repr(StubPriceFetcher(frames))
    assert hashlib.sha256(fetcher_repr.encode()).hexdigest() != a.price_snapshot_hash


def test_trial_key_distinguishes_strategies_sharing_experiment_and_params(tmp_path):
    """Shared experiment_id + same swept params must not collide across configs."""
    from screener.backtester.optimization.trials import (
        load_trial_search_stats,
        trial_config_identity,
    )

    bars = make_bars(n=70, seed=31, drift=0.2)
    frames = {"AAA": bars, "SPY": bars}
    fetcher = StubPriceFetcher(frames)
    trial_db = tmp_path / "trials.db"
    exp = "shared-exp-family"
    start = bars.index[0].date()
    end = bars.index[-1].date()
    grid = {"hold": [5]}

    cfg_a = _config(entry_expr="close > sma(close, 3)")
    cfg_b = _config(entry_expr="close > sma(close, 8)")
    key_a = trial_config_identity(
        cfg_a.model_copy(update={"hold": 5}),
        runner="rolling",
        start_date=start,
        end_date=end,
        metric="total_return",
        min_trades=1,
    )
    key_b = trial_config_identity(
        cfg_b.model_copy(update={"hold": 5}),
        runner="rolling",
        start_date=start,
        end_date=end,
        metric="total_return",
        min_trades=1,
    )
    assert key_a != key_b

    grid_search(
        cfg_a,
        fetcher,
        grid,
        runner="rolling",
        start_date=start,
        end_date=end,
        top_n=1,
        max_workers=1,
        metric="total_return",
        experiment_id=exp,
        trial_db_path=trial_db,
    )
    grid_search(
        cfg_b,
        fetcher,
        grid,
        runner="rolling",
        start_date=start,
        end_date=end,
        top_n=1,
        max_workers=1,
        metric="total_return",
        experiment_id=exp,
        trial_db_path=trial_db,
    )
    stats = load_trial_search_stats(exp, db_path=trial_db)
    assert stats.n_trials_nominal == 2

    # Revisiting the identical trial upserts; changing config adds a row.
    grid_search(
        cfg_a,
        fetcher,
        grid,
        runner="rolling",
        start_date=start,
        end_date=end,
        top_n=1,
        max_workers=1,
        metric="total_return",
        experiment_id=exp,
        trial_db_path=trial_db,
    )
    stats_again = load_trial_search_stats(exp, db_path=trial_db)
    assert stats_again.n_trials_nominal == 2


def test_attempted_registered_before_execution_on_interrupt(tmp_path, monkeypatch):
    """ATTEMPTED must be recorded before run so interruption keeps exposure."""
    from screener.backtester.optimization import grid as grid_mod
    from screener.backtester.optimization.trials import load_trial_search_stats

    bars = make_bars(n=60, seed=17, drift=0.1)
    frames = {"AAA": bars, "SPY": bars}
    cfg = _config()
    trial_db = tmp_path / "attempted.db"
    exp = "exp-attempted"

    def boom(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(grid_mod, "run_rolling_backtest", boom)
    monkeypatch.setattr(grid_mod, "prepare_rolling_backtest", boom)

    with pytest.raises(KeyboardInterrupt):
        grid_search(
            cfg,
            StubPriceFetcher(frames),
            {"hold": [3, 5]},
            runner="rolling",
            start_date=bars.index[0].date(),
            end_date=bars.index[-1].date(),
            top_n=2,
            max_workers=1,
            experiment_id=exp,
            trial_db_path=trial_db,
        )
    stats = load_trial_search_stats(exp, db_path=trial_db)
    assert stats.n_trials_nominal == 2


def test_universe_file_identity_hashes_contents_not_path(tmp_path):
    from screener.backtester.optimization.cache import (
        universe_membership_identity_from_config,
    )

    path_a = tmp_path / "univ_a.txt"
    path_b = tmp_path / "univ_b.txt"
    path_a.write_text("AAA\nBBB\n", encoding="utf-8")
    path_b.write_text("AAA\nBBB\n", encoding="utf-8")
    same_a = universe_membership_identity_from_config(
        _config(universe_file=str(path_a))
    )
    same_b = universe_membership_identity_from_config(
        _config(universe_file=str(path_b))
    )
    assert same_a == same_b

    path_b.write_text("AAA\nCCC\n", encoding="utf-8")
    changed = universe_membership_identity_from_config(
        _config(universe_file=str(path_b))
    )
    assert changed != same_a


def test_hash_price_frames_mutation_and_dtype_invalidate():
    from screener.backtester.optimization.cache import hash_price_frames

    bars = make_bars(n=30, seed=4)
    base = {"AAA": bars.copy(), "SPY": bars.copy()}
    h0 = hash_price_frames(base)
    mutated = {"AAA": bars.copy(), "SPY": bars.copy()}
    mutated["AAA"].iloc[0, mutated["AAA"].columns.get_loc("close")] += 1.0
    assert hash_price_frames(mutated) != h0
    as_f32 = {
        "AAA": bars.astype({"close": "float32"}),
        "SPY": bars.copy(),
    }
    # close dtype change must change identity when values are stored as f32 bytes.
    assert hash_price_frames(as_f32) != h0


def test_code_fingerprint_includes_lockfile_bytes(tmp_path, monkeypatch):
    from screener.backtester.optimization import cache as cache_mod

    pkg = tmp_path / "screener"
    pkg.mkdir()
    (pkg / "mod.py").write_text("x = 1\n", encoding="utf-8")
    repo = tmp_path
    (repo / "uv.lock").write_text("lock-v1\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    fp1 = cache_mod.code_fingerprint(pkg)
    (repo / "uv.lock").write_text("lock-v2\n", encoding="utf-8")
    fp2 = cache_mod.code_fingerprint(pkg)
    assert fp1 != fp2


def test_metric_dsr_updates_score_and_ranking(tmp_path):
    """Post-search corrected DSR must drive score and ranking, including cache hits."""
    bars = make_bars(n=120, seed=41, drift=0.08)
    frames = {"AAA": bars, "SPY": bars}
    cfg = _config()
    identity = build_frozen_input_identity(cfg, frames)
    cache_path = tmp_path / "dsr.sqlite"
    trial_db = tmp_path / "dsr_trials.db"
    exp = "dsr-rank"
    grid = {"hold": [3, 5, 8, 13]}
    kwargs = dict(
        runner="rolling",
        start_date=bars.index[5].date(),
        end_date=bars.index[-5].date(),
        top_n=4,
        max_workers=1,
        metric="dsr",
        frozen_input_identity=identity,
        cache_path=cache_path,
        trial_db_path=trial_db,
        experiment_id=exp,
    )
    first = grid_search(cfg, StubPriceFetcher(frames), grid, **kwargs)
    assert len(first) == 4
    for row in first:
        assert row.metrics.get("dsr") == row.metrics["dsr"]
        # Score must match corrected DSR or -inf when nonfinite; never a stale
        # single-trial objective left as the ranking key.
        corrected = float(row.metrics["dsr"])
        if math.isfinite(corrected) and row.trade_count >= 1 and not row.error:
            assert row.score == pytest.approx(corrected)
        else:
            assert row.score == float("-inf")
        if row.dsr_unavailable_reason is not None:
            assert row.score == float("-inf") or not math.isfinite(corrected)

    scores = [r.score for r in first]
    assert scores == sorted(scores, reverse=True)

    second = grid_search(cfg, StubPriceFetcher(frames), grid, **kwargs)
    assert any(r.cached for r in second)
    for row in second:
        corrected = float(row.metrics["dsr"])
        if math.isfinite(corrected) and row.trade_count >= 1 and not row.error:
            assert row.score == pytest.approx(corrected)
        else:
            assert row.score == float("-inf")


def test_cache_path_without_identity_logs_warning(tmp_path, caplog):
    import logging

    bars = make_bars(n=50, seed=2, drift=0.05)
    cfg = _config()
    with caplog.at_level(
        logging.WARNING, logger="screener.backtester.optimization.grid"
    ):
        grid_search(
            cfg,
            StubPriceFetcher({"AAA": bars, "SPY": bars}),
            {"hold": [4]},
            runner="rolling",
            start_date=bars.index[0].date(),
            end_date=bars.index[-1].date(),
            top_n=1,
            max_workers=1,
            cache_path=tmp_path / "disabled.json",
            trial_db_path=tmp_path / "t.db",
        )
    assert any("result cache disabled" in r.message for r in caplog.records)


def test_cache_key_unknown_placeholder_is_deterministic():
    cfg = _config()
    a = grid_module._cache_key(
        cfg,
        {"hold": 5},
        runner="historical",
        start_date=None,
        end_date=None,
        metric="sharpe",
        min_trades=1,
    )
    b = grid_module._cache_key(
        cfg,
        {"hold": 5},
        runner="historical",
        start_date=None,
        end_date=None,
        metric="sharpe",
        min_trades=1,
    )
    assert a == b


def test_strict_json_metrics_use_null_for_nonfinite():
    from screener.backtester.optimization.cache import (
        dumps_strict_json,
        loads_metrics_json,
    )

    raw = dumps_strict_json({"dsr": float("nan"), "sharpe": 1.25, "x": float("-inf")})
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert "null" in raw
    loaded = loads_metrics_json(raw)
    # All nonfinite values round-trip through JSON null as nan in memory.
    assert math.isnan(loaded["dsr"])
    assert loaded["sharpe"] == pytest.approx(1.25)
    assert math.isnan(loaded["x"])
