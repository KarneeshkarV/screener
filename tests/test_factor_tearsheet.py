"""Unit tests for IC / quantile factor tearsheet pure math."""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from rich.console import Console

import screener.backtester.factor_tearsheet as ft
from screener.backtester.factor_tearsheet import (
    ICSummary,
    QuantileResult,
    analyze_horizon,
    build_score_and_close_matrices,
    daily_spearman_ic,
    forward_returns,
    quantile_mean_returns,
    summarize_ic,
    top_quantile_turnover,
)


def _dates(n: int = 30) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


def test_forward_returns_no_lookahead_on_score_alignment() -> None:
    idx = _dates(5)
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=idx)
    fwd = forward_returns(close, horizon=1)
    # Return from t0 to t1 uses close[t1]/close[t0]-1; last row NaN.
    assert fwd.iloc[0, 0] == pytest.approx(0.10)
    assert math.isnan(fwd.iloc[-1, 0])


def test_positive_ic_for_predictive_factor() -> None:
    """Construct scores that perfectly rank next-day returns → IC ~ 1."""
    idx = _dates(40)
    # Three names; next-day return order is always A > B > C.
    # close paths: A grows fastest, C slowest.
    rng = np.arange(len(idx), dtype=float)
    close = pd.DataFrame(
        {
            "A": 100 * (1.02**rng),
            "B": 100 * (1.01**rng),
            "C": 100 * (1.005**rng),
        },
        index=idx,
    )
    # Score at t equals the *realized* 1-day forward return (oracle) — IC should
    # be near +1. This is only for unit-test synthetic data.
    fwd = forward_returns(close, 1)
    scores = fwd.copy()  # perfect foresight scores
    # Drop the last NaN row for a fair series.
    ic = daily_spearman_ic(scores.iloc[:-1], fwd.iloc[:-1])
    summary = summarize_ic(ic, horizon=1)
    assert summary.n_days > 10
    assert summary.ic_mean == pytest.approx(1.0, abs=1e-9)
    assert summary.pct_positive == pytest.approx(1.0)


def test_quantile_top_minus_bottom_positive() -> None:
    idx = _dates(50)
    # Scores rank names A > B > C > D > E every day; returns follow the same order.
    tickers = list("ABCDE")
    scores = pd.DataFrame(
        {t: float(i) for i, t in enumerate(tickers)},
        index=idx,
    )
    # Constant cross-section of scores every day.
    for t in tickers:
        scores[t] = float(ord(t) - ord("A"))
    fwd = pd.DataFrame(
        {t: float(ord(t) - ord("A")) * 0.01 for t in tickers},
        index=idx,
    )
    means, spread = quantile_mean_returns(scores, fwd, n_quantiles=5)
    assert spread > 0
    assert means[5] > means[1]


def test_top_quantile_turnover_full_churn() -> None:
    idx = _dates(4)
    # Alternate which name is the top score each day → turnover ≈ 1.
    scores = pd.DataFrame(
        {
            "A": [10.0, 1.0, 10.0, 1.0],
            "B": [1.0, 10.0, 1.0, 10.0],
        },
        index=idx,
    )
    # With only 2 names and 2 quantiles, top quantile is a singleton alternating.
    turnover = top_quantile_turnover(scores, n_quantiles=2)
    assert turnover == pytest.approx(1.0)


def test_analyze_horizon_bundle() -> None:
    idx = _dates(20)
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 120, len(idx)),
            "B": np.linspace(100, 110, len(idx)),
            "C": np.linspace(100, 105, len(idx)),
            "D": np.linspace(100, 102, len(idx)),
        },
        index=idx,
    )
    scores = pd.DataFrame(
        {
            "A": 4.0,
            "B": 3.0,
            "C": 2.0,
            "D": 1.0,
        },
        index=idx,
    )
    summary, qres, ic = analyze_horizon(scores, close, horizon=1, n_quantiles=4)
    assert summary.horizon == 1
    assert qres.n_quantiles == 4
    assert len(ic) == len(idx)


def test_build_score_and_close_matrices() -> None:
    idx = _dates(3)
    bars = {
        "AAA": pd.DataFrame(
            {
                "close": [1.0, 2.0, 3.0],
                "rank_score": [0.5, 0.6, 0.7],
            },
            index=idx,
        ),
        "BBB": pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=idx),
    }
    scores, closes = build_score_and_close_matrices(bars)
    assert list(scores.columns) == ["AAA"]
    assert set(closes.columns) == {"AAA", "BBB"}


def test_math_validation_and_degenerate_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="horizon"):
        forward_returns(pd.DataFrame(), 0)
    with pytest.raises(ValueError, match="n_quantiles"):
        quantile_mean_returns(pd.DataFrame(), pd.DataFrame(), n_quantiles=1)
    with pytest.raises(ValueError, match="n_quantiles"):
        top_quantile_turnover(pd.DataFrame(), n_quantiles=1)

    assert daily_spearman_ic(pd.DataFrame(), pd.DataFrame()).empty
    idx = _dates(2)
    tiny = pd.DataFrame({"A": [1.0, np.nan], "B": [2.0, 2.0]}, index=idx)
    assert daily_spearman_ic(tiny, tiny).isna().all()

    empty = summarize_ic(pd.Series(dtype=float), horizon=3)
    assert empty.n_days == 0 and math.isnan(empty.ic_mean)
    single = summarize_ic(pd.Series([0.25]), horizon=1)
    assert single.ic_mean == pytest.approx(0.25)
    assert math.isnan(single.ic_std) and math.isnan(single.ic_ir)

    too_small = pd.Series([1.0, np.nan], index=["A", "B"])
    assert ft._quantile_labels(too_small, 2).isna().all()
    monkeypatch.setattr(
        pd, "qcut", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError())
    )
    assert ft._quantile_labels(pd.Series([1.0, 2.0]), 2).isna().all()
    monkeypatch.setattr(
        pd,
        "qcut",
        lambda *args, **kwargs: pd.Series([0, 0], index=args[0].index),
    )
    assert ft._quantile_labels(pd.Series([1.0, 2.0]), 2).isna().all()


def test_quantile_empty_results_and_turnover_reset() -> None:
    idx = _dates(3)
    scores = pd.DataFrame({"A": [2.0, np.nan, 2.0], "B": [1.0, np.nan, 1.0]}, index=idx)
    means, spread = quantile_mean_returns(
        scores, pd.DataFrame(np.nan, index=idx, columns=["A", "B"]), n_quantiles=2
    )
    assert all(math.isnan(value) for value in means.values())
    assert math.isnan(spread)
    assert math.isnan(top_quantile_turnover(scores, n_quantiles=2))


def test_matrix_builder_skips_bad_frames() -> None:
    scores, closes = build_score_and_close_matrices(
        {
            "NONE": None,  # type: ignore[dict-item]
            "EMPTY": pd.DataFrame(),
            "NO_CLOSE": pd.DataFrame({"rank_score": [1.0]}),
        }
    )
    assert scores.empty and closes.empty


@pytest.mark.parametrize(
    ("raw", "message"),
    [("", "at least one"), ("1,nope", "integers"), ("1,0", ">= 1")],
)
def test_parse_int_list_errors(raw: str, message: str) -> None:
    with pytest.raises(click.UsageError, match=message):
        ft._parse_int_list(raw, name="horizons")
    assert ft._parse_int_list("1, 5,21", name="horizons") == [1, 5, 21]


def test_resolve_tickers_all_sources(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    end = date(2024, 6, 30)
    assert ft._resolve_tickers(
        "us",
        tickers=" AAPL, MSFT ",
        universe=None,
        universe_file=None,
        end_date=end,
        no_universe_cache=False,
    ) == (("AAPL", "MSFT"), None)
    with pytest.raises(click.UsageError, match="tickers is empty"):
        ft._resolve_tickers(
            "us",
            tickers=",",
            universe=None,
            universe_file=None,
            end_date=end,
            no_universe_cache=False,
        )

    path = tmp_path / "universe.txt"
    path.write_text("# comment\nAAA\n\nBBB\n")
    symbols, note = ft._resolve_tickers(
        "us",
        tickers=None,
        universe=None,
        universe_file=str(path),
        end_date=end,
        no_universe_cache=False,
    )
    assert symbols == ("AAA", "BBB") and note == f"file:{path}"
    path.write_text("# only comment\n")
    with pytest.raises(click.UsageError, match="universe file is empty"):
        ft._resolve_tickers(
            "us",
            tickers=None,
            universe=None,
            universe_file=str(path),
            end_date=end,
            no_universe_cache=False,
        )

    monkeypatch.setattr(
        ft, "get_market", lambda market: SimpleNamespace(default_universe="sp500")
    )
    monkeypatch.setattr(
        ft,
        "load_current_universe",
        lambda *args, **kwargs: SimpleNamespace(
            name="sp500", symbols=("X",), source="stub", cached_path="cache.json"
        ),
    )
    symbols, note = ft._resolve_tickers(
        "us",
        tickers=None,
        universe=None,
        universe_file=None,
        end_date=end,
        no_universe_cache=True,
    )
    assert symbols == ("X",) and "cache=cache.json" in str(note)


class _PanelFetcher:
    def __init__(self, panel: dict[str, pd.DataFrame]) -> None:
        self.panel = panel
        self.calls: list[tuple[list[str], date, date]] = []

    def fetch(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        self.calls.append((symbols, start, end))
        return self.panel


def test_load_factor_panels_prepares_and_trims(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.bdate_range("2023-12-20", "2024-01-10")
    bars = pd.DataFrame({"close": 100.0, "rank_score": 1.0}, index=idx)
    fetcher = _PanelFetcher({"AAA": bars})
    monkeypatch.setattr(ft, "tv_to_yf", lambda tv, market: tv)
    monkeypatch.setattr(
        ft, "get_market", lambda market: SimpleNamespace(benchmark="SPY")
    )
    import screener.strategies.spec as spec

    monkeypatch.setattr(spec, "discover_plugins", lambda: None)
    resolved = SimpleNamespace(
        entry="entry",
        required_lookback=None,
        prepare_bars=None,
    )
    resolve_calls: list[str] = []

    def resolve(name: str):
        resolve_calls.append(name)
        return resolved

    monkeypatch.setattr(spec, "resolve_strategy_spec", resolve)
    scores, closes = ft.load_factor_panels(
        market="us",
        strategy_name="factor",
        tickers=("AAA",),
        start=date(2024, 1, 2),
        end=date(2024, 1, 8),
        fetcher=fetcher,
        warnings=[],
    )
    assert scores.index.min() == pd.Timestamp("2024-01-02")
    assert closes.index.max() == pd.Timestamp("2024-01-08")
    assert fetcher.calls[0][1] < date(2023, 1, 2)
    assert resolve_calls == ["factor"]

    resolved.required_lookback = lambda: 10
    ft.load_factor_panels(
        market="us",
        strategy_name="combo:x=1",
        tickers=("AAA",),
        start=date(2024, 1, 2),
        end=date(2024, 1, 8),
        fetcher=fetcher,
        warnings=[],
    )
    assert resolve_calls == ["factor", "combo:x=1"]


def _sample_results() -> tuple[list[ICSummary], list[QuantileResult]]:
    return (
        [
            ICSummary(1, 0.1, 0.2, 0.5, 2.0, 0.75, 4),
            ICSummary(5, *(float("nan"),) * 5, 0),
        ],
        [
            QuantileResult(1, 2, {1: -0.01, 2: 0.02}, 0.03, 0.5),
            QuantileResult(
                5, 2, {1: float("nan"), 2: float("nan")}, float("nan"), float("nan")
            ),
        ],
    )


def test_output_serialization_rendering_and_csv(
    tmp_path: pytest.TempPathFactory,
) -> None:
    summaries, quantiles = _sample_results()
    payload = ft.tearsheet_to_dict(
        strategy="factor",
        market="us",
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        quantiles=2,
        ic_summaries=summaries,
        quantile_results=quantiles,
        warnings=["stub warning"],
    )
    assert payload["ic"][1]["ic_mean"] is None
    assert payload["quantiles_by_horizon"][1]["top_minus_bottom"] is None
    assert ft._finite_or_none(None) is None  # type: ignore[arg-type]

    console = Console(record=True, width=120)
    ft.print_tearsheet(
        strategy="factor",
        ic_summaries=summaries,
        quantile_results=quantiles,
        warnings=["stub warning"],
        console=console,
    )
    rendered = console.export_text()
    assert "Factor IC" in rendered and "Top − Bottom" in rendered
    assert "stub warning" in rendered and "n/a" in rendered

    path = tmp_path / "tearsheet.csv"
    ft.write_tearsheet_csv(path, summaries, quantiles)
    frame = pd.read_csv(path)
    assert {"ic", "quantile"} == set(frame["section"])
    assert "top_quantile_turnover" in set(frame["metric"])


def test_factor_tearsheet_cli_success_and_errors(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    idx = _dates(4)
    scores = pd.DataFrame({"A": [2.0] * 4, "B": [1.0] * 4, "C": [0.0] * 4}, index=idx)
    closes = pd.DataFrame(
        {"A": [1, 2, 3, 4], "B": [1, 1, 1, 1], "C": [4, 3, 2, 1]}, index=idx
    )
    monkeypatch.setattr(
        ft,
        "_resolve_tickers",
        lambda *args, **kwargs: (("A", "B", "C"), "stub universe"),
    )
    monkeypatch.setattr(ft, "get_price_fetcher", lambda *args, **kwargs: object())
    monkeypatch.setattr(ft, "load_factor_panels", lambda **kwargs: (scores, closes))
    monkeypatch.setattr(ft, "print_tearsheet", lambda **kwargs: None)
    csv_path = tmp_path / "nested" / "out.csv"
    json_path = tmp_path / "nested" / "out.json"
    result = runner.invoke(
        ft.factor_tearsheet,
        [
            "--strategy",
            "factor",
            "--tickers",
            "A,B,C",
            "--start",
            "2024-01-02",
            "--end",
            "2024-01-05",
            "--horizons",
            "1",
            "--quantiles",
            "3",
            "--csv",
            str(csv_path),
            "--json",
            str(json_path),
        ],
        obj={},
    )
    assert result.exit_code == 0, result.output
    assert csv_path.exists() and json_path.exists()
    assert "Wrote CSV" in result.output and "Wrote JSON" in result.output

    assert (
        runner.invoke(
            ft.factor_tearsheet, ["--strategy", "x", "--quantiles", "1"]
        ).exit_code
        == 2
    )
    assert (
        runner.invoke(
            ft.factor_tearsheet,
            ["--strategy", "x", "--start", "2024-02-01", "--end", "2024-01-01"],
        ).exit_code
        == 2
    )

    monkeypatch.setattr(
        ft, "load_factor_panels", lambda **kwargs: (pd.DataFrame(), closes)
    )
    empty_scores = runner.invoke(
        ft.factor_tearsheet, ["--strategy", "x", "--tickers", "A"]
    )
    assert empty_scores.exit_code == 2 and "no rank_score" in empty_scores.output
    monkeypatch.setattr(
        ft, "load_factor_panels", lambda **kwargs: (scores, pd.DataFrame())
    )
    empty_closes = runner.invoke(
        ft.factor_tearsheet, ["--strategy", "x", "--tickers", "A"]
    )
    assert empty_closes.exit_code == 2 and "no close prices" in empty_closes.output


def test_load_factor_panels_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown factor strategy"):
        ft.load_factor_panels(
            market="us",
            strategy_name="definitely-not-a-strategy",
            tickers=["AAA"],
            start=date(2024, 1, 1),
            end=date(2024, 3, 1),
            fetcher=None,
            warnings=[],
        )
