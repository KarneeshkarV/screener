"""Offline coverage tests for ``screener.backtester.vbt_sweep``.

vectorbt is an optional dependency and is **not** installed in CI. The
functions that call into vectorbt (``run_combo_backtest``,
``_build_indicator_signal_panels``, ``_portfolio_chunk_metrics``,
``run_parameter_sweep``) are exercised here against a small, numerically
faithful **fake vbt** that is injected via ``_require_vectorbt`` and a
fake ``vectorbt.generic.nb`` module in ``sys.modules``.

Everything is deterministic and offline; no network, no real vectorbt.
"""

from __future__ import annotations


import sys


import types


import warnings


from datetime import date


import numpy as np


import pandas as pd


import pytest


import screener.backtester.vbt_sweep as vs


from screener.backtester.optimization.walk_forward import (
    WalkForwardWindow,
    generate_walk_forward_windows,
)


from tests.conftest import StubPriceFetcher, make_bars


def _crossed_above_nb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` crosses strictly above ``b`` (prev a<=b, now a>b). 2D arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros(a.shape, dtype=bool)
    prev_a = a[:-1]
    prev_b = b[:-1]
    cur_a = a[1:]
    cur_b = b[1:]
    with np.errstate(invalid="ignore"):
        crossed = (prev_a <= prev_b) & (cur_a > cur_b)
    crossed = crossed & np.isfinite(prev_a) & np.isfinite(prev_b)
    crossed = crossed & np.isfinite(cur_a) & np.isfinite(cur_b)
    out[1:] = crossed
    return out


class _MARun:
    def __init__(self, ma: pd.DataFrame) -> None:
        self.ma = ma


class _FakeMA:
    @staticmethod
    def run(close: pd.DataFrame, window):  # noqa: ANN001
        if isinstance(window, (list, tuple)):
            frames = {}
            for w in window:
                rolled = close.rolling(int(w)).mean()
                for col in close.columns:
                    frames[(w, col)] = rolled[col]
            ma = pd.DataFrame(frames)
            ma.columns = pd.MultiIndex.from_tuples(
                list(frames.keys()), names=["ma_window", None]
            )
            return _MARun(ma)
        # Real vbt tags single-window output with an ``ma_window`` level too;
        # mirror that so the MultiIndex branch in ``_sma`` is exercised.
        ma = close.rolling(int(window)).mean()
        ma.columns = pd.MultiIndex.from_tuples(
            [(int(window), c) for c in ma.columns], names=["ma_window", None]
        )
        return _MARun(ma)


def _metric_series(group_names, group_index, fn):  # noqa: ANN001
    """Build a Series indexed by a MultiIndex of ``group_index`` tuples."""
    mi = pd.MultiIndex.from_tuples(group_index, names=group_names)
    return pd.Series([fn(g) for g in group_index], index=mi)


class _Trades:
    def __init__(self, portfolio) -> None:  # noqa: ANN001
        self._pf = portfolio

    def win_rate(self):
        return self._pf._reduce(self._pf._win_rate_of)

    def count(self):
        return self._pf._reduce(self._pf._count_of)


class _FakePortfolio:
    """Minimal Portfolio that computes simple, deterministic metrics.

    Returns are computed as a naive long-only mark-to-market over the close
    panel using the (shifted) entries/exits as a binary holding mask. Grouped
    portfolios (``group_by`` is a list) return Series keyed by the group
    MultiIndex so the production reduction paths are exercised.
    """

    def __init__(self, close, entries, exits, group_by):  # noqa: ANN001
        self._close = close
        self._entries = entries.astype(bool)
        self._exits = exits.astype(bool)
        self._group_by = group_by
        if isinstance(group_by, list):
            # Distinct group labels in column order.
            tuples = [tuple(col[: len(group_by)]) for col in close.columns]
            seen: list = []
            for t in tuples:
                if t not in seen:
                    seen.append(t)
            self._groups = seen
            self._single = False
        else:
            # group_by=True -> one combined group. Return single-element Series
            # so the Series-reduction branch in run_combo_backtest is exercised.
            self._groups = None
            self._single = True

    @classmethod
    def from_signals(
        cls,
        close,  # noqa: ANN001
        entries,  # noqa: ANN001
        exits,  # noqa: ANN001
        *,
        price=None,
        init_cash=0.0,
        fees=0.0,
        slippage=0.0,
        group_by=True,
        cash_sharing=True,
        freq="1D",
    ):
        return cls(close, entries, exits, group_by)

    @property
    def trades(self):
        return _Trades(self)

    def _holding_mask(self, cols) -> np.ndarray:
        """Binary held-state mask for the given columns (forward-fill of entry
        until an exit). Returns array of shape (n_days, n_cols)."""
        ent = self._entries[cols].to_numpy(dtype=bool)
        ex = self._exits[cols].to_numpy(dtype=bool)
        held = np.zeros(ent.shape, dtype=bool)
        state = np.zeros(ent.shape[1], dtype=bool)
        for i in range(ent.shape[0]):
            state = state & ~ex[i]
            state = state | ent[i]
            held[i] = state
        return held

    def _group_cols(self, group):  # noqa: ANN001
        if self._groups is None:
            return list(self._close.columns)
        n = len(self._group_by)
        return [c for c in self._close.columns if tuple(c[:n]) == group]

    def _ret_of(self, group) -> float:  # noqa: ANN001
        cols = self._group_cols(group)
        held = self._holding_mask(cols)
        close = self._close[cols].to_numpy(dtype=float)
        rets = np.zeros(close.shape)
        rets[1:] = (close[1:] - close[:-1]) / close[:-1]
        rets = np.where(np.isfinite(rets), rets, 0.0)
        port = (held[:-1] * rets[1:]).mean(axis=1) if held.shape[1] else np.zeros(0)
        return float(np.prod(1.0 + port) - 1.0) if port.size else 0.0

    def _count_of(self, group) -> int:  # noqa: ANN001
        cols = self._group_cols(group)
        return int(self._entries[cols].to_numpy(dtype=bool).sum())

    def _win_rate_of(self, group) -> float:  # noqa: ANN001
        cnt = self._count_of(group)
        if cnt == 0:
            return float("nan")
        return 0.5

    def _reduce(self, fn):  # noqa: ANN001
        if self._groups is None:
            return pd.Series([fn(None)], index=["group"])
        return _metric_series(list(self._group_by), self._groups, fn)

    def sharpe_ratio(self):
        return self._reduce(lambda g: self._ret_of(g) * 2.0)

    def total_return(self):
        return self._reduce(self._ret_of)

    def calmar_ratio(self):
        return self._reduce(lambda g: self._ret_of(g) * 1.5)

    def max_drawdown(self):
        return self._reduce(lambda g: -abs(self._ret_of(g)) * 0.1)


class _VbtAccessor:
    """Implements ``df.vbt.crossed_above`` / ``crossed_below``."""

    def __init__(self, obj) -> None:  # noqa: ANN001
        self._obj = obj

    def crossed_above(self, other):  # noqa: ANN001
        a = self._obj.to_numpy(dtype=float)
        b = np.asarray(other.to_numpy(dtype=float), dtype=float)
        res = _crossed_above_nb(a, b)
        return pd.DataFrame(res, index=self._obj.index, columns=self._obj.columns)

    def crossed_below(self, other):  # noqa: ANN001
        a = self._obj.to_numpy(dtype=float)
        b = np.asarray(other.to_numpy(dtype=float), dtype=float)
        res = _crossed_above_nb(b, a)
        return pd.DataFrame(res, index=self._obj.index, columns=self._obj.columns)


def _make_fake_vbt() -> types.SimpleNamespace:
    return types.SimpleNamespace(MA=_FakeMA, Portfolio=_FakePortfolio)


@pytest.fixture
def fake_vbt(monkeypatch):
    """Install a fake vbt + fake ``vectorbt.generic.nb`` and a ``.vbt`` accessor."""
    fake = _make_fake_vbt()
    monkeypatch.setattr("screener.backtester.vbt.sweep._require_vectorbt", lambda: fake)
    monkeypatch.setattr("screener.backtester.vbt.cli._require_vectorbt", lambda: fake)

    # Fake ``from vectorbt.generic.nb import crossed_above_nb``.
    nb_mod = types.ModuleType("vectorbt.generic.nb")
    nb_mod.crossed_above_nb = _crossed_above_nb
    generic_mod = types.ModuleType("vectorbt.generic")
    generic_mod.nb = nb_mod
    root_mod = types.ModuleType("vectorbt")
    root_mod.generic = generic_mod
    monkeypatch.setitem(sys.modules, "vectorbt", root_mod)
    monkeypatch.setitem(sys.modules, "vectorbt.generic", generic_mod)
    monkeypatch.setitem(sys.modules, "vectorbt.generic.nb", nb_mod)

    # Register the ``.vbt`` accessor on DataFrame, keeping any existing
    # registration (real vectorbt's, when the extra is installed) so teardown
    # can restore it instead of leaking the fake into later tests.
    prev_accessor = pd.DataFrame.__dict__.get("vbt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pd.api.extensions.register_dataframe_accessor("vbt")(_VbtAccessor)
    yield fake
    if prev_accessor is None:
        delattr(pd.DataFrame, "vbt")
    else:
        setattr(pd.DataFrame, "vbt", prev_accessor)


def _panels(n: int = 120, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    cols = ["AAA", "BBB"]
    close = pd.DataFrame(
        {
            c: 100.0 + i * 5 + np.cumsum(rng.normal(0.05, 1.0, n))
            for i, c in enumerate(cols)
        },
        index=idx,
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    noise = pd.DataFrame(
        rng.uniform(0.5, 2.0, size=close.shape), index=idx, columns=cols
    )
    high = close + noise
    low = close - noise
    volume = pd.DataFrame(
        rng.uniform(1e6, 5e6, size=close.shape), index=idx, columns=cols
    )
    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
    }


def _results_df():
    return pd.DataFrame(
        {
            "indicator": ["sma", "ema"],
            "fast": [10, 20],
            "slow": [50.0, float("nan")],
            "hold": [0, 5],
            "sharpe": [1.2, float("nan")],
            "total_return": [0.1, -0.2],
            "calmar": [0.5, float("nan")],
            "max_drawdown": [-0.1, -0.3],
            "win_rate": [0.6, float("nan")],
            "trades": [5, 0],
        }
    )


def _wf_close(periods: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=periods)
    return pd.DataFrame(
        {
            "AAA": 100.0 + np.cumsum(rng.normal(0.05, 1.0, periods)),
            "BBB": 50.0 + np.cumsum(rng.normal(0.02, 0.8, periods)),
        },
        index=idx,
    )


def _stub_sweep_fn():
    def sweep(close, *, fast_values, slow_values, hold_values, indicators=None, **_):  # noqa: ANN001
        rows = []
        for fast in fast_values:
            for slow in slow_values:
                if slow <= fast:
                    continue
                for hold in hold_values:
                    score = (fast + slow) / 100.0
                    rows.append(
                        {
                            "indicator": (indicators or ["sma"])[0],
                            "fast": fast,
                            "slow": slow,
                            "hold": hold,
                            "sharpe": score,
                            "total_return": score / 10.0,
                            "calmar": score,
                            "max_drawdown": -0.1,
                            "win_rate": 0.5,
                            "trades": 4,
                        }
                    )
        return pd.DataFrame(rows)

    return sweep


def _cli_env() -> StubPriceFetcher:
    a = make_bars(start="2022-01-03", n=600, seed=1, open_base=100.0)
    b = make_bars(start="2022-01-03", n=600, seed=2, open_base=50.0)
    spy = make_bars(start="2022-01-03", n=600, seed=3, open_base=400.0)
    return StubPriceFetcher({"AAA": a, "BBB": b, "SPY": spy})


def test_run_walk_forward_sweep_skips_empty_window():
    close = _wf_close(400)
    windows = generate_walk_forward_windows(
        close.index[0].date(),
        close.index[-1].date(),
        train_days=360,
        test_days=90,
    )
    # Add a window that falls entirely outside the data -> skipped.
    bad = WalkForwardWindow(
        train_start=date(2099, 1, 1),
        train_end=date(2099, 2, 1),
        test_start=date(2099, 3, 1),
        test_end=date(2099, 4, 1),
    )
    summary = vs.run_walk_forward_sweep(
        close,
        windows=[bad, *windows],
        metric="sharpe",
        grid={
            "fast_values": [10, 20],
            "slow_values": [50, 100],
            "hold_values": [0],
            "indicators": ["sma"],
        },
        sweep_fn=_stub_sweep_fn(),
    )
    assert len(summary.windows) == len(windows)
    assert (summary.windows["fast"] == 20).all()


def test_run_walk_forward_sweep_no_windows():
    close = _wf_close(200)
    summary = vs.run_walk_forward_sweep(
        close,
        windows=[],
        metric="sharpe",
        grid={
            "fast_values": [10, 20],
            "slow_values": [50, 100],
            "hold_values": [0],
            "indicators": ["sma"],
        },
        sweep_fn=_stub_sweep_fn(),
    )
    assert summary.windows.empty
    assert summary.aggregate_is_score == 0.0
    assert summary.efficiency == 0.0


def test_print_walk_forward_sweep_table_empty(capsys):
    from rich.console import Console

    summary = vs.WalkForwardSweepSummary(
        metric="sharpe",
        windows=pd.DataFrame(),
        aggregate_is_score=0.0,
        aggregate_oos_score=0.0,
        efficiency=0.0,
        parameter_stability=0.0,
        aggregate_oos_metrics={},
    )
    vs.print_walk_forward_sweep_table(summary, console=Console())
    out = capsys.readouterr().out
    assert "No walk-forward windows" in out


def test_print_walk_forward_sweep_table_partial_agg(capsys):
    from rich.console import Console

    windows = pd.DataFrame(
        [
            {
                "train_start": date(2022, 1, 1),
                "train_end": date(2022, 6, 1),
                "test_start": date(2022, 6, 2),
                "test_end": date(2022, 9, 1),
                "indicator": "sma",
                "fast": 20,
                "slow": 100,
                "hold": 0,
                "is_score": 1.2,
                "oos_score": 0.6,
                "oos_total_return": 0.05,
                "oos_trades": 4,
            }
        ]
    )
    # aggregate_oos_metrics intentionally omits most WALK_FORWARD_OOS_COLUMNS
    # to exercise the ``continue`` skip branch.
    summary = vs.WalkForwardSweepSummary(
        metric="sharpe",
        windows=windows,
        aggregate_is_score=1.2,
        aggregate_oos_score=0.6,
        efficiency=0.5,
        parameter_stability=1.0,
        aggregate_oos_metrics={"sharpe": 0.6, "trades": 4.0},
    )
    vs.print_walk_forward_sweep_table(summary, console=Console())
    out = capsys.readouterr().out
    assert "sharpe=" in out


def test_print_walk_forward_sweep_table_populated(capsys):
    from rich.console import Console

    close = _wf_close(400)
    windows = generate_walk_forward_windows(
        close.index[0].date(),
        close.index[-1].date(),
        train_days=360,
        test_days=90,
    )
    summary = vs.run_walk_forward_sweep(
        close,
        windows=windows,
        metric="sharpe",
        grid={
            "fast_values": [10, 20],
            "slow_values": [50, 100],
            "hold_values": [0],
            "indicators": ["sma"],
        },
        sweep_fn=_stub_sweep_fn(),
    )
    vs.print_walk_forward_sweep_table(summary, console=Console())
    out = capsys.readouterr().out
    assert "Walk-Forward Sweep" in out
    assert "Aggregate OOS" in out


def test_cli_basic_table(monkeypatch):
    from click.testing import CliRunner
    from main import cli

    def fake_sweep(close, **kwargs):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "indicator": "sma",
                    "fast": 10,
                    "slow": 50,
                    "hold": 0,
                    "sharpe": 1.0,
                    "total_return": 0.1,
                    "calmar": 0.5,
                    "max_drawdown": -0.1,
                    "win_rate": 0.6,
                    "trades": 3,
                }
            ]
        )

    monkeypatch.setattr(vs, "run_parameter_sweep", fake_sweep)
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2022-06-01",
            "--end",
            "2023-06-01",
            "--indicator",
            "sma",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output
    assert "Window:" in res.output


def test_cli_csv_output(monkeypatch):
    from click.testing import CliRunner
    from main import cli

    def fake_sweep(close, **kwargs):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "indicator": "sma",
                    "fast": 10,
                    "slow": 50,
                    "hold": 0,
                    "sharpe": 1.0,
                    "total_return": 0.1,
                    "calmar": 0.5,
                    "max_drawdown": -0.1,
                    "win_rate": 0.6,
                    "trades": 3,
                }
            ]
        )

    monkeypatch.setattr(vs, "run_parameter_sweep", fake_sweep)
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2022-06-01",
            "--end",
            "2023-06-01",
            "--csv",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output
    assert "sharpe" in res.output


def test_cli_with_hl_volume_indicators(monkeypatch):
    from click.testing import CliRunner
    from main import cli

    captured = {}

    def fake_sweep(close, **kwargs):  # noqa: ANN001
        captured["high"] = kwargs.get("high")
        captured["volume"] = kwargs.get("volume")
        return pd.DataFrame(
            [
                {
                    "indicator": "supertrend",
                    "fast": 7,
                    "slow": float("nan"),
                    "hold": 0,
                    "sharpe": 1.0,
                    "total_return": 0.1,
                    "calmar": 0.5,
                    "max_drawdown": -0.1,
                    "win_rate": 0.6,
                    "trades": 3,
                }
            ]
        )

    monkeypatch.setattr(vs, "run_parameter_sweep", fake_sweep)
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2022-06-01",
            "--end",
            "2023-06-01",
            "--indicator",
            "supertrend,vol_breakout,keltner,obv_trend,breakout,bbands,macd,rsi",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output
    assert captured["high"] is not None
    assert captured["volume"] is not None


def test_cli_universe_file(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from main import cli

    uni = tmp_path / "u.txt"
    uni.write_text("# header\nAAA\nBBB\n")

    monkeypatch.setattr(
        vs,
        "run_parameter_sweep",
        lambda close, **kwargs: pd.DataFrame(
            [
                {
                    "indicator": "sma",
                    "fast": 10,
                    "slow": 50,
                    "hold": 0,
                    "sharpe": 1.0,
                    "total_return": 0.1,
                    "calmar": 0.5,
                    "max_drawdown": -0.1,
                    "win_rate": 0.6,
                    "trades": 3,
                }
            ]
        ),
    )
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--universe-file",
            str(uni),
            "--start",
            "2022-06-01",
            "--end",
            "2023-06-01",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output


def test_cli_universe_load(monkeypatch):
    from click.testing import CliRunner
    from main import cli

    class _Loaded:
        name = "sp500"
        symbols = ["AAA", "BBB"]
        source = "test"
        cached_path = "/tmp/x"

    monkeypatch.setattr(
        "screener.backtester.vbt.cli.load_current_universe", lambda *a, **k: _Loaded()
    )
    monkeypatch.setattr(
        vs,
        "run_parameter_sweep",
        lambda close, **kwargs: pd.DataFrame(
            [
                {
                    "indicator": "sma",
                    "fast": 10,
                    "slow": 50,
                    "hold": 0,
                    "sharpe": 1.0,
                    "total_return": 0.1,
                    "calmar": 0.5,
                    "max_drawdown": -0.1,
                    "win_rate": 0.6,
                    "trades": 3,
                }
            ]
        ),
    )
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--start",
            "2022-06-01",
            "--end",
            "2023-06-01",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output
    assert "Universe:" in res.output


def test_cli_walk_forward_window_overflow():
    from click.testing import CliRunner
    from main import cli

    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2022-01-03",
            "--end",
            "2022-04-01",
            "--walk-forward",
            "12:3",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code != 0
    assert "do not fit" in res.output


def test_cli_walk_forward_with_universe(monkeypatch):
    from click.testing import CliRunner
    from main import cli

    class _Loaded:
        name = "sp500"
        symbols = ["AAA", "BBB"]
        source = "test"
        cached_path = "/tmp/x"

    monkeypatch.setattr(
        "screener.backtester.vbt.cli.load_current_universe", lambda *a, **k: _Loaded()
    )
    monkeypatch.setattr(vs, "run_parameter_sweep", _stub_sweep_fn())
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--start",
            "2022-01-03",
            "--end",
            "2024-04-01",
            "--walk-forward",
            "12:3",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output
    assert "Universe:" in res.output


def test_cli_no_tickers_errors(monkeypatch):
    from click.testing import CliRunner
    from main import cli

    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            " , ",
            "--start",
            "2022-06-01",
            "--end",
            "2023-06-01",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code != 0


def test_cli_walk_forward(monkeypatch):
    from click.testing import CliRunner
    from main import cli

    monkeypatch.setattr(vs, "run_parameter_sweep", _stub_sweep_fn())
    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2022-01-03",
            "--end",
            "2024-04-01",
            "--walk-forward",
            "12:3",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output
    assert "Walk-forward:" in res.output
    assert "Walk-Forward Sweep" in res.output
