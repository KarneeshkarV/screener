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


# ---------------------------------------------------------------------------
# Fake vectorbt machinery
# ---------------------------------------------------------------------------


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
    monkeypatch.setattr(vs, "_require_vectorbt", lambda: fake)

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

    # Register the ``.vbt`` accessor on DataFrame.
    try:
        pd.api.extensions.register_dataframe_accessor("vbt")(_VbtAccessor)
    except Exception:  # pragma: no cover - already registered
        pass
    return fake


# ---------------------------------------------------------------------------
# Synthetic panels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pure helpers (no vbt needed)
# ---------------------------------------------------------------------------


def test_require_vectorbt_raises_when_absent(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vectorbt":
            raise ImportError("no vectorbt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import click

    with pytest.raises(click.ClickException):
        vs._require_vectorbt()


def test_require_vectorbt_success(monkeypatch):
    # Place a fake ``vectorbt`` module in sys.modules so the genuine
    # ``_require_vectorbt`` import succeeds and returns it (covers the return).
    fake = types.ModuleType("vectorbt")
    monkeypatch.setitem(sys.modules, "vectorbt", fake)
    assert vs._require_vectorbt() is fake


def test_iter_param_combos_direct():
    # Include a slow<=fast pair (100<=100, 50<=100) to exercise the skip branch.
    combos = vs.iter_param_combos([50, 100], [50, 100], [0])
    assert all(slow > fast for fast, slow, _h in combos)
    assert (50, 100, 0) in combos
    assert (100, 50, 0) not in combos


def test_parse_walk_forward_direct():
    import click

    assert vs.parse_walk_forward("12:3") == (12, 3)
    with pytest.raises(click.UsageError):
        vs.parse_walk_forward("12")
    with pytest.raises(click.UsageError):
        vs.parse_walk_forward("a:b")
    with pytest.raises(click.UsageError):
        vs.parse_walk_forward("0:3")


def test_single_combo_sweep_kwargs_unknown():
    with pytest.raises(ValueError):
        vs._single_combo_sweep_kwargs("nope", 1, 2, 3)


def test_parse_int_list_errors():
    import click

    with pytest.raises(click.UsageError):
        vs.parse_int_list("", name="fast")
    with pytest.raises(click.UsageError):
        vs.parse_int_list("a,b", name="fast")


def test_parse_indicator_list_variants():
    import click

    assert vs.parse_indicator_list("all") == list(vs.VALID_INDICATORS)
    # dedupe preserves order
    assert vs.parse_indicator_list("sma, sma, ema") == ["sma", "ema"]
    with pytest.raises(click.UsageError):
        vs.parse_indicator_list("")
    with pytest.raises(click.UsageError):
        vs.parse_indicator_list("not_a_real_indicator")


def test_fixed_hold_exits_paths():
    idx = pd.bdate_range("2021-01-04", periods=5)
    cols = ["AAA", "BBB"]
    entries = pd.DataFrame(False, index=idx, columns=cols)
    entries.iloc[0, 0] = True
    entries.iloc[3, 1] = True
    # hold=0 -> all False
    zero = vs._fixed_hold_exits(entries, 0)
    assert not zero.to_numpy().any()
    # no entries -> all False
    empty = vs._fixed_hold_exits(pd.DataFrame(False, index=idx, columns=cols), 2)
    assert not empty.to_numpy().any()
    # hold=2 -> exit two bars after each entry that stays in range
    held = vs._fixed_hold_exits(entries, 2)
    assert held.iloc[2, 0]  # entry row 0 + 2
    # entry at row 3 + 2 = row 5 is out of range -> dropped
    assert not held.iloc[:, 1].to_numpy().any()


def test_sma_non_multiindex_branch():
    # A vbt whose MA.run returns a plain (non-MultiIndex) columns frame, to
    # exercise the non-MultiIndex return in ``_sma``.
    idx = pd.bdate_range("2021-01-04", periods=10)
    close = pd.DataFrame(
        {"AAA": np.arange(10.0), "BBB": np.arange(10.0) + 1}, index=idx
    )

    class _PlainMA:
        @staticmethod
        def run(c, window):  # noqa: ANN001
            return types.SimpleNamespace(ma=c.rolling(int(window)).mean())

    fake = types.SimpleNamespace(MA=_PlainMA)
    out = vs._sma(close, 3, fake)
    assert list(out.columns) == ["AAA", "BBB"]


def test_run_combo_backtest_scalar_trade_count(monkeypatch):
    # A portfolio whose trades.count() returns a scalar -> exercises the
    # ``else: int(trade_count)`` branch in run_combo_backtest.
    idx = pd.bdate_range("2021-01-04", periods=10)
    close = pd.DataFrame({"AAA": np.arange(10.0) + 1}, index=idx)

    class _ScalarTrades:
        def win_rate(self):
            return 0.5

        def count(self):
            return 2  # plain int

    class _ScalarPf:
        trades = _ScalarTrades()

        def sharpe_ratio(self):
            return 1.0

        def total_return(self):
            return 0.1

        def calmar_ratio(self):
            return 0.5

        def max_drawdown(self):
            return -0.1

    class _ScalarPortfolio:
        @staticmethod
        def from_signals(*a, **k):  # noqa: ANN001
            return _ScalarPf()

    def _entries_exits(c, fast, slow, hold, vbt):  # noqa: ANN001
        e = pd.DataFrame(False, index=c.index, columns=c.columns)
        e.iloc[2] = True
        x = pd.DataFrame(False, index=c.index, columns=c.columns)
        return e, x

    monkeypatch.setitem(vs.STRATEGY_BUILDERS, "sma_cross", _entries_exits)
    fake = types.SimpleNamespace(Portfolio=_ScalarPortfolio)
    res = vs.run_combo_backtest(close, 3, 5, 0, vbt=fake)
    assert res["trades"] == 2


def test_ma_for_window_non_multiindex():
    idx = pd.bdate_range("2021-01-04", periods=3)
    panel = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    out = vs._ma_for_window(panel, 5)
    pd.testing.assert_frame_equal(out, panel)


def test_scalar_metric_variants():
    assert np.isnan(vs._scalar_metric(None))
    assert vs._scalar_metric(pd.Series([2.0, 3.0])) == 2.0
    assert np.isnan(vs._scalar_metric(pd.Series([], dtype=float)))
    assert vs._scalar_metric(np.array([7.0, 8.0])) == 7.0
    assert np.isnan(vs._scalar_metric(np.array([], dtype=float)))
    assert vs._scalar_metric(5) == 5.0


def test_combo_metric_scalar_and_series():
    s = pd.Series(
        [1.0, 2.0],
        index=pd.MultiIndex.from_tuples(
            [("sma", 10, 50, 0), ("ema", 10, 50, 0)],
            names=["indicator", "fast", "slow", "hold"],
        ),
    )
    assert vs._combo_metric(s, "ema", 10, 50, 0) == 2.0
    assert vs._combo_metric(3.0, "x", 1, 2, 3) == 3.0


def test_build_column_panel_no_usable_raises():
    with pytest.raises(ValueError):
        vs.build_close_panel(
            {"AAA": pd.DataFrame()},
            ["AAA"],
            start=pd.Timestamp("2021-01-01"),
            end=pd.Timestamp("2021-02-01"),
        )
    # missing column / out-of-window also raise
    df = make_bars(start="2021-01-04", n=10)
    with pytest.raises(ValueError):
        vs._build_column_panel(
            {"AAA": df},
            ["AAA"],
            column="close",
            start=pd.Timestamp("2030-01-01"),
            end=pd.Timestamp("2030-02-01"),
        )


def test_build_column_panel_skips_missing_symbol():
    df = make_bars(start="2021-01-04", n=20)
    panel = vs.build_high_panel(
        {"AAA": df},
        ["AAA", "ZZZ"],  # ZZZ absent -> skipped via continue
        start=pd.Timestamp("2021-01-04"),
        end=pd.Timestamp("2021-03-01"),
    )
    assert list(panel.columns) == ["AAA"]


def test_indicator_helpers_numeric():
    p = _panels(40)
    rsi = vs._rsi_wilder(p["close"], 14)
    assert rsi.shape == p["close"].shape
    atr = vs._atr_wilder(p["high"], p["low"], p["close"], 14)
    assert atr.shape == p["close"].shape
    obv = vs._obv(p["close"], p["volume"])
    assert obv.shape == p["close"].shape
    # supertrend with period >= n exercises the early-return branch
    n = p["close"].shape[0]
    e, x = vs._supertrend_signals_np(
        p["close"].to_numpy(),
        p["high"].to_numpy(),
        p["low"].to_numpy(),
        n + 5,
        3.0,
    )
    assert not e.any() and not x.any()
    # normal supertrend with small period exercises the state machine
    e2, x2 = vs._supertrend_signals_np(
        p["close"].to_numpy(),
        p["high"].to_numpy(),
        p["low"].to_numpy(),
        7,
        3.0,
    )
    assert e2.shape == p["close"].shape


def test_iter_indicator_combos_all_and_unknown():
    combos = vs.iter_indicator_combos(
        list(vs.VALID_INDICATORS),
        [10, 20],
        [50, 100],
        [0, 5],
    )
    inds = {c[0] for c in combos}
    assert inds == set(vs.VALID_INDICATORS)
    with pytest.raises(ValueError):
        vs.iter_indicator_combos(["bogus"], [10], [50], [0])


def test_single_combo_sweep_kwargs_macd_and_window():
    macd = vs._single_combo_sweep_kwargs("macd", 12, 26, 0)
    assert macd["indicators"] == ["macd"]
    assert "breakout_windows" not in macd
    bb = vs._single_combo_sweep_kwargs("bbands", 20, 0, 5)
    assert bb["bbands_windows"] == [20]


def test_slice_panel_none_and_value():
    assert vs._slice_panel(None, date(2021, 1, 1), date(2021, 2, 1)) is None
    p = _panels(30)["close"]
    sliced = vs._slice_panel(p, p.index[5].date(), p.index[10].date())
    assert sliced.shape[0] >= 1


def test_fmt_int_or_dash():
    assert vs._fmt_int_or_dash(3.0) == "3"
    assert vs._fmt_int_or_dash(float("nan")) == "—"
    assert vs._fmt_int_or_dash("x") == "—"


# ---------------------------------------------------------------------------
# vbt-backed paths (fake_vbt)
# ---------------------------------------------------------------------------


def test_sma_crossover_signals_hold_branches(fake_vbt):
    p = _panels(80)
    close = p["close"]
    ent, ex = vs.sma_crossover_signals(close, 5, 20, 0, fake_vbt)
    assert ent.shape == close.shape
    ent2, ex2 = vs.sma_crossover_signals(close, 5, 20, 3, fake_vbt)
    # hold>0 adds extra exits
    assert ex2.to_numpy().sum() >= ex.to_numpy().sum()


def test_run_combo_backtest_with_and_without_open(fake_vbt):
    p = _panels(80)
    res = vs.run_combo_backtest(
        p["close"], 5, 20, 3, vbt=fake_vbt, open_=p["open"], initial_capital=50_000
    )
    assert set(res) >= {"fast", "slow", "hold", "sharpe", "trades", "win_rate"}
    assert res["fast"] == 5 and res["slow"] == 20
    # without open_ (fill at close path)
    res2 = vs.run_combo_backtest(p["close"], 5, 20, 0, vbt=fake_vbt)
    assert "sharpe" in res2


def test_run_parameter_sweep_no_combos_raises(fake_vbt):
    p = _panels(60)
    with pytest.raises(ValueError):
        # slow <= fast for every pair -> empty combos
        vs.run_parameter_sweep(
            p["close"],
            fast_values=[50],
            slow_values=[10],
            hold_values=[0],
            indicators=["sma"],
            open_=p["open"],
        )


def test_run_parameter_sweep_default_indicator(fake_vbt):
    p = _panels(120)
    df = vs.run_parameter_sweep(
        p["close"],
        fast_values=[5, 10],
        slow_values=[20, 40],
        hold_values=[0, 5],
        open_=p["open"],
    )
    assert (df["indicator"] == "sma").all()
    assert {"sharpe", "total_return", "trades"}.issubset(df.columns)


def test_run_parameter_sweep_all_indicators(fake_vbt):
    p = _panels(160)
    df = vs.run_parameter_sweep(
        p["close"],
        fast_values=[5, 10],
        slow_values=[20, 40],
        hold_values=[3],
        indicators=list(vs.VALID_INDICATORS),
        breakout_windows=[10, 20],
        bbands_windows=[20],
        supertrend_periods=[7],
        keltner_windows=[20],
        rsi_thresholds=[50],
        obv_ema_windows=[20],
        high=p["high"],
        low=p["low"],
        volume=p["volume"],
        open_=p["open"],
    )
    assert set(df["indicator"].unique()) == set(vs.VALID_INDICATORS)
    # breakout-family slow is NaN
    bsub = df[df["indicator"] == "breakout"]
    assert bsub["slow"].isna().all()


def test_run_parameter_sweep_explicit_chunk_size(fake_vbt):
    p = _panels(120)
    df = vs.run_parameter_sweep(
        p["close"],
        fast_values=[5, 10],
        slow_values=[20, 40],
        hold_values=[0],
        indicators=["sma"],
        open_=p["open"],
        chunk_size=1,
    )
    assert len(df) == len(vs.iter_indicator_combos(["sma"], [5, 10], [20, 40], [0]))


def test_run_parameter_sweep_scalar_chunk_metrics(fake_vbt, monkeypatch):
    # Force the chunk metrics to be plain scalars (not Series) so the ``_concat``
    # fallback (``cleaned`` empty -> ``parts[0]``) branch is exercised.
    def fake_chunk(
        close, fill_price, entries_chunk, exits_chunk, *, vbt, initial_capital
    ):  # noqa: ANN001
        return (0.5, 0.1, 0.4, -0.1, 0.5, 3)

    monkeypatch.setattr(vs, "_portfolio_chunk_metrics", fake_chunk)
    p = _panels(80)
    df = vs.run_parameter_sweep(
        p["close"],
        fast_values=[5],
        slow_values=[20],
        hold_values=[0],
        indicators=["sma"],
        open_=p["open"],
    )
    assert len(df) == 1
    assert df.iloc[0]["sharpe"] == 0.5
    assert df.iloc[0]["trades"] == 3


def test_build_indicator_signal_panels_requires_hl(fake_vbt):
    p = _panels(60)
    combos = [("supertrend", 7, 0, 0)]
    with pytest.raises(ValueError):
        vs._build_indicator_signal_panels(p["close"], combos, vbt=fake_vbt)


def test_build_indicator_signal_panels_requires_volume(fake_vbt):
    p = _panels(60)
    combos = [("obv_trend", 20, 0, 0)]
    with pytest.raises(ValueError):
        vs._build_indicator_signal_panels(p["close"], combos, vbt=fake_vbt)


def test_build_indicator_signal_panels_unknown(fake_vbt):
    p = _panels(60)
    combos = [("bogus", 1, 0, 0)]
    with pytest.raises(ValueError):
        vs._build_indicator_signal_panels(p["close"], combos, vbt=fake_vbt)


# ---------------------------------------------------------------------------
# Printing + walk-forward
# ---------------------------------------------------------------------------


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


def test_rank_results_unknown_metric():
    with pytest.raises(ValueError):
        vs.rank_results(_results_df(), "bogus")  # type: ignore[arg-type]


def test_print_results_table_with_and_without_indicator(capsys):
    from rich.console import Console

    df = _results_df()
    vs.print_results_table(df, top_n=5, metric="sharpe", console=Console())
    # branch where there is no indicator column
    df2 = df.drop(columns="indicator")
    vs.print_results_table(df2, top_n=5, metric="sharpe")
    out = capsys.readouterr().out
    assert "Top" in out


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


# ---------------------------------------------------------------------------
# CLI paths (Click runner, stubbed sweep + stub fetcher)
# ---------------------------------------------------------------------------


def _cli_env() -> StubPriceFetcher:
    a = make_bars(start="2022-01-03", n=600, seed=1, open_base=100.0)
    b = make_bars(start="2022-01-03", n=600, seed=2, open_base=50.0)
    spy = make_bars(start="2022-01-03", n=600, seed=3, open_base=400.0)
    return StubPriceFetcher({"AAA": a, "BBB": b, "SPY": spy})


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

    monkeypatch.setattr(vs, "load_current_universe", lambda *a, **k: _Loaded())
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

    monkeypatch.setattr(vs, "load_current_universe", lambda *a, **k: _Loaded())
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


def test_cli_walk_forward_csv(monkeypatch):
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
            "--csv",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output
    assert "is_score" in res.output


def test_cli_panel_value_errors(monkeypatch):
    from click.testing import CliRunner
    from main import cli

    # Force the open / high / low / volume panel builders to raise ValueError so
    # the CLI's fallback (set panel to None) branches are exercised.
    monkeypatch.setattr(
        vs, "build_open_panel", lambda *a, **k: (_ for _ in ()).throw(ValueError())
    )
    monkeypatch.setattr(
        vs, "build_high_panel", lambda *a, **k: (_ for _ in ()).throw(ValueError())
    )
    monkeypatch.setattr(
        vs, "build_low_panel", lambda *a, **k: (_ for _ in ()).throw(ValueError())
    )
    monkeypatch.setattr(
        vs, "build_volume_panel", lambda *a, **k: (_ for _ in ()).throw(ValueError())
    )

    captured = {}

    def fake_sweep(close, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
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
            "supertrend,vol_breakout",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code == 0, res.output
    assert captured["open_"] is None
    assert captured["high"] is None
    assert captured["volume"] is None


def test_cli_end_before_start_errors():
    from click.testing import CliRunner
    from main import cli

    res = CliRunner().invoke(
        cli,
        [
            "vbt-sweep",
            "--tickers",
            "AAA",
            "--start",
            "2023-01-01",
            "--end",
            "2022-01-01",
        ],
        obj=_cli_env(),
    )
    assert res.exit_code != 0
    assert "--end must be on or after --start" in res.output
