"""
Phase 6 – No-lookahead correctness suite.

Black-box perturbation tests: overwrite bars STRICTLY AFTER a decision
point T and assert every decision made at or before T is byte-identical.
A real lookahead bug causes a pre-T decision to change → caught here.

Run with:
    uv run pytest tests/correctness/test_lookahead_blindness.py -q
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.core import simulate_ticker
from screener.backtester.engine import (
    run_backtest,
    run_rolling_backtest,
    select_candidates,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import parse

from tests.conftest import StubPriceFetcher, make_bars


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ENTRY_EXPR = "close > sma(close, 3)"
_BENCH = "SPY"


def _cfg(**overrides) -> BacktestConfig:
    """Build a minimal BacktestConfig; override any field via kwargs."""
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=5,
        entry_expr=_ENTRY_EXPR,
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark=_BENCH,
        strategy_name=None,
        tickers=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _perturb_after(bars: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return a copy of bars where every row with index > cutoff is overwritten
    with obviously-garbage values (multiplied by 1000 for OHLC, 0 for volume).
    The perturbed frame keeps the same DatetimeIndex; only bar *values* change.
    """
    perturbed = bars.copy()
    mask = perturbed.index > cutoff
    if not mask.any():
        return perturbed
    perturbed.loc[mask, "open"] = perturbed.loc[mask, "open"] * 1000.0
    perturbed.loc[mask, "high"] = perturbed.loc[mask, "high"] * 1000.0
    perturbed.loc[mask, "low"] = perturbed.loc[mask, "low"] * 1000.0
    perturbed.loc[mask, "close"] = perturbed.loc[mask, "close"] * 1000.0
    perturbed.loc[mask, "volume"] = 0.0
    return perturbed


# ---------------------------------------------------------------------------
# T1 – select_candidates: selection at as_of=D is independent of bars after D
# ---------------------------------------------------------------------------


class TestT1SelectCandidates:
    """Perturb bars with date > as_of and assert selection set is unchanged."""

    def _run(self, bars_by_ticker: dict[str, pd.DataFrame], as_of: pd.Timestamp):
        entry_ast = parse(_ENTRY_EXPR)
        return select_candidates(
            bars_by_ticker=bars_by_ticker,
            entry_ast=entry_ast,
            as_of=as_of,
            top_n=5,
            lookback_required=3,
        )

    def test_selected_tickers_unchanged(self):
        """Tickers chosen at as_of are identical whether future bars are garbage or not."""
        as_of = pd.Timestamp("2024-03-01")

        # Build 4 tickers with 60 bars each.  Some will fire the entry signal.
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        baseline: dict[str, pd.DataFrame] = {
            t: make_bars(start="2024-01-01", n=60, seed=i)
            for i, t in enumerate(tickers)
        }

        # Perturbed copies: bars after as_of are garbage.
        perturbed: dict[str, pd.DataFrame] = {
            t: _perturb_after(df, as_of) for t, df in baseline.items()
        }

        sel_base, _ = self._run(baseline, as_of)
        sel_pert, _ = self._run(perturbed, as_of)

        # The set of selected tickers must be identical.
        assert set(sel_base["ticker"].tolist()) == set(sel_pert["ticker"].tolist()), (
            "LOOKAHEAD BUG: selected ticker set changed when future bars were perturbed. "
            f"baseline={sel_base['ticker'].tolist()!r}, "
            f"perturbed={sel_pert['ticker'].tolist()!r}"
        )

    def test_as_of_close_and_volume_unchanged(self):
        """as_of_close and as_of_dollar_vol in the selection are identical."""
        as_of = pd.Timestamp("2024-03-01")
        # Use 4 tickers (same seeds as test_selected_tickers_unchanged) so that
        # at least one candidate fires the entry signal on this date.
        tickers = ["AA", "BB", "CC", "DD"]
        baseline: dict[str, pd.DataFrame] = {
            t: make_bars(start="2024-01-01", n=60, seed=i)
            for i, t in enumerate(tickers)
        }
        perturbed: dict[str, pd.DataFrame] = {
            t: _perturb_after(df, as_of) for t, df in baseline.items()
        }

        sel_base, _ = self._run(baseline, as_of)
        sel_pert, _ = self._run(perturbed, as_of)

        if sel_base.empty and sel_pert.empty:
            pytest.skip("no candidates on this date; adjust seed if needed")

        # Align on ticker and compare numeric columns.
        for _, row_b in sel_base.iterrows():
            tk = row_b["ticker"]
            rows_p = sel_pert[sel_pert["ticker"] == tk]
            assert not rows_p.empty, (
                f"LOOKAHEAD BUG: ticker {tk!r} present in baseline selection but "
                "absent from perturbed selection"
            )
            row_p = rows_p.iloc[0]
            assert row_b["as_of_close"] == pytest.approx(
                row_p["as_of_close"], abs=1e-9
            ), (
                f"LOOKAHEAD BUG: as_of_close for {tk!r} changed after future perturbation"
            )
            assert row_b["as_of_dollar_vol"] == pytest.approx(
                row_p["as_of_dollar_vol"], abs=1e-9
            ), (
                f"LOOKAHEAD BUG: as_of_dollar_vol for {tk!r} changed after future perturbation"
            )


# ---------------------------------------------------------------------------
# T2 – simulate_ticker: entry_date / entry_price invariant after entry bar
# ---------------------------------------------------------------------------


class TestT2SimulateTicker:
    """Entry fill depends only on bars up to entry_idx; later bars are irrelevant."""

    def _make_cfg(self) -> BacktestConfig:
        return _cfg(hold=5)

    def test_entry_date_unchanged(self):
        """entry_date is the same whether tail bars are garbage or not."""
        bars = make_bars(start="2024-01-01", n=40, seed=42)
        signal_idx = 5
        cfg = self._make_cfg()

        outcome_base = simulate_ticker(bars, signal_idx=signal_idx, cfg=cfg)
        assert outcome_base.trade is not None, (
            "baseline trade is None; check signal_idx"
        )

        # Perturb everything strictly after the entry bar.
        entry_ts = pd.Timestamp(outcome_base.trade.entry_date)
        perturbed = _perturb_after(bars, entry_ts)

        outcome_pert = simulate_ticker(perturbed, signal_idx=signal_idx, cfg=cfg)
        assert outcome_pert.trade is not None

        assert outcome_base.trade.entry_date == outcome_pert.trade.entry_date, (
            f"LOOKAHEAD BUG: entry_date changed under tail perturbation. "
            f"baseline={outcome_base.trade.entry_date!r}, "
            f"perturbed={outcome_pert.trade.entry_date!r}"
        )

    def test_entry_price_unchanged(self):
        """entry_price is bit-identical whether tail bars are garbage or not."""
        bars = make_bars(start="2024-01-01", n=40, seed=7)
        signal_idx = 4
        cfg = self._make_cfg()

        outcome_base = simulate_ticker(bars, signal_idx=signal_idx, cfg=cfg)
        assert outcome_base.trade is not None

        entry_ts = pd.Timestamp(outcome_base.trade.entry_date)
        perturbed = _perturb_after(bars, entry_ts)

        outcome_pert = simulate_ticker(perturbed, signal_idx=signal_idx, cfg=cfg)
        assert outcome_pert.trade is not None

        assert outcome_base.trade.entry_price == pytest.approx(
            outcome_pert.trade.entry_price, abs=1e-9
        ), (
            f"LOOKAHEAD BUG: entry_price changed under tail perturbation. "
            f"baseline={outcome_base.trade.entry_price!r}, "
            f"perturbed={outcome_pert.trade.entry_price!r}"
        )

    def test_signal_date_unchanged(self):
        """signal_date (the bar that fired the entry) is invariant."""
        bars = make_bars(start="2024-01-01", n=40, seed=99)
        signal_idx = 6
        cfg = self._make_cfg()

        outcome_base = simulate_ticker(bars, signal_idx=signal_idx, cfg=cfg)
        assert outcome_base.trade is not None

        entry_ts = pd.Timestamp(outcome_base.trade.entry_date)
        perturbed = _perturb_after(bars, entry_ts)

        outcome_pert = simulate_ticker(perturbed, signal_idx=signal_idx, cfg=cfg)
        assert outcome_pert.trade is not None

        assert outcome_base.trade.signal_date == outcome_pert.trade.signal_date, (
            f"LOOKAHEAD BUG: signal_date changed under tail perturbation. "
            f"baseline={outcome_base.trade.signal_date!r}, "
            f"perturbed={outcome_pert.trade.signal_date!r}"
        )


# ---------------------------------------------------------------------------
# T3 – run_backtest: pre-cutoff trades are byte-identical in a perturbed run
# ---------------------------------------------------------------------------


class TestT3RunBacktest:
    """Full historical backtest: every trade whose entry precedes the cutoff
    must have the same signal_date, entry_date, entry_price in both runs."""

    # Spy/benchmark stub – needs at least one bar per day in the simulation range.
    _BENCH_START = "2023-06-01"
    _BENCH_N = 300  # ~15 months; covers warmup + hold horizon

    def _build_stub(
        self,
        ticker_map: dict[str, pd.DataFrame],
        bench_bars: pd.DataFrame,
    ) -> StubPriceFetcher:
        data = dict(ticker_map)
        data[_BENCH] = bench_bars
        return StubPriceFetcher(data)

    def test_pre_cutoff_trades_identical(self):
        """Trades entered before a cutoff are unchanged when post-cutoff bars are garbage."""
        as_of = date(2024, 3, 1)
        # 6 tickers; enough bars for warmup + holding window.
        tickers = ("AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF")
        bar_frames_base: dict[str, pd.DataFrame] = {
            t: make_bars(start=self._BENCH_START, n=220, seed=i + 20)
            for i, t in enumerate(tickers)
        }
        bench_base = make_bars(start=self._BENCH_START, n=220, seed=999)

        cfg = _cfg(
            as_of=as_of,
            hold=5,
            top=3,
            tickers=tickers,
            benchmark=_BENCH,
        )

        # --- Baseline run ---
        fetcher_base = self._build_stub(bar_frames_base, bench_base)
        result_base = run_backtest(cfg, fetcher_base)

        # Sanity: we need at least one trade to validate anything.
        assert result_base.trades, (
            "No trades produced in baseline; tweak seed/dates so at least one trade fires."
        )

        # Cutoff = entry_date of the earliest trade.
        earliest_entry_ts = min(pd.Timestamp(t.entry_date) for t in result_base.trades)
        # Perturb everything strictly after the earliest entry.
        bar_frames_pert: dict[str, pd.DataFrame] = {
            t: _perturb_after(df, earliest_entry_ts)
            for t, df in bar_frames_base.items()
        }
        bench_pert = _perturb_after(bench_base, earliest_entry_ts)

        fetcher_pert = self._build_stub(bar_frames_pert, bench_pert)
        result_pert = run_backtest(cfg, fetcher_pert)

        # Build lookup: (ticker, signal_date) → trade from perturbed run.
        pert_lookup: dict[tuple[str, date], object] = {
            (tr.ticker, tr.signal_date): tr for tr in result_pert.trades
        }

        matched = 0
        for tr_b in result_base.trades:
            key = (tr_b.ticker, tr_b.signal_date)
            tr_p = pert_lookup.get(key)
            if tr_p is None:
                # Trade not found in perturbed run — only a bug if its entry is
                # at or before the cutoff (i.e. it should be unaffected).
                if pd.Timestamp(tr_b.entry_date) <= earliest_entry_ts:
                    pytest.fail(
                        f"LOOKAHEAD BUG: trade {key!r} present in baseline but "
                        "absent in perturbed run even though its entry precedes "
                        f"the perturbation cutoff ({earliest_entry_ts.date()!r})."
                    )
                continue

            # Only assert invariance for trades whose entry is at/before cutoff.
            if pd.Timestamp(tr_b.entry_date) > earliest_entry_ts:
                continue

            assert tr_b.signal_date == tr_p.signal_date, (
                f"LOOKAHEAD BUG in run_backtest: signal_date changed for {key!r}. "
                f"baseline={tr_b.signal_date!r}, perturbed={tr_p.signal_date!r}"
            )
            assert tr_b.entry_date == tr_p.entry_date, (
                f"LOOKAHEAD BUG in run_backtest: entry_date changed for {key!r}. "
                f"baseline={tr_b.entry_date!r}, perturbed={tr_p.entry_date!r}"
            )
            assert tr_b.entry_price == pytest.approx(tr_p.entry_price, abs=1e-9), (
                f"LOOKAHEAD BUG in run_backtest: entry_price changed for {key!r}. "
                f"baseline={tr_b.entry_price!r}, perturbed={tr_p.entry_price!r}"
            )
            matched += 1

        assert matched >= 1, (
            "No trades fell at/before the cutoff to validate; "
            "choose a tighter cutoff or more tickers."
        )


# ---------------------------------------------------------------------------
# T4 – run_rolling_backtest: (ticker, signal_date) set for signal_date <= D
#       is unchanged when tail (dates > D) is perturbed
# ---------------------------------------------------------------------------


class TestT4RollingEngine:
    """Rolling backtest: signals fired on or before D are invariant under tail perturbation."""

    _START = "2023-09-01"
    _N = 180  # ~9 months of daily bars

    def _build_stub(
        self,
        ticker_map: dict[str, pd.DataFrame],
        bench_bars: pd.DataFrame,
    ) -> StubPriceFetcher:
        data = dict(ticker_map)
        data[_BENCH] = bench_bars
        return StubPriceFetcher(data)

    def test_signal_set_unchanged_before_cutoff(self):
        """(ticker, signal_date) pairs with signal_date <= D are the same in both runs."""
        roll_start = date(2024, 1, 15)
        roll_end = date(2024, 4, 30)
        as_of = date(2024, 4, 30)  # cfg.as_of unused by rolling but required

        tickers = ("RR1", "RR2", "RR3", "RR4")
        bar_frames_base: dict[str, pd.DataFrame] = {
            t: make_bars(start=self._START, n=self._N, seed=i + 50)
            for i, t in enumerate(tickers)
        }
        bench_base = make_bars(start=self._START, n=self._N, seed=777)

        cfg = _cfg(
            as_of=as_of,
            hold=5,
            top=2,
            tickers=tickers,
            benchmark=_BENCH,
        )

        # --- Baseline run ---
        fetcher_base = self._build_stub(bar_frames_base, bench_base)
        result_base = run_rolling_backtest(
            cfg, fetcher_base, start_date=roll_start, end_date=roll_end
        )

        if result_base.selection.empty:
            pytest.skip("No candidates in rolling baseline; adjust seeds/dates.")

        # Cutoff D = 80 % through the signal-date range so a decent fraction
        # of signals are "before D" (invariant set) and some are "after D".
        all_signal_dates = pd.to_datetime(result_base.selection["signal_date"])
        d_cutoff = all_signal_dates.quantile(0.6).normalize()

        # Perturb all ticker bars strictly after d_cutoff.
        bar_frames_pert: dict[str, pd.DataFrame] = {
            t: _perturb_after(df, d_cutoff) for t, df in bar_frames_base.items()
        }
        bench_pert = _perturb_after(bench_base, d_cutoff)

        fetcher_pert = self._build_stub(bar_frames_pert, bench_pert)
        result_pert = run_rolling_backtest(
            cfg, fetcher_pert, start_date=roll_start, end_date=roll_end
        )

        # Extract (ticker, signal_date) sets restricted to signal_date <= d_cutoff.
        def _pre_cutoff_set(result) -> set[tuple[str, date]]:
            if result.selection.empty:
                return set()
            sel = result.selection.copy()
            sel["signal_date"] = pd.to_datetime(sel["signal_date"])
            sel = sel[sel["signal_date"] <= d_cutoff]
            return {
                (row["ticker"], row["signal_date"].date()) for _, row in sel.iterrows()
            }

        set_base = _pre_cutoff_set(result_base)
        set_pert = _pre_cutoff_set(result_pert)

        extra_in_pert = set_pert - set_base
        missing_in_pert = set_base - set_pert

        assert not extra_in_pert and not missing_in_pert, (
            "LOOKAHEAD BUG in run_rolling_backtest: "
            "(ticker, signal_date) set for signal_date <= cutoff changed under tail perturbation.\n"
            f"cutoff={d_cutoff.date()!r}\n"
            f"extra_in_perturbed={extra_in_pert!r}\n"
            f"missing_in_perturbed={missing_in_pert!r}"
        )

    def test_signal_dates_monotone_with_rolling_window(self):
        """Signals issued only within [roll_start, roll_end]; no future signal leakage."""
        roll_start = date(2024, 1, 15)
        roll_end = date(2024, 3, 31)
        as_of = date(2024, 3, 31)

        tickers = ("SS1", "SS2", "SS3")
        bar_frames: dict[str, pd.DataFrame] = {
            t: make_bars(start=self._START, n=self._N, seed=i + 70)
            for i, t in enumerate(tickers)
        }
        bench = make_bars(start=self._START, n=self._N, seed=888)

        cfg = _cfg(
            as_of=as_of,
            hold=5,
            top=2,
            tickers=tickers,
            benchmark=_BENCH,
        )
        fetcher = self._build_stub(bar_frames, bench)
        result = run_rolling_backtest(
            cfg, fetcher, start_date=roll_start, end_date=roll_end
        )

        if result.selection.empty:
            pytest.skip("No candidates; adjust seeds/dates.")

        signal_dates = pd.to_datetime(result.selection["signal_date"])
        roll_start_ts = pd.Timestamp(roll_start)
        roll_end_ts = pd.Timestamp(roll_end)

        out_of_window = signal_dates[
            (signal_dates < roll_start_ts) | (signal_dates > roll_end_ts)
        ]
        assert out_of_window.empty, (
            f"LOOKAHEAD BUG: signal_dates outside [{roll_start!r}, {roll_end!r}] found: "
            f"{out_of_window.tolist()!r}"
        )
