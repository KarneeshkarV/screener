"""Corporate-action correctness: splits + dividends in the backtester.

All scenarios are offline/synthetic. They pin the three fixes:

* H-2 — ``split_factor`` is now read: a flat-priced series spanning a real
  2:1 split must stay flat in ``splits_only`` mode (no phantom -50% step).
* H-3 — cash dividends are threaded into ``build_equity_curve`` and into
  ``Trade.pnl`` so the per-trade total and the equity-curve endpoint agree to
  machine precision in ``splits_only`` mode, while ``full`` mode is untouched.
* M-1 — FMP-served frames (adjClose, no Stock Splits) cannot have a split
  reliably recovered (adjClose conflates splits + dividends), so they pass
  through unadjusted with a loud warning instead of silently — and a pure
  dividend is never mistaken for a split.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from screener.backtester.data import (
    apply_splits_only_adjustment,
    warn_unadjustable_fmp_frames,
)
from screener.backtester.engine import run_backtest
from screener.backtester.models import BacktestConfig

from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=1,
        entry_expr="entry_signal > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        strategy_name=None,
        tickers=("AAA",),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _split_frame(n: int = 30, split_at: int = 15, ratio: float = 2.0) -> pd.DataFrame:
    """Flat-economic-value series spanning a ratio:1 split at ``split_at``.

    Pre-split bars trade at ``100 * ratio`` (raw, unadjusted) and post-split at
    ``100``; the back-adjust ``split_factor`` (as emitted by ``_normalize_frame``)
    is ``ratio`` before the ex-day and ``1.0`` from the ex-day on. After
    ``apply_splits_only_adjustment`` the close is a flat ``100`` throughout.
    """
    idx = pd.bdate_range("2024-01-01", periods=n)
    pre = 100.0 * ratio
    close = [pre] * split_at + [100.0] * (n - split_at)
    factor = [ratio] * split_at + [1.0] * (n - split_at)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000.0] * n,
            "dividend": [0.0] * n,
            "split_factor": factor,
            "entry_signal": [0.0] * n,
        },
        index=idx,
    )
    return df


# ── (1) flat-price 2:1 split → equity curve ~constant ────────────────


def test_flat_price_split_has_no_phantom_drop():
    """A flat-value 2:1 split in splits_only must not show a -50% step."""
    bars = _split_frame(n=30, split_at=15, ratio=2.0)
    bars.iat[2, bars.columns.get_loc("entry_signal")] = 1.0  # enter pre-split
    spy = make_bars(n=30, seed=99, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(
        as_of=bars.index[2].date(),
        hold=20,  # hold spans the split
        price_adjustment="splits_only",
    )
    result = run_backtest(cfg, fetcher)

    equity = result.equity_curve
    assert not equity.empty
    base = float(equity.iloc[0])
    # The economic value is flat across the split, so every point sits within a
    # rounding tolerance of the start. A phantom split (the H-2 bug) would halve
    # the curve at the ex-day — far outside this band.
    assert (equity / base - 1.0).abs().max() < 1e-6, "phantom split drop in equity"
    # And explicitly: no single bar is anywhere near a -50% drop.
    assert equity.min() > base * 0.99


def test_apply_splits_only_adjustment_back_adjusts_and_fast_paths():
    """Unit: OHLC/dividend divided, volume multiplied; factor==1 untouched."""
    bars = _split_frame(n=10, split_at=5, ratio=2.0)
    bars.iat[3, bars.columns.get_loc("dividend")] = 4.0  # pre-split per-share div
    adjusted = apply_splits_only_adjustment({"AAA": bars})["AAA"]
    # Pre-split close 200 / factor 2 == 100 (flat with post-split bars).
    assert adjusted["close"].tolist() == [100.0] * 10
    # Pre-split per-share dividend back-adjusts 4 / 2 == 2.
    assert adjusted["dividend"].iloc[3] == 2.0
    # Pre-split volume doubles (1000 * 2), post-split unchanged.
    assert adjusted["volume"].iloc[0] == 2_000.0
    assert adjusted["volume"].iloc[5] == 1_000.0

    # Fast path: an all-ones factor returns the frame unchanged (same object).
    flat = bars.copy()
    flat["split_factor"] = 1.0
    out = apply_splits_only_adjustment({"AAA": flat})["AAA"]
    assert out is flat


# ── (2) dividend over a held position: pnl and equity agree ──────────


def test_dividend_pnl_and_equity_endpoint_agree_splits_only():
    """splits_only: equity endpoint and Trade.pnl both rise by shares*D."""
    bars = make_bars(n=20, seed=61, open_base=100.0)
    bars["entry_signal"] = 0.0
    bars.iat[4, bars.columns.get_loc("entry_signal")] = 1.0
    bars["dividend"] = 0.0
    div_per_share = 1.25
    bars.iat[7, bars.columns.get_loc("dividend")] = div_per_share
    spy = make_bars(n=20, seed=62, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    cfg = _cfg(
        as_of=bars.index[4].date(),
        hold=6,
        price_adjustment="splits_only",
    )
    result = run_backtest(cfg, fetcher)

    assert result.trades
    trade = result.trades[0]
    shares = trade.shares

    # The dividend was credited once over the hold.
    assert abs(trade.dividend_income - shares * div_per_share) < 1e-9
    # Trade.pnl now bakes in the dividend income (capital pnl + dividends).
    capital_pnl = trade.exit_value - trade.entry_cost
    assert abs(trade.pnl - (capital_pnl + shares * div_per_share)) < 1e-9

    # Equity endpoint == initial_capital + total per-trade pnl, to 1e-9.
    endpoint = float(result.equity_curve.iloc[-1])
    total_pnl = sum(t.pnl for t in result.trades)
    assert abs(endpoint - (cfg.initial_capital + total_pnl)) < 1e-9


# ── (3) full mode: dividend column has no effect ─────────────────────


def test_full_mode_ignores_dividend_column():
    """full mode: dividend_income==0 and the equity curve ignores dividends."""
    bars = make_bars(n=20, seed=61, open_base=100.0)
    bars["entry_signal"] = 0.0
    bars.iat[4, bars.columns.get_loc("entry_signal")] = 1.0
    spy = make_bars(n=20, seed=62, open_base=400.0)

    # Two identical runs in full mode: one with a dividend column, one without.
    bars_with_div = bars.copy()
    bars_with_div["dividend"] = 0.0
    bars_with_div.iat[7, bars_with_div.columns.get_loc("dividend")] = 5.0

    cfg = _cfg(as_of=bars.index[4].date(), hold=6, price_adjustment="full")

    result_div = run_backtest(cfg, StubPriceFetcher({"AAA": bars_with_div, "SPY": spy}))
    result_plain = run_backtest(cfg, StubPriceFetcher({"AAA": bars, "SPY": spy}))

    assert result_div.trades
    # full mode never credits explicit dividends.
    assert all(t.dividend_income == 0.0 for t in result_div.trades)
    # The dividend column is inert: identical trades and identical equity curve.
    assert [t.pnl for t in result_div.trades] == [t.pnl for t in result_plain.trades]
    assert result_div.equity_curve.tolist() == result_plain.equity_curve.tolist()


# ── (M-1) FMP split-factor reconstruction ────────────────────────────


def test_fmp_frame_passes_through_unadjusted_and_warns():
    """FMP frame (adjClose, no Stock Splits) is left untouched and a warning fires.

    ``adj_close`` is back-adjusted for *both* splits and dividends, so a
    split_factor cannot be reliably recovered from the adjClose/close ratio. We
    deliberately do not fabricate one: the frame passes through unchanged.
    """
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = [200.0] * 5 + [100.0] * 5  # contains a real 2:1 split
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000.0] * 10,
            "adj_close": [100.0] * 10,
        },
        index=idx,
    )
    out = warn_unadjustable_fmp_frames({"AAA": frame})["AAA"]
    # No split_factor fabricated; prices are returned exactly as supplied.
    assert "split_factor" not in out.columns
    assert out["close"].tolist() == close
    # apply_splits_only_adjustment then leaves the (split_factor-less) frame as-is.
    adjusted = apply_splits_only_adjustment({"AAA": out})["AAA"]
    assert adjusted["close"].tolist() == close


def test_dividend_step_in_adj_close_is_not_mistaken_for_a_split():
    """Regression: a pure dividend (smooth adjClose drift) must NOT corrupt prices.

    The old adjClose-ratio reconstruction read any dividend divergence as a fake
    split and baked dividend return into the OHLC. With reconstruction removed,
    a flat $100 close whose adjClose drifts with quarterly dividends is passed
    through with prices unchanged and no split_factor attached.
    """
    idx = pd.bdate_range("2024-01-01", periods=8)
    close = [100.0] * 8
    # adjClose below the raw close and stepping down further as dividends accrue
    # back through history — exactly the shape that fooled the old heuristic.
    adj_close = [98.5, 98.5, 99.0, 99.0, 99.5, 99.5, 100.0, 100.0]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000.0] * 8,
            "adj_close": adj_close,
        },
        index=idx,
    )
    out = warn_unadjustable_fmp_frames({"AAA": frame})["AAA"]
    assert "split_factor" not in out.columns
    # Prices are completely untouched — no phantom split adjustment.
    assert out["close"].tolist() == close
    adjusted = apply_splits_only_adjustment({"AAA": out})["AAA"]
    assert adjusted["close"].tolist() == close
