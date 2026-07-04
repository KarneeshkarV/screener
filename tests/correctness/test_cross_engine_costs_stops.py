"""Phase 5 (extended) — Cross-engine reconciliation *with costs and stops*.

The sibling module ``test_cross_engine_reconciliation.py`` proves the
event-driven engine (``historical.py`` / ``core.py`` / ``portfolio.py``) and
vectorbt agree on the *frictionless* regime (single ticker, 1 slot, SMA
crossover, ``fees=0``, ``slippage=0``, no stops). This module extends that proof
to the regimes that actually matter for realism:

    1. commission            (engine ``commission_bps`` ⇄ vbt ``fees``)
    2. slippage              (engine ``slippage_bps``   ⇄ vbt ``slippage``)
    3. commission + slippage (combined)
    4. stop-loss             (engine ``stop_loss``      ⇄ vbt ``sl_stop``)
    5. take-profit           (engine ``take_profit``    ⇄ vbt ``tp_stop``)
    6. trailing stop         (engine ``trailing_stop``  ⇄ vbt ``sl_stop`` + ``sl_trail``)
    7. stop-loss + costs     (exercises the slippage-adjusted stop base)

plus a **control** proving the comparison is non-trivial (feeding vbt the raw
basis-point number as a fraction diverges by >1.0 absolute).

Installed vectorbt is **1.0.0** (``vbt.Portfolio.from_signals`` with the
``sl_stop`` / ``tp_stop`` / ``sl_trail`` / ``stop_entry_price`` /
``stop_exit_price`` keyword family). Everything below was confirmed against the
installed API, not a remembered 0.x signature.

RECONCILIATION RULES (why a green test means the *math* agrees)
--------------------------------------------------------------
Unit conversion
~~~~~~~~~~~~~~~
The engine takes basis points; vectorbt takes fractions. ``bps_to_fraction``
(10 bps → 0.001) is applied on every hand-off. The control test deliberately
skips it and shows the result diverges wildly, so the conversion is load-bearing
rather than cosmetic.

Per-side cost application (verified in ``portfolio.py`` / ``fills.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Slippage widens the *fill price*: buy ``ref·(1+s)``, sell ``ref·(1−s)`` — the
  price recorded on the ``Trade`` already includes it. vectorbt does the same
  (``price·(1±slippage)``), so recorded entry/exit prices match to ``rtol=1e-9``.
* Commission is a fraction of notional charged on *each* side:
  ``entry_cost = shares·entry_price·(1+c)``, ``exit_value = shares·exit_price·(1−c)``.
  vectorbt's ``fees`` is applied identically to each order's value.

Net portfolio-return comparison (the crux)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The engine's per-slot sizing does **not** compound (each trade is sized against
a fixed ``slot_capital``); vectorbt reinvests the whole cash pool. They are
nevertheless *exactly* comparable because each trade multiplies capital by the
same capital-independent factor:

    exit_value_i / entry_cost_i
        = [shares·exit_price·(1−c)] / [shares·entry_price·(1+c)]
        = exit_price·(1−c) / (entry_price·(1+c))          ← shares cancel

vectorbt's reinvested pool multiplies by exactly this factor each trade too
(buy spends ``C`` on ``C/(price·(1+c))`` shares; sell returns
``shares·price·(1−c)``). Chaining the ratios (``net_compound_return``) therefore
equals ``pf.total_return()`` to machine precision — observed ≤ 1.4e-15 across all
cost scenarios, far inside the ≤1e-8 target.

Exit-date shift depends on the exit *mechanism* (documented, not a bug)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``exit_expr`` (SMA crossunder) exits carry the **+1 business-day** shift from
  the frictionless test: the engine exits on the signal bar at ``close``; vbt
  shifts the exit signal +1 and fills at the next ``open`` (equal price because
  the frame is built with ``open[t] = close[t-1]``). The commission / slippage /
  combined tests use crossunder exits and inherit this shift.
* **Stop / target / trailing** exits fill *intrabar on the bar the level is
  hit*, in **both** engines — so their exit dates match **exactly, with no
  shift**. The stop tests assert this stronger equality.

Stop reconciliation specifics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Stop *base* price: the engine measures ``stop_loss`` / ``take_profit`` /
  ``trailing_stop`` against the **slippage-adjusted entry fill**
  (``stop_ref = entry_fill·(1−stop_loss)``). vectorbt's default
  ``stop_entry_price`` is ``Close`` (closing price of the entry bar), which does
  **not** match; we pass ``StopEntryPrice.FillPrice`` so vbt measures against the
  same slipped fill. Test 7 (stop + slippage + commission) is the one that would
  fail if this base were wrong.
* Stop *exit* price + slippage: the engine fills the stop at the stop reference
  and then applies sell-side slippage. vectorbt's default ``stop_exit_price`` is
  ``StopLimit`` (slippage **not** applied); we pass ``StopExitPrice.StopMarket``
  so vbt applies slippage on the stop fill too. (With zero slippage the two vbt
  modes coincide, so tests 4–6 would pass under either.)
* Gap handling: the engine's ``gap_fills=True`` fills a bar that *opens* through
  the stop at the open, matching vectorbt's StopMarket "if the stop was hit
  before, use the opening price" rule. To keep the comparison a clean intrabar
  fill *at the stop reference* (independent of gap semantics), the stop frames
  are built so the trigger bar's open is on the safe side of the level and only
  the intrabar ``low``/``high`` pierces it — no bar gaps through a stop. The gap
  divergence itself is already covered by ``test_hand_computed_trades.py`` and is
  not re-litigated here.
* Isolating the stop: tests 4–7 set ``exit_expr=None`` (engine) and pass an
  all-``False`` exit mask (vbt) so the stop is the *sole* exit path. This is
  required for the stop-loss frame, whose declining tail would otherwise trip an
  SMA crossunder before the stop; it is applied uniformly to TP/trailing too so
  every stop test isolates exactly the mechanism it names.

Every stop test asserts the recorded ``exit_reason`` (``stop`` / ``target`` /
``trail``) so it cannot pass vacuously by exiting some other way.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

# Guard: skip the entire module if vectorbt is not installed.
pytest.importorskip("vectorbt")

from vectorbt.portfolio.enums import (  # noqa: E402
    StopEntryPrice,
    StopExitPrice,
)

from screener.backtester.historical import run_backtest  # noqa: E402
from screener.backtester.models import BacktestConfig  # noqa: E402

from .reference_adapters import (  # noqa: E402
    bps_to_fraction,
    net_compound_return,
)

# ---------------------------------------------------------------------------
# Shared constants — identical crossover definition to the frictionless test.
# ---------------------------------------------------------------------------

_FAST: int = 10
_SLOW: int = 30
_ENTRY_EXPR: str = (
    f"crossover(close, sma(close, {_SLOW})) and close > sma(close, {_FAST})"
)
_EXIT_EXPR: str = f"crossunder(close, sma(close, {_SLOW}))"

#: Entry-signal bar index of the FIRST SMA(10,30) crossover in the sinusoid
#: warm-up used for the stop frames; the trade fills at bar 70's open (verified
#: below in ``_spliced_close``). Kept as a constant so the stop frames can splice
#: a controlled tail immediately after the entry bar.
_STOP_ENTRY_SIGNAL_IDX: int = 69


class _StubFetcher:
    """Minimal offline price fetcher — identical interface to StubPriceFetcher."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = {k: v.copy() for k, v in data.items()}

    def fetch(
        self,
        tickers,
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        for t in tickers:
            frame = self._data.get(t, pd.DataFrame())
            if frame.empty:
                out[t] = frame
                continue
            out[t] = frame.loc[(frame.index >= s) & (frame.index <= e)]
        return out


# ---------------------------------------------------------------------------
# Deterministic frames (no RNG anywhere).
# ---------------------------------------------------------------------------


def _frame_from_close(close: np.ndarray) -> pd.DataFrame:
    """OHLCV frame with ``open[t] = close[t-1]`` and ±0.3 high/low padding.

    The ``open[t] = close[t-1]`` construction makes an ``exit_expr`` fill on the
    signal-bar close equal to a next-bar-open fill, and — for the stop frames —
    keeps every bar gap-free so a stop is a clean intrabar fill at its reference.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.bdate_range("2020-01-01", periods=n)
    open_ = np.concatenate(([close[0] - 0.5], close[:-1]))
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    volume = np.full(n, 50_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_cost_frame() -> pd.DataFrame:
    """The same deterministic 300-bar sinusoid frame as the frictionless test.

    Produces exactly three SMA(10,30) crossover trades that all exit via
    crossunder before the last bar (no terminal force-close).
    """
    t = np.arange(300)
    close = 100.0 + 15.0 * np.sin(2.0 * np.pi * t / 80.0) + 0.05 * t
    return _frame_from_close(close)


def _spliced_close(tail: np.ndarray) -> np.ndarray:
    """Sinusoid warm-up through the first entry bar, then a controlled tail.

    Bars 0..70 are the frictionless-test sinusoid: the first crossover signal is
    at bar ``_STOP_ENTRY_SIGNAL_IDX`` (=69) and the trade fills at bar 70's open.
    ``tail`` supplies the post-entry close path that triggers the stop/target.
    """
    n_warm = _STOP_ENTRY_SIGNAL_IDX + 2  # bars 0..70 inclusive
    t = np.arange(n_warm)
    warm = 100.0 + 15.0 * np.sin(2.0 * np.pi * t / 80.0) + 0.05 * t
    return np.concatenate([warm, np.asarray(tail, dtype=float)])


# ---------------------------------------------------------------------------
# Config builders.
# ---------------------------------------------------------------------------


def _cost_cfg(**overrides) -> BacktestConfig:
    """Frictionless base config for the 300-bar cost frame (crossunder exits)."""
    defaults: dict = dict(
        market="us",
        as_of=date(2020, 4, 7),  # first crossover signal date in the cost frame
        hold=300,
        top=1,
        entry_expr=_ENTRY_EXPR,
        exit_expr=_EXIT_EXPR,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        gap_fills=False,
        tickers=("A",),
        min_price=None,
        min_avg_dollar_volume=None,
        allow_reentry=True,
        max_reentries=10,
        reinvest=True,
        reserve_multiple=1,
        entry_order_type="moo",
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _stop_cfg(**overrides) -> BacktestConfig:
    """Config for the spliced stop frames: single trade, stop is the only exit.

    ``exit_expr=None`` and ``allow_reentry=False`` guarantee exactly one trade
    governed solely by the configured stop/target. ``gap_fills=True`` matches
    vectorbt's StopMarket gap rule (immaterial on these gap-free frames but kept
    honest). ``hold=200`` is large enough that no time-exit fires yet small
    enough to avoid the ``hold*2`` day-range overflow in ``run_backtest``.
    """
    defaults: dict = dict(
        market="us",
        as_of=date(2020, 4, 7),
        hold=200,
        top=1,
        entry_expr=_ENTRY_EXPR,
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        gap_fills=True,
        tickers=("A",),
        min_price=None,
        min_avg_dollar_volume=None,
        allow_reentry=False,
        max_reentries=0,
        reinvest=True,
        reserve_multiple=1,
        entry_order_type="moo",
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ---------------------------------------------------------------------------
# Engine drivers.
# ---------------------------------------------------------------------------


def _run_event(bars: pd.DataFrame, cfg: BacktestConfig):
    """Drive ``run_backtest`` and return the list of ``Trade`` objects sorted by
    entry date. Full ``Trade`` objects are returned (not tuples) so tests can
    read ``exit_reason`` / ``entry_cost`` / ``exit_value``."""
    fetcher = _StubFetcher({"A": bars, "SPY": bars})
    result = run_backtest(cfg, fetcher)
    return sorted(result.trades, key=lambda tr: tr.entry_date)


def _vbt_signals(bars: pd.DataFrame, vbt):
    """Shifted (entries, exits, close, open, high, low) frames matching the
    engine's MOO next-open fill and the frictionless test's +1-bar signal shift.
    """
    from screener.backtester.vbt_sweep import sma_crossover_signals

    idx = bars.index
    close_df = pd.DataFrame({"A": bars["close"].to_numpy()}, index=idx)
    open_df = pd.DataFrame({"A": bars["open"].to_numpy()}, index=idx)
    high_df = pd.DataFrame({"A": bars["high"].to_numpy()}, index=idx)
    low_df = pd.DataFrame({"A": bars["low"].to_numpy()}, index=idx)
    entries, exits = sma_crossover_signals(close_df, _FAST, _SLOW, 0, vbt)
    entries_s = entries.astype(bool).shift(1, fill_value=False).astype(bool)
    exits_s = exits.astype(bool).shift(1, fill_value=False).astype(bool)
    return entries_s, exits_s, close_df, open_df, high_df, low_df


def _run_vbt_costs(
    bars: pd.DataFrame,
    *,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> tuple[list[tuple[date, date, float, float]], float]:
    """vbt SMA-crossover run with fees/slippage (crossunder exits). Returns
    ``([(entry_date, exit_date, entry_price, exit_price)...], total_return)``."""
    from screener.backtester.vbt_sweep import _require_vectorbt

    vbt = _require_vectorbt()
    entries_s, exits_s, close_df, open_df, _high, _low = _vbt_signals(bars, vbt)
    pf = vbt.Portfolio.from_signals(
        close_df,
        entries_s,
        exits_s,
        price=open_df,
        init_cash=100_000.0,
        fees=fees,
        slippage=slippage,
        group_by=True,
        cash_sharing=True,
        freq="1D",
    )
    idx = bars.index
    trades = sorted(
        [
            (
                idx[int(rec["entry_idx"])].date(),
                idx[int(rec["exit_idx"])].date(),
                float(rec["entry_price"]),
                float(rec["exit_price"]),
            )
            for rec in pf.trades.records_arr
        ],
        key=lambda x: x[0],
    )
    return trades, float(pf.total_return())


def _run_vbt_stop(
    bars: pd.DataFrame,
    *,
    sl_stop: float = float("nan"),
    tp_stop: float = float("nan"),
    sl_trail: bool = False,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> tuple[list[tuple[date, date, float, float]], float]:
    """vbt stop-only run: entries from the SMA crossover, exits solely from the
    configured stop (all-``False`` exit mask). Stop base = slipped fill price;
    stop exit applies slippage (StopMarket) — see module docstring."""
    from screener.backtester.vbt_sweep import _require_vectorbt

    vbt = _require_vectorbt()
    entries_s, _exits, close_df, open_df, high_df, low_df = _vbt_signals(bars, vbt)
    no_exits = pd.DataFrame(False, index=bars.index, columns=["A"])
    pf = vbt.Portfolio.from_signals(
        close_df,
        entries_s,
        no_exits,
        price=open_df,
        open=open_df,
        high=high_df,
        low=low_df,
        init_cash=100_000.0,
        fees=fees,
        slippage=slippage,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        sl_trail=sl_trail,
        stop_entry_price=StopEntryPrice.FillPrice,
        stop_exit_price=StopExitPrice.StopMarket,
        group_by=True,
        cash_sharing=True,
        freq="1D",
    )
    idx = bars.index
    trades = sorted(
        [
            (
                idx[int(rec["entry_idx"])].date(),
                idx[int(rec["exit_idx"])].date(),
                float(rec["entry_price"]),
                float(rec["exit_price"]),
            )
            for rec in pf.trades.records_arr
        ],
        key=lambda x: x[0],
    )
    return trades, float(pf.total_return())


# ---------------------------------------------------------------------------
# Assertion helpers.
# ---------------------------------------------------------------------------


def _assert_prices_match(
    event_trades,
    vbt_trades: list[tuple[date, date, float, float]],
    *,
    exit_shift_bdays: int,
) -> None:
    """Assert identical trade count, entry dates, entry/exit prices (rtol=1e-9),
    and that exit dates differ by exactly ``exit_shift_bdays`` business days
    (0 for stop/target/trail exits, 1 for crossunder exits)."""
    assert len(event_trades) == len(vbt_trades), (
        f"trade count: event={len(event_trades)} vbt={len(vbt_trades)}"
    )
    assert len(event_trades) >= 1, "expected at least one trade"
    for i, (ev, vb) in enumerate(zip(event_trades, vbt_trades)):
        vb_entry_date, vb_exit_date, vb_entry_px, vb_exit_px = vb
        assert ev.entry_date == vb_entry_date, (
            f"trade {i}: entry date event={ev.entry_date} vbt={vb_entry_date}"
        )
        assert math.isclose(ev.entry_price, vb_entry_px, rel_tol=1e-9), (
            f"trade {i}: entry px event={ev.entry_price:.10f} vbt={vb_entry_px:.10f}"
        )
        assert math.isclose(ev.exit_price, vb_exit_px, rel_tol=1e-9), (
            f"trade {i}: exit px event={ev.exit_price:.10f} vbt={vb_exit_px:.10f}"
        )
        gap = len(pd.bdate_range(ev.exit_date, vb_exit_date)) - 1
        if pd.Timestamp(ev.exit_date) > pd.Timestamp(vb_exit_date):
            gap = -(len(pd.bdate_range(vb_exit_date, ev.exit_date)) - 1)
        assert gap == exit_shift_bdays, (
            f"trade {i}: exit-date gap expected {exit_shift_bdays} bday(s), got "
            f"{gap} (event={ev.exit_date} vbt={vb_exit_date})"
        )


def _assert_net_return_matches(event_trades, vbt_total_return: float) -> None:
    """Net compounded per-trade return equals vbt ``total_return`` to ≤1e-8."""
    event_net = net_compound_return(event_trades)
    assert math.isclose(event_net, vbt_total_return, abs_tol=1e-8), (
        f"net total_return: event={event_net:.12f} vbt={vbt_total_return:.12f} "
        f"abs_diff={abs(event_net - vbt_total_return):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 1 — Commission
# ---------------------------------------------------------------------------


def test_commission_reconciles() -> None:
    """``commission_bps=10`` ⇄ vbt ``fees=0.001``: identical fills, net return
    matches to ≤1e-8 (observed ~5e-16). Exit dates carry the +1-bday crossunder
    shift."""
    bars = _make_cost_frame()
    comm_bps = 10.0
    event_trades = _run_event(bars, _cost_cfg(commission_bps=comm_bps))
    vbt_trades, vbt_total = _run_vbt_costs(bars, fees=bps_to_fraction(comm_bps))

    _assert_prices_match(event_trades, vbt_trades, exit_shift_bdays=1)
    _assert_net_return_matches(event_trades, vbt_total)
    # Commission shows up in the trade accounting: entry_cost > gross notional.
    for tr in event_trades:
        gross = tr.shares * tr.entry_price
        assert tr.entry_cost > gross, "commission should inflate entry_cost"


# ---------------------------------------------------------------------------
# Test 2 — Slippage
# ---------------------------------------------------------------------------


def test_slippage_reconciles() -> None:
    """``slippage_bps=20`` ⇄ vbt ``slippage=0.002``: slipped fills match to
    ``rtol=1e-9`` and the net return matches to ≤1e-8 (observed ~4e-16)."""
    bars = _make_cost_frame()
    slip_bps = 20.0
    event_trades = _run_event(bars, _cost_cfg(slippage_bps=slip_bps))
    vbt_trades, vbt_total = _run_vbt_costs(bars, slippage=bps_to_fraction(slip_bps))

    _assert_prices_match(event_trades, vbt_trades, exit_shift_bdays=1)
    _assert_net_return_matches(event_trades, vbt_total)
    # Buy-side slippage lifts the entry fill above the raw bar open.
    raw = _make_cost_frame()
    for tr in event_trades:
        bar_open = float(raw.loc[pd.Timestamp(tr.entry_date), "open"])
        assert tr.entry_price > bar_open, "buy slippage should raise entry fill"


# ---------------------------------------------------------------------------
# Test 3 — Commission + slippage combined
# ---------------------------------------------------------------------------


def test_commission_plus_slippage_reconciles() -> None:
    """Both frictions together (``comm=10bps`` + ``slip=20bps``) reconcile to
    ≤1e-8 (observed ~1e-15)."""
    bars = _make_cost_frame()
    comm_bps, slip_bps = 10.0, 20.0
    event_trades = _run_event(
        bars, _cost_cfg(commission_bps=comm_bps, slippage_bps=slip_bps)
    )
    vbt_trades, vbt_total = _run_vbt_costs(
        bars,
        fees=bps_to_fraction(comm_bps),
        slippage=bps_to_fraction(slip_bps),
    )

    _assert_prices_match(event_trades, vbt_trades, exit_shift_bdays=1)
    _assert_net_return_matches(event_trades, vbt_total)


# ---------------------------------------------------------------------------
# Test 4 — Stop-loss
# ---------------------------------------------------------------------------


def test_stop_loss_reconciles() -> None:
    """``stop_loss=0.05`` ⇄ vbt ``sl_stop=0.05``: the single trade stops out
    intrabar at the stop reference, on the *same* date in both engines (no
    +1 shift), price to ``rtol=1e-9`` and net return to ≤1e-8."""
    bars = _frame_from_close(_spliced_close(np.linspace(92.0, 70.0, 30)))
    sl = 0.05
    event_trades = _run_event(bars, _stop_cfg(stop_loss=sl))
    vbt_trades, vbt_total = _run_vbt_stop(bars, sl_stop=sl)

    assert len(event_trades) == 1
    assert event_trades[0].exit_reason == "stop", (
        f"expected a stop-out, got {event_trades[0].exit_reason}"
    )
    _assert_prices_match(event_trades, vbt_trades, exit_shift_bdays=0)
    _assert_net_return_matches(event_trades, vbt_total)
    # Stop reference = slipped entry fill · (1 − sl); here slippage=0.
    assert math.isclose(
        event_trades[0].exit_price,
        event_trades[0].entry_price * (1.0 - sl),
        rel_tol=1e-12,
    )


# ---------------------------------------------------------------------------
# Test 5 — Take-profit
# ---------------------------------------------------------------------------


def test_take_profit_reconciles() -> None:
    """``take_profit=0.10`` ⇄ vbt ``tp_stop=0.10``: target hit intrabar, same
    exit date, price to ``rtol=1e-9`` and net return to ≤1e-8."""
    bars = _frame_from_close(_spliced_close(np.linspace(92.5, 120.0, 30)))
    tp = 0.10
    event_trades = _run_event(bars, _stop_cfg(take_profit=tp))
    vbt_trades, vbt_total = _run_vbt_stop(bars, tp_stop=tp)

    assert len(event_trades) == 1
    assert event_trades[0].exit_reason == "target", (
        f"expected a target hit, got {event_trades[0].exit_reason}"
    )
    _assert_prices_match(event_trades, vbt_trades, exit_shift_bdays=0)
    _assert_net_return_matches(event_trades, vbt_total)
    assert math.isclose(
        event_trades[0].exit_price,
        event_trades[0].entry_price * (1.0 + tp),
        rel_tol=1e-12,
    )


# ---------------------------------------------------------------------------
# Test 6 — Trailing stop
# ---------------------------------------------------------------------------


def test_trailing_stop_reconciles() -> None:
    """``trailing_stop=0.08`` ⇄ vbt ``sl_stop=0.08, sl_trail=True``.

    The frame rises to a peak then pulls back >8%. Both engines ratchet the stop
    up on new highs and exit at ``peak·(1−0.08)`` on the same bar. Reconciles to
    ≤1e-8 (observed ~3e-17). Pinned frame: the peak is reached before the
    pullback and no bar gaps through the trailing level, so the ratchet timing is
    unambiguous."""
    bars = _frame_from_close(
        _spliced_close(
            np.concatenate([np.linspace(92.5, 115.0, 20), np.linspace(114.0, 95.0, 15)])
        )
    )
    tr = 0.08
    event_trades = _run_event(bars, _stop_cfg(trailing_stop=tr))
    vbt_trades, vbt_total = _run_vbt_stop(bars, sl_stop=tr, sl_trail=True)

    assert len(event_trades) == 1
    assert event_trades[0].exit_reason == "trail", (
        f"expected a trailing-stop exit, got {event_trades[0].exit_reason}"
    )
    _assert_prices_match(event_trades, vbt_trades, exit_shift_bdays=0)
    _assert_net_return_matches(event_trades, vbt_total)


# ---------------------------------------------------------------------------
# Test 7 — Stop-loss WITH costs (exercises the slippage-adjusted stop base)
# ---------------------------------------------------------------------------


def test_stop_loss_with_costs_reconciles() -> None:
    """Stop-loss combined with commission + slippage.

    This is the test that pins the two stop-base reconciliation rules:
    ``StopEntryPrice.FillPrice`` (vbt measures ``sl_stop`` against the *slipped*
    entry fill, matching ``stop_ref = entry_fill·(1−sl)``) and
    ``StopExitPrice.StopMarket`` (vbt applies sell-side slippage on the stop
    fill, matching the engine). Reconciles to ≤1e-8 (observed ~1e-16)."""
    bars = _frame_from_close(_spliced_close(np.linspace(92.0, 70.0, 30)))
    sl, comm_bps, slip_bps = 0.05, 10.0, 20.0
    event_trades = _run_event(
        bars,
        _stop_cfg(stop_loss=sl, commission_bps=comm_bps, slippage_bps=slip_bps),
    )
    vbt_trades, vbt_total = _run_vbt_stop(
        bars,
        sl_stop=sl,
        fees=bps_to_fraction(comm_bps),
        slippage=bps_to_fraction(slip_bps),
    )

    assert len(event_trades) == 1
    assert event_trades[0].exit_reason == "stop"
    _assert_prices_match(event_trades, vbt_trades, exit_shift_bdays=0)
    _assert_net_return_matches(event_trades, vbt_total)
    # Stop reference is measured off the SLIPPED entry fill, then slipped again on
    # the sell side: exit = entry_fill·(1−sl)·(1−slip).
    slip = bps_to_fraction(slip_bps)
    expected_exit = event_trades[0].entry_price * (1.0 - sl) * (1.0 - slip)
    assert math.isclose(event_trades[0].exit_price, expected_exit, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Test 8 — Control: wrong fee units must diverge (comparison is non-trivial)
# ---------------------------------------------------------------------------


def test_control_wrong_fee_units_diverge() -> None:
    """Feeding vbt the *raw basis-point number* as a fraction (skipping
    ``bps_to_fraction``) diverges from the engine by >1.0 absolute — proving the
    unit conversion in tests 1/3/7 is load-bearing and the matches are not
    coincidental."""
    bars = _make_cost_frame()
    comm_bps = 10.0
    event_trades = _run_event(bars, _cost_cfg(commission_bps=comm_bps))
    event_net = net_compound_return(event_trades)

    # WRONG: fees=10.0 (i.e. 1000%), not 10 bps = 0.001.
    _vbt_trades, vbt_total_wrong = _run_vbt_costs(bars, fees=comm_bps)
    assert abs(event_net - vbt_total_wrong) > 1.0, (
        "wrong fee units should diverge massively; got "
        f"event={event_net:.6f} vbt_wrong={vbt_total_wrong:.6f}"
    )

    # SANITY: with the correct conversion the same frame reconciles tightly.
    _vbt_ok, vbt_total_ok = _run_vbt_costs(bars, fees=bps_to_fraction(comm_bps))
    assert math.isclose(event_net, vbt_total_ok, abs_tol=1e-8)
