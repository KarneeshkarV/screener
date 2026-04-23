"""Backtest Pine strategies from github.com/Alorse/pinescript-strategies.

Each strategy from that repo has been re-implemented in pandas/numpy here so
we can run it against our own OHLCV cache rather than TradingView.

Nine strategies are ported (all long-only, close-based entries/exits — intra-bar
stops and multi-timeframe ``request.security`` calls are dropped to keep the
port self-contained):

  supertrend        strategies/trend/Supertrend.pine
                    entry: ta.supertrend flips bullish
                    exit:  ta.supertrend flips bearish
  supertrend_rsi    strategies/trend/Supertrend + RSI.pine
                    entry: inLong AND RSI crosses above 50
                    exit:  RSI > 72 OR supertrend flips bearish
  macd_rsi          strategies/momentum/MACD+RSI.pine
                    entry: MACD crosses over signal AND RSI was < 30 in last 5
                    exit:  MACD crosses under signal AND RSI was > 70 in last 5
  rsi_ema           strategies/momentum/RSI + EMA.pine
                    entry: RSI < 30 AND EMA150 > EMA600 (bull regime)
                    exit:  RSI > 70
  ma_cross          strategies/trend/MA Cross + DMI.pine
                    entry: EMA10 crosses over EMA20
                    exit:  EMA10 crosses under EMA20
  bb_breakout       strategies/mean-reversion/Bollinger Breakout [kodify].pine
                    entry: close crosses over SMA350 + 2.5σ
                    exit:  close crosses under SMA350
  ma_cross_regime   ma_cross gated by the rsi_ema bull regime
  ma_cross_st_entry ma_cross AND supertrend bullish
  ma_cross_st_exit  ma_cross entry, supertrend bearish exit

Indicator math is shared with the Pine AST evaluator via :mod:`screener.indicators`
to keep the two implementations from drifting.

The universe must be supplied explicitly via ``--universe-file`` (a
newline-separated list of tickers). We deliberately do NOT pull from the live
TradingView scanner, because a current top-N-by-volume list injects
survivorship bias into a historical backtest (tickers that existed throughout
the window but were delisted, halted, or deleveraged as of today would be
silently excluded).

Usage:
    uv run python run_pinescript_strategies.py --market us --years 3 \\
        --universe-file us_universe.txt
"""
from __future__ import annotations

import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import click
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from screener import indicators
from screener.backtester.data import YFinancePriceFetcher, tv_to_yf

BENCHMARKS = {"us": "SPY", "india": "^NSEI"}

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date,
    market: str,
    fetcher: YFinancePriceFetcher,
) -> pd.DataFrame | None:
    yf_sym = ticker if ticker.startswith("^") else tv_to_yf(ticker, market)
    frames = fetcher.fetch([yf_sym], start, end)
    df = frames.get(yf_sym)
    if df is None or df.empty:
        return None
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    return df


def load_universe_file(path: Path) -> list[str]:
    """Read a newline-separated ticker list. Blank lines and ``#`` comments
    are ignored."""
    content = path.read_text()
    return [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# ───────────────────────── indicators ───────────────────────────────────
# Thin numpy adapters over the shared screener.indicators module so the
# supertrend routine (which does an explicit loop) keeps its ndarray inputs.


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(indicators.ema(x, n))


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(indicators.sma(x, n))


def _stdev(x: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(indicators.stdev(x, n))


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    return np.asarray(indicators.rsi(close, n))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    return np.asarray(indicators.atr(high, low, close, n))


def _supertrend_dir(high, low, close, period=10, mult=3.0) -> np.ndarray:
    """Return direction array matching Pine ``ta.supertrend`` semantics:
    direction < 0 → uptrend (inLong); direction > 0 → downtrend."""
    n = len(close)
    hl2 = (high + low) / 2.0
    atr = _atr(high, low, close, period)
    upper_b = hl2 + mult * atr
    lower_b = hl2 - mult * atr
    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    direction = np.ones(n, dtype=np.int8)  # down-trend by convention before first flip

    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = upper_b[i]
            final_lower[i] = lower_b[i]
            continue
        if upper_b[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_b[i]
        else:
            final_upper[i] = final_upper[i - 1]
        if lower_b[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_b[i]
        else:
            final_lower[i] = final_lower[i - 1]
        if close[i] > final_upper[i - 1]:
            direction[i] = -1
        elif close[i] < final_lower[i - 1]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
    return direction


# ───────────────────────── trade model + walker ────────────────────────

@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp

    @property
    def ret(self) -> float:
        return self.exit_px / self.entry_px - 1.0 if self.entry_px > 0 else 0.0


def _walk(entries: np.ndarray, exits: np.ndarray, close: np.ndarray, dates) -> list[Trade]:
    """Long-only round-trip walker. Entry fires on bar close; exit fires on
    bar close. Open position at end-of-history is force-closed on the last bar."""
    trades: list[Trade] = []
    in_pos = False
    entry_i = -1
    entry_px = 0.0
    n = len(close)
    for i in range(n):
        if not in_pos:
            if entries[i]:
                in_pos = True
                entry_i = i
                entry_px = float(close[i])
        else:
            if exits[i]:
                trades.append(Trade(
                    entry_i, i, entry_px, float(close[i]),
                    pd.Timestamp(dates[entry_i]), pd.Timestamp(dates[i]),
                ))
                in_pos = False
    if in_pos:
        trades.append(Trade(
            entry_i, n - 1, entry_px, float(close[-1]),
            pd.Timestamp(dates[entry_i]), pd.Timestamp(dates[-1]),
        ))
    return trades


# ──────────────────────── ported strategies ────────────────────────────

def strat_supertrend(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    dp = np.concatenate(([d[0]], d[:-1]))
    entries = (d < 0) & (dp >= 0)
    exits   = (d > 0) & (dp <= 0)
    return _walk(entries, exits, close, df["date"].values)


def strat_supertrend_rsi(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    rsi = _rsi(close, 14)
    inLong = d < 0
    rsi_prev = np.concatenate(([np.nan], rsi[:-1]))
    entries = inLong & (rsi_prev < 50) & (rsi > 50)
    dp = np.concatenate(([d[0]], d[:-1]))
    flip_down = (d > 0) & (dp <= 0)
    exits = (rsi > 72) | flip_down
    return _walk(entries, exits, close, df["date"].values)


def strat_macd_rsi(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    macd = _ema(close, 12) - _ema(close, 26)
    sig  = _ema(macd, 9)
    rsi = _rsi(close, 14)
    mp = np.concatenate(([macd[0]], macd[:-1]))
    sp = np.concatenate(([sig[0]],  sig[:-1]))
    cross_over  = (mp <= sp) & (macd > sig)
    cross_under = (mp >= sp) & (macd < sig)
    # rolling "rsi was below 30 in last 5 bars" (window excludes current bar)
    n = len(close)
    was_down = np.zeros(n, dtype=bool)
    was_up   = np.zeros(n, dtype=bool)
    lookback = 5
    for i in range(1, n):
        lo = max(0, i - lookback)
        w = rsi[lo:i]
        if w.size:
            if np.any(w <= 30):
                was_down[i] = True
            if np.any(w >= 70):
                was_up[i] = True
    entries = cross_over & was_down
    exits   = cross_under & was_up
    return _walk(entries, exits, close, df["date"].values)


def strat_rsi_ema(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    rsi = _rsi(close, 14)
    regime = _ema(close, 150) > _ema(close, 600)
    entries = (rsi < 30) & regime
    exits   = rsi > 70
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    entries = (mfp <= msp) & (mf > ms)
    exits   = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)


def strat_bb_breakout(df: pd.DataFrame) -> list[Trade]:
    close = df["close"].to_numpy(dtype=float)
    s  = _sma(close, 350)
    sd = _stdev(close, 350)
    upper = s + 2.5 * sd
    cp = np.concatenate(([close[0]], close[:-1]))
    up = np.concatenate(([upper[0]], upper[:-1]))
    sp = np.concatenate(([s[0]],     s[:-1]))
    entries = (cp <= up) & (close > upper)
    exits   = (cp >= sp) & (close < s)
    valid = ~np.isnan(upper)
    entries &= valid
    exits   &= valid
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross_regime(df: pd.DataFrame) -> list[Trade]:
    """ma_cross entries gated by rsi_ema's EMA150 > EMA600 bull regime."""
    close = df["close"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    regime = _ema(close, 150) > _ema(close, 600)
    entries = (mfp <= msp) & (mf > ms) & regime
    exits   = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross_st_entry(df: pd.DataFrame) -> list[Trade]:
    """Entry = ma_cross AND supertrend bullish; exit = ma_cross bearish."""
    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    entries = (mfp <= msp) & (mf > ms) & (d < 0)
    exits   = (mfp >= msp) & (mf < ms)
    return _walk(entries, exits, close, df["date"].values)


def strat_ma_cross_st_exit(df: pd.DataFrame) -> list[Trade]:
    """Entry = ma_cross bullish; exit = supertrend flips bearish."""
    close = df["close"].to_numpy(dtype=float)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    mf = _ema(close, 10)
    ms = _ema(close, 20)
    mfp = np.concatenate(([mf[0]], mf[:-1]))
    msp = np.concatenate(([ms[0]], ms[:-1]))
    d = _supertrend_dir(high, low, close, period=10, mult=3.0)
    dp = np.concatenate(([d[0]], d[:-1]))
    entries = (mfp <= msp) & (mf > ms)
    exits   = (d > 0) & (dp <= 0)
    return _walk(entries, exits, close, df["date"].values)


STRATEGIES = {
    "supertrend":     strat_supertrend,
    "supertrend_rsi": strat_supertrend_rsi,
    "macd_rsi":       strat_macd_rsi,
    "rsi_ema":        strat_rsi_ema,
    "ma_cross":       strat_ma_cross,
    "bb_breakout":    strat_bb_breakout,
    "ma_cross_regime":   strat_ma_cross_regime,
    "ma_cross_st_entry": strat_ma_cross_st_entry,
    "ma_cross_st_exit":  strat_ma_cross_st_exit,
}


# ─────────────────────────── aggregation ───────────────────────────────

def _compound(trades: list[Trade]) -> float:
    r = 1.0
    for t in trades:
        r *= (1 + t.ret)
    return r - 1.0


def _run_ticker(df: pd.DataFrame, window_start: pd.Timestamp, strategy_fn) -> dict | None:
    """Run one strategy on one ticker. Indicators warm up on pre-window bars
    but trades are counted only if the entry falls in [window_start, end]."""
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 50:
        return None
    trades = strategy_fn(df)
    in_win = [t for t in trades if t.entry_date >= window_start]
    n_bars_window = int((pd.to_datetime(df["date"]) >= window_start).sum())
    exposure = sum(t.exit_idx - t.entry_idx for t in in_win)
    return {
        "n_trades":     len(in_win),
        "n_bars":       n_bars_window,
        "exposure":     exposure,
        "total_return": _compound(in_win),
        "wins":         sum(1 for t in in_win if t.ret > 0),
        "trades":       in_win,
    }


# ──────────────────────────── CLI ──────────────────────────────────────

@click.command()
@click.option("--market", type=click.Choice(["us", "india"]), default="us")
@click.option("--years", type=int, default=3, help="Backtest window length (years).")
@click.option(
    "--universe-file",
    "universe_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Newline-separated ticker list. Required: a live TradingView scan "
    "would inject survivorship bias into historical results.",
)
@click.option("--limit", type=int, default=0, help="Cap universe size (0 = all).")
@click.option("--trades-json", type=str, default=None,
              help="If set, write per-strategy top-trader ticker lists to this JSON file.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose progress logging.")
def main(
    market: str,
    years: int,
    universe_file: Path,
    limit: int,
    trades_json: str | None,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    today = date.today()
    window_start_ts = pd.Timestamp(today) - pd.DateOffset(years=years)
    window_start_ts = window_start_ts.normalize()
    # EMA600 ≈ 2.4y; give it extra warmup
    fetch_start = (pd.Timestamp(today) - pd.DateOffset(years=years + 4)).date()
    fetch_end = today

    tickers = load_universe_file(universe_file)
    if limit and limit < len(tickers):
        tickers = tickers[:limit]
    logger.info("Universe:   %s (%d tickers from %s)", market, len(tickers), universe_file)
    logger.info("Window:     %s → %s (%dy)", window_start_ts.date(), today, years)
    logger.info("Warmup pad: %s → %s", fetch_start, window_start_ts.date())
    logger.info("Strategies: %s", ", ".join(STRATEGIES))

    fetcher = YFinancePriceFetcher()

    # ── fetch ───────────────────────────────────────────────────────────
    ohlcv: dict[str, pd.DataFrame] = {}

    def _fetch(t: str):
        df = fetch_ohlcv(t, fetch_start, fetch_end, market, fetcher)
        return t, df

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            t, df = fut.result()
            if df is not None and not df.empty:
                ohlcv[t] = df
            if i % 50 == 0 or i == len(tickers):
                logger.info("  fetched %d/%d (%d have data)", i, len(tickers), len(ohlcv))

    # ── benchmark buy-and-hold over window ──────────────────────────────
    bench_sym = BENCHMARKS[market]
    bench_df = fetch_ohlcv(bench_sym, fetch_start, fetch_end, market, fetcher)
    bench_return: float | None = None
    if bench_df is not None and not bench_df.empty:
        b = bench_df.sort_values("date")
        b = b[pd.to_datetime(b["date"]) >= window_start_ts]
        if len(b) > 1:
            bench_return = float(b["adj_close"].iloc[-1] / b["adj_close"].iloc[0] - 1.0)
    if bench_return is None:
        logger.warning("  benchmark %s missing — alpha column will be blank", bench_sym)

    # ── run every strategy on every ticker ──────────────────────────────
    per_strat: dict[str, list[dict]] = {n: [] for n in STRATEGIES}
    err_counts: dict[str, int] = {n: 0 for n in STRATEGIES}
    for i, (t, df) in enumerate(ohlcv.items(), 1):
        for name, fn in STRATEGIES.items():
            try:
                res = _run_ticker(df, window_start_ts, fn)
            except Exception:
                err_counts[name] += 1
                logger.debug("  %s failed on %s", name, t, exc_info=True)
                continue
            if res is None:
                continue
            per_strat[name].append(res | {"ticker": t})
        if i % 100 == 0 or i == len(ohlcv):
            logger.info("  backtested %d/%d tickers", i, len(ohlcv))

    # ── output table ────────────────────────────────────────────────────
    HDR = (f"{'Strategy':<18} {'Tkrs':>5} {'Trades':>7} {'Tr/Tk':>6} "
           f"{'Basket':>9} {'Median':>9} {'Bench':>9} {'Alpha':>9} "
           f"{'Win%':>6} {'Exp%':>6}")
    print()
    print("=" * (len(HDR) + 2))
    print(f"{market.upper()}  |  window {window_start_ts.date()} → {today}  |  "
          f"bench={bench_sym}={'-' if bench_return is None else f'{bench_return:+.1%}'}")
    print("=" * (len(HDR) + 2))
    print(HDR)
    print("-" * len(HDR))
    rows = []
    for name in STRATEGIES:
        results = per_strat[name]
        if not results:
            print(f"{name:<18}  no results  (errors: {err_counts[name]})")
            continue
        n_t = len(results)
        returns = [r["total_return"] for r in results]
        total_trades = sum(r["n_trades"] for r in results)
        total_wins   = sum(r["wins"]     for r in results)
        total_exp    = sum(r["exposure"] for r in results)
        total_bars   = sum(r["n_bars"]   for r in results) or 1
        basket = float(np.mean(returns))
        med    = float(np.median(returns))
        win    = (total_wins / total_trades) if total_trades else float("nan")
        alpha  = (basket - bench_return) if bench_return is not None else float("nan")
        rows.append({
            "strategy": name, "n": n_t, "trades": total_trades,
            "basket": basket, "median": med, "alpha": alpha,
            "win_rate": win, "exposure": total_exp / total_bars,
        })
        print(
            f"{name:<18} {n_t:>5} {total_trades:>7} "
            f"{total_trades/n_t:>6.1f} "
            f"{basket:>+9.1%} {med:>+9.1%} "
            f"{('-' if bench_return is None else f'{bench_return:+.1%}'):>9} "
            f"{('-' if np.isnan(alpha) else f'{alpha:+.1%}'):>9} "
            f"{win:>6.1%} {total_exp/total_bars:>6.1%}"
        )
    print()

    # ── ranking block ───────────────────────────────────────────────────
    if rows:
        best_alpha  = max(rows, key=lambda r: r["alpha"] if not np.isnan(r["alpha"]) else -9e9)
        best_basket = max(rows, key=lambda r: r["basket"])
        best_win    = max(rows, key=lambda r: r["win_rate"])
        print("Best in this market:")
        print(f"  highest alpha:       {best_alpha['strategy']:<18} "
              f"alpha={best_alpha['alpha']:+.1%}  basket={best_alpha['basket']:+.1%}")
        print(f"  highest basket rtn:  {best_basket['strategy']:<18} "
              f"basket={best_basket['basket']:+.1%}")
        print(f"  highest win rate:    {best_win['strategy']:<18} "
              f"win={best_win['win_rate']:.1%}  trades={best_win['trades']}")
        print()

    # ── per-strategy ticker dump ────────────────────────────────────────
    if trades_json:
        import json
        payload = {
            "market": market,
            "window_start": str(window_start_ts.date()),
            "window_end": str(today),
            "strategies": {},
        }
        for name, results in per_strat.items():
            traded = [r for r in results if r["n_trades"] > 0]
            traded.sort(key=lambda r: r["total_return"], reverse=True)
            payload["strategies"][name] = {
                "n_tickers_traded": len(traded),
                "tickers": [
                    {
                        "ticker": r["ticker"],
                        "n_trades": r["n_trades"],
                        "wins": r["wins"],
                        "return": round(r["total_return"], 4),
                    }
                    for r in traded
                ],
            }
        with open(trades_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("wrote traded-ticker dump → %s", trades_json)


if __name__ == "__main__":
    main()
