"""Backtest Pine strategies from github.com/Alorse/pinescript-strategies.

Each strategy from that repo has been re-implemented in numpy/pandas here so we
can run it on our own OHLCV cache (fetch_ohlcv) rather than TradingView.

Six strategies are ported (all long-only, close-based entries/exits — intra-bar
stops and multi-timeframe request.security calls are dropped to keep the port
self-contained):

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

For each (market, strategy) we walk every ticker in the default universe over a
*window*, collect round-trip trades (indicators warm up on data before the
window), and report basket compounded return vs buy-and-hold benchmark.

Usage:
    uv run python run_pinescript_strategies.py --market us --years 3
    uv run python run_pinescript_strategies.py --market india --years 5
    uv run python run_pinescript_strategies.py --market us --years 3 --limit 50
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date

import click
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.scanner import scan as _tv_scan

BENCHMARKS = {"us": "SPY", "india": "^NSEI"}

_FETCHER = YFinancePriceFetcher()


def fetch_ohlcv(ticker, start, end, market, refresh=False):
    yf_sym = ticker if ticker.startswith("^") else tv_to_yf(ticker, market)
    frames = _FETCHER.fetch([yf_sym], start, end)
    df = frames.get(yf_sym)
    if df is None or df.empty:
        return None
    df = df.reset_index()
    first = df.columns[0]
    if first.lower() in {"date", "index"}:
        df = df.rename(columns={first: "date"})
    else:
        df = df.rename(columns={first: "date"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    return df


def load_universe(market, _unused=None):
    from tradingview_screener import col
    stock_only = [col("type") == "stock"]
    _total, df = _tv_scan(market=market, filters=stock_only, limit=500, order_by="volume")
    return [str(t) for t in df["name"].dropna().tolist()]


# ───────────────────────── indicators (numpy) ──────────────────────────

def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's RMA — matches Pine ta.rma."""
    out = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < n:
        return out
    alpha = 1.0 / n
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1)
    out = np.empty(len(x), dtype=np.float64)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _stdev(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).std(ddof=0).to_numpy()


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    rma_up = _rma(up, n)
    rma_dn = _rma(dn, n)
    rs = np.where(rma_dn > 0, rma_up / np.maximum(rma_dn, 1e-12), np.inf)
    rsi = 100 - 100 / (1 + rs)
    rsi[rma_dn == 0] = 100
    return rsi


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return _rma(tr, n)


def _supertrend_dir(high, low, close, period=10, mult=3.0) -> np.ndarray:
    """Return direction array matching Pine ta.supertrend semantics:
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
        # Seed the final bands on the first valid bar (or after any NaN gap).
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = upper_b[i]
            final_lower[i] = lower_b[i]
            continue
        # Sticky upper ratchets down until close breaks above it.
        if upper_b[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_b[i]
        else:
            final_upper[i] = final_upper[i - 1]
        # Sticky lower ratchets up until close breaks below it.
        if lower_b[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_b[i]
        else:
            final_lower[i] = final_lower[i - 1]
        # Flip direction on close vs prior final band.
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
            if np.any(w <= 30): was_down[i] = True
            if np.any(w >= 70): was_up[i]   = True
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
    # combined strategies
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
@click.option("--limit", type=int, default=0, help="Cap universe size (0 = all).")
@click.option("--refresh", is_flag=True, help="Force re-fetch OHLCV.")
@click.option("--trades-json", type=str, default=None,
              help="If set, write per-strategy top-trader ticker lists to this JSON file.")
def main(market: str, years: int, limit: int, refresh: bool, trades_json: str | None) -> None:
    today = date.today()
    window_start_ts = pd.Timestamp(today) - pd.DateOffset(years=years)
    window_start_ts = window_start_ts.normalize()
    # EMA600 ≈ 2.4y; give it extra warmup
    fetch_start = (pd.Timestamp(today) - pd.DateOffset(years=years + 4)).date()
    fetch_end = today

    tickers = load_universe(market, None)
    if limit and limit < len(tickers):
        tickers = tickers[:limit]
    print(f"Universe:   {market} ({len(tickers)} tickers)", file=sys.stderr)
    print(f"Window:     {window_start_ts.date()} → {today} ({years}y)", file=sys.stderr)
    print(f"Warmup pad: {fetch_start} → {window_start_ts.date()}", file=sys.stderr)
    print(f"Strategies: {', '.join(STRATEGIES)}", file=sys.stderr)

    # ── fetch ───────────────────────────────────────────────────────────
    ohlcv: dict[str, pd.DataFrame] = {}

    def _fetch(t: str):
        df = fetch_ohlcv(t, fetch_start, fetch_end, market, refresh=refresh)
        return t, df

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futs), 1):
            t, df = fut.result()
            if df is not None and not df.empty:
                ohlcv[t] = df
            if i % 50 == 0 or i == len(tickers):
                print(f"  fetched {i}/{len(tickers)} ({len(ohlcv)} have data)",
                      file=sys.stderr, flush=True)

    # ── benchmark buy-and-hold over window ──────────────────────────────
    bench_sym = BENCHMARKS[market]
    bench_df = fetch_ohlcv(bench_sym, fetch_start, fetch_end, market, refresh=refresh)
    bench_return: float | None = None
    if bench_df is not None and not bench_df.empty:
        b = bench_df.sort_values("date")
        b = b[pd.to_datetime(b["date"]) >= window_start_ts]
        if len(b) > 1:
            bench_return = float(b["adj_close"].iloc[-1] / b["adj_close"].iloc[0] - 1.0)
    if bench_return is None:
        print(f"  benchmark {bench_sym} missing — alpha column will be blank",
              file=sys.stderr)

    # ── run every strategy on every ticker ──────────────────────────────
    per_strat: dict[str, list[dict]] = {n: [] for n in STRATEGIES}
    err_counts: dict[str, int] = {n: 0 for n in STRATEGIES}
    for i, (t, df) in enumerate(ohlcv.items(), 1):
        for name, fn in STRATEGIES.items():
            try:
                res = _run_ticker(df, window_start_ts, fn)
            except Exception:
                err_counts[name] += 1
                continue
            if res is None:
                continue
            per_strat[name].append(res | {"ticker": t})
        if i % 100 == 0 or i == len(ohlcv):
            print(f"  backtested {i}/{len(ohlcv)} tickers", file=sys.stderr, flush=True)

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
        print(f"wrote traded-ticker dump → {trades_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
