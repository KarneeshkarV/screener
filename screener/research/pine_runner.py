"""Backtest implemented research strategies over market universes."""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import click
import numpy as np
import pandas as pd

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.logging_config import get_logger
from screener.scanner import scan as _tv_scan
from screener.strategies.registry import STRATEGIES
from screener.strategies.trades import Trade

warnings.filterwarnings("ignore")

log = get_logger("pine_runner")

BENCHMARKS = {"us": "SPY", "india": "^NSEI"}

_FETCHER = YFinancePriceFetcher()


def fetch_ohlcv(ticker, start, end, market, refresh=False):
    yf_sym = ticker if ticker.startswith("^") else tv_to_yf(ticker, market)
    fetcher = YFinancePriceFetcher(refresh=True) if refresh else _FETCHER
    frames = fetcher.fetch([yf_sym], start, end)
    df = frames.get(yf_sym)
    if df is None or df.empty:
        return None
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    return df


def load_universe(market, _unused=None):
    from tradingview_screener import col

    # Price floor strips OTC sub-penny tickers that volume-rank to the top.
    price_floor = {"us": 5.0, "india": 50.0}.get(market, 5.0)
    filters = [col("type") == "stock", col("close") >= price_floor]
    _total, df = _tv_scan(market=market, filters=filters, limit=500, order_by="volume")
    return [str(t) for t in df["name"].dropna().tolist()]


def _compound(trades: list[Trade]) -> float:
    r = 1.0
    for t in trades:
        r *= 1 + t.ret
    return r - 1.0


def _run_ticker(df: pd.DataFrame, window_start: pd.Timestamp, strategy_fn) -> dict | None:
    """Run one strategy on one ticker with pre-window indicator warmup."""
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 50:
        return None
    trades = strategy_fn(df)
    in_win = [t for t in trades if t.entry_date >= window_start]
    n_bars_window = int((pd.to_datetime(df["date"]) >= window_start).sum())
    exposure = sum(t.exit_idx - t.entry_idx for t in in_win)
    return {
        "n_trades": len(in_win),
        "n_bars": n_bars_window,
        "exposure": exposure,
        "total_return": _compound(in_win),
        "wins": sum(1 for t in in_win if t.ret > 0),
        "trades": in_win,
    }


@click.command()
@click.option("--market", type=click.Choice(["us", "india"]), default="us")
@click.option("--years", type=int, default=3, help="Backtest window length (years).")
@click.option("--limit", type=int, default=0, help="Cap universe size (0 = all).")
@click.option("--refresh", is_flag=True, help="Force re-fetch OHLCV.")
@click.option(
    "--trades-json",
    type=str,
    default=None,
    help="If set, write per-strategy top-trader ticker lists to this JSON file.",
)
def main(market: str, years: int, limit: int, refresh: bool, trades_json: str | None) -> None:
    today = date.today()
    window_start_ts = pd.Timestamp(today) - pd.DateOffset(years=years)
    window_start_ts = window_start_ts.normalize()
    fetch_start = (pd.Timestamp(today) - pd.DateOffset(years=years + 4)).date()
    fetch_end = today

    tickers = load_universe(market, None)
    if limit and limit < len(tickers):
        tickers = tickers[:limit]
    log.info(
        "backtest.run_started",
        market=market,
        tickers=len(tickers),
        window_start=str(window_start_ts.date()),
        window_end=str(today),
        years=years,
        warmup_start=str(fetch_start),
        strategies=list(STRATEGIES),
    )

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
                log.info(
                    "backtest.fetch_progress",
                    fetched=i,
                    total=len(tickers),
                    with_data=len(ohlcv),
                )

    bench_sym = BENCHMARKS[market]
    bench_df = fetch_ohlcv(bench_sym, fetch_start, fetch_end, market, refresh=refresh)
    bench_return: float | None = None
    if bench_df is not None and not bench_df.empty:
        b = bench_df.sort_values("date")
        b = b[pd.to_datetime(b["date"]) >= window_start_ts]
        if len(b) > 1:
            bench_return = float(b["adj_close"].iloc[-1] / b["adj_close"].iloc[0] - 1.0)
    if bench_return is None:
        log.warning("backtest.benchmark_missing", benchmark=bench_sym)

    per_strat: dict[str, list[dict]] = {n: [] for n in STRATEGIES}
    err_counts: dict[str, int] = {n: 0 for n in STRATEGIES}
    for i, (t, df) in enumerate(ohlcv.items(), 1):
        for name, fn in STRATEGIES.items():
            try:
                res = _run_ticker(df, window_start_ts, fn)
            except (ValueError, KeyError, TypeError, RuntimeError, IndexError):
                err_counts[name] += 1
                continue
            if res is None:
                continue
            per_strat[name].append(res | {"ticker": t})
        if i % 100 == 0 or i == len(ohlcv):
            log.info("backtest.iter_progress", processed=i, total=len(ohlcv))

    hdr = (
        f"{'Strategy':<18} {'Tkrs':>5} {'Trades':>7} {'Tr/Tk':>6} "
        f"{'Basket':>9} {'Median':>9} {'Bench':>9} {'Alpha':>9} "
        f"{'Win%':>6} {'Exp%':>6}"
    )
    print()
    print("=" * (len(hdr) + 2))
    print(
        f"{market.upper()}  |  window {window_start_ts.date()} -> {today}  |  "
        f"bench={bench_sym}={'-' if bench_return is None else f'{bench_return:+.1%}'}"
    )
    print("=" * (len(hdr) + 2))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name in STRATEGIES:
        results = per_strat[name]
        if not results:
            print(f"{name:<18}  no results  (errors: {err_counts[name]})")
            continue
        n_t = len(results)
        returns = [r["total_return"] for r in results]
        total_trades = sum(r["n_trades"] for r in results)
        total_wins = sum(r["wins"] for r in results)
        total_exp = sum(r["exposure"] for r in results)
        total_bars = sum(r["n_bars"] for r in results) or 1
        basket = float(np.mean(returns))
        med = float(np.median(returns))
        win = (total_wins / total_trades) if total_trades else float("nan")
        alpha = (basket - bench_return) if bench_return is not None else float("nan")
        rows.append(
            {
                "strategy": name,
                "n": n_t,
                "trades": total_trades,
                "basket": basket,
                "median": med,
                "alpha": alpha,
                "win_rate": win,
                "exposure": total_exp / total_bars,
            }
        )
        print(
            f"{name:<18} {n_t:>5} {total_trades:>7} "
            f"{total_trades / n_t:>6.1f} "
            f"{basket:>+9.1%} {med:>+9.1%} "
            f"{('-' if bench_return is None else f'{bench_return:+.1%}'):>9} "
            f"{('-' if np.isnan(alpha) else f'{alpha:+.1%}'):>9} "
            f"{win:>6.1%} {total_exp / total_bars:>6.1%}"
        )
    print()

    if rows:
        best_alpha = max(
            rows, key=lambda r: r["alpha"] if not np.isnan(r["alpha"]) else -9e9
        )
        best_basket = max(rows, key=lambda r: r["basket"])
        best_win = max(rows, key=lambda r: r["win_rate"])
        print("Best in this market:")
        print(
            f"  highest alpha:       {best_alpha['strategy']:<18} "
            f"alpha={best_alpha['alpha']:+.1%}  basket={best_alpha['basket']:+.1%}"
        )
        print(
            f"  highest basket rtn:  {best_basket['strategy']:<18} "
            f"basket={best_basket['basket']:+.1%}"
        )
        print(
            f"  highest win rate:    {best_win['strategy']:<18} "
            f"win={best_win['win_rate']:.1%}  trades={best_win['trades']}"
        )
        print()

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
        log.info("backtest.trades_dump_written", path=trades_json)


if __name__ == "__main__":
    main()
