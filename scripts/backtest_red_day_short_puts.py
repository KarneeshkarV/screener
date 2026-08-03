from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from screener.backtester.data import build_price_fetcher


@dataclass(frozen=True)
class MarketConfig:
    name: str
    tickers: tuple[str, ...]
    display_names: dict[str, str]
    contract_multipliers: dict[str, int]
    currency: str
    initial_capital: float
    commission_per_contract_per_side: float


MARKETS = {
    "us": MarketConfig(
        name="US leveraged ETFs",
        tickers=("TQQQ", "SOXL", "NVDL"),
        display_names={"TQQQ": "TQQQ", "SOXL": "SOXL", "NVDL": "NVDL"},
        contract_multipliers={"TQQQ": 100, "SOXL": 100, "NVDL": 100},
        currency="$",
        initial_capital=100_000.0,
        commission_per_contract_per_side=0.65,
    ),
    "india": MarketConfig(
        name="India index options",
        tickers=("^NSEI", "^NSEBANK", "NIFTY_FIN_SERVICE.NS"),
        display_names={
            "^NSEI": "NIFTY",
            "^NSEBANK": "BANKNIFTY",
            "NIFTY_FIN_SERVICE.NS": "FINNIFTY",
        },
        contract_multipliers={
            "^NSEI": 65,
            "^NSEBANK": 30,
            "NIFTY_FIN_SERVICE.NS": 60,
        },
        currency="INR",
        initial_capital=10_000_000.0,
        commission_per_contract_per_side=20.0,
    ),
}
WINDOW_YEARS = (5, 3, 2, 1)
RISK_FREE_RATE = 0.04
TARGET_PUT_DELTA = -0.30
DTE = 30
IV_MULTIPLIER = 1.25
MIN_IV = 0.35
MAX_IV = 2.50


@dataclass(frozen=True)
class Position:
    ticker: str
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    strike: float
    entry_premium: float
    contracts: int
    contract_multiplier: int
    reserve: float


@dataclass(frozen=True)
class Trade:
    ticker: str
    entry_date: date
    exit_date: date
    entry_underlying: float
    exit_underlying: float
    strike: float
    entry_premium: float
    exit_premium: float
    contracts: int
    pnl: float
    return_on_reserve: float
    exit_reason: str
    dte_held: int


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_ppf(p: float) -> float:
    # Acklam inverse-normal approximation. Accurate enough for strike selection.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def put_price(spot: float, strike: float, years: float, iv: float) -> float:
    if years <= 0:
        return max(strike - spot, 0.0)
    vol_sqrt_t = iv * sqrt(years)
    if vol_sqrt_t <= 0:
        return max(strike * exp(-RISK_FREE_RATE * years) - spot, 0.0)
    d1 = (log(spot / strike) + (RISK_FREE_RATE + 0.5 * iv * iv) * years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return strike * exp(-RISK_FREE_RATE * years) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def strike_for_delta(spot: float, years: float, iv: float, put_delta: float) -> float:
    d1 = norm_ppf(put_delta + 1.0)
    return spot / exp(d1 * iv * sqrt(years) - (RISK_FREE_RATE + 0.5 * iv * iv) * years)


def modeled_iv(frame: pd.DataFrame) -> pd.Series:
    returns = np.log(frame["close"].astype(float)).diff()
    rv20 = returns.rolling(20).std() * sqrt(252)
    rv60 = returns.rolling(60).std() * sqrt(252)
    iv = pd.concat([rv20, rv60], axis=1).max(axis=1) * IV_MULTIPLIER
    return iv.clip(lower=MIN_IV, upper=MAX_IV).ffill()


def next_trading_day_on_or_after(
    index: pd.DatetimeIndex, ts: pd.Timestamp
) -> pd.Timestamp | None:
    loc = index.searchsorted(ts, side="left")
    if loc >= len(index):
        return None
    return index[loc]


def run_window(
    bars_by_ticker: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: MarketConfig,
    initial_capital: float,
) -> tuple[pd.Series, list[Trade], dict[str, str]]:
    common_index = sorted(
        set().union(
            *[
                set(frame.loc[(frame.index >= start) & (frame.index <= end)].index)
                for frame in bars_by_ticker.values()
                if not frame.empty
            ]
        )
    )
    cash = {ticker: initial_capital / len(config.tickers) for ticker in config.tickers}
    positions: dict[str, Position | None] = {ticker: None for ticker in config.tickers}
    trades: list[Trade] = []
    equity_points: dict[pd.Timestamp, float] = {}
    data_notes: dict[str, str] = {}

    for current in common_index:
        total_equity = 0.0
        for ticker, frame in bars_by_ticker.items():
            sliced = frame.loc[frame.index <= current]
            if sliced.empty or current not in frame.index:
                total_equity += cash[ticker]
                continue
            row = frame.loc[current]
            spot = float(row["close"])
            iv = float(row["iv"])
            pos = positions[ticker]

            if pos is not None:
                years_left = max((pos.expiry_date - current).days, 0) / 365.0
                mark = put_price(spot, pos.strike, years_left, iv)
                exit_reason = None
                if mark <= 0.5 * pos.entry_premium:
                    exit_reason = "50pct_profit"
                elif current >= pos.expiry_date:
                    mark = max(pos.strike - spot, 0.0)
                    exit_reason = "expiration"
                if exit_reason:
                    gross = (
                        (pos.entry_premium - mark)
                        * pos.contract_multiplier
                        * pos.contracts
                    )
                    fees = config.commission_per_contract_per_side * pos.contracts * 2
                    pnl = gross - fees
                    cash[ticker] += pos.reserve + pnl
                    trades.append(
                        Trade(
                            ticker=ticker,
                            entry_date=pos.entry_date.date(),
                            exit_date=current.date(),
                            entry_underlying=float(frame.loc[pos.entry_date, "close"]),
                            exit_underlying=spot,
                            strike=pos.strike,
                            entry_premium=pos.entry_premium,
                            exit_premium=mark,
                            contracts=pos.contracts,
                            pnl=pnl,
                            return_on_reserve=pnl / pos.reserve,
                            exit_reason=exit_reason,
                            dte_held=(current - pos.entry_date).days,
                        )
                    )
                    positions[ticker] = None
                else:
                    total_equity += (
                        cash[ticker]
                        + pos.reserve
                        + (pos.entry_premium - mark)
                        * pos.contract_multiplier
                        * pos.contracts
                    )
                    continue

            pos = positions[ticker]
            prev = sliced.iloc[-2] if len(sliced) >= 2 else None
            if pos is None and prev is not None and spot < float(prev["close"]):
                years = DTE / 365.0
                contract_multiplier = config.contract_multipliers[ticker]
                strike = strike_for_delta(spot, years, iv, TARGET_PUT_DELTA)
                premium = put_price(spot, strike, years, iv)
                reserve_per_contract = (
                    strike * contract_multiplier - premium * contract_multiplier
                )
                contracts = int(cash[ticker] // reserve_per_contract)
                expiry = next_trading_day_on_or_after(
                    frame.index, current + timedelta(days=DTE)
                )
                if contracts > 0 and expiry is not None and premium > 0:
                    reserve = reserve_per_contract * contracts
                    cash[ticker] -= reserve
                    positions[ticker] = Position(
                        ticker=ticker,
                        entry_date=current,
                        expiry_date=expiry,
                        strike=strike,
                        entry_premium=premium,
                        contracts=contracts,
                        contract_multiplier=contract_multiplier,
                        reserve=reserve,
                    )
                    total_equity += cash[ticker] + reserve
                else:
                    total_equity += cash[ticker]
            else:
                total_equity += cash[ticker]
        equity_points[current] = total_equity

    for ticker, frame in bars_by_ticker.items():
        available = frame.loc[(frame.index >= start) & (frame.index <= end)]
        if available.empty:
            data_notes[ticker] = "no bars"
        else:
            data_notes[ticker] = (
                f"{available.index.min().date()} to {available.index.max().date()}"
            )
    return pd.Series(equity_points).sort_index(), trades, data_notes


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def summarize_window(
    years: int, equity: pd.Series, trades: list[Trade], initial_capital: float
) -> dict[str, object]:
    wins = [t for t in trades if t.pnl > 0]
    expirations = [t for t in trades if t.exit_reason == "expiration"]
    total_return = (
        float(equity.iloc[-1] / equity.iloc[0] - 1.0) if not equity.empty else 0.0
    )
    monthly = (1 + total_return) ** (1 / max(years * 12, 1)) - 1
    return {
        "window_years": years,
        "start": equity.index[0].date() if not equity.empty else None,
        "end": equity.index[-1].date() if not equity.empty else None,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr(equity) * 100,
        "modeled_monthly_pct": monthly * 100,
        "max_drawdown_pct": max_drawdown(equity) * 100,
        "trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
        "expiration_exits": len(expirations),
        "avg_days_held": statistics.mean([t.dte_held for t in trades])
        if trades
        else 0.0,
        "ending_equity": float(equity.iloc[-1])
        if not equity.empty
        else initial_capital,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=sorted(MARKETS), default="us")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated display tickers to run, e.g. NIFTY or TQQQ,SOXL.",
    )
    args = parser.parse_args()

    config = MARKETS[args.market]
    if args.only:
        wanted = {item.strip().upper() for item in args.only.split(",") if item.strip()}
        tickers = tuple(
            ticker
            for ticker in config.tickers
            if config.display_names.get(ticker, ticker).upper() in wanted
            or ticker.upper() in wanted
        )
        if not tickers:
            raise SystemExit(f"No tickers matched --only={args.only!r}")
        config = MarketConfig(
            name=f"{config.name} subset",
            tickers=tickers,
            display_names=config.display_names,
            contract_multipliers=config.contract_multipliers,
            currency=config.currency,
            initial_capital=config.initial_capital,
            commission_per_contract_per_side=config.commission_per_contract_per_side,
        )
    initial_capital = (
        args.capital if args.capital is not None else config.initial_capital
    )
    fetch_start = date.today() - timedelta(days=365 * (max(WINDOW_YEARS) + 1))
    fetcher = build_price_fetcher(provider="yfinance", auto_adjust=True, refresh=False)
    raw = fetcher.fetch(config.tickers, fetch_start, date.today())
    bars_by_ticker = {}
    for ticker, frame in raw.items():
        frame = frame.copy()
        frame["iv"] = modeled_iv(frame)
        frame = frame.dropna(subset=["close", "iv"])
        bars_by_ticker[ticker] = frame

    latest_end = max(
        frame.index.max() for frame in bars_by_ticker.values() if not frame.empty
    )
    summaries: list[dict[str, object]] = []
    all_trade_rows: list[dict[str, object]] = []
    notes_by_window: dict[int, dict[str, str]] = {}

    for years in WINDOW_YEARS:
        start = latest_end - pd.DateOffset(years=years)
        equity, trades, notes = run_window(
            bars_by_ticker, start, latest_end, config, initial_capital
        )
        summaries.append(summarize_window(years, equity, trades, initial_capital))
        notes_by_window[years] = notes
        for trade in trades:
            all_trade_rows.append(
                {
                    "window_years": years,
                    **trade.__dict__,
                    "ticker": config.display_names.get(trade.ticker, trade.ticker),
                }
            )

    summary_df = pd.DataFrame(summaries)
    print("ASSUMPTIONS")
    print(
        f"market={config.name} "
        f"tickers={','.join(config.display_names.get(t, t) for t in config.tickers)} "
        f"capital={initial_capital:.0f} {config.currency} "
        f"DTE={DTE} put_delta={TARGET_PUT_DELTA} close_profit=50% "
        f"iv=rolling_realized_vol*{IV_MULTIPLIER} clipped {MIN_IV:.0%}-{MAX_IV:.0%} "
        f"commission={config.commission_per_contract_per_side:g} "
        f"{config.currency}/contract/side"
    )
    print(
        f"data_source=yfinance adjusted daily OHLCV latest_common_end={latest_end.date()}"
    )
    print()
    print("SUMMARY")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print()
    print("DATA_COVERAGE")
    for years, notes in notes_by_window.items():
        print(
            f"{years}y: "
            + "; ".join(
                f"{config.display_names.get(ticker, ticker)}={note}"
                for ticker, note in notes.items()
            )
        )
    print()
    if all_trade_rows:
        trade_df = pd.DataFrame(all_trade_rows)
        print("TICKER_DETAIL")
        detail = (
            trade_df.groupby(["window_years", "ticker"])
            .agg(
                trades=("pnl", "size"),
                pnl=("pnl", "sum"),
                win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100),
                avg_days_held=("dte_held", "mean"),
                expirations=("exit_reason", lambda s: (s == "expiration").sum()),
            )
            .reset_index()
        )
        print(detail.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))


if __name__ == "__main__":
    main()
