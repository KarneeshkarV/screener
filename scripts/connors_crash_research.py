"""Offline Connors crash short research with dated index membership and gap ledgers.

This is not the long-only screener engine and does not establish stock borrowability.
All signals use the previous completed market session; unfilled entries are GTC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ConnorsMarketData:
    """Daily nominal execution prices and adjusted signals on a benchmark calendar."""

    market: str
    calendar: pd.DatetimeIndex
    symbols: list[str]
    arrays: dict[str, np.ndarray]
    membership: np.ndarray
    coverage: pd.DataFrame
    missing_sessions: pd.DataFrame
    observations: pd.DataFrame
    anomalies: pd.DataFrame
    sources: list[dict]
    evidence: dict = field(
        default_factory=lambda: {
            "classification": "observed_index_proxy",
            "executable": False,
        }
    )


def lean_wilder_rsi(values: np.ndarray, period: int) -> np.ndarray:
    """Match LEAN RSI's SMA seed, Wilder smoothing, and zero-loss value of 100."""
    result = np.full(len(values), np.nan)
    gain = loss = 0.0
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        up, down = max(delta, 0.0), max(-delta, 0.0)
        if i <= period:
            gain += up / period
            loss += down / period
        else:
            gain += (up - gain) / period
            loss += (down - loss) / period
        if i >= period:
            result[i] = 100.0 if round(loss, 10) == 0 else 100 - 100 / (1 + gain / loss)
    return result


def connors_crash_indicators(close: pd.Series) -> pd.DataFrame:
    """Match current LEAN CRSI, including unchanged streaks on equal closes.

    Percent rank compares today's return to the preceding 100 returns, strictly.
    Custom volatility is population SD of 99 log returns from 100 closes.
    """
    values = close.to_numpy(dtype=float)
    streak = np.zeros(len(values))
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        previous = streak[i - 1]
        if delta > 0:
            streak[i] = max(previous, 0) + 1
        elif delta < 0:
            streak[i] = min(previous, 0) - 1
        else:
            streak[i] = previous
    returns = close.pct_change(fill_method=None).fillna(0).to_numpy()
    rank = np.full(len(values), np.nan)
    # The first ready LEAN value has no ready previous rank window and uses zero.
    if len(values) >= 100:
        rank[99] = 0.0
    for i in range(100, len(values)):
        rank[i] = np.count_nonzero(returns[i - 100 : i] < returns[i])
    crsi = (lean_wilder_rsi(values, 3) + lean_wilder_rsi(streak, 2) + rank) / 3
    volatility = np.log(close).diff().rolling(99).std(ddof=0) * np.sqrt(252) * 100
    return pd.DataFrame({"crsi": crsi, "volatility": volatility}, index=close.index)


def connors_contiguous_indicators(close: pd.Series) -> pd.DataFrame:
    """Reset daily indicators after missing data instead of compressing missing sessions."""
    result = pd.DataFrame(np.nan, index=close.index, columns=["crsi", "volatility"])
    groups = close.isna().cumsum()
    for _, segment in close[close.notna()].groupby(groups[close.notna()]):
        result.loc[segment.index] = connors_crash_indicators(segment)
    return result


def connors_nominal_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover nominal OHLC from retained turnover and explicit future split factors.

    Legacy adjusted-only caches cannot establish nominal execution and are refused.
    Yahoo volume and raw close are split adjusted, unlike the trade-date nominal price.
    """
    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "split_factor",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            "Connors nominal price reconstruction lacks turnover or split factors"
        )
    result = frame.copy()
    split = frame["split_factor"]
    nominal_close = frame["dollar_volume"] / frame["volume"].replace(0, np.nan) * split
    ratio = nominal_close / frame["close"]
    for column in ("open", "high", "low", "close"):
        result[column] = frame[column] * ratio
    result["turnover"] = frame["dollar_volume"]
    result["dividend"] = (
        frame.get("dividend", pd.Series(0.0, index=frame.index)) * split
    )
    result["split"] = frame.get(
        "stock_splits", pd.Series(0.0, index=frame.index)
    ).replace(0, 1)
    result["signal_close"] = frame["close"]
    return result


def _source_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_connors_market(
    market: str,
    cache: Path,
    membership_path: Path,
    end: pd.Timestamp,
    earliest_start: pd.Timestamp,
    allow_fmp_proxy: bool = False,
) -> ConnorsMarketData:
    """Load only versioned adjusted caches; record every unavailable member-session."""
    members = pd.read_csv(membership_path)
    members["effective_date"] = pd.to_datetime(members["effective_date"])
    members = members[members.effective_date <= end]
    dates = sorted(members.effective_date.unique())
    previous = max(pd.Timestamp(d) for d in dates if d <= earliest_start)
    members = members[members.effective_date >= previous]
    symbols = sorted(members.symbol.unique())
    if market == "india":
        from screener.symbols import tv_to_yf

        symbols_yf = {symbol: tv_to_yf(symbol, "india") for symbol in symbols}
    else:
        symbols_yf = {symbol: symbol for symbol in symbols}
    benchmark = "SPY" if market == "us" else "^NSEI"
    benchmark_path = cache / f"{benchmark}__full_turnover_v2.parquet"
    benchmark_frame = pd.read_parquet(benchmark_path)
    history_start = earliest_start - pd.Timedelta(days=400)
    calendar = pd.DatetimeIndex(benchmark_frame.loc[history_start:end].index)
    shape = (len(calendar), len(symbols))
    keys = (
        "open",
        "high",
        "low",
        "close",
        "turnover",
        "dividend",
        "split",
        "crsi",
        "volatility",
    )
    arrays = {key: np.full(shape, np.nan) for key in keys}
    arrays["complete_window"] = np.zeros(shape, dtype=bool)
    membership = np.zeros(shape, dtype=bool)
    observations = []
    symbol_indices = {s: i for i, s in enumerate(symbols)}
    grouped = list(members.groupby("effective_date"))
    for i, (stamp, group) in enumerate(grouped):
        next_stamp = (
            grouped[i + 1][0] if i + 1 < len(grouped) else end + pd.Timedelta(days=1)
        )
        rows = (calendar >= stamp) & (calendar < next_stamp)
        cols = [symbol_indices[s] for s in group.symbol]
        membership[np.ix_(rows, cols)] = True
        observations.append(
            {
                "market": market,
                "observation_date": stamp.date().isoformat(),
                "next_observation_or_end": next_stamp.date().isoformat(),
                "gap_days": (next_stamp - stamp).days,
                "members": len(group),
                "gap_at_least_180_days": (next_stamp - stamp).days >= 180,
            }
        )
    coverage, gaps, anomalies = [], [], []
    sources = [_source_record(membership_path), _source_record(benchmark_path)]
    for j, symbol in enumerate(symbols):
        path = cache / f"{symbols_yf[symbol]}__full_turnover_v2.parquet"
        status, first, last = "missing_cache", None, None
        invalid = 0
        fmp_path = (
            cache.parent
            / "fmp_prices"
            / f"fmp_{symbols_yf[symbol]}__full_turnover_v2.parquet"
        )
        proxy = False
        if allow_fmp_proxy and fmp_path.exists():
            # Select one source per symbol, never splice incompatible price scales.
            # Fixed fallback policy, never choose using future-window coverage.
            if not path.exists():
                path, proxy = fmp_path, True
        if path.exists():
            frame = pd.read_parquet(path).sort_index()
            frame = frame[~frame.index.duplicated(keep="last")].loc[history_start:end]
            try:
                if proxy:
                    nominal = frame.copy()
                    nominal["signal_close"] = frame["close"]
                    nominal["turnover"] = frame["dollar_volume"]
                    nominal["dividend"] = 0.0
                    nominal["split"] = 1.0
                else:
                    nominal = connors_nominal_bars(frame)
                valid = (
                    np.isfinite(
                        nominal[
                            ["open", "high", "low", "close", "signal_close", "turnover"]
                        ]
                    ).all(axis=1)
                    & (
                        nominal[
                            ["open", "high", "low", "close", "signal_close", "turnover"]
                        ]
                        > 0
                    ).all(axis=1)
                    & (nominal.high >= nominal[["open", "close"]].max(axis=1))
                    & (nominal.low <= nominal[["open", "close"]].min(axis=1))
                )
                invalid = int((~valid).sum())
                # Corporate actions remain binding even when the quote is invalid.
                action_series = nominal[["dividend", "split"]].copy()
                nominal = nominal[valid]
                signal_close = nominal.signal_close.reindex(calendar)
                features = connors_contiguous_indicators(signal_close)
                observed = signal_close.notna().astype(int)
                arrays["complete_window"][:, j] = (
                    observed.rolling(101).sum().eq(101).to_numpy()
                )
                for key in keys:
                    if key in action_series:
                        series = action_series[key]
                    else:
                        series = features[key] if key in features else nominal[key]
                    arrays[key][:, j] = series.reindex(calendar).to_numpy()
                status = (
                    ("fmp_total_return_proxy" if proxy else "available")
                    if len(nominal)
                    else "no_valid_bars"
                )
                if len(nominal):
                    first, last = nominal.index.min(), nominal.index.max()
                    daily_return = nominal.signal_close.pct_change(fill_method=None)
                    for stamp, value in daily_return[daily_return.abs() > 0.5].items():
                        anomalies.append(
                            {
                                "market": market,
                                "symbol": symbol,
                                "date": str(stamp.date()),
                                "adjusted_close_return": value,
                            }
                        )
                sources.append(_source_record(path))
            except ValueError as exc:
                status = str(exc)
        present = np.isfinite(arrays["close"][:, j])
        expected = membership[:, j] & (calendar >= earliest_start)
        missing = expected & ~present
        unready = expected & present & ~np.isfinite(arrays["crsi"][:, j])
        coverage.append(
            {
                "market": market,
                "symbol": symbol,
                "price_symbol": symbols_yf[symbol],
                "status": status,
                "first_bar": str(first.date()) if first is not None else "",
                "last_bar": str(last.date()) if last is not None else "",
                "expected_member_sessions": int(expected.sum()),
                "missing_member_sessions": int(missing.sum()),
                "unready_indicator_sessions": int(unready.sum()),
                "ready_but_incomplete_101_session_windows": int(
                    (
                        expected
                        & present
                        & ~arrays["complete_window"][:, j]
                        & np.isfinite(arrays["crsi"][:, j])
                    ).sum()
                ),
                "invalid_bars": invalid,
            }
        )
        for stamp in calendar[missing]:
            location = "no_usable_history"
            if first is not None and last is not None:
                location = (
                    "before_first_bar"
                    if stamp < first
                    else "after_last_bar"
                    if stamp > last
                    else "internal_gap"
                )
            gaps.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "date": str(stamp.date()),
                    "location": location,
                }
            )
    return ConnorsMarketData(
        market,
        calendar,
        symbols,
        arrays,
        membership,
        pd.DataFrame(coverage),
        pd.DataFrame(gaps),
        pd.DataFrame(observations),
        pd.DataFrame(anomalies),
        sources,
    )


@dataclass
class ConnorsShortPosition:
    """A short lot in nominal shares; positive quantity means shares owed."""

    symbol_index: int
    quantity: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_notional: float
    entry_fee: float
    signal_date: pd.Timestamp
    crsi: float
    volatility: float
    borrow: float = 0.0
    dividends: float = 0.0
    exit_reason: str = ""


@dataclass
class ConnorsLimitOrder:
    """A persistent short entry, priced and sized from its submission-day prior close."""

    symbol_index: int
    quantity: float
    limit: float
    submitted: pd.Timestamp
    signal_date: pd.Timestamp
    crsi: float
    volatility: float


@dataclass
class ConnorsSimulationResult:
    """Account reconciliation plus complete trades, orders, and marked daily equity."""

    summary: dict
    trades: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    equity: list[dict] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)


def simulate_connors_crash(
    data: ConnorsMarketData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float = 100_000,
    annual_borrow: float = 0.0,
    execution_bps: float = 0.0,
    max_shorts: int = 40,
    require_short_permissions: bool = False,
) -> ConnorsSimulationResult:
    """Simulate nominal short cash flows with prior-bar signals and GTC limit fills.

    Missing bars never fill. Last known marks remain visible as stale, not exits.
    A cover queued this morning still occupies its slot for that morning's entries.
    """
    calendar, a, members = data.calendar, data.arrays, data.membership
    if max_shorts <= 0 or initial_capital <= 0 or end <= start:
        raise ValueError("Connors invalid account, slot count, or date window")
    if require_short_permissions and "short_capacity" not in a:
        raise ValueError("Connors historical short permissions required but missing")
    cash = initial_capital
    positions: dict[int, ConnorsShortPosition] = {}
    pending: dict[int, ConnorsLimitOrder] = {}
    marks = np.full(len(data.symbols), np.nan)
    trades, orders, equity, decisions = [], [], [], []
    total_execution = total_borrow = total_dividends = 0.0
    submitted = fills = stale_position_days = unfilled_exit_days = 0
    exit_indicator_unavailable_days = 0
    candidates_seen = eligible_highvol = 0
    entry_ages = []
    previous_stamp = start
    fee_rate = execution_bps / 10_000

    def account_equity() -> float:
        return cash - sum(p.quantity * marks[j] for j, p in positions.items())

    for i, stamp in enumerate(calendar):
        if stamp < start or stamp > end or i == 0:
            continue
        prior_stamp = calendar[i - 1]
        if "delisted" in a and any(a["delisted"][i, j] for j in positions):
            raise ValueError(
                "Connors held delisting requires verified settlement; cannot carry stale valuation"
            )
        quote_available = a.get("prior_quote_available", np.ones_like(members))[i]
        finite_prior = np.isfinite(a["close"][i - 1]) & quote_available
        marks[finite_prior] = a["close"][i - 1, finite_prior]
        occupied = set(positions) | set(pending)
        available = a.get("prior_bar_available", np.ones_like(members))[i]
        if positions and any(
            np.isfinite(a["close"][i - 1, j]) and not quote_available[j]
            for j in positions
        ):
            raise ValueError(
                "Connors held valuation unavailable at decision; repair evidence before continuing"
            )
        # Borrow is charged on the previous close over elapsed calendar days.
        for j, p in positions.items():
            rate = annual_borrow
            if require_short_permissions:
                # No invented recall fill or automatic contract renewal.
                post_split_quantity = p.quantity * (
                    a["split"][i, j] if np.isfinite(a["split"][i, j]) else 1
                )
                if a["short_capacity"][i, j] < post_split_quantity:
                    raise ValueError(
                        f"Connors held short lacks permission: {data.symbols[j]} {stamp.date()}"
                    )
                rate = a["borrow_rate"][i, j]
            cost = p.quantity * marks[j] * rate * (stamp - previous_stamp).days / 365
            p.borrow += cost
            cash -= cost
            total_borrow += cost
        # Rebase every prior mark, including names that can receive a new order.
        splits = np.where(np.isfinite(a["split"][i]), a["split"][i], 1.0)
        marks /= splits
        # Process split and dividend obligations before morning orders.
        for j, p in positions.items():
            split = splits[j]
            if split != 1:
                p.quantity *= split
            dividend = a["dividend"][i, j]
            if np.isfinite(dividend):
                cost = p.quantity * dividend
                cash -= cost
                p.dividends += cost
                total_dividends += cost
        for j, order in pending.items():
            split = a["split"][i, j]
            if np.isfinite(split) and split != 1:
                order.quantity *= split
                order.limit /= split
        # Daily price/turnover removal also closes shorts and cancels entry orders.
        liquid = (marks > 5) & (a["turnover"][i - 1] > 1_000_000)
        universe = members[i] & (liquid | ~finite_prior | ~quote_available)
        for j in list(pending):
            if not universe[j]:
                order = pending.pop(j)
                orders.append(
                    {
                        "symbol": data.symbols[j],
                        "date": str(stamp.date()),
                        "event": "cancel_universe_removal",
                        "submitted": str(order.submitted.date()),
                    }
                )
        for j, p in positions.items():
            exit_indicator_available = available[j] and np.isfinite(a["crsi"][i - 1, j])
            if not exit_indicator_available:
                exit_indicator_unavailable_days += 1
            if not universe[j]:
                p.exit_reason = p.exit_reason or "universe_removal"
            elif not exit_indicator_available:
                # A reset must not silently freeze a short until indicator warmup ends.
                p.exit_reason = p.exit_reason or "exit_indicator_unavailable"
            elif a["crsi"][i - 1, j] < 30:
                p.exit_reason = p.exit_reason or "crsi_below_30"
        morning_marks = marks.copy()
        good_open = np.isfinite(a["open"][i])
        # Size from the previous close, not a future open or an exit fill.
        sizing_equity = account_equity()
        for j in list(positions):
            p = positions[j]
            if p.exit_reason and good_open[j]:
                price = a["open"][i, j]
                fee = p.quantity * price * fee_rate
                cash -= p.quantity * price + fee
                total_execution += fee
                pnl = (
                    p.entry_notional
                    - p.quantity * price
                    - p.entry_fee
                    - fee
                    - p.borrow
                    - p.dividends
                )
                trades.append(
                    {
                        "symbol": data.symbols[j],
                        "signal_date": str(p.signal_date.date()),
                        "entry_date": str(p.entry_date.date()),
                        "exit_date": str(stamp.date()),
                        "entry_price": p.entry_price,
                        "exit_price": price,
                        "exit_quantity": p.quantity,
                        "entry_notional": p.entry_notional,
                        "pnl": pnl,
                        "return_on_entry_notional": pnl / p.entry_notional,
                        "holding_calendar_days": (stamp - p.entry_date).days,
                        "entry_crsi": p.crsi,
                        "entry_volatility": p.volatility,
                        "execution_cost": p.entry_fee + fee,
                        "borrow_cost": p.borrow,
                        "dividends_owed": p.dividends,
                        "exit_reason": p.exit_reason,
                    }
                )
                del positions[j]
            elif p.exit_reason:
                unfilled_exit_days += 1
        prior_ready = (
            finite_prior
            & np.isfinite(a["crsi"][i - 1])
            & np.isfinite(a["volatility"][i - 1])
        )
        complete = a.get("complete_window", np.ones_like(members))[i - 1]
        quality = (
            a.get("entry_quality", np.ones_like(members))[i - 1] & complete & available
        )
        permission = (
            a["short_capacity"][i] > 0
            if require_short_permissions
            else np.ones(len(data.symbols), dtype=bool)
        )
        for j in np.flatnonzero(members[i]):
            reason = (
                "bar_not_available_at_decision"
                if not available[j]
                else "missing_prior_quote"
                if not finite_prior[j]
                else "incomplete_indicator_window"
                if not complete[j] or not prior_ready[j]
                else "quality_quarantine"
                if not quality[j]
                else "short_permission_missing"
                if not permission[j]
                else "liquidity_filter"
                if not liquid[j]
                else "signal_not_triggered"
                if a["volatility"][i - 1, j] <= 100 or a["crsi"][i - 1, j] <= 90
                else "occupied"
                if j in occupied
                else "candidate"
            )
            decisions.append(
                {
                    "symbol": data.symbols[j],
                    "date": str(stamp.date()),
                    "prior_session": str(prior_stamp.date()),
                    "reason": reason,
                }
            )
        for j in list(pending):
            if not quality[j] or not permission[j]:
                order = pending.pop(j)
                orders.append(
                    {
                        "symbol": data.symbols[j],
                        "date": str(stamp.date()),
                        "event": "cancel_data_quality",
                        "submitted": str(order.submitted.date()),
                    }
                )
        eligible = (
            members[i]
            & prior_ready
            & liquid
            & quality
            & permission
            & (a["volatility"][i - 1] > 100)
        )
        eligible_highvol += int(eligible.sum())
        candidates = [
            j
            for j in np.flatnonzero(eligible & (a["crsi"][i - 1] > 90))
            if j not in occupied
        ]
        candidates_seen += len(candidates)
        candidates.sort(
            key=lambda j: (
                a["crsi"][i - 1, j],
                a["volatility"][i - 1, j],
                data.symbols[j],
            )
        )
        slots = max(0, max_shorts - len(occupied))
        chosen = candidates[-slots:] if slots else []
        if sizing_equity > 0:
            for j in chosen:
                price = morning_marks[j]
                quantity = int(sizing_equity / max_shorts / price)
                if require_short_permissions:
                    quantity = min(quantity, int(a["short_capacity"][i, j]))
                if quantity <= 0:
                    continue
                limit = round(price * 1.03, 2)
                order = ConnorsLimitOrder(
                    j,
                    quantity,
                    limit,
                    stamp,
                    prior_stamp,
                    a["crsi"][i - 1, j],
                    a["volatility"][i - 1, j],
                )
                pending[j] = order
                submitted += 1
                orders.append(
                    {
                        "symbol": data.symbols[j],
                        "date": str(stamp.date()),
                        "event": "submit",
                        "quantity": quantity,
                        "limit": limit,
                        "crsi": order.crsi,
                        "volatility": order.volatility,
                    }
                )
        for j in list(pending):
            order = pending[j]
            if require_short_permissions and order.quantity > a["short_capacity"][i, j]:
                orders.append(
                    {
                        "symbol": data.symbols[j],
                        "date": str(stamp.date()),
                        "event": "cancel_short_capacity",
                        "quantity": order.quantity,
                    }
                )
                del pending[j]
                continue
            if good_open[j] and a["high"][i, j] >= order.limit:
                price = max(a["open"][i, j], order.limit)
                notional = order.quantity * price
                fee = notional * fee_rate
                cash += notional - fee
                total_execution += fee
                positions[j] = ConnorsShortPosition(
                    j,
                    order.quantity,
                    stamp,
                    price,
                    notional,
                    fee,
                    order.signal_date,
                    order.crsi,
                    order.volatility,
                )
                fills += 1
                entry_ages.append((stamp - order.submitted).days)
                orders.append(
                    {
                        "symbol": data.symbols[j],
                        "date": str(stamp.date()),
                        "event": "fill",
                        "submitted": str(order.submitted.date()),
                        "quantity": order.quantity,
                        "price": price,
                    }
                )
                del pending[j]
        good_close = np.isfinite(a["close"][i])
        marks[good_close] = a["close"][i, good_close]
        stale = sum(not good_close[j] for j in positions)
        stale_position_days += stale
        value = account_equity()
        gross = sum(p.quantity * marks[j] for j, p in positions.items())
        equity.append(
            {
                "date": str(stamp.date()),
                "equity": value,
                "cash": cash,
                "gross_short_notional": gross,
                "gross_exposure": gross / value if value > 0 else np.nan,
                "positions": len(positions),
                "pending_entries": len(pending),
                "stale_positions": stale,
            }
        )
        assert len(positions) + len(pending) <= max_shorts, (
            "Connors short slot cap violated"
        )
        previous_stamp = stamp
    for j in positions:
        delisted_on = data.evidence.get("delisting_dates", {}).get(data.symbols[j])
        if delisted_on and pd.Timestamp(delisted_on) <= end:
            raise ValueError("Connors held delisting requires verified settlement")
    final = equity[-1]["equity"] if equity else initial_capital
    curve = np.array([initial_capital] + [row["equity"] for row in equity])
    returns = np.diff(curve) / curve[:-1]
    drawdown = curve / np.maximum.accumulate(curve) - 1
    closed_pnl = sum(t["pnl"] for t in trades)
    open_rows = []
    for j, p in positions.items():
        pnl = (
            p.entry_notional
            - p.quantity * marks[j]
            - p.entry_fee
            - p.borrow
            - p.dividends
        )
        open_rows.append(
            {
                "symbol": data.symbols[j],
                "entry_date": str(p.entry_date.date()),
                "quantity": p.quantity,
                "entry_notional": p.entry_notional,
                "last_mark": marks[j],
                "pnl": pnl,
                "borrow_cost": p.borrow,
                "dividends_owed": p.dividends,
                "queued_exit_reason": p.exit_reason,
            }
        )
    reconciliation = (
        final - initial_capital - closed_pnl - sum(p["pnl"] for p in open_rows)
    )
    if abs(reconciliation) > 1e-6:
        raise AssertionError(f"Connors account reconciliation failed: {reconciliation}")
    winners = sum(t["pnl"] > 0 for t in trades)
    gains = sum(max(t["pnl"], 0) for t in trades)
    losses = -sum(min(t["pnl"], 0) for t in trades)
    annual_years = (end - start).days / 365.25
    summary = {
        "market": data.market,
        "evidence_classification": data.evidence["classification"],
        "executable": False,
        "historical_short_permissions_required": require_short_permissions,
        "blocked_decision_count": sum(
            d["reason"]
            not in {"candidate", "occupied", "signal_not_triggered", "liquidity_filter"}
            for d in decisions
        ),
        "start": str(start.date()),
        "end": str(end.date()),
        "first_session": equity[0]["date"] if equity else None,
        "initial_capital": initial_capital,
        "final_equity": final,
        "return_pct": (final / initial_capital - 1) * 100,
        "cagr_pct": ((final / initial_capital) ** (1 / annual_years) - 1) * 100
        if final > 0
        else None,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "sharpe_zero_hurdle": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else None,
        "closed_trades": len(trades),
        "win_rate_pct": winners / len(trades) * 100 if trades else None,
        "profit_factor": gains / losses if losses else None,
        "orders_submitted": submitted,
        "entries_filled": fills,
        "open_positions_at_end": len(positions),
        "pending_entries_at_end": len(pending),
        "average_positions": float(np.mean([row["positions"] for row in equity]))
        if equity
        else 0,
        "max_positions": max((row["positions"] for row in equity), default=0),
        "average_gross_exposure_pct": float(
            np.mean([row["gross_exposure"] for row in equity]) * 100
        )
        if equity
        else 0,
        "max_gross_exposure_pct": max(
            (row["gross_exposure"] * 100 for row in equity), default=0
        ),
        "execution_bps_per_side": execution_bps,
        "annual_borrow_pct": None if require_short_permissions else annual_borrow * 100,
        "borrow_rate_mode": "dated_permission_rate"
        if require_short_permissions
        else "assumed_constant",
        "execution_cost": total_execution,
        "borrow_cost": total_borrow,
        "dividends_owed": total_dividends,
        "closed_pnl": closed_pnl,
        "open_pnl": sum(p["pnl"] for p in open_rows),
        "reconciliation_error": reconciliation,
        "stale_position_days": stale_position_days,
        "unfilled_exit_days": unfilled_exit_days,
        "exit_indicator_unavailable_days": exit_indicator_unavailable_days,
        "missing_exit_indicator_policy": "queue cover at next valid open; preserve first exit reason",
        "highvol_eligible_symbol_days": eligible_highvol,
        "entry_candidate_symbol_days": candidates_seen,
        "max_entry_wait_calendar_days": max(entry_ages, default=0),
    }
    return ConnorsSimulationResult(
        summary, trades, orders, equity, open_rows, decisions
    )


def run_connors_research() -> None:
    """Write eight fresh-account windows plus declared borrow-cost sensitivity runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/connors_crash"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".screener/prices")
    parser.add_argument(
        "--us-membership",
        type=Path,
        default=Path.home() / ".screener/universes/sp500_pit_snapshots.csv",
    )
    parser.add_argument(
        "--india-membership",
        type=Path,
        default=Path("data/universes/nifty500_pit_snapshots.csv"),
    )
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--pit-bundles",
        type=Path,
        help="Timestamped evidence root with us/ and india/ bundles; never uses legacy caches.",
    )
    parser.add_argument(
        "--require-short-permissions",
        action="store_true",
        help="Require dated capacity for every entry and holding; abort on unsupported recalls.",
    )
    parser.add_argument(
        "--allow-fmp-proxy",
        action="store_true",
        help="Use FMP adjusted proxies only when the Yahoo cache is absent; never select by future coverage.",
    )
    parser.add_argument(
        "--quarantine-extreme-returns",
        action="store_true",
        help="Block entries for 100 market sessions after absolute adjusted returns above 50%% and require 101 consecutive valid market-session bars.",
    )
    args = parser.parse_args()
    if args.pit_bundles and args.allow_fmp_proxy:
        parser.error("Connors timestamped bundles cannot use adjusted FMP proxies")
    if args.pit_bundles and not args.end:
        parser.error("Connors timestamped bundles require an explicit --end")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.end:
        end = pd.Timestamp(args.end)
    else:
        end = min(
            pd.read_parquet(args.cache / f"{s}__full_turnover_v2.parquet").index.max()
            for s in ("SPY", "^NSEI")
        )
    earliest = end - pd.DateOffset(years=5)
    summaries, all_sources = [], []
    for market, membership_path in (
        ("us", args.us_membership),
        ("india", args.india_membership),
    ):
        print(f"Loading {market}, end={end.date()}", flush=True)
        if args.pit_bundles:
            from scripts.connors_pit_data import load_connors_pit_bundle

            data = load_connors_pit_bundle(
                args.pit_bundles / market, market, end, earliest
            )
        else:
            data = load_connors_market(
                market,
                args.cache,
                membership_path,
                end,
                earliest,
                allow_fmp_proxy=args.allow_fmp_proxy,
            )
        (args.output / f"{market}_evidence.json").write_text(
            json.dumps(data.evidence, indent=2) + "\n"
        )
        all_sources.extend(data.sources)
        if args.quarantine_extreme_returns:
            quality = np.ones_like(data.membership)
            for anomaly in data.anomalies.to_dict("records"):
                index = data.calendar.searchsorted(pd.Timestamp(anomaly["date"]))
                column = data.symbols.index(anomaly["symbol"])
                quality[index : index + 100, column] = False
            # Missing bars cannot let contaminated valid-bar windows outlive quarantine.
            quality &= data.arrays["complete_window"]
            data.arrays["entry_quality"] = quality
        for name in ("coverage", "missing_sessions", "observations", "anomalies"):
            getattr(data, name).to_csv(
                args.output / f"{market}_{name}.csv", index=False
            )
        for years in (5, 3, 2, 1):
            start = end - pd.DateOffset(years=years)
            for label, borrow, execution in (
                ("gross", 0.0, 0.0),
                ("cost10", 0.10, 5.0),
                ("cost30", 0.30, 5.0),
            ):
                result = simulate_connors_crash(
                    data,
                    start,
                    end,
                    annual_borrow=borrow,
                    execution_bps=execution,
                    require_short_permissions=args.require_short_permissions,
                )
                result.summary.update({"years": years, "scenario": label})
                in_window = (
                    (data.missing_sessions.date >= str(start.date()))
                    if len(data.missing_sessions)
                    else []
                )
                result.summary["missing_member_sessions"] = int(sum(in_window))
                result.summary["symbols_with_missing_sessions"] = (
                    int(data.missing_sessions.loc[in_window, "symbol"].nunique())
                    if len(data.missing_sessions)
                    else 0
                )
                result.summary["member_symbol_count"] = int(
                    data.membership[data.calendar >= start].any(axis=0).sum()
                )
                summaries.append(result.summary)
                folder = args.output / f"{market}_{years}y_{label}"
                folder.mkdir(exist_ok=True)
                for name in (
                    "trades",
                    "orders",
                    "equity",
                    "open_positions",
                    "decisions",
                ):
                    pd.DataFrame(getattr(result, name)).to_csv(
                        folder / f"{name}.csv", index=False
                    )
                (folder / "summary.json").write_text(
                    json.dumps(result.summary, indent=2, allow_nan=False) + "\n"
                )
                print(
                    f"{market} {years}y {label}: return={result.summary['return_pct']:.3f}% closed={len(result.trades)} fills={result.summary['entries_filled']}",
                    flush=True,
                )
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    manifest = {
        "end": str(end.date()),
        "classification": "timestamped_signal_research"
        if args.pit_bundles
        else "observed_index_proxy",
        "executable": False,
        "short_permissions_required": args.require_short_permissions,
        "research_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "pit_loader_sha256": hashlib.sha256(
            Path(__file__).with_name("connors_pit_data.py").read_bytes()
        ).hexdigest(),
        "extreme_return_quarantine": args.quarantine_extreme_returns,
        "fmp_total_return_proxy": args.allow_fmp_proxy,
        "fmp_proxy_limitations": "FMP symbols use adjusted execution prices and synthetic shares; dividends are implicit in total-return price changes, no verified nominal price cutoff or whole-share sizing. One provider per symbol, never spliced.",
        "sources": all_sources,
        "universe_mode": "supplied daily evidence; see market evidence JSON"
        if args.pit_bundles
        else "observed index snapshots, not full liquid-stock universes",
        "initial_capital": "USD 100000 US; INR 100000 India",
        "liquidity_threshold": "prior close > 5; prior daily turnover > 1000000, each in local currency",
        "price_mode": "causal total-return signals; supplied nominal fills and explicit actions"
        if args.pit_bundles
        else "dividend-adjusted signals; reconstructed nominal fills; explicit short dividends and splits",
        "timing": "previous market-session signals; covers at next open; sell limit max(open, limit) on high >= limit",
        "pending_entries": "GTC until fill or observed universe removal; stale signals do not cancel orders",
        "end_policy": "retain positions at last close; no invented end liquidation",
        "scenarios": "dated permission rates in all scenarios; gross zero execution cost, cost10/cost30 5bps per side"
        if args.require_short_permissions
        else "gross includes dividend obligations; cost10/cost30 add 5bps execution charges per side and 10%/30% annual borrow",
        "unmodeled": [
            "historical borrow availability and rates",
            "SLB eligibility and contracts",
            "US locate and SSR",
            "intraday fill sequencing and queues",
            "partial fills",
            "market impact",
            "broker margin and forced buy-ins",
            "India circuit bands",
            "date-aware taxes and broker fees",
            "cash interest",
            "cash in lieu for fractional split shares",
        ],
        "membership_dates": "explicit daily evidence"
        if args.pit_bundles
        else "observations, not verified exchange effective dates; stale final snapshot is carried to end",
        "indicator_reference": "https://github.com/QuantConnect/Lean/blob/master/Indicators/ConnorsRelativeStrengthIndex.cs fetched 2026-10-20",
        "indicator_warmup": "up to 400 calendar days pre-window; continuous per-symbol indicators, not reset on universe re-entry",
        "indicator_gap_policy": "Indicators reset after missing sessions; entries and pending limits require a complete 101-session window. Held shorts queue a cover when their prior exit indicator is unavailable; fill only at a valid open. Count affected position-days as exit_indicator_unavailable_days.",
        "provider_selection_policy": "Fixed Yahoo source; optional FMP fallback only if Yahoo path is absent. Observed caches still lack historical availability and cannot establish PIT.",
        "calendar": "supplied independent sessions"
        if args.pit_bundles
        else "available SPY / Nifty50 benchmark sessions, not an independent exchange calendar",
        "coverage_interpretation": "unavailable member sessions may be IPO, suspension, rename, delisting, or cache gaps; reasons are not established",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    run_connors_research()
