"""Panel-backed options screening criteria and their shared pipeline runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
import logging

import click
import pandas as pd

from screener.options.panels import read_options_panel
from screener.symbols import tv_to_yf

LOG = logging.getLogger(__name__)

MIN_TRAILING_VOLUME_DAYS = 6
MIN_IV_HISTORY_DAYS = 5
UNUSUAL_OPTIONS_MULTIPLE = 2.0
HIGH_IV_RANK = 80.0
LOW_IV_RANK = 20.0


@dataclass(frozen=True)
class OptionsCriterionResult:
    frame: pd.DataFrame
    message: str


def latest_panel_rows(
    panel: pd.DataFrame, *, as_of: date | None = None
) -> pd.DataFrame:
    """Return the last point-in-time row per symbol on or before ``as_of``."""
    if panel.empty:
        return panel.copy()
    required = {"as_of", "SYMBOL"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"options panel missing columns: {sorted(missing)}")
    rows = panel.copy()
    rows["as_of"] = pd.to_datetime(rows["as_of"], errors="coerce").dt.normalize()
    rows = rows[rows["as_of"].notna()]
    if as_of is not None:
        rows = rows[rows["as_of"] <= pd.Timestamp(as_of)]
    if rows.empty:
        return rows
    rows["SYMBOL"] = rows["SYMBOL"].astype(str).str.upper()
    return (
        rows.sort_values(["SYMBOL", "as_of"])
        .groupby("SYMBOL", as_index=False, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(float("nan"), index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _unusual_options(latest: pd.DataFrame) -> OptionsCriterionResult:
    history = _numeric(latest, "history_days")
    ratio = _numeric(latest, "unusual_options_ratio")
    covered = history >= MIN_TRAILING_VOLUME_DAYS
    if not covered.any():
        return OptionsCriterionResult(
            latest.iloc[0:0].copy(),
            "Options panel is thin: unusual_options needs at least 6 daily "
            "snapshots per symbol (5 prior days plus today).",
        )
    selected = latest[covered & ratio.ge(UNUSUAL_OPTIONS_MULTIPLE)].copy()
    selected["signal"] = "unusual_options"
    selected["coverage_days"] = history.loc[selected.index].astype(int)
    return OptionsCriterionResult(
        selected.sort_values("unusual_options_ratio", ascending=False),
        f"Volume multiple uses each symbol's own trailing 20-day mean; "
        f"minimum coverage={MIN_TRAILING_VOLUME_DAYS} days.",
    )


def _bullish_oi_buildup(latest: pd.DataFrame) -> OptionsCriterionResult:
    call_change = _numeric(latest, "call_oi_change")
    put_change = _numeric(latest, "put_oi_change")
    call_writing = _numeric(latest, "call_writing_near_spot")
    put_writing = _numeric(latest, "put_writing_near_spot")

    exact = (
        call_change.gt(0) & put_writing.gt(0) & put_writing.gt(call_writing.fillna(0))
    )
    proxy = (
        call_change.gt(0)
        & put_change.gt(0)
        & put_change.gt(call_change)
        & put_writing.isna()
    )
    selected = latest[exact | proxy].copy()
    if selected.empty and call_change.isna().all():
        return OptionsCriterionResult(
            selected,
            "Options panel has no OI-change baseline yet; take another daily "
            "snapshot before using bullish_oi_buildup.",
        )
    selected["signal"] = "bullish_oi_buildup"
    selected["oi_signal_basis"] = [
        "exact_put_writing" if bool(exact.loc[index]) else "snapshot_diff_proxy"
        for index in selected.index
    ]
    selected["coverage_days"] = _numeric(selected, "history_days").fillna(1).astype(int)
    selected["bullish_oi_score"] = (
        _numeric(selected, "put_writing_near_spot")
        .fillna(_numeric(selected, "put_oi_change"))
        .fillna(0)
    )
    return OptionsCriterionResult(
        selected.sort_values("bullish_oi_score", ascending=False),
        "India uses exact bhavcopy OI/premium changes; US uses consecutive "
        "snapshot OI differences when writing direction is unavailable.",
    )


def _iv_rank(latest: pd.DataFrame, *, high: bool) -> OptionsCriterionResult:
    history = _numeric(latest, "iv_history_days")
    rank = _numeric(latest, "iv_rank")
    covered = history >= MIN_IV_HISTORY_DAYS
    name = "high_iv_rank" if high else "low_iv_rank"
    if not covered.any():
        return OptionsCriterionResult(
            latest.iloc[0:0].copy(),
            f"Options panel is thin: {name} needs at least "
            f"{MIN_IV_HISTORY_DAYS} usable IV days per symbol.",
        )
    qualifies = rank.ge(HIGH_IV_RANK) if high else rank.le(LOW_IV_RANK)
    selected = latest[covered & qualifies].copy()
    selected["signal"] = name
    selected["coverage_days"] = history.loc[selected.index].astype(int)
    return OptionsCriterionResult(
        selected.sort_values("iv_rank", ascending=not high),
        f"IV rank is expanding and point-in-time; each row reports its usable "
        f"history (minimum {MIN_IV_HISTORY_DAYS} days).",
    )


def realized_earnings_moves(
    symbols: list[str],
    *,
    market: str,
    as_of: date,
    earnings_fetcher: Callable[..., pd.DataFrame] | None = None,
    price_fetcher: Callable[..., Mapping[str, pd.DataFrame]] | None = None,
) -> dict[str, tuple[float, int]]:
    """Median absolute close-to-next-close move around past earnings events."""
    if not symbols:
        return {}
    if earnings_fetcher is None:
        from screener.earnings_backtest.earnings_dates import collect_earnings_events

        earnings_fetcher = collect_earnings_events
    if price_fetcher is None:
        from screener.earnings_backtest.data import fetch_price_data

        price_fetcher = fetch_price_data

    yf_by_symbol = {symbol: tv_to_yf(symbol, market) for symbol in symbols}
    tickers = list(dict.fromkeys(yf_by_symbol.values()))
    events = earnings_fetcher(tickers, years=3, batch_size=50, market=market)
    if events is None or events.empty or "earnings_date" not in events.columns:
        return {}
    events = events.copy()
    events["earnings_date"] = pd.to_datetime(
        events["earnings_date"], errors="coerce"
    ).dt.normalize()
    events = events[
        events["earnings_date"].notna()
        & (events["earnings_date"] <= pd.Timestamp(as_of))
    ]
    if events.empty:
        return {}
    start = events["earnings_date"].min().date() - timedelta(days=7)
    end = min(as_of, events["earnings_date"].max().date() + timedelta(days=7))
    prices = price_fetcher(tickers, start, end)
    moves_by_ticker: dict[str, list[float]] = {}
    for ticker, ticker_events in events.groupby("ticker"):
        bars = prices.get(str(ticker))
        if bars is None or bars.empty or "close" not in bars.columns:
            continue
        frame = bars.sort_index().copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        closes = pd.to_numeric(frame["close"], errors="coerce")
        moves: list[float] = []
        for event_date in ticker_events["earnings_date"]:
            before_pos = int(frame.index.searchsorted(event_date, side="left")) - 1
            after_pos = int(frame.index.searchsorted(event_date, side="right"))
            if before_pos < 0 or after_pos >= len(frame):
                continue
            before = float(closes.iloc[before_pos])
            after = float(closes.iloc[after_pos])
            if before > 0 and pd.notna(after):
                moves.append(abs(after / before - 1.0) * 100.0)
        if moves:
            moves_by_ticker[str(ticker)] = moves

    out: dict[str, tuple[float, int]] = {}
    for symbol, ticker in yf_by_symbol.items():
        moves = moves_by_ticker.get(ticker, [])
        if moves:
            out[symbol] = (float(pd.Series(moves).median()), len(moves))
    return out


def _cheap_earnings_vol(
    latest: pd.DataFrame,
    *,
    market: str,
    as_of: date,
    earnings_fetcher: Callable[..., pd.DataFrame] | None,
    price_fetcher: Callable[..., Mapping[str, pd.DataFrame]] | None,
) -> OptionsCriterionResult:
    implied = _numeric(latest, "implied_move_pct")
    candidates = latest[implied.notna() & implied.gt(0)].copy()
    if candidates.empty:
        return OptionsCriterionResult(
            candidates,
            "No front-expiry ATM straddle quotes are available for cheap_earnings_vol.",
        )
    try:
        realized = realized_earnings_moves(
            candidates["SYMBOL"].astype(str).tolist(),
            market=market,
            as_of=as_of,
            earnings_fetcher=earnings_fetcher,
            price_fetcher=price_fetcher,
        )
    except Exception as exc:  # noqa: BLE001 - optional overlay degrades cleanly
        LOG.warning("realized earnings moves unavailable: %s", exc)
        return OptionsCriterionResult(
            candidates.iloc[0:0].copy(),
            f"Historical earnings moves are unavailable: {exc}.",
        )
    candidates["realized_earnings_move_pct"] = candidates["SYMBOL"].map(
        lambda symbol: realized.get(str(symbol), (float("nan"), 0))[0]
    )
    candidates["earnings_events"] = candidates["SYMBOL"].map(
        lambda symbol: realized.get(str(symbol), (float("nan"), 0))[1]
    )
    candidates["vol_edge_pct"] = candidates[
        "realized_earnings_move_pct"
    ] - pd.to_numeric(candidates["implied_move_pct"], errors="coerce")
    selected = candidates[
        candidates["earnings_events"].ge(2) & candidates["vol_edge_pct"].gt(0)
    ].copy()
    selected["signal"] = "cheap_earnings_vol"
    selected["coverage_days"] = selected["earnings_events"].astype(int)
    return OptionsCriterionResult(
        selected.sort_values("vol_edge_pct", ascending=False),
        "Implied move is the front ATM straddle/spot; realized baseline is the "
        "median absolute close-to-next-close move across past earnings (2+ events).",
    )


def screen_options_criterion(
    name: str,
    *,
    market: str,
    limit: int,
    as_of: date | None = None,
    panel: pd.DataFrame | None = None,
    earnings_fetcher: Callable[..., pd.DataFrame] | None = None,
    price_fetcher: Callable[..., Mapping[str, pd.DataFrame]] | None = None,
) -> OptionsCriterionResult:
    """Evaluate one registered options criterion without Click/provider coupling."""
    effective_date = as_of or date.today()
    source = read_options_panel(market) if panel is None else panel
    latest = latest_panel_rows(source, as_of=effective_date)
    if latest.empty:
        return OptionsCriterionResult(
            latest,
            f"No {market.upper()} options panel rows exist on or before "
            f"{effective_date}; run `screener options snapshot` or `build-panel` first.",
        )
    if name == "unusual_options":
        result = _unusual_options(latest)
    elif name == "bullish_oi_buildup":
        result = _bullish_oi_buildup(latest)
    elif name == "high_iv_rank":
        result = _iv_rank(latest, high=True)
    elif name == "low_iv_rank":
        result = _iv_rank(latest, high=False)
    elif name == "cheap_earnings_vol":
        result = _cheap_earnings_vol(
            latest,
            market=market,
            as_of=effective_date,
            earnings_fetcher=earnings_fetcher,
            price_fetcher=price_fetcher,
        )
    else:
        raise ValueError(f"unknown options criterion: {name}")
    if limit > 0:
        return OptionsCriterionResult(result.frame.head(limit), result.message)
    return result


def run_options_criterion(
    name: str,
    *,
    market: str,
    limit: int,
    output_csv: bool,
) -> None:
    """Render the panel criterion through the generic ``screen`` pipeline."""
    result = screen_options_criterion(name, market=market, limit=limit)
    if result.frame.empty:
        click.echo(result.message, err=output_csv)
        return
    if output_csv:
        click.echo(result.frame.to_csv(index=False), nl=False)
        return
    preferred = [
        "as_of",
        "SYMBOL",
        "signal",
        "source",
        "unusual_options_ratio",
        "iv_rank",
        "pcr",
        "call_oi_change",
        "put_oi_change",
        "oi_signal_basis",
        "implied_move_pct",
        "realized_earnings_move_pct",
        "vol_edge_pct",
        "coverage_days",
    ]
    columns = [column for column in preferred if column in result.frame.columns]
    click.echo(result.frame[columns].to_string(index=False))
    click.echo(result.message)


__all__ = [
    "OptionsCriterionResult",
    "latest_panel_rows",
    "realized_earnings_moves",
    "run_options_criterion",
    "screen_options_criterion",
]
