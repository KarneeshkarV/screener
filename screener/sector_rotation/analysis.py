"""Monthly equal-weight research baskets with explicit missing-data failures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from screener.backtester.data import PriceFetcher
from screener.sector_rotation.catalogue import SectorGroup, SectorSelection
from screener.symbols import tv_to_yf


@dataclass(frozen=True)
class SectorRotationResult:
    """Descriptive current-membership analysis, never historical trading signals."""

    selection: SectorSelection
    start: date
    end: date
    benchmark: str
    summary: pd.DataFrame
    members: pd.DataFrame
    curves: pd.DataFrame
    warnings: tuple[str, ...]


def _usable_close(frame: pd.DataFrame | None) -> pd.Series:
    if frame is None or frame.empty or "close" not in frame:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    close.index = (
        pd.DatetimeIndex(pd.to_datetime(close.index)).tz_localize(None).normalize()
    )
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close.where(np.isfinite(close) & (close > 0))


def _missing_benchmark_sessions(
    selection: SectorSelection,
    prices: Mapping[str, pd.DataFrame],
    benchmark_sessions: pd.DatetimeIndex,
    end: date,
) -> list[pd.Timestamp]:
    """Member trading sessions inside the window that the benchmark omits."""
    stock_sessions: set[pd.Timestamp] = set()
    anchor = benchmark_sessions[0]
    for group in selection.groups:
        for ticker in group.members:
            frame = prices.get(tv_to_yf(ticker, selection.catalogue.market))
            close = _usable_close(frame)
            # Providers sometimes emit flat, zero-volume holiday placeholders.
            # Those are not evidence that the exchange held a session.
            if frame is not None and "volume" in frame:
                volume = frame.copy()
                volume["close"] = volume["volume"]
                close = close.where(_usable_close(volume).reindex(close.index).notna())
            stock_sessions.update(close.loc[anchor : str(end)].dropna().index)
    return sorted(stock_sessions - set(benchmark_sessions))


def _format_sessions(sessions: list[pd.Timestamp]) -> str:
    shown = ", ".join(session.date().isoformat() for session in sessions[:5])
    return shown + (f" and {len(sessions) - 5} more" if len(sessions) > 5 else "")


def _benchmark_window_sessions(
    frame: pd.DataFrame | None, start: date, end: date
) -> pd.DatetimeIndex:
    close = _usable_close(frame).loc[: str(end)].dropna()
    anchors = close.index[close.index <= pd.Timestamp(start)]
    index = pd.DatetimeIndex(close.index)
    return index[index >= anchors[-1]] if len(anchors) else index[:0]


def build_sector_basket(
    closes: pd.DataFrame, *, minimum_coverage: float = 0.8
) -> tuple[pd.Series, pd.Series]:
    """Equal weight at the anchor and each month end, for the next session.

    Weights drift between rebalances. Unpriced constituents cannot be silently
    sold or assigned a zero return. A missing held price invalidates the rest
    of the curve. Selection uses only prices known at the rebalance close.
    Coverage counts quoted members, including those not held yet.
    """
    if not 0 < minimum_coverage <= 1 or closes.shape[1] == 0:
        raise ValueError("Sector rotation: coverage must be in (0, 1] with members")
    if not closes.index.is_monotonic_increasing or not closes.index.is_unique:
        raise ValueError("Sector rotation: basket sessions must be unique and sorted")
    closes = closes.astype(float).where(np.isfinite(closes) & (closes > 0))
    coverage = closes.notna().sum(axis=1) / closes.shape[1]
    curve = pd.Series(np.nan, index=closes.index, dtype=float)
    if closes.empty or float(coverage.iloc[0]) < minimum_coverage:
        return curve, coverage
    holdings = closes.iloc[0].notna()
    units = (1.0 / int(holdings.sum())) / closes.iloc[0][holdings]
    curve.iloc[0] = 100.0
    value = 1.0
    for i in range(1, len(closes)):
        previous_date = closes.index[i - 1]
        current_date = closes.index[i]
        if (previous_date.year, previous_date.month) != (
            current_date.year,
            current_date.month,
        ):
            previous = closes.iloc[i - 1]
            if float(coverage.iloc[i - 1]) < minimum_coverage:
                break
            holdings = previous.notna()
            units = (value / int(holdings.sum())) / previous[holdings]
        marks = closes.iloc[i].reindex(units.index)
        if marks.isna().any() or float(coverage.iloc[i]) < minimum_coverage:
            break
        value = float((units * marks).sum())
        curve.iloc[i] = value * 100.0
    return curve, coverage


def _return_percent(close: pd.Series) -> float:
    if len(close) < 2 or close.isna().any():
        return float("nan")
    return float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)


def _member_metrics(
    group: SectorGroup, histories: pd.DataFrame, window: pd.DataFrame
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ticker in group.members:
        values = window[ticker]
        history = histories[ticker].loc[: window.index[-1]]
        latest = float(values.iloc[-1])
        row: dict[str, object] = {
            "group_id": group.id,
            "ticker": ticker,
            "return_pct": _return_percent(values),
            "coverage_pct": float(values.notna().mean() * 100),
            "latest_close": latest,
        }
        for days in (50, 200):
            average = history.rolling(days, min_periods=days).mean().iloc[-1]
            row[f"above_sma{days}"] = (
                bool(latest > average)
                if pd.notna(latest) and pd.notna(average)
                else None
            )
        rows.append(row)
    return rows


def compute_sector_rotation(
    selection: SectorSelection,
    prices: Mapping[str, pd.DataFrame],
    *,
    start: date,
    end: date,
    benchmark: str | None = None,
    minimum_coverage: float = 0.8,
) -> SectorRotationResult:
    """Analyze one common benchmark window without filling missing stock prices."""
    if start >= end:
        raise ValueError("Sector rotation: start must be before end")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("Sector rotation: minimum coverage must be in (0, 1]")
    benchmark = benchmark or selection.catalogue.benchmark
    if not selection.groups:
        raise ValueError("Sector rotation: at least one group is required")
    benchmark_close = _usable_close(prices.get(benchmark)).loc[
        str(start - timedelta(days=400)) : str(end)
    ]
    if benchmark_close.empty:
        raise ValueError("Sector rotation: benchmark prices are missing or invalid")
    anchors = benchmark_close.index[benchmark_close.index <= pd.Timestamp(start)]
    if len(anchors) == 0:
        raise ValueError(
            "Sector rotation: benchmark does not cover the requested start"
        )
    anchor = anchors[-1]
    benchmark_window = benchmark_close.loc[anchor:]
    if benchmark_window.isna().any():
        raise ValueError(
            "Sector rotation: benchmark prices are missing or invalid in the analysis window"
        )
    missing_sessions = _missing_benchmark_sessions(
        selection, prices, pd.DatetimeIndex(benchmark_window.index), end
    )
    warnings = [selection.catalogue.membership_note]
    if missing_sessions:
        # Keep member sessions in the common calendar so month-end rebalances
        # stay where the stocks traded; the benchmark holds its last close.
        warnings.append(
            f"Benchmark {benchmark} has no price for "
            f"{len(missing_sessions)} session(s) where member stocks traded "
            f"({_format_sessions(missing_sessions)}); its last close was carried "
            "forward on those sessions."
        )
        benchmark_close = benchmark_close.reindex(
            benchmark_close.index.union(missing_sessions)
        ).ffill()
        benchmark_window = benchmark_close.loc[anchor:]
    if len(benchmark_window) < 2:
        raise ValueError("Sector rotation: fewer than two benchmark sessions")
    if (pd.Timestamp(start) - anchor).days > 7:
        raise ValueError(
            "Sector rotation: benchmark start is stale by more than seven days"
        )
    if (pd.Timestamp(end) - benchmark_window.index[-1]).days > 7:
        raise ValueError(
            "Sector rotation: benchmark end is stale by more than seven days"
        )
    if start < selection.catalogue.observed_on:
        warnings.append(
            "Historical view uses current curated members and has survivorship bias."
        )
    basket_curves = {"benchmark": benchmark_window / benchmark_window.iloc[0] * 100}
    benchmark_return = _return_percent(benchmark_window)
    summary_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for group in selection.groups:
        histories = pd.DataFrame(
            {
                ticker: _usable_close(
                    prices.get(tv_to_yf(ticker, selection.catalogue.market))
                ).reindex(benchmark_close.index)
                for ticker in group.members
            },
            index=benchmark_close.index,
        )
        window = histories.loc[anchor:]
        curve, coverage = build_sector_basket(window, minimum_coverage=minimum_coverage)
        basket_curves[group.id] = curve
        group_return = _return_percent(curve)
        members = _member_metrics(group, histories, window)
        member_rows.extend(members)
        status = "ok" if pd.notna(group_return) else "insufficient_data"
        row: dict[str, object] = {
            "group_id": group.id,
            "group": group.name,
            "parent": group.parent,
            "kind": group.kind,
            "market": selection.catalogue.market,
            "description": group.description,
            "drivers": "; ".join(group.drivers),
            "risks": "; ".join(group.risks),
            "notes": group.notes,
            "sources": "; ".join(group.sources),
            "start": anchor.date().isoformat(),
            "end": benchmark_window.index[-1].date().isoformat(),
            "return_pct": group_return,
            "benchmark_return_pct": benchmark_return,
            "excess_return_pp": group_return - benchmark_return,
            "relative_strength_pct": (
                ((1 + group_return / 100) / (1 + benchmark_return / 100) - 1) * 100
            ),
            "members": len(group.members),
            "priced_members": int(window.iloc[-1].notna().sum()),
            "minimum_coverage_pct": float(coverage.min() * 100),
            "status": status,
            "concentrated": len(group.members) < 3,
        }
        for days in (50, 200):
            available = [
                member[f"above_sma{days}"]
                for member in members
                if member[f"above_sma{days}"] is not None
            ]
            row[f"breadth_sma{days}_pct"] = (
                sum(bool(value) for value in available) / len(available) * 100
                if available
                else float("nan")
            )
            row[f"breadth_sma{days}_members"] = len(available)
        summary_rows.append(row)
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            ["excess_return_pp", "group_id"],
            ascending=[False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    summary.insert(
        0,
        "rank",
        summary["excess_return_pp"].rank(ascending=False, method="min").astype("Int64"),
    )
    incomplete = int((summary.status != "ok").sum())
    if incomplete:
        warnings.append(
            f"{incomplete} group(s) unranked because coverage or held prices are missing."
        )
    warnings.append("Baskets with fewer than three selected members are concentrated.")
    return SectorRotationResult(
        selection,
        anchor.date(),
        benchmark_window.index[-1].date(),
        benchmark,
        summary,
        pd.DataFrame(member_rows),
        pd.DataFrame(basket_curves),
        tuple(warnings),
    )


def fetch_sector_rotation(
    selection: SectorSelection,
    fetcher: PriceFetcher,
    *,
    start: date,
    end: date,
    benchmark: str | None = None,
    minimum_coverage: float = 0.8,
    fallback: PriceFetcher | None = None,
) -> SectorRotationResult:
    """Fetch adjusted history once, including 400 calendar days for breadth warmup.

    ``fallback`` (FMP) supplies the entire history of any ticker the primary
    provider returns empty. When the benchmark omits sessions where members
    traded, the whole benchmark history comes from ``fallback`` if that
    provider has them all. Mixing two providers inside one series would splice
    different adjustment bases, so the fallback replaces a ticker or is unused.
    """
    benchmark = benchmark or selection.catalogue.benchmark
    tickers = sorted(
        {
            tv_to_yf(ticker, selection.catalogue.market)
            for group in selection.groups
            for ticker in group.members
        }
        | {benchmark}
    )
    prices = fetcher.fetch(tickers, start - timedelta(days=400), end)
    fallback_warnings: list[str] = []
    empty = [ticker for ticker in tickers if _usable_close(prices.get(ticker)).empty]
    if empty and fallback is not None:
        replacements = fallback.fetch(empty, start - timedelta(days=400), end)
        recovered = [
            ticker
            for ticker in empty
            if not _usable_close(replacements.get(ticker)).empty
        ]
        prices = {**prices, **{ticker: replacements[ticker] for ticker in recovered}}
        if recovered:
            fallback_warnings.append(
                f"yfinance returned no prices for {len(recovered)} ticker(s); "
                f"FMP history was used for the entire ticker: {', '.join(recovered)}."
            )
        empty = [ticker for ticker in empty if ticker not in recovered]
    if empty:
        fallback_warnings.append(
            f"No prices for {len(empty)} ticker(s)"
            + ("" if fallback is not None else " and FMP_API_KEY is not set")
            + f": {', '.join(empty)}."
        )
    benchmark_close = _usable_close(prices.get(benchmark)).loc[: str(end)]
    if not benchmark_close.empty:
        # Anchor calendar periods to the effective end session, not a weekend
        # or holiday requested by the caller. Keep requested end for stale checks.
        start = benchmark_close.index[-1].date() - (end - start)
    fallback_warning: str | None = None
    sessions = _benchmark_window_sessions(prices.get(benchmark), start, end)
    missing = (
        _missing_benchmark_sessions(selection, prices, sessions, end)
        if len(sessions)
        else []
    )
    if missing and fallback is None:
        fallback_warning = (
            f"Benchmark {benchmark}: FMP fallback not tried because FMP_API_KEY "
            "is not set."
        )
    elif missing and fallback is not None:
        fallback_warning = (
            f"Benchmark {benchmark}: FMP fallback does not cover the missing "
            "session(s), so the primary provider history was kept."
        )
        replacement = fallback.fetch([benchmark], start - timedelta(days=400), end).get(
            benchmark
        )
        replacement_sessions = _benchmark_window_sessions(replacement, start, end)
        if (
            replacement is not None
            and len(replacement_sessions)
            and replacement_sessions[0] <= sessions[0]
            and replacement_sessions[-1] >= sessions[-1]
            and not _missing_benchmark_sessions(
                selection, prices, replacement_sessions, end
            )
        ):
            prices = {**prices, benchmark: replacement}
            fallback_warning = (
                f"Benchmark {benchmark}: the primary provider has no price for "
                f"{len(missing)} session(s) where member stocks traded "
                f"({_format_sessions(missing)}); FMP history was used for the "
                "entire benchmark."
            )
    result = compute_sector_rotation(
        selection,
        prices,
        start=start,
        end=end,
        benchmark=benchmark,
        minimum_coverage=minimum_coverage,
    )
    if fallback_warning is not None:
        fallback_warnings.append(fallback_warning)
    if not fallback_warnings:
        return result
    return replace(result, warnings=(*result.warnings, *fallback_warnings))
