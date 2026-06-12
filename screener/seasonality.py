"""Calendar seasonality statistics computed from daily OHLCV bars.

Three views over a single ticker's close series:

1. Per-calendar-month returns across years (mean, median, win rate,
   best, worst).
2. Turn-of-month effect — mean daily return over the last 3 + first 3
   trading days of each month vs all other trading days.
3. Day-of-week mean daily returns.

Pure computation lives here; the click command in
``screener/commands/seasonality.py`` handles fetching and output.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import cast

import pandas as pd
from rich.console import Console
from rich.table import Table

TURN_OF_MONTH_WINDOW = 3
TURN_OF_MONTH_LABEL = "Turn of month"
OTHER_DAYS_LABEL = "Other days"
DAY_LABELS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@dataclass(frozen=True)
class GroupStats:
    """Summary statistics for one bucket of returns."""

    label: str
    count: int
    mean: float
    median: float
    win_rate: float
    best: float
    worst: float


@dataclass(frozen=True)
class SeasonalityReport:
    ticker: str
    start: date
    end: date
    monthly: list[GroupStats]
    turn_of_month: list[GroupStats]
    day_of_week: list[GroupStats]


def _group_stats(label: str, returns: pd.Series) -> GroupStats:
    return GroupStats(
        label=label,
        count=int(returns.count()),
        mean=float(returns.mean()),
        median=float(returns.median()),
        win_rate=float((returns > 0).mean()),
        best=float(returns.max()),
        worst=float(returns.min()),
    )


def _monthly_stats(close: pd.Series) -> list[GroupStats]:
    month_end = close.resample("ME").last().dropna()
    monthly_returns = month_end.pct_change().dropna()
    out: list[GroupStats] = []
    month_idx = cast(pd.DatetimeIndex, monthly_returns.index)
    grouped = dict(iter(monthly_returns.groupby(month_idx.month)))
    for month in range(1, 13):
        returns = grouped.get(month)
        if returns is None or returns.empty:
            continue
        out.append(_group_stats(calendar.month_abbr[month], returns))
    return out


def _turn_of_month_stats(close: pd.Series, daily: pd.Series) -> list[GroupStats]:
    marker = pd.Series(0, index=close.index)
    close_idx = cast(pd.DatetimeIndex, close.index)
    grouped = marker.groupby(close_idx.to_period("M"))
    position = grouped.cumcount()
    reverse_position = grouped.cumcount(ascending=False)
    tom_mask = (position < TURN_OF_MONTH_WINDOW) | (
        reverse_position < TURN_OF_MONTH_WINDOW
    )
    aligned = tom_mask.reindex(daily.index).fillna(False).astype(bool)
    return [
        _group_stats(TURN_OF_MONTH_LABEL, daily[aligned]),
        _group_stats(OTHER_DAYS_LABEL, daily[~aligned]),
    ]


def _day_of_week_stats(daily: pd.Series) -> list[GroupStats]:
    daily_idx = cast(pd.DatetimeIndex, daily.index)
    grouped = dict(iter(daily.groupby(daily_idx.dayofweek)))
    out: list[GroupStats] = []
    for day in range(7):
        returns = grouped.get(day)
        if returns is None or returns.empty:
            continue
        out.append(_group_stats(DAY_LABELS[day], returns))
    return out


def compute_seasonality(bars: pd.DataFrame, ticker: str) -> SeasonalityReport:
    """Compute the full seasonality report from an OHLCV frame.

    Raises ``ValueError`` when there is not enough history to compute
    daily returns.
    """
    close = bars["close"].dropna().sort_index()
    if len(close) < 2:
        raise ValueError(f"Not enough price history for {ticker} (need >= 2 bars).")
    daily = close.pct_change().dropna()
    return SeasonalityReport(
        ticker=ticker,
        start=close.index[0].date(),
        end=close.index[-1].date(),
        monthly=_monthly_stats(close),
        turn_of_month=_turn_of_month_stats(close, daily),
        day_of_week=_day_of_week_stats(daily),
    )


_SECTIONS: list[tuple[str, str, str]] = [
    ("monthly", "Monthly returns across years", "Month"),
    ("turn_of_month", "Turn-of-month effect (daily returns)", "Bucket"),
    ("day_of_week", "Day-of-week (daily returns)", "Day"),
]


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def render_report(report: SeasonalityReport, console: Console) -> None:
    console.print(
        f"[bold]Seasonality — {report.ticker}[/bold]  ({report.start} → {report.end})"
    )
    for attr, title, label_header in _SECTIONS:
        rows: list[GroupStats] = getattr(report, attr)
        table = Table(title=title)
        table.add_column(label_header)
        table.add_column("N", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Win rate", justify="right")
        table.add_column("Best", justify="right")
        table.add_column("Worst", justify="right")
        for stats in rows:
            mean_style = "green" if stats.mean >= 0 else "red"
            table.add_row(
                stats.label,
                str(stats.count),
                f"[{mean_style}]{_pct(stats.mean)}[/{mean_style}]",
                _pct(stats.median),
                f"{stats.win_rate * 100:.1f}%",
                _pct(stats.best),
                _pct(stats.worst),
            )
        console.print(table)


def report_to_csv(report: SeasonalityReport) -> str:
    rows = []
    for attr, _title, _header in _SECTIONS:
        for stats in getattr(report, attr):
            rows.append(
                {
                    "section": attr,
                    "label": stats.label,
                    "count": stats.count,
                    "mean_return": stats.mean,
                    "median_return": stats.median,
                    "win_rate": stats.win_rate,
                    "best": stats.best,
                    "worst": stats.worst,
                }
            )
    return pd.DataFrame(rows).to_csv(index=False)
