"""Point-in-time options-panel joins for Pine-like backtest expressions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from types import NoneType
from typing import get_args
from zoneinfo import ZoneInfo

import pandas as pd

from screener.backtester.pine import Node, collect_names
from screener.markets import get_market
from screener.options.models import ChainMetrics
from screener.options.panels import INTRADAY_PANEL_FIELDS, read_options_panel
from screener.symbols import tv_to_nse, tv_to_yf

# Regular-session close (market-local). Daily panel rows are stamped at midnight;
# on intraday targets they only become available at this clock (see H5).
_SESSION_CLOSE_LOCAL: dict[str, time] = {
    "us": time(16, 0),
    "india": time(15, 30),
}


def _numeric_metrics_fields() -> frozenset[str]:
    numeric_types = {int, float}
    return frozenset(
        name
        for name, field in ChainMetrics.model_fields.items()
        if (set(get_args(field.annotation) or (field.annotation,)) - {NoneType})
        and (set(get_args(field.annotation) or (field.annotation,)) - {NoneType})
        <= numeric_types
    )


PANEL_DERIVED_EXPRESSION_FIELDS = frozenset(
    {
        "options_volume",
        "iv_rank",
        "iv_percentile",
        "iv_history_days",
        "options_volume_avg_20",
        "unusual_options_ratio",
        "history_days",
        # Phase 3.4 intraday-derived columns (present on the panel schema, NaN
        # until a store reduction fills them); expressions may reference them.
        *INTRADAY_PANEL_FIELDS,
    }
)
OPTION_EXPRESSION_FIELDS = _numeric_metrics_fields() | PANEL_DERIVED_EXPRESSION_FIELDS


@dataclass(frozen=True)
class OptionsJoinResult:
    bars_by_tv: dict[str, pd.DataFrame]
    fields: tuple[str, ...]
    covered_symbols: tuple[str, ...]
    panel_rows: int


def referenced_options_fields(*nodes: Node | None) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        if node is not None:
            names |= collect_names(node)
    return names & OPTION_EXPRESSION_FIELDS


def _symbol_key(symbol: str, market: str) -> str:
    if market == "india":
        return tv_to_nse(symbol, strip_suffix=True)
    return tv_to_yf(symbol, market).upper().replace(".", "-")


def _index_is_intraday(index: pd.DatetimeIndex) -> bool:
    """True when any timestamp carries a non-midnight clock (intraday bars)."""
    if len(index) == 0:
        return False
    return bool(
        (index.hour != 0).any()
        or (index.minute != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
    )


def _session_close_availability(
    as_of_index: pd.DatetimeIndex,
    *,
    market: str,
    target_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Map midnight-stamped daily panel rows to session-close availability.

    Daily option metrics are end-of-session observations stamped at midnight.
    On an *intraday* target index they must only ffill onto bars at/after that
    day's regular close (local), converted to the target's timezone convention
    (naive UTC for the bar store, or the target's tz when it is aware).
    Daily (midnight-normalized) targets keep the midnight stamp unchanged.
    """
    if not _index_is_intraday(target_index):
        return as_of_index

    close_local = _SESSION_CLOSE_LOCAL.get(market, time(16, 0))
    market_tz = ZoneInfo(get_market(market).timezone)
    target_tz = target_index.tz
    shifted: list[pd.Timestamp] = []
    for ts in as_of_index:
        day: date = pd.Timestamp(ts).date()
        local_close = datetime.combine(day, close_local, tzinfo=market_tz)
        if target_tz is not None:
            converted = local_close.astimezone(target_tz)
            shifted.append(pd.Timestamp(converted))
        else:
            # Canonical bar store: naive UTC.
            utc_naive = local_close.astimezone(timezone.utc).replace(tzinfo=None)
            shifted.append(pd.Timestamp(utc_naive))
    return pd.DatetimeIndex(shifted)


def merge_options_into_bars(
    bars_by_tv: dict[str, pd.DataFrame],
    *,
    market: str,
    fields: set[str],
    panel: pd.DataFrame | None = None,
) -> OptionsJoinResult:
    """Forward-fill only panel rows whose ``as_of`` is <= each price bar.

    Requested fields are added as NaN for symbols/dates with no coverage, so a
    comparison evaluates false rather than aborting the entire backtest with an
    unknown-name error. Existing injected bar columns are retained where the
    canonical panel has no value.

    Daily panel rows are midnight-stamped end-of-session observations. When the
    target bar index is intraday, each row becomes available only at that day's
    session close (no same-day lookahead of closing IV/OI onto morning bars).
    Daily-bar targets keep midnight availability (byte-identical to prior
    behavior).
    """
    requested = tuple(sorted(fields & OPTION_EXPRESSION_FIELDS))
    if not requested:
        return OptionsJoinResult(dict(bars_by_tv), (), (), 0)
    source = read_options_panel(market) if panel is None else panel.copy()
    usable = source.copy()
    if usable.empty or not {"as_of", "SYMBOL"} <= set(usable.columns):
        usable = pd.DataFrame(columns=["as_of", "SYMBOL", *requested])
    else:
        usable["as_of"] = pd.to_datetime(
            usable["as_of"], errors="coerce"
        ).dt.normalize()
        usable = usable[usable["as_of"].notna()].copy()
        usable["_symbol_key"] = usable["SYMBOL"].map(
            lambda value: _symbol_key(str(value), market)
        )
        usable = usable.drop_duplicates(["as_of", "_symbol_key"], keep="last")

    out: dict[str, pd.DataFrame] = {}
    covered: list[str] = []
    for tv_symbol, original in bars_by_tv.items():
        if original is None or original.empty:
            out[tv_symbol] = original
            continue
        bars = original.copy()
        key = _symbol_key(tv_symbol, market)
        symbol_rows = (
            usable[usable["_symbol_key"] == key].sort_values("as_of")
            if "_symbol_key" in usable.columns
            else usable.iloc[0:0]
        )
        has_coverage = False
        for field in requested:
            existing = (
                pd.to_numeric(bars[field], errors="coerce")
                if field in bars.columns
                else pd.Series(float("nan"), index=bars.index, dtype=float)
            )
            if field not in symbol_rows.columns or symbol_rows.empty:
                bars[field] = existing
                continue
            series = pd.to_numeric(
                symbol_rows.set_index("as_of")[field], errors="coerce"
            )
            series = series[~series.index.duplicated(keep="last")].sort_index()
            target_index = pd.DatetimeIndex(pd.to_datetime(bars.index))
            series.index = pd.DatetimeIndex(series.index)
            # Shift midnight daily rows to session close on intraday targets
            # before timezone alignment / ffill.
            series.index = _session_close_availability(
                series.index, market=market, target_index=target_index
            )
            if target_index.tz is not None and series.index.tz is None:
                series.index = series.index.tz_localize(target_index.tz)
            elif target_index.tz is None and series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            aligned = series.reindex(target_index, method="ffill")
            aligned.index = bars.index
            bars[field] = aligned.combine_first(existing)
            has_coverage = has_coverage or bool(aligned.notna().any())
        if has_coverage:
            covered.append(tv_symbol)
        out[tv_symbol] = bars
    return OptionsJoinResult(
        bars_by_tv=out,
        fields=requested,
        covered_symbols=tuple(sorted(covered)),
        panel_rows=len(usable),
    )


def merge_referenced_options(
    bars_by_tv: dict[str, pd.DataFrame],
    *,
    market: str,
    entry_ast: Node,
    exit_ast: Node | None,
    warnings: list[str],
) -> dict[str, pd.DataFrame]:
    """Join only identifiers actually referenced by the strategy ASTs."""
    fields = referenced_options_fields(entry_ast, exit_ast)
    if not fields:
        return bars_by_tv
    result = merge_options_into_bars(
        bars_by_tv,
        market=market,
        fields=fields,
    )
    covered = len(result.covered_symbols)
    total = len(
        [
            frame
            for frame in bars_by_tv.values()
            if frame is not None and not frame.empty
        ]
    )
    field_note = ", ".join(result.fields)
    if covered == 0:
        warnings.append(
            f"options panel has no point-in-time coverage for [{field_note}]; "
            "affected entry/exit comparisons evaluate false"
        )
    else:
        warnings.append(
            f"options panel joined point-in-time fields [{field_note}] for "
            f"{covered}/{total} ticker(s); uncovered dates remain NaN"
        )
    return result.bars_by_tv


__all__ = [
    "OPTION_EXPRESSION_FIELDS",
    "OptionsJoinResult",
    "merge_options_into_bars",
    "merge_referenced_options",
    "referenced_options_fields",
]
