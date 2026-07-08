"""Rolling backtest candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _RollingCandidateMatrices:
    """Precomputed per-day matrices for vectorized candidate selection.

    The numpy mirrors (``*_np``, ``row_by_day``, ``col_by_ticker``, ``tickers``)
    are derived once from the DataFrames so the per-day scan in
    :func:`_candidate_rows_for_day` runs on raw arrays and integer positions
    instead of pandas label lookups (``.loc[day]`` / ``.at[day, ticker]``).
    """

    signal_mat: pd.DataFrame
    lookback_ok_mat: pd.DataFrame
    filter_mat: pd.DataFrame | None
    dollar_vol_mat: pd.DataFrame
    close_mat: pd.DataFrame
    volume_mat: pd.DataFrame
    bar_idx_mat: pd.DataFrame
    # Optional cross-sectional factor score. When a strategy's ``prepare_bars``
    # hook writes a ``rank_score`` column into its bars, candidates on each day
    # are ranked by this score (descending) instead of by signal-day dollar
    # volume — turning the backtester into a real factor-portfolio selector.
    # ``None`` when no ticker carries the column, which preserves the legacy
    # dollar-volume ranking byte-for-byte.
    rank_score_mat: pd.DataFrame | None
    tickers: tuple[str, ...]
    row_by_day: dict[pd.Timestamp, int]
    col_by_ticker: dict[str, int]
    signal_np: np.ndarray
    lookback_ok_np: np.ndarray
    filter_np: np.ndarray | None
    dollar_vol_np: np.ndarray
    close_np: np.ndarray
    volume_np: np.ndarray
    bar_idx_np: np.ndarray
    # Numpy mirror of ``rank_score_mat`` (``None`` when no factor scores), kept
    # aligned column-for-column with the other ``*_np`` arrays.
    rank_score_np: np.ndarray | None


def _build_rolling_candidate_matrices(
    bars_by_tv: dict[str, pd.DataFrame],
    entry_signals_by_tv: dict[str, pd.Series],
    filter_signals_by_tv: dict[str, pd.Series],
    master_dates: list[pd.Timestamp],
    lookback_required: int,
    membership_added: dict[str, date] | None = None,
    regime_allowed: pd.Series | None = None,
    warnings: list[str] | None = None,
) -> _RollingCandidateMatrices:
    """Build once-per-run matrices for daily candidate scans."""
    master_ix = pd.DatetimeIndex(master_dates)
    valid_tickers = [
        tv for tv, bars in bars_by_tv.items() if bars is not None and not bars.empty
    ]
    signal_mat = (
        pd.DataFrame(entry_signals_by_tv)
        .reindex(master_ix)
        .reindex(columns=valid_tickers)
        .eq(True)
    )
    # Point-in-time eligibility: suppress entry signals before a symbol's
    # index "date added" so today's constituents are not backtested through
    # history they were never selectable in.
    if membership_added:
        for tv, added in membership_added.items():
            if tv in signal_mat.columns:
                signal_mat.loc[master_ix < pd.Timestamp(added), tv] = False
    # Benchmark-regime gate: suppress every entry signal on days whose
    # benchmark regime is not allowed (days missing from the benchmark
    # calendar inherit the most recent prior regime; warmup days are blocked).
    if regime_allowed is not None:
        allowed = (
            regime_allowed.reindex(master_ix, method="ffill").fillna(False).astype(bool)
        )
        signal_mat.loc[~allowed.to_numpy(), :] = False
    # Empty dict sentinel: no min-price / ADV filters configured.
    filter_mat: pd.DataFrame | None
    if filter_signals_by_tv:
        filter_mat = (
            pd.DataFrame(filter_signals_by_tv)
            .reindex(master_ix)
            .reindex(columns=valid_tickers)
            .eq(True)
        )
    else:
        filter_mat = None

    bar_cols: dict[str, np.ndarray] = {}
    lookback_cols: dict[str, np.ndarray] = {}
    close_cols: dict[str, np.ndarray] = {}
    volume_cols: dict[str, np.ndarray] = {}
    # Cross-sectional factor scores (as-of the signal bar), only populated for
    # tickers whose prepared bars carry a ``rank_score`` column.
    score_cols: dict[str, np.ndarray] = {}
    any_score = False
    missing_score: list[str] = []
    for tv in valid_tickers:
        bars = bars_by_tv[tv]
        close = bars["close"].astype(float).to_numpy()
        volume = bars["volume"].astype(float).to_numpy()
        pos = bars.index.searchsorted(master_ix, side="right") - 1
        pos = np.where(pos < 0, -1, pos)
        n = len(bars)
        has_bar = pos >= 0
        bar_cols[tv] = pos
        lookback_cols[tv] = (pos + 1 >= lookback_required + 1) & (pos + 1 < n) & has_bar
        close_cols[tv] = np.where(has_bar, close[pos], np.nan)
        volume_cols[tv] = np.where(has_bar, volume[pos], np.nan)
        if "rank_score" in bars.columns:
            any_score = True
            score = bars["rank_score"].astype(float).to_numpy()
            score_cols[tv] = np.where(has_bar, score[pos], np.nan)
        else:
            missing_score.append(tv)
            score_cols[tv] = np.full(len(master_ix), np.nan)

    # Mixed universe: a factor strategy is ranking by ``rank_score`` but some
    # tickers never received the column (e.g. a partial/heterogeneous prepare).
    # Those names get an all-NaN score and are silently dropped at selection
    # time, so surface them explicitly rather than excluding them without trace.
    if any_score and missing_score and warnings is not None:
        preview = ", ".join(sorted(missing_score)[:5])
        more = "" if len(missing_score) <= 5 else f" (+{len(missing_score) - 5} more)"
        warnings.append(
            f"factor ranking active but {len(missing_score)} ticker(s) lack a "
            f"rank_score column; excluded from selection: {preview}{more}"
        )

    bar_idx_mat = pd.DataFrame(bar_cols, index=master_ix)
    lookback_ok_mat = pd.DataFrame(lookback_cols, index=master_ix).astype(bool)
    close_mat = pd.DataFrame(close_cols, index=master_ix)
    volume_mat = pd.DataFrame(volume_cols, index=master_ix)
    dollar_vol_mat = close_mat * volume_mat
    rank_score_mat = pd.DataFrame(score_cols, index=master_ix) if any_score else None
    return _RollingCandidateMatrices(
        signal_mat=signal_mat,
        lookback_ok_mat=lookback_ok_mat,
        filter_mat=filter_mat,
        dollar_vol_mat=dollar_vol_mat,
        close_mat=close_mat,
        volume_mat=volume_mat,
        bar_idx_mat=bar_idx_mat,
        rank_score_mat=rank_score_mat,
        tickers=tuple(valid_tickers),
        row_by_day={ts: i for i, ts in enumerate(master_ix)},
        col_by_ticker={tv: j for j, tv in enumerate(valid_tickers)},
        signal_np=signal_mat.to_numpy(),
        lookback_ok_np=lookback_ok_mat.to_numpy(),
        filter_np=filter_mat.to_numpy() if filter_mat is not None else None,
        dollar_vol_np=dollar_vol_mat.to_numpy(),
        close_np=close_mat.to_numpy(),
        volume_np=volume_mat.to_numpy(),
        bar_idx_np=bar_idx_mat.to_numpy(),
        rank_score_np=rank_score_mat.to_numpy() if rank_score_mat is not None else None,
    )


def _candidate_rows_for_day(
    day: pd.Timestamp,
    matrices: _RollingCandidateMatrices,
    *,
    exclude: set[str],
) -> tuple[list[dict], list[str]]:
    """Evaluate entry signals for the full universe on one trading day."""
    warnings: list[str] = []
    row = matrices.row_by_day[day]
    eligible = matrices.signal_np[row] & matrices.lookback_ok_np[row]
    if matrices.filter_np is not None:
        eligible = eligible & matrices.filter_np[row]
    if exclude:
        for ticker in exclude:
            col = matrices.col_by_ticker.get(ticker)
            if col is not None:
                eligible[col] = False
    dollar_vol = matrices.dollar_vol_np[row]
    eligible = eligible & ~np.isnan(dollar_vol)
    # Factor portfolios rank by cross-sectional score; everything else keeps the
    # legacy signal-day dollar-volume ordering. The dollar-volume value is still
    # surfaced as ``as_of_dollar_vol`` either way for reporting/liquidity audit.
    if matrices.rank_score_np is not None:
        rank_score = matrices.rank_score_np[row]
        eligible = eligible & ~np.isnan(rank_score)
        eligible_cols = np.nonzero(eligible)[0]
        if eligible_cols.size == 0:
            return [], warnings
        # Rank by cross-sectional factor score (descending), breaking ties by
        # signal-day dollar volume so equal-score names resolve by liquidity —
        # a principled deterministic fallback — rather than by arbitrary
        # universe/column insertion order. The DataFrame rows are in ascending
        # column order (``eligible_cols``), so a stable mergesort reproduces the
        # legacy pandas ranking byte-for-byte.
        order = (
            pd.DataFrame(
                {
                    "rank_score": rank_score[eligible_cols],
                    "dollar_vol": dollar_vol[eligible_cols],
                }
            )
            .sort_values(
                ["rank_score", "dollar_vol"], ascending=False, kind="mergesort"
            )
            .index.to_numpy()
        )
    else:
        eligible_cols = np.nonzero(eligible)[0]
        if eligible_cols.size == 0:
            return [], warnings
        # Match ``sort_values(ascending=False, kind="mergesort")`` exactly: pandas
        # reverses the stable ascending order, so ties land in reversed column
        # order (see pandas ``nargsort``).
        order = dollar_vol[eligible_cols].argsort(kind="mergesort")[::-1]
    rows: list[dict] = []
    for rank, col in enumerate(eligible_cols[order], start=1):
        rows.append(
            {
                "ticker": matrices.tickers[col],
                "signal_idx": int(matrices.bar_idx_np[row, col]),
                "as_of_close": float(matrices.close_np[row, col]),
                "as_of_volume": float(matrices.volume_np[row, col]),
                "as_of_dollar_vol": float(dollar_vol[col]),
                "rank": rank,
                "role": "active",
            }
        )
    return rows, warnings
