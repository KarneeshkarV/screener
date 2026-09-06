"""Rolling backtest candidate selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


def _preview_warning(symbols: list[str], template: str) -> str:
    """Format a count plus a stable five-symbol preview for warning messages."""
    preview = ", ".join(sorted(symbols)[:5])
    more = "" if len(symbols) <= 5 else f" (+{len(symbols) - 5} more)"
    return template.format(count=len(symbols), preview=f"{preview}{more}")


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
    # Percentile floor a candidate's ``setup_score`` must clear, on the 0-100
    # scale. ``None`` disables the gate. It rides on the matrices rather than
    # on the per-day call so every caller of :func:`_candidate_rows_for_day`
    # gets the run's gate without having to remember to pass it.
    min_score: float | None = None


def _sector_neutralize_scores(
    rank_score_mat: pd.DataFrame,
    sector_by_tv: dict[str, str],
) -> pd.DataFrame:
    """Z-score ``rank_score`` within each (day, sector) group.

    Vectorized: stack → groupby transform → unstack. Population std (ddof=0);
    sectors with std 0 or a single non-NaN name get neutralized score 0.
    NaN input scores stay NaN (still ineligible at selection time).
    """
    if rank_score_mat.empty:
        return rank_score_mat
    # Map each column to its sector; missing tickers fall into UNKNOWN.
    sectors = {col: sector_by_tv.get(col, "UNKNOWN") for col in rank_score_mat.columns}
    long = pd.Series(
        rank_score_mat.stack(future_stack=True),
        dtype=float,
    )
    long.index = long.index.set_names(["date", "ticker"])
    sector = pd.Series(
        long.index.get_level_values("ticker").map(sectors),
        index=long.index,
    )
    date = pd.Series(long.index.get_level_values("date"), index=long.index)
    group_keys = [date, sector]
    mu = long.groupby(group_keys, sort=False).transform("mean")
    # Population variance via mean of squared deviations (handles n=1 as 0).
    centered = long - mu
    var = centered.pow(2).groupby(group_keys, sort=False).transform("mean")
    sigma = var.pow(0.5)
    zscore = centered / sigma
    # std 0 / single-name / all-equal → 0; preserve NaN inputs.
    zscore = zscore.mask(~(sigma > 0), 0.0)
    zscore = zscore.mask(long.isna(), np.nan)
    neutralized = zscore.unstack(level="ticker")
    # Preserve original column order (unstack may reorder alphabetically).
    if isinstance(neutralized, pd.Series):
        neutralized = neutralized.to_frame().T
    return neutralized.reindex(
        index=rank_score_mat.index, columns=rank_score_mat.columns
    )


def _signal_mask_matrix(
    signals_by_tv: Mapping[str, pd.Series | np.ndarray],
    bars_by_tv: dict[str, pd.DataFrame],
    master_ix: pd.DatetimeIndex,
    valid_tickers: list[str],
) -> pd.DataFrame:
    """Assemble a master-calendar bool matrix from per-ticker Series or arrays.

    Bool ndarrays are assumed aligned to ``bars_by_tv[tv].index`` (the panel
    evaluator contract). Series carry their own index.

    Both spellings agree on what an unknown value means: a missing or NaN
    entry is "no signal". That has to be stated because the two branches are
    easy to make disagree - ``np.asarray([np.nan], dtype=bool)`` is ``True``,
    so a plain cast on the array branch would invent an entry the caller never
    signalled, while the Series branch reads the same input as ``False``.

    Alignment is by exact label. ``Index.get_indexer`` returns -1 for a master
    date the ticker has no bar on, and those columns keep the ``False`` the
    zeroed block already holds, which is what ``reindex(...).fillna(False)``
    used to produce one Series at a time. Doing it positionally is about 4.5x
    faster across a 5,000-name field, where the per-ticker Series round trip
    cost 0.83s.

    Resist the temptation to use ``searchsorted(side="right")`` here, which is
    what the close and volume matrices below do. Those want the last known
    bar; a signal must not carry into a session the ticker did not trade, and
    searchsorted would do exactly that.
    """
    n_days = len(master_ix)
    n_tickers = len(valid_tickers)
    block = np.zeros((n_days, n_tickers), dtype=bool)
    for column, tv in enumerate(valid_tickers):
        signal = signals_by_tv.get(tv)
        if signal is None:
            continue
        if isinstance(signal, np.ndarray):
            index = bars_by_tv[tv].index
            if signal.dtype == bool:
                values = signal
            else:
                # Route a non-bool array through pandas rather than casting it,
                # so NaN reads as "no signal" exactly as it does for a Series.
                # A direct ``np.asarray(..., dtype=bool)`` maps NaN to True.
                values = (
                    pd.Series(signal).fillna(False).astype(bool).to_numpy(dtype=bool)
                )
            # An array carries no labels, so a length that disagrees with the
            # ticker's bars means the caller aligned it to something else.
            # ``get_indexer`` would not raise, it would silently build the mask
            # from the first ``len(index)`` entries.
            if len(values) != len(index):
                raise ValueError(
                    f"signal array for {tv} has {len(values)} values against "
                    f"{len(index)} bars"
                )
        else:
            index = signal.index
            values = signal.fillna(False).astype(bool).to_numpy(dtype=bool)
        if len(values) == n_days and index.equals(master_ix):
            block[:, column] = values
            continue
        if not index.is_unique:
            # A duplicated label is a caller bug, and the pandas path raises a
            # message that names it. ``get_indexer`` raises something less
            # useful. A Series cannot reach the length check above, because its
            # values are built from its own index.
            block[:, column] = (
                pd.Series(values, index=index, copy=False)
                .reindex(master_ix)
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            continue
        positions = index.get_indexer(master_ix)
        found = positions >= 0
        block[found, column] = values[positions[found]]
    return pd.DataFrame(block, index=master_ix, columns=valid_tickers, copy=False)


def _build_rolling_candidate_matrices(
    bars_by_tv: dict[str, pd.DataFrame],
    entry_signals_by_tv: dict[str, pd.Series | np.ndarray],
    filter_signals_by_tv: dict[str, pd.Series],
    master_dates: list[pd.Timestamp],
    lookback_required: int,
    membership_added: dict[str, date] | None = None,
    membership_windows: tuple[tuple[str, date, date | None], ...] = (),
    dynamic_universe_size: int | None = None,
    dynamic_universe_lookback: int = 60,
    dynamic_universe_rebalance: str = "monthly",
    regime_allowed: pd.Series | None = None,
    earnings_blackout: dict[str, list[date]] | None = None,
    earnings_blackout_days: int | None = None,
    warnings: list[str] | None = None,
    *,
    sector_neutral: bool = False,
    sector_by_tv: dict[str, str] | None = None,
    require_next_bar: bool = True,
    min_score: float | None = None,
) -> _RollingCandidateMatrices:
    """Build once-per-run matrices for daily candidate scans.

    ``require_next_bar`` is the backtester's rule that a signal is only a
    candidate if a later bar exists to fill the entry on. A screen asks the
    opposite question - "who fires on the newest bar I have?" - and for it the
    signal bar is always the last one, so the rule would reject every answer.
    Screens pass ``False``; every backtest path keeps the default.
    """
    master_ix = pd.DatetimeIndex(master_dates)
    valid_tickers = [
        tv for tv, bars in bars_by_tv.items() if bars is not None and not bars.empty
    ]
    signal_mat = _signal_mask_matrix(
        entry_signals_by_tv, bars_by_tv, master_ix, valid_tickers
    )
    # Point-in-time eligibility: suppress entry signals before a symbol's
    # index "date added" so today's constituents are not backtested through
    # history they were never selectable in.
    if membership_added:
        for tv, added in membership_added.items():
            if tv in signal_mat.columns:
                signal_mat.loc[master_ix < pd.Timestamp(added), tv] = False
    if membership_windows:
        membership_mask = pd.DataFrame(
            False, index=master_ix, columns=valid_tickers, dtype=bool
        )
        for tv, effective_from, effective_to in membership_windows:
            if tv not in membership_mask.columns:
                continue
            eligible = master_ix >= pd.Timestamp(effective_from)
            if effective_to is not None:
                eligible &= master_ix < pd.Timestamp(effective_to)
            membership_mask.loc[eligible, tv] = True
        signal_mat &= membership_mask
    # Benchmark-regime gate: suppress every entry signal on days whose
    # benchmark regime is not allowed (days missing from the benchmark
    # calendar inherit the most recent prior regime; warmup days are blocked).
    if regime_allowed is not None:
        allowed = (
            regime_allowed.reindex(master_ix, method="ffill").fillna(False).astype(bool)
        )
        signal_mat.loc[~allowed.to_numpy(), :] = False
    # Earnings blackout gate: suppress entries on calendar days within N days
    # before (and including) a known earnings date for that ticker. Tickers with
    # no known earnings dates are left untouched (and warned about below).
    if (
        earnings_blackout is not None
        and earnings_blackout_days is not None
        and earnings_blackout_days >= 0
        and not signal_mat.empty
    ):
        day_ord = master_ix.map(pd.Timestamp.toordinal).to_numpy(dtype=np.int64)
        missing_earnings: list[str] = []
        for tv in valid_tickers:
            edates = earnings_blackout.get(tv) or []
            if not edates:
                missing_earnings.append(tv)
                continue
            ed_ord = np.fromiter(
                (pd.Timestamp(d).toordinal() for d in edates),
                dtype=np.int64,
                count=len(edates),
            )
            # In blackout when some earnings date E satisfies 0 <= E - day <= N.
            diffs = ed_ord[None, :] - day_ord[:, None]
            in_blackout = ((diffs >= 0) & (diffs <= earnings_blackout_days)).any(axis=1)
            if in_blackout.any():
                signal_mat.loc[in_blackout, tv] = False
        if missing_earnings and warnings is not None:
            warnings.append(
                _preview_warning(
                    missing_earnings,
                    "earnings blackout active but {count} ticker(s) lack earnings "
                    "dates; not gated: {preview}",
                )
            )
    # Empty dict sentinel: no min-price / ADV filters configured.
    #
    # Same assembly as the entry mask above, and for the same reason. This was
    # the untouched twin of that mask: on a 5,000-name field over 1,500 master
    # days with mixed listing dates, the ``.eq(True)`` round trip measured
    # 0.726s against 0.405s here. Every run that sets --min-price or
    # --min-avg-dollar-volume was handing part of the entry mask's saving back
    # four lines later.
    #
    # ``_precompute_filter_signals`` emits bool Series with no NaN, so this is
    # value-for-value what ``.eq(True)`` produced. The two disagree only on a
    # float signal, where ``eq(True)`` means "== 1.0" and the mask means
    # "!= 0"; that shape cannot reach here.
    filter_mat: pd.DataFrame | None
    if filter_signals_by_tv:
        filter_mat = _signal_mask_matrix(
            filter_signals_by_tv, bars_by_tv, master_ix, valid_tickers
        )
    else:
        filter_mat = None

    n_days = len(master_ix)
    n_tickers = len(valid_tickers)
    bar_idx_np = np.empty((n_days, n_tickers), dtype=np.int64)
    lookback_ok_np = np.empty((n_days, n_tickers), dtype=bool)
    close_np = np.empty((n_days, n_tickers), dtype=float)
    volume_np = np.empty((n_days, n_tickers), dtype=float)
    # Cross-sectional factor scores (as-of the signal bar), only populated for
    # tickers whose prepared bars carry a ``rank_score`` column.
    rank_score_np = np.full((n_days, n_tickers), np.nan, dtype=float)
    dynamic_score_np = (
        np.full((n_days, n_tickers), np.nan, dtype=float)
        if dynamic_universe_size is not None
        else None
    )
    any_score = False
    missing_score: list[str] = []
    for column, tv in enumerate(valid_tickers):
        bars = bars_by_tv[tv]
        close = bars["close"].astype(float).to_numpy()
        volume = bars["volume"].astype(float).to_numpy()
        pos = bars.index.searchsorted(master_ix, side="right") - 1
        pos = np.where(pos < 0, -1, pos)
        n = len(bars)
        has_bar = pos >= 0
        bar_idx_np[:, column] = pos
        enough_history = (pos + 1 >= lookback_required + 1) & has_bar
        lookback_ok_np[:, column] = (
            (enough_history & (pos + 1 < n)) if require_next_bar else enough_history
        )
        close_np[:, column] = np.where(has_bar, close[pos], np.nan)
        volume_np[:, column] = np.where(has_bar, volume[pos], np.nan)
        if dynamic_universe_size is not None:
            lagged_adv = (
                (bars["close"].astype(float) * bars["volume"].astype(float))
                .shift(1)
                .rolling(
                    dynamic_universe_lookback, min_periods=dynamic_universe_lookback
                )
                .mean()
                .to_numpy()
            )
            assert dynamic_score_np is not None
            dynamic_score_np[:, column] = np.where(has_bar, lagged_adv[pos], np.nan)
        if "rank_score" in bars.columns:
            any_score = True
            score = bars["rank_score"].astype(float).to_numpy()
            rank_score_np[:, column] = np.where(has_bar, score[pos], np.nan)
        else:
            missing_score.append(tv)

    # Mixed universe: a factor strategy is ranking by ``rank_score`` but some
    # tickers never received the column (e.g. a partial/heterogeneous prepare).
    # Those names get an all-NaN score and are silently dropped at selection
    # time, so surface them explicitly rather than excluding them without trace.
    if any_score and missing_score and warnings is not None:
        warnings.append(
            _preview_warning(
                missing_score,
                "factor ranking active but {count} ticker(s) lack a rank_score "
                "column; excluded from selection: {preview}",
            )
        )

    bar_idx_mat = pd.DataFrame(
        bar_idx_np, index=master_ix, columns=valid_tickers, copy=False
    )
    lookback_ok_mat = pd.DataFrame(
        lookback_ok_np, index=master_ix, columns=valid_tickers, copy=False
    )
    close_mat = pd.DataFrame(
        close_np, index=master_ix, columns=valid_tickers, copy=False
    )
    volume_mat = pd.DataFrame(
        volume_np, index=master_ix, columns=valid_tickers, copy=False
    )
    dollar_vol_np = close_np * volume_np
    dollar_vol_mat = pd.DataFrame(
        dollar_vol_np, index=master_ix, columns=valid_tickers, copy=False
    )
    if dynamic_universe_size is not None:
        assert dynamic_score_np is not None
        dynamic_score_mat = pd.DataFrame(
            dynamic_score_np, index=master_ix, columns=valid_tickers, copy=False
        )
        signal_mat &= _dynamic_eligibility_mask(
            dynamic_score_mat,
            size=dynamic_universe_size,
            rebalance=dynamic_universe_rebalance,
        )
    rank_score_mat = (
        pd.DataFrame(rank_score_np, index=master_ix, columns=valid_tickers, copy=False)
        if any_score
        else None
    )
    # Sector neutralization is a no-op when ranking is inactive (no rank_score)
    # or the flag is off. When active, z-score within each sector per day.
    if sector_neutral and rank_score_mat is not None:
        sector_map = sector_by_tv or {}
        unknown = [
            tv
            for tv in rank_score_mat.columns
            if sector_map.get(tv, "UNKNOWN") == "UNKNOWN"
        ]
        if unknown and warnings is not None:
            warnings.append(
                _preview_warning(
                    unknown,
                    "sector neutralization: {count} ticker(s) mapped to UNKNOWN "
                    "sector: {preview}",
                )
            )
        rank_score_mat = _sector_neutralize_scores(rank_score_mat, sector_map)
    final_rank_score_np = (
        rank_score_mat.to_numpy()
        if sector_neutral and rank_score_mat is not None
        else (rank_score_np if rank_score_mat is not None else None)
    )
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
        lookback_ok_np=lookback_ok_np,
        filter_np=filter_mat.to_numpy() if filter_mat is not None else None,
        dollar_vol_np=dollar_vol_np,
        close_np=close_np,
        volume_np=volume_np,
        bar_idx_np=bar_idx_np,
        rank_score_np=final_rank_score_np,
        min_score=min_score,
    )


def _dynamic_eligibility_mask(
    scores: pd.DataFrame, *, size: int, rebalance: str
) -> pd.DataFrame:
    """Select the top lagged scores on each rebalance date and hold membership."""
    mask = pd.DataFrame(False, index=scores.index, columns=scores.columns, dtype=bool)
    if scores.empty:
        return mask
    if rebalance == "daily":
        rebalance_rows = np.arange(len(scores), dtype=int)
    else:
        freq = {"weekly": "W-FRI", "monthly": "M", "quarterly": "Q"}[rebalance]
        periods = pd.DatetimeIndex(scores.index).to_period(freq)
        rebalance_rows = np.flatnonzero(
            np.r_[True, periods[1:].to_numpy() != periods[:-1].to_numpy()]
        )
    selected = np.zeros(len(scores.columns), dtype=bool)
    rebalance_set = set(int(row) for row in rebalance_rows)
    raw = scores.to_numpy(dtype=float)
    for row in range(len(scores)):
        if row in rebalance_set:
            finite = np.flatnonzero(np.isfinite(raw[row]))
            selected = np.zeros(len(scores.columns), dtype=bool)
            if finite.size:
                order = np.argsort(-raw[row, finite], kind="stable")
                selected[finite[order[:size]]] = True
        mask.iloc[row] = selected
    return mask


#: ``setup_score`` runs 0-100, like every other percentile the project reports.
_PERCENTILE_SCALE = 100.0


def _setup_scores(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile of ``values``, over ``eligible`` only, 0-100.

    The field a name is scored against is the eligible set *before* ``exclude``
    and before ``limit``: a score says where a name stands among the names that
    cleared the gates, not among the ones the caller had room for. Ties share
    the average rank, so the score never depends on column order.

    Names outside ``eligible`` score 0. They are dropped anyway, and 0 is what
    :func:`screener.scoring.components.percentile` gives a missing value.
    """
    scores = np.zeros(values.shape, dtype=float)
    cols = np.nonzero(eligible)[0]
    if cols.size == 0:
        return scores
    ranked = pd.Series(values[cols]).rank(pct=True).to_numpy(dtype=float)
    scores[cols] = ranked * _PERCENTILE_SCALE
    return scores


def _apply_min_score(
    eligible: np.ndarray, setup_score: np.ndarray, min_score: float | None
) -> np.ndarray:
    """``eligible`` with the names below ``min_score`` removed."""
    if min_score is None:
        return eligible
    return eligible & (setup_score >= min_score)


def _drop_excluded(
    eligible: np.ndarray, col_by_ticker: dict[str, int], exclude: set[str]
) -> None:
    """Clear ``eligible`` in place for every name in ``exclude``.

    Applied after the percentile so a name the caller already holds still
    counts towards the field the others are ranked against.
    """
    for ticker in exclude:
        col = col_by_ticker.get(ticker)
        if col is not None:
            eligible[col] = False


def _candidate_rows_for_day(
    day: pd.Timestamp,
    matrices: _RollingCandidateMatrices,
    *,
    exclude: set[str],
    limit: int | None = None,
) -> tuple[list[dict], list[str]]:
    """Evaluate entry signals for the full universe on one trading day.

    ``limit`` caps how many ranked candidates are materialised into ``list[dict]``.
    Ranking still considers the whole eligible set, so ranks 1..limit match the
    uncapped path; only the dict-building tail is skipped. Pass ``None`` (the
    default) to materialise every eligible name — used by tests that assert on
    full-day rankings.
    """
    warnings: list[str] = []
    row = matrices.row_by_day[day]
    eligible = matrices.signal_np[row] & matrices.lookback_ok_np[row]
    if matrices.filter_np is not None:
        eligible = eligible & matrices.filter_np[row]
    dollar_vol = matrices.dollar_vol_np[row]
    eligible = eligible & ~np.isnan(dollar_vol)
    # Factor portfolios rank by cross-sectional score; everything else keeps the
    # legacy signal-day dollar-volume ordering. The dollar-volume value is still
    # surfaced as ``as_of_dollar_vol`` either way for reporting/liquidity audit.
    if matrices.rank_score_np is not None:
        rank_score = matrices.rank_score_np[row]
        eligible = eligible & ~np.isnan(rank_score)
        setup_score = _setup_scores(rank_score, eligible)
        eligible = _apply_min_score(eligible, setup_score, matrices.min_score)
        _drop_excluded(eligible, matrices.col_by_ticker, exclude)
        eligible_cols = np.nonzero(eligible)[0]
        if eligible_cols.size == 0:
            return [], warnings
        # Rank by cross-sectional factor score (descending), breaking ties by
        # signal-day dollar volume so equal-score names resolve by liquidity -
        # a principled deterministic fallback - rather than by arbitrary
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
        setup_score = _setup_scores(dollar_vol, eligible)
        eligible = _apply_min_score(eligible, setup_score, matrices.min_score)
        _drop_excluded(eligible, matrices.col_by_ticker, exclude)
        eligible_cols = np.nonzero(eligible)[0]
        if eligible_cols.size == 0:
            return [], warnings
        # Match ``sort_values(ascending=False, kind="mergesort")`` exactly: pandas
        # reverses the stable ascending order, so ties land in reversed column
        # order (see pandas ``nargsort``).
        order = dollar_vol[eligible_cols].argsort(kind="mergesort")[::-1]
    if limit is not None and limit >= 0:
        order = order[:limit]
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
                "setup_score": float(setup_score[col]),
            }
        )
    return rows, warnings
