"""Wilder's RMA, matching Pine ``ta.rma``."""

from __future__ import annotations

import numpy as np

from screener.indicators.registry import indicator


@indicator("rma")
def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's running average: SMA seed, then ``alpha = 1/n``.

    The seed is the mean of the first ``n`` *observations* and lands at the
    position of the ``n``-th one, which is not index ``n - 1`` when the input
    opens with undefined values. ``ta.change`` is undefined on bar 0, so
    ``rsi`` feeds exactly such an input, and counting positions instead of
    observations would seed off ``n - 1`` real values one bar early.

    Past the seed the recursion propagates NaN forward, as ``ema`` does: a gap
    in the middle of the input is missing information, not a zero.

    ``x`` may be one series or a ``(bars, symbols)`` panel of them, and the two
    forms agree to the last bit -- ``tests/test_bar_column_panel.py`` pins that.
    They are written out separately because the loop body is the whole cost: a
    row of numpy work is worth it for a panel and several times too expensive
    for a lone series.
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim > 1:
        return _rma_panel(values, n)
    out = np.full(len(values), np.nan)
    observed = np.flatnonzero(~np.isnan(values))
    if observed.size < n:
        return out
    seed_at = observed[n - 1]
    out[seed_at] = np.mean(values[observed[:n]])
    alpha = 1.0 / n
    for i in range(seed_at + 1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _rma_panel(values: np.ndarray, n: int) -> np.ndarray:
    """:func:`rma` over a ``(bars, symbols)`` panel, one Python step per bar.

    Columns are independent: each seeds at its own ``n``-th observation, and
    NaN never crosses between them.
    """
    out = np.full(values.shape, np.nan, dtype=np.float64)
    observed = ~np.isnan(values)
    ranks = np.cumsum(observed, axis=0)
    seed_rows = observed & (ranks == n)
    if not seed_rows.any():
        return out

    # Gather each column's first ``n`` observations into its own contiguous
    # row, so the mean is summed in the same order -- and so to the last bit
    # the same value -- as a mean over that one column taken on its own.
    gathered_rows, gathered_columns = np.nonzero(observed & (ranks <= n))
    gathered = np.full((values.shape[1], n), np.nan, dtype=np.float64)
    gathered[gathered_columns, ranks[gathered_rows, gathered_columns] - 1] = values[
        gathered_rows, gathered_columns
    ]
    out[seed_rows] = gathered.mean(axis=1)[np.nonzero(seed_rows)[1]]

    alpha = 1.0 / n
    first_seed = int(np.argmax(seed_rows.any(axis=1)))
    for i in range(first_seed + 1, values.shape[0]):
        # ``where=isnan`` leaves a column's own seed row untouched, and leaves
        # a column that has not seeded yet at NaN, since its previous value is
        # NaN.
        np.copyto(
            out[i],
            alpha * values[i] + (1.0 - alpha) * out[i - 1],
            where=np.isnan(out[i]),
        )
    return out
