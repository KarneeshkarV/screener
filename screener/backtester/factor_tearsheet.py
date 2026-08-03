"""IC / quantile factor tearsheet: pure math + CLI.

The pure metric routines are intentionally kept isolated at the top of this
module so they can move to ``factor_metrics.py`` when the factor-reporting API
next changes; splitting them now would add import churn without changing behavior.

Score at day ``t`` predicts the forward return from ``t`` to ``t+h``
(close[t+h] / close[t] - 1). The score itself is causal (no lookahead);
only the *evaluation* labels use future closes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.data import build_price_fetcher, tv_to_yf
from screener.markets import get_market, get_price_fetcher, market_option
from screener.universes import available_universes, load_current_universe

# ── pure computation ────────────────────────────────────────────────


def forward_returns(close_mat: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return dates × tickers matrix of h-bar forward returns.

    ``fwd[t] = close[t+h] / close[t] - 1``. The last ``horizon`` rows are NaN.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    close = close_mat.astype(float)
    return close.shift(-horizon) / close - 1.0


def daily_spearman_ic(scores: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank IC per day (score vs forward return)."""
    aligned_scores, aligned_fwd = scores.align(fwd_returns, join="inner")
    if aligned_scores.empty:
        return pd.Series(dtype=float, name="ic")

    def _day_ic(row_score: pd.Series) -> float:
        day = row_score.name
        s = row_score
        r = cast("pd.Series[Any]", aligned_fwd.loc[cast(Any, day)])
        mask = s.notna() & r.notna()
        if int(mask.sum()) < 3:
            return float("nan")
        # Spearman = Pearson of ranks.
        return float(s[mask].rank().corr(r[mask].rank()))

    ic = aligned_scores.apply(_day_ic, axis=1)
    ic.name = "ic"
    return ic


@dataclass(frozen=True)
class ICSummary:
    horizon: int
    ic_mean: float
    ic_std: float
    ic_ir: float
    t_stat: float
    pct_positive: float
    n_days: int


def summarize_ic(ic: pd.Series, *, horizon: int) -> ICSummary:
    """Summary stats over a daily IC series (NaNs dropped)."""
    clean = ic.dropna().astype(float)
    n = len(clean)
    if n == 0:
        return ICSummary(
            horizon=horizon,
            ic_mean=float("nan"),
            ic_std=float("nan"),
            ic_ir=float("nan"),
            t_stat=float("nan"),
            pct_positive=float("nan"),
            n_days=0,
        )
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n > 1 else float("nan")
    ir = mean / std if std and std > 0 and math.isfinite(std) else float("nan")
    t_stat = (
        mean / (std / math.sqrt(n))
        if std and std > 0 and math.isfinite(std)
        else float("nan")
    )
    pct_pos = float((clean > 0).mean())
    return ICSummary(
        horizon=horizon,
        ic_mean=mean,
        ic_std=std,
        ic_ir=ir,
        t_stat=t_stat,
        pct_positive=pct_pos,
        n_days=n,
    )


@dataclass(frozen=True)
class QuantileResult:
    horizon: int
    n_quantiles: int
    mean_return_by_quantile: dict[int, float]
    top_minus_bottom: float
    top_quantile_turnover: float


def _quantile_labels(row: pd.Series, n_quantiles: int) -> pd.Series:
    """Assign 1..N quantile labels by score (1 = lowest, N = highest).

    Uses rank-based binning so equal-score ties don't blow up ``qcut`` on
    small cross-sections. Names with NaN scores stay NaN.
    """
    valid = row.dropna()
    if len(valid) < n_quantiles:
        return pd.Series(np.nan, index=row.index, dtype=float)
    # Rank then map into n equal-count bins.
    ranks = valid.rank(method="first")
    # pd.qcut on ranks is stable when values are unique after rank(method=first).
    try:
        bins = pd.qcut(ranks, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=row.index, dtype=float)
    if bins.nunique(dropna=True) < n_quantiles:
        return pd.Series(np.nan, index=row.index, dtype=float)
    out = pd.Series(np.nan, index=row.index, dtype=float)
    out.loc[bins.index] = bins.astype(float) + 1.0  # 1..N
    return out


def quantile_mean_returns(
    scores: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    *,
    n_quantiles: int,
) -> tuple[dict[int, float], float]:
    """Mean forward return per daily score quantile + top-minus-bottom spread."""
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    aligned_scores, aligned_fwd = scores.align(fwd_returns, join="inner")
    labels = aligned_scores.apply(
        lambda row: _quantile_labels(row, n_quantiles), axis=1
    )
    # Per-quantile pooled mean of all (day, name) cells.
    means: dict[int, float] = {}
    for q in range(1, n_quantiles + 1):
        mask = labels == q
        vals = aligned_fwd.where(mask)
        flat = cast(pd.Series, vals.stack(future_stack=True)).dropna()
        means[q] = float(flat.mean()) if not flat.empty else float("nan")
    top = means.get(n_quantiles, float("nan"))
    bottom = means.get(1, float("nan"))
    spread = (
        top - bottom if math.isfinite(top) and math.isfinite(bottom) else float("nan")
    )
    return means, spread


def top_quantile_turnover(scores: pd.DataFrame, *, n_quantiles: int) -> float:
    """Mean one-day membership turnover of the top score quantile.

    Turnover on day t is ``1 - |Q_t ∩ Q_{t-1}| / |Q_t|`` when both sets are
    non-empty; days that cannot form full quantiles are skipped.
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    labels = scores.apply(lambda row: _quantile_labels(row, n_quantiles), axis=1)
    turnovers: list[float] = []
    prev: set[str] | None = None
    for day in labels.index:
        row = labels.loc[day]
        members = set(row.index[row == n_quantiles].astype(str))
        if not members:
            prev = None
            continue
        if prev is not None and prev:
            overlap = len(members & prev)
            turnovers.append(1.0 - overlap / len(members))
        prev = members
    if not turnovers:
        return float("nan")
    return float(np.mean(turnovers))


def analyze_horizon(
    scores: pd.DataFrame,
    close_mat: pd.DataFrame,
    *,
    horizon: int,
    n_quantiles: int,
) -> tuple[ICSummary, QuantileResult, pd.Series]:
    """IC summary + quantile stats + raw IC series for one horizon."""
    fwd = forward_returns(close_mat, horizon)
    ic = daily_spearman_ic(scores, fwd)
    summary = summarize_ic(ic, horizon=horizon)
    means, spread = quantile_mean_returns(scores, fwd, n_quantiles=n_quantiles)
    turnover = top_quantile_turnover(scores, n_quantiles=n_quantiles)
    qres = QuantileResult(
        horizon=horizon,
        n_quantiles=n_quantiles,
        mean_return_by_quantile=means,
        top_minus_bottom=spread,
        top_quantile_turnover=turnover,
    )
    return summary, qres, ic


def build_score_and_close_matrices(
    bars_by_tv: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack prepared bars into dates × tickers score and close matrices."""
    score_cols: dict[str, pd.Series] = {}
    close_cols: dict[str, pd.Series] = {}
    for tv, bars in bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        if "close" not in bars.columns:
            continue
        close_cols[tv] = bars["close"].astype(float)
        if "rank_score" in bars.columns:
            score_cols[tv] = bars["rank_score"].astype(float)
    if not close_cols:
        return pd.DataFrame(), pd.DataFrame()
    close_mat = pd.DataFrame(close_cols)
    score_mat = (
        pd.DataFrame(score_cols).reindex(index=close_mat.index)
        if score_cols
        else pd.DataFrame(index=close_mat.index)
    )
    return score_mat, close_mat


# ── data loading (CLI path) ──────────────────────────────────────────


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise click.UsageError(f"--{name} must list at least one integer")
    out: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise click.UsageError(
                f"--{name} entries must be integers, got {part!r}"
            ) from exc
        if value < 1:
            raise click.UsageError(f"--{name} entries must be >= 1, got {value}")
        out.append(value)
    return out


def _resolve_tickers(
    market: str,
    *,
    tickers: str | None,
    universe: str | None,
    universe_file: str | None,
    end_date: date,
    no_universe_cache: bool,
) -> tuple[tuple[str, ...], str | None]:
    if tickers:
        resolved = tuple(t.strip() for t in tickers.split(",") if t.strip())
        if not resolved:
            raise click.UsageError("--tickers is empty")
        return resolved, None
    if universe_file:
        content = Path(universe_file).read_text()
        resolved = tuple(
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        if not resolved:
            raise click.UsageError(f"universe file is empty: {universe_file}")
        return resolved, f"file:{universe_file}"
    market_meta = get_market(market)
    resolved_universe = universe or market_meta.default_universe
    loaded = load_current_universe(
        resolved_universe,
        as_of=end_date,
        use_cache=not no_universe_cache,
    )
    note = (
        f"{loaded.name}: {len(loaded.symbols)} symbols from {loaded.source}; "
        f"cache={loaded.cached_path}"
    )
    return loaded.symbols, note


def load_factor_panels(
    *,
    market: str,
    strategy_name: str,
    tickers: Sequence[str],
    start: date,
    end: date,
    fetcher: Any,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch prices, run strategy ``prepare_bars``, return score + close matrices."""
    from screener.backtester.core import prepare_strategy_bars
    from screener.strategies.spec import discover_plugins, resolve_strategy_spec

    discover_plugins()
    spec = resolve_strategy_spec(strategy_name)
    if spec is None or getattr(spec, "entry", None) is None:
        raise ValueError(f"unknown factor strategy {strategy_name!r}")
    # Warmup: use the strategy lookback when available, else a conservative floor.
    lookback_fn = getattr(spec, "required_lookback", None)
    lookback = lookback_fn() if lookback_fn else 252
    warmup_days = max(lookback * 3 + 30, 365)
    fetch_start = start - timedelta(days=warmup_days)

    yf_by_tv = {tv: tv_to_yf(tv, market) for tv in tickers}
    yf_symbols = list(dict.fromkeys(yf_by_tv.values()))
    price_panel = fetcher.fetch(yf_symbols, fetch_start, end)
    # See rolling_simulation: dict.get's default is eager.
    bars_by_tv = {}
    for tv in tickers:
        panel_bars = price_panel.get(yf_by_tv[tv])
        bars_by_tv[tv] = pd.DataFrame() if panel_bars is None else panel_bars
    benchmark = get_market(market).benchmark
    bars_by_tv, _ = prepare_strategy_bars(
        spec,
        bars_by_tv,
        price_panel,
        list(tickers),
        fetch_start,
        end,
        fetcher,
        warnings,
        market=market,
        benchmark=benchmark,
    )
    scores, closes = build_score_and_close_matrices(bars_by_tv)
    # Restrict analysis window to the requested [start, end] (warmup kept only
    # for causal factor construction).
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if not scores.empty:
        scores = scores.loc[(scores.index >= start_ts) & (scores.index <= end_ts)]
    if not closes.empty:
        closes = closes.loc[(closes.index >= start_ts) & (closes.index <= end_ts)]
    return scores, closes


# ── output ───────────────────────────────────────────────────────────


def _finite_or_none(value: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def tearsheet_to_dict(
    *,
    strategy: str,
    market: str,
    start: date,
    end: date,
    quantiles: int,
    ic_summaries: Sequence[ICSummary],
    quantile_results: Sequence[QuantileResult],
    warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "market": market,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "quantiles": quantiles,
        "ic": [
            {
                "horizon": s.horizon,
                "ic_mean": _finite_or_none(s.ic_mean),
                "ic_std": _finite_or_none(s.ic_std),
                "ic_ir": _finite_or_none(s.ic_ir),
                "t_stat": _finite_or_none(s.t_stat),
                "pct_positive": _finite_or_none(s.pct_positive),
                "n_days": s.n_days,
            }
            for s in ic_summaries
        ],
        "quantiles_by_horizon": [
            {
                "horizon": q.horizon,
                "n_quantiles": q.n_quantiles,
                "mean_return_by_quantile": {
                    str(k): _finite_or_none(v)
                    for k, v in q.mean_return_by_quantile.items()
                },
                "top_minus_bottom": _finite_or_none(q.top_minus_bottom),
                "top_quantile_turnover": _finite_or_none(q.top_quantile_turnover),
            }
            for q in quantile_results
        ],
        "warnings": list(warnings),
    }


def print_tearsheet(
    *,
    strategy: str,
    ic_summaries: Sequence[ICSummary],
    quantile_results: Sequence[QuantileResult],
    warnings: Sequence[str],
    console: Console | None = None,
) -> None:
    console = console or Console()
    ic_table = Table(
        title=f"Factor IC — {strategy}",
        show_header=True,
        header_style="bold",
    )
    for col in [
        "Horizon",
        "IC Mean",
        "IC Std",
        "IC IR",
        "t-stat",
        "% Pos",
        "N days",
    ]:
        ic_table.add_column(col, justify="right")
    for s in ic_summaries:
        ic_table.add_row(
            str(s.horizon),
            f"{s.ic_mean:.4f}" if math.isfinite(s.ic_mean) else "n/a",
            f"{s.ic_std:.4f}" if math.isfinite(s.ic_std) else "n/a",
            f"{s.ic_ir:.3f}" if math.isfinite(s.ic_ir) else "n/a",
            f"{s.t_stat:.2f}" if math.isfinite(s.t_stat) else "n/a",
            f"{s.pct_positive * 100:.1f}%" if math.isfinite(s.pct_positive) else "n/a",
            str(s.n_days),
        )
    console.print(ic_table)

    for q in quantile_results:
        q_table = Table(
            title=f"Quantile mean fwd return — h={q.horizon}",
            show_header=True,
            header_style="bold",
        )
        q_table.add_column("Quantile", justify="right")
        q_table.add_column("Mean Return", justify="right")
        for k in sorted(q.mean_return_by_quantile):
            v = q.mean_return_by_quantile[k]
            label = f"Q{k}"
            if k == 1:
                label += " (low)"
            elif k == q.n_quantiles:
                label += " (high)"
            q_table.add_row(
                label,
                f"{v * 100:.3f}%" if math.isfinite(v) else "n/a",
            )
        spread = q.top_minus_bottom
        turnover = q.top_quantile_turnover
        q_table.add_row(
            "Top − Bottom",
            f"{spread * 100:.3f}%" if math.isfinite(spread) else "n/a",
        )
        q_table.add_row(
            "Top Q turnover",
            f"{turnover * 100:.1f}%" if math.isfinite(turnover) else "n/a",
        )
        console.print(q_table)

    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def write_tearsheet_csv(
    path: Path,
    ic_summaries: Sequence[ICSummary],
    quantile_results: Sequence[QuantileResult],
) -> None:
    rows: list[dict[str, Any]] = []
    for s in ic_summaries:
        rows.append(
            {
                "section": "ic",
                "horizon": s.horizon,
                "metric": "ic_mean",
                "value": s.ic_mean,
            }
        )
        rows.append(
            {
                "section": "ic",
                "horizon": s.horizon,
                "metric": "ic_std",
                "value": s.ic_std,
            }
        )
        rows.append(
            {
                "section": "ic",
                "horizon": s.horizon,
                "metric": "ic_ir",
                "value": s.ic_ir,
            }
        )
        rows.append(
            {
                "section": "ic",
                "horizon": s.horizon,
                "metric": "t_stat",
                "value": s.t_stat,
            }
        )
        rows.append(
            {
                "section": "ic",
                "horizon": s.horizon,
                "metric": "pct_positive",
                "value": s.pct_positive,
            }
        )
        rows.append(
            {
                "section": "ic",
                "horizon": s.horizon,
                "metric": "n_days",
                "value": s.n_days,
            }
        )
    for q in quantile_results:
        for k, v in q.mean_return_by_quantile.items():
            rows.append(
                {
                    "section": "quantile",
                    "horizon": q.horizon,
                    "metric": f"q{k}_mean_return",
                    "value": v,
                }
            )
        rows.append(
            {
                "section": "quantile",
                "horizon": q.horizon,
                "metric": "top_minus_bottom",
                "value": q.top_minus_bottom,
            }
        )
        rows.append(
            {
                "section": "quantile",
                "horizon": q.horizon,
                "metric": "top_quantile_turnover",
                "value": q.top_quantile_turnover,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


# ── CLI ──────────────────────────────────────────────────────────────


@click.command(name="factor-tearsheet")
@market_option(default="us", help="Market for universe / symbol mapping.")
@click.option(
    "--strategy",
    "strategy_name",
    required=True,
    help="Named factor strategy (must emit rank_score), or combo:name=w,...",
)
@click.option(
    "--universe",
    type=click.Choice(list(available_universes())),
    default=None,
    help="Index universe. Defaults to market default when --tickers omitted.",
)
@click.option("--tickers", default=None, help="Comma-separated ticker list.")
@click.option("--universe-file", default=None, help="Newline-separated ticker file.")
@click.option(
    "--no-universe-cache",
    is_flag=True,
    default=False,
    help="Force live constituent refresh.",
)
@click.option(
    "--start", "start_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--end", "end_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--years",
    type=int,
    default=3,
    show_default=True,
    help="Trailing calendar years when --start is omitted.",
)
@click.option(
    "--quantiles",
    type=int,
    default=5,
    show_default=True,
    help="Number of daily score quantiles.",
)
@click.option(
    "--horizons",
    default="1,5,21",
    show_default=True,
    help="Comma-separated forward-return horizons in bars.",
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write summary metrics CSV to this path.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write full tearsheet JSON to this path.",
)
@click.pass_context
def factor_tearsheet(
    ctx: click.Context,
    market: str,
    strategy_name: str,
    universe: str | None,
    tickers: str | None,
    universe_file: str | None,
    no_universe_cache: bool,
    start_arg: datetime | None,
    end_arg: datetime | None,
    years: int,
    quantiles: int,
    horizons: str,
    csv_path: Path | None,
    json_path: Path | None,
) -> None:
    """Compute factor IC and quantile tearsheet for a named strategy."""
    if quantiles < 2:
        raise click.UsageError("--quantiles must be >= 2")
    horizon_list = _parse_int_list(horizons, name="horizons")

    end_date = (
        end_arg.date() if isinstance(end_arg, datetime) else (end_arg or date.today())
    )
    start_date = (
        start_arg.date()
        if isinstance(start_arg, datetime)
        else (start_arg or (end_date - timedelta(days=365 * int(years))))
    )
    if end_date < start_date:
        raise click.UsageError("--end must be >= --start")

    ticker_tuple, universe_note = _resolve_tickers(
        market,
        tickers=tickers,
        universe=universe,
        universe_file=universe_file,
        end_date=end_date,
        no_universe_cache=no_universe_cache,
    )
    warnings: list[str] = []
    if universe_note:
        warnings.append(universe_note)

    fetcher = get_price_fetcher(
        ctx.obj if ctx is not None else None,
        builder=build_price_fetcher,
    )
    scores, closes = load_factor_panels(
        market=market,
        strategy_name=strategy_name,
        tickers=ticker_tuple,
        start=start_date,
        end=end_date,
        fetcher=fetcher,
        warnings=warnings,
    )
    if scores.empty or scores.notna().sum().sum() == 0:
        raise click.UsageError(
            f"strategy {strategy_name!r} produced no rank_score values; "
            "is it a factor strategy with prepare_bars?"
        )
    if closes.empty:
        raise click.UsageError("no close prices available for the requested universe")

    ic_summaries: list[ICSummary] = []
    quantile_results: list[QuantileResult] = []
    for h in horizon_list:
        summary, qres, _ic = analyze_horizon(
            scores, closes, horizon=h, n_quantiles=quantiles
        )
        ic_summaries.append(summary)
        quantile_results.append(qres)

    print_tearsheet(
        strategy=strategy_name,
        ic_summaries=ic_summaries,
        quantile_results=quantile_results,
        warnings=warnings,
    )
    payload = tearsheet_to_dict(
        strategy=strategy_name,
        market=market,
        start=start_date,
        end=end_date,
        quantiles=quantiles,
        ic_summaries=ic_summaries,
        quantile_results=quantile_results,
        warnings=warnings,
    )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_tearsheet_csv(csv_path, ic_summaries, quantile_results)
        click.echo(f"Wrote CSV: {csv_path}")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))
        click.echo(f"Wrote JSON: {json_path}")
