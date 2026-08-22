#!/usr/bin/env python
"""Evaluate every trend filter against the default screen, one at a time.

Reads the cached panels from ``filter_panels.py`` and, for every feature at
every parameter setting and every keep-fraction, reports the full metric set
both in-sample and out-of-sample.

Three questions are kept separate on purpose:

1. **Does the filter add anything?** Every arm is compared against the base arm
   evaluated over the identical dates through the identical code path.
2. **Does it survive out of sample?** Expanding walk-forward folds; the
   headline table reports pooled OOS only. In-sample numbers are written too,
   but only so the gap between them can be read.
3. **Is it stable?** A feature is scored across its whole parameter grid. One
   setting with a standout Sharpe and a poor grid median is the signature of
   overfitting, and is ranked as such rather than celebrated.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screener.research.features import registry
from screener.research.filter_study import (
    PanelSet,
    setting_key,
    evaluate_mask,
    selection_mask,
    walk_forward_folds,
)

DEFAULT_OUT = Path.home() / "grill-me-filters"
# Keep-fractions. Every filter is "hold the top q of what the base screen
# already passed", so q is part of the stability grid, not a tuned constant.
KEEP_FRACTIONS = (0.7, 0.5, 0.3)
BASE_ARM = "__base__"
# An arm holding fewer names than this is not a screen result, it is a handful
# of idiosyncratic stocks. Its Sharpe is dominated by single-name noise, so it
# is reported but excluded from every verdict.
MIN_SURVIVORS = 20.0


def _arms(panels: PanelSet) -> list[tuple[str, str | None, float | None]]:
    """``(arm_id, feature_setting_key, q)``, base first."""
    arms: list[tuple[str, str | None, float | None]] = [(BASE_ARM, None, None)]
    for key in sorted(panels.features):
        for q in KEEP_FRACTIONS:
            arms.append((f"{key}@q{int(q * 100)}", key, q))
    return arms


def _ascending_for(setting_key: str, panels: PanelSet) -> bool:
    """Rank ascending when a low value is the strong end of the feature."""
    name = panels.settings[setting_key]["feature"]
    return not registry[name].higher_is_stronger


def evaluate_all(
    panels: PanelSet, *, rebalance: int, cost_bps: float, n_folds: int, min_train: int
) -> pd.DataFrame:
    dates = panels.base.index
    folds = walk_forward_folds(dates, n_folds=n_folds, min_train=min_train)
    if not folds:
        raise SystemExit(
            f"not enough dates ({len(dates)}) for {n_folds} folds after "
            f"{min_train} training bars"
        )
    oos_dates = dates[min_train:]
    arms = _arms(panels)
    rows: list[dict[str, Any]] = []
    started = time.time()

    for position, (arm_id, key, q) in enumerate(arms, start=1):
        scores = None if key is None else panels.features[key]
        ascending = False if key is None else _ascending_for(key, panels)
        mask = selection_mask(panels.base, scores, q or 1.0, ascending=ascending)

        common = {
            "arm": arm_id,
            "feature": "base" if key is None else panels.settings[key]["feature"],
            "setting": key or "",
            "q": q if q is not None else 1.0,
            "category": (
                "base"
                if key is None
                else registry[panels.settings[key]["feature"]].category
            ),
        }
        # Pooled out-of-sample is the headline; in-sample is kept only so the
        # in/out gap is visible.
        rows.append(
            {
                **common,
                "split": "oos",
                **evaluate_mask(
                    mask,
                    panels,
                    rebalance=rebalance,
                    cost_bps=cost_bps,
                    dates=oos_dates,
                ),
            }
        )
        rows.append(
            {
                **common,
                "split": "is",
                **evaluate_mask(
                    mask,
                    panels,
                    rebalance=rebalance,
                    cost_bps=cost_bps,
                    dates=dates[:min_train],
                ),
            }
        )
        for fold_index, (_, test) in enumerate(folds):
            rows.append(
                {
                    **common,
                    "split": f"fold{fold_index}",
                    **evaluate_mask(
                        mask, panels, rebalance=rebalance, cost_bps=cost_bps, dates=test
                    ),
                }
            )
        if position % 50 == 0:
            print(
                f"  {position}/{len(arms)} arms ({time.time() - started:.0f}s)",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: grid-wide stability, best setting, and the verdict.

    The ranking rule, stated once:

    - **strong**            median grid Sharpe beats base and the majority of
                            settings beat base. It works, and not by luck of one
                            parameter.
    - **regime_only**       flat or negative overall, but clearly positive in one
                            market state.
    - **likely_overfit**    one setting beats base handsomely while the grid
                            median does not. The classic single-parameter spike.
    - **no_improvement**    neither the median nor any setting clears base by a
                            margin worth the turnover.
    """
    oos = results[results.split == "oos"]
    base_rows = oos[oos.arm == BASE_ARM]
    if base_rows.empty:
        raise SystemExit("no base arm in results")
    base = base_rows.iloc[0]
    base_sharpe = float(base.sharpe)

    fold_cols = sorted({s for s in results.split.unique() if s.startswith("fold")})
    records: list[dict[str, Any]] = []
    for feature, group in oos[oos.arm != BASE_ARM].groupby("feature"):
        degenerate = group.mean_survivors < MIN_SURVIVORS
        usable = group[~degenerate]
        sharpes = usable.sharpe.dropna()
        if sharpes.empty:
            # Every setting collapsed below the survivor floor: report the
            # feature as unusable rather than dropping it silently.
            records.append(
                {
                    "feature": feature,
                    "category": group.category.iloc[0],
                    "n_settings": 0,
                    "n_degenerate": int(degenerate.sum()),
                    "verdict_override": "degenerate",
                    "median_sharpe": np.nan,
                    "d_median_sharpe": np.nan,
                    "d_best_sharpe": np.nan,
                    "share_beating_base": np.nan,
                    "folds_beating_base": np.nan,
                }
            )
            continue
        group = usable
        best = group.loc[group.sharpe.idxmax()]
        folds = results[(results.feature == feature) & results.split.isin(fold_cols)]
        fold_beat = (
            folds.groupby("split").sharpe.median()
            > results[(results.arm == BASE_ARM) & results.split.isin(fold_cols)]
            .set_index("split")
            .sharpe
        )
        record = {
            "feature": feature,
            "category": group.category.iloc[0],
            "n_settings": int(len(group)),
            "n_degenerate": int(degenerate.sum()),
            "verdict_override": "",
            "median_sharpe": float(sharpes.median()),
            "best_sharpe": float(sharpes.max()),
            "worst_sharpe": float(sharpes.min()),
            "sharpe_spread": float(sharpes.max() - sharpes.min()),
            "share_beating_base": float((sharpes > base_sharpe).mean()),
            "folds_beating_base": float(fold_beat.mean()) if len(fold_beat) else np.nan,
            "best_setting": str(best.setting),
            "best_q": float(best.q),
            "d_median_sharpe": float(sharpes.median() - base_sharpe),
            "d_best_sharpe": float(sharpes.max() - base_sharpe),
            "median_survivors": float(group.mean_survivors.median()),
            "median_turnover": float(group.turnover.median()),
            "median_fwd_3m": float(group.fwd_3m_mean.median()),
            "median_calmar": float(group.calmar.median(skipna=True)),
        }
        for state in ("bull", "bear", "sideways", "high_vol", "low_vol"):
            column = f"sharpe_{state}"
            if column in group and group[column].notna().any():
                record[f"d_{column}"] = float(
                    group[column].median() - float(base.get(column, np.nan))
                )
        records.append(record)

    summary = pd.DataFrame(records)
    if summary.empty:
        return summary

    regime_cols = [c for c in summary.columns if c.startswith("d_sharpe_")]

    def verdict(row: pd.Series) -> str:
        if row.get("verdict_override"):
            return str(row["verdict_override"])
        median_edge = row.d_median_sharpe
        best_edge = row.d_best_sharpe
        majority = row.share_beating_base >= 0.6
        consistent = (row.folds_beating_base or 0.0) >= 0.6
        if median_edge > 0.10 and majority and consistent:
            return "strong"
        if best_edge > 0.25 and median_edge <= 0.05:
            return "likely_overfit"
        if (
            regime_cols
            and max(row.get(c, np.nan) or np.nan for c in regime_cols) > 0.25
        ):
            return "regime_only"
        if median_edge > 0.03 and majority:
            return "useful"
        return "no_improvement"

    summary["verdict"] = summary.apply(verdict, axis=1)
    summary["base_sharpe"] = base_sharpe
    return summary.sort_values("d_median_sharpe", ascending=False)


def redundancy(panels: PanelSet, out_dir: Path) -> pd.DataFrame:
    """Cross-sectional correlation between features at their default settings.

    A feature that adds nothing beyond one already kept is redundant however
    well it scores on its own, so the ranking needs this alongside the metrics.
    """
    defaults: dict[str, pd.Series] = {}
    for key, meta in panels.settings.items():
        name = meta["feature"]
        spec = registry[name]
        if key != setting_key(name, spec.params):
            continue
        frame = panels.features[key].where(panels.base)
        defaults[name] = frame.stack(future_stack=True)
    if not defaults:
        return pd.DataFrame()
    matrix = pd.DataFrame(defaults).corr(method="spearman")
    matrix.to_csv(out_dir / "feature_correlation.csv")
    return matrix


def _flag_redundant(summary: pd.DataFrame, correlation: pd.DataFrame) -> pd.DataFrame:
    """Demote a feature to `redundant` when a better one already says the same.

    "The same" is |Spearman| >= 0.9 against a feature that scored at least as
    well. Correlation alone is not a verdict: two features can agree closely and
    still both be worth keeping if neither dominates, so only the weaker of a
    correlated pair is demoted.
    """
    summary = summary.copy()
    if correlation.empty or summary.empty:
        summary["redundant_with"] = ""
        return summary
    ranked = summary.sort_values("d_median_sharpe", ascending=False)
    kept: list[str] = []
    partner: dict[str, str] = {}
    for name in ranked.feature:
        if name not in correlation.columns:
            kept.append(name)
            continue
        clash = next(
            (
                other
                for other in kept
                if other in correlation.columns
                and abs(correlation.at[name, other]) >= 0.9
            ),
            None,
        )
        if clash is None:
            kept.append(name)
        else:
            partner[name] = clash
    summary["redundant_with"] = summary.feature.map(partner).fillna("")
    demote = summary.redundant_with.ne("") & summary.verdict.isin({"strong", "useful"})
    summary.loc[demote, "verdict"] = "redundant"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="core")
    parser.add_argument(
        "--rebalance", type=int, default=21, help="Bars between refreshes."
    )
    parser.add_argument(
        "--cost-bps", type=float, default=20.0, help="One-way cost per rebalance."
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--min-train",
        type=int,
        default=250,
        help="Bars reserved for the first training window (excluded from OOS).",
    )
    args = parser.parse_args()

    path = args.out_dir / f"panels_{args.tag}.pkl"
    with path.open("rb") as handle:
        panels: PanelSet = pickle.load(handle)
    print(
        f"loaded {path.name}: {panels.meta['n_tickers']} tickers, "
        f"{panels.meta['n_dates']} dates, {len(panels.features)} settings",
        flush=True,
    )

    results = evaluate_all(
        panels,
        rebalance=args.rebalance,
        cost_bps=args.cost_bps,
        n_folds=args.folds,
        min_train=args.min_train,
    )
    results.to_csv(args.out_dir / f"filter_results_{args.tag}.csv", index=False)
    summary = summarize(results)
    summary.to_csv(args.out_dir / f"filter_summary_{args.tag}.csv", index=False)
    (args.out_dir / f"filter_meta_{args.tag}.json").write_text(
        json.dumps(
            {
                **panels.meta,
                "rebalance": args.rebalance,
                "cost_bps": args.cost_bps,
                "folds": args.folds,
                "min_train": args.min_train,
                "keep_fractions": list(KEEP_FRACTIONS),
                "n_arms": int(results.arm.nunique()),
            },
            indent=2,
            default=str,
        )
    )
    correlation = redundancy(panels, args.out_dir)
    if not summary.empty:
        summary = _flag_redundant(summary, correlation)
        summary.to_csv(args.out_dir / f"filter_summary_{args.tag}.csv", index=False)
        print(summary.verdict.value_counts().to_string())
    print(f"wrote filter_results_{args.tag}.csv and filter_summary_{args.tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
