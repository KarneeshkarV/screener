#!/usr/bin/env python
"""Do any filters stack, or does the first one take all the available edge?

Only worth running once the individual pass has named its survivors. Filters
are applied in sequence: rank the base survivors by the first feature and keep
the top ``q``, then rank *those* by the second and keep the top ``q`` again.

Each leg uses the same ``q``, so a k-filter combination keeps ``q**k`` of the
base. That is the honest way to compare a stack against a single filter: a
2-filter combination at q=0.7 keeps 0.49, close to a single filter at q=0.5, so
a difference between them is about *which* names were dropped rather than how
many.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from screener.research.features import registry
from screener.research.filter_study import (
    PanelSet,
    evaluate_mask,
    selection_mask,
    setting_key,
)

DEFAULT_OUT = Path.home() / "grill-me-filters"


def _default_setting(panels: PanelSet, feature: str) -> str | None:
    key = setting_key(feature, registry[feature].params)
    return key if key in panels.features else None


def stacked_mask(
    panels: PanelSet, features: tuple[str, ...], q: float
) -> pd.DataFrame | None:
    mask = panels.base
    for name in features:
        key = _default_setting(panels, name)
        if key is None:
            return None
        mask = selection_mask(
            mask,
            panels.features[key],
            q,
            ascending=not registry[name].higher_is_stronger,
        )
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="core")
    parser.add_argument(
        "--anchor",
        default="downside_vol",
        help="The filter that survived on its own; every combination builds on it.",
    )
    parser.add_argument(
        "--partners",
        default="",
        help="Comma-separated features to try on top. Empty = best of each category.",
    )
    parser.add_argument("--q", type=float, default=0.7)
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--min-train", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=2)
    args = parser.parse_args()

    with (args.out_dir / f"panels_{args.tag}.pkl").open("rb") as handle:
        panels: PanelSet = pickle.load(handle)
    summary = pd.read_csv(args.out_dir / f"filter_summary_{args.tag}.csv")
    oos_dates = panels.base.index[args.min_train :]

    if args.partners:
        partners = tuple(p.strip() for p in args.partners.split(",") if p.strip())
    else:
        # Best-scoring feature per category, excluding the anchor's own.
        anchor_category = registry[args.anchor].category
        best = (
            summary[summary.category != anchor_category]
            .sort_values("d_median_sharpe", ascending=False)
            .groupby("category")
            .head(1)
        )
        partners = tuple(best.feature)

    def row(label: str, mask: pd.DataFrame) -> dict[str, Any]:
        stats = evaluate_mask(
            mask,
            panels,
            rebalance=args.rebalance,
            cost_bps=args.cost_bps,
            dates=oos_dates,
        )
        return {"combo": label, "depth": label.count("+") + 1, **stats}

    rows: list[dict[str, Any]] = [row("base", panels.base)]
    anchor_mask = stacked_mask(panels, (args.anchor,), args.q)
    if anchor_mask is None:
        raise SystemExit(f"anchor {args.anchor} not in panel")
    rows.append(row(args.anchor, anchor_mask))
    # A single filter at q**depth, as the honest like-for-like control: it holds
    # the same number of names a stack would, using one signal instead of two.
    solo = stacked_mask(panels, (args.anchor,), args.q**2)
    if solo is not None:
        rows.append(row(f"{args.anchor}@tight", solo))

    for depth in range(1, args.max_depth):
        for combination in itertools.combinations(partners, depth):
            features = (args.anchor, *combination)
            mask = stacked_mask(panels, features, args.q)
            if mask is None:
                continue
            rows.append(row("+".join(features), mask))

    results = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    dest = args.out_dir / f"filter_combos_{args.tag}.csv"
    results.to_csv(dest, index=False)
    (args.out_dir / f"filter_combos_{args.tag}.json").write_text(
        json.dumps(
            {
                "anchor": args.anchor,
                "partners": list(partners),
                "q": args.q,
                "tag": args.tag,
                "n_combos": len(results),
            },
            indent=2,
        )
    )
    columns = [
        "combo",
        "mean_survivors",
        "sharpe",
        "cagr",
        "max_drawdown",
        "calmar",
        "turnover",
        "fwd_3m_mean",
    ]
    print(results[columns].round(3).to_string(index=False))
    print(f"wrote {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
