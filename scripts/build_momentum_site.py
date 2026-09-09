#!/usr/bin/env python
"""Assemble the momentum study's browsable site from the run JSONs.

    uv run python scripts/run_momentum_study.py     # produce the runs
    uv run python scripts/build_momentum_site.py    # assemble the site
    uv run python scripts/serve_momentum_site.py    # serve it

The page itself is a static asset (``screener/assets/momentum_site.html``) that
fetches its data at runtime, so this script only has to write the data and copy
the page. Keeping the markup out of Python means the page can be edited as HTML
rather than as a string literal.

Two derived quantities are added here rather than in the engine, because they
exist only to be displayed:

* ``benchmark_cagr`` - the engine reports the benchmark's total return over the
  window; annualizing it makes it comparable with the strategy's CAGR in the
  same table.
* ``order`` - the display order of a strategy inside its family, taken from the
  study's own strategy list so the site and the runner cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.run_momentum_study import (
    DEFAULT_OUT_DIR,
    FAMILY_TITLES,
    LEVERS,
    MARKETS,
    PERIODS,
    REGIME_FILTERS,
    STRATEGIES,
)

PAGE_ASSET = Path("screener/assets/momentum_site.html")
MARKET_SHORT = {"india": "India", "us": "US"}
# Write-ups served alongside the data so the page can show the study's
# conclusions next to the numbers they came from.
DOCS = (
    ("meta", "Meta-analysis", Path("docs/momentum-study-meta-analysis.md")),
    ("findings", "Verified findings", Path("docs/momentum-study-findings.md")),
)


def _benchmark_cagr(total_return: float | None, years: int) -> float | None:
    """Annualize a benchmark's total return over the run's window."""
    if total_return is None or years <= 0:
        return None
    growth = 1.0 + float(total_return)
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years)) - 1.0


def is_baseline(payload: dict[str, Any]) -> bool:
    """True when a run uses the strategy's own settings with no sweep applied.

    The overview shows baselines only; everything else belongs in the variant
    table on a strategy's own page, where it is compared against the baseline it
    was meant to improve on.
    """
    return (
        not payload.get("regime")
        and not payload.get("lever")
        and payload["hold"] == payload.get("default_hold", payload["hold"])
    )


def drawdown_profile(curve: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the depth, dates and duration of the worst peak-to-trough decline.

    The momentum literature reports peak loss far more consistently than time
    under water, and mixes conventions when it does report depth: Antonacci
    measures month-end values, Faber peak-to-trough on monthly data, and daily
    strategies use daily observations, which are not comparable with each other.
    A -20% month-end drawdown hides every intra-month low, so it reads shallower
    than the same path measured daily.

    This returns both conventions plus the two durations the papers omit: bars
    from peak to trough, and bars from trough back to the previous high.
    ``recovered`` is False when the run ended still under water, in which case
    ``recovery_days`` counts the days elapsed so far rather than a completed
    recovery.
    """
    empty: dict[str, Any] = {
        "max_drawdown_daily": None,
        "max_drawdown_monthly": None,
        "peak_date": None,
        "trough_date": None,
        "recovery_date": None,
        "decline_days": None,
        "recovery_days": None,
        "recovered": None,
    }
    if not curve:
        return empty

    dates = [pd.Timestamp(point["date"]) for point in curve]
    values = [float(point["value"]) for point in curve]

    peak_value = values[0]
    peak_index = 0
    worst = 0.0
    worst_peak_index = 0
    worst_trough_index = 0
    for i, value in enumerate(values):
        if value > peak_value:
            peak_value = value
            peak_index = i
        drawdown = value / peak_value - 1.0 if peak_value > 0 else 0.0
        if drawdown < worst:
            worst = drawdown
            worst_peak_index = peak_index
            worst_trough_index = i

    # Month-end observations only, the convention Antonacci and Faber report.
    monthly = pd.Series(values, index=pd.DatetimeIndex(dates)).resample("ME").last()
    monthly_dd = (
        float((monthly / monthly.cummax() - 1.0).min()) if len(monthly) else 0.0
    )

    trough_value = values[worst_trough_index]
    peak_at_trough = values[worst_peak_index]
    recovery_index: int | None = None
    for i in range(worst_trough_index + 1, len(values)):
        if values[i] >= peak_at_trough:
            recovery_index = i
            break

    recovered = recovery_index is not None
    end_index = recovery_index if recovered else len(values) - 1
    return {
        "max_drawdown_daily": worst,
        "max_drawdown_monthly": monthly_dd,
        "peak_date": dates[worst_peak_index].date().isoformat(),
        "trough_date": dates[worst_trough_index].date().isoformat(),
        "recovery_date": dates[end_index].date().isoformat() if recovered else None,
        "decline_days": (dates[worst_trough_index] - dates[worst_peak_index]).days,
        "recovery_days": (dates[end_index] - dates[worst_trough_index]).days,
        "recovered": recovered,
        "trough_value": trough_value,
    }


def build(out_dir: Path) -> int:
    runs_dir = out_dir / "runs"
    site_dir = out_dir / "site"
    data_dir = site_dir / "data"
    run_data_dir = data_dir / "runs"
    run_data_dir.mkdir(parents=True, exist_ok=True)

    order = {strategy.name: i for i, strategy in enumerate(STRATEGIES)}
    summaries: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        metrics["benchmark_cagr"] = _benchmark_cagr(
            metrics.get("benchmark_return"), int(payload["years"])
        )
        drawdown = drawdown_profile(payload.get("equity_curve", []))
        payload["metrics"] = metrics
        payload["drawdown"] = drawdown
        # The page looks a run up in the index by key, so the payload carries it
        # rather than re-deriving the filename from its own fields.
        payload["key"] = path.stem
        (run_data_dir / path.name).write_text(json.dumps(payload), encoding="utf-8")
        summaries.append(
            {
                "key": path.stem,
                "strategy": payload["strategy"],
                "label": payload["label"],
                "family": payload["family"],
                "paper": payload["paper"],
                "note": payload["note"],
                "market": payload["market"],
                "market_label": payload["market_label"],
                "benchmark": payload["benchmark"],
                "years": payload["years"],
                "start": payload["start"],
                "end": payload["end"],
                "top": payload["top"],
                "hold": payload["hold"],
                "cost_model": payload["cost_model"],
                "slippage_bps": payload["slippage_bps"],
                # Runs written before the sweep dimensions existed carry none of
                # these, and read as the baseline they in fact were.
                "regime": payload.get("regime", ""),
                "regime_label": payload.get("regime_label", "No overlay"),
                "lever": payload.get("lever", ""),
                "lever_label": payload.get("lever_label", "Baseline"),
                "default_hold": payload.get("default_hold", payload["hold"]),
                "baseline": is_baseline(payload),
                "order": order.get(payload["strategy"], 99),
                "metrics": metrics,
                "drawdown": drawdown,
            }
        )

    if not summaries:
        print(f"no runs found in {runs_dir}")
        return 1

    present_markets = {s["market"] for s in summaries}
    present_years = sorted({int(s["years"]) for s in summaries})
    present_regimes = {s["regime"] for s in summaries}
    present_levers = {s["lever"] for s in summaries}
    index = {
        "generated": date.today().isoformat(),
        "markets": [
            {"id": name, "short": MARKET_SHORT.get(name, name), "label": spec.label}
            for name, spec in MARKETS.items()
            if name in present_markets
        ],
        "periods": [y for y in PERIODS if y in present_years],
        "regimes": [
            {"id": regime.key, "label": regime.label}
            for regime in REGIME_FILTERS
            if regime.key in present_regimes
        ],
        "levers": [
            {"id": lever.key, "label": lever.label, "why": lever.why}
            for lever in LEVERS
            if lever.key in present_levers
        ],
        "families": FAMILY_TITLES,
        "docs": [
            {"id": doc_id, "title": title}
            for doc_id, title, source in DOCS
            if source.exists()
        ],
        "runs": summaries,
    }
    (data_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")

    docs_dir = data_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, _title, source in DOCS:
        if source.exists():
            shutil.copyfile(source, docs_dir / f"{doc_id}.md")
        else:
            # A missing write-up must not fail the build, but it must be
            # visible: the page lists only what the index advertises.
            print(f"warning: {source} not found, its reader button is omitted")

    shutil.copyfile(PAGE_ASSET, site_dir / "index.html")
    print(f"wrote {site_dir}/index.html with {len(summaries)} runs")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    return build(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
