#!/usr/bin/env python
"""Build CSV + HTML comparison from the full-factorial temp dir."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path.home() / "tmp" / "screener-india-pit-compare-2026-08-17"


def load_rows(out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    metrics_dir = out_dir / "metrics"
    if metrics_dir.exists():
        for path in sorted(metrics_dir.glob("*.jsonl")):
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        for path in (out_dir / "runs").glob("*/*/*/*.json"):
            payload = json.loads(path.read_text())
            metrics = payload.get("metrics") or {}
            rows.append(
                {
                    "strategy": payload.get("strategy"),
                    "family": payload.get("family"),
                    "universe": payload.get("universe"),
                    "window": payload.get("window"),
                    "hold": payload.get("hold"),
                    "base_hold": payload.get("base_hold"),
                    "stop_loss": payload.get("stop_loss"),
                    "trailing_stop": payload.get("trailing_stop"),
                    "regime": payload.get("regime"),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "n_trades": payload.get("n_trades"),
                    "cagr": metrics.get("cagr"),
                    "sharpe": metrics.get("sharpe"),
                    "sortino": metrics.get("sortino"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "hit_rate": metrics.get("hit_rate"),
                    "total_return": metrics.get("total_return"),
                    "exposure": metrics.get("exposure"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "error": payload.get("error") or "",
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "error" not in frame.columns:
        frame["error"] = ""
    return frame.drop_duplicates(
        subset=["strategy", "universe", "window", "hold", "stop_loss", "trailing_stop", "regime"],
        keep="last",
    )


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_base(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["hold"] == frame["base_hold"])
        & frame["stop_loss"].isna()
        & frame["trailing_stop"].isna()
        & (frame["regime"] == "off")
    )


def cfg_label(row: pd.Series) -> str:
    parts = [f"h{int(row['hold'])}"]
    if pd.notna(row.get("stop_loss")):
        parts.append(f"sl{int(round(float(row['stop_loss']) * 100))}")
    if pd.notna(row.get("trailing_stop")):
        parts.append(f"tr{int(round(float(row['trailing_stop']) * 100))}")
    parts.append(f"rg{row.get('regime')}")
    return " ".join(parts)


def best_rows(ok: pd.DataFrame) -> pd.DataFrame:
    if ok.empty:
        return ok
    work = ok.copy()
    work["sharpe_n"] = _num(work["sharpe"]).fillna(-99)
    work["cagr_n"] = _num(work["cagr"]).fillna(-99)
    work["dd_n"] = _num(work["max_drawdown"]).fillna(-1)
    work = work.sort_values(["sharpe_n", "cagr_n", "dd_n"], ascending=[False, False, False])
    return work.groupby(["strategy", "universe", "window"], as_index=False).head(1)


def write_summary(frame: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if frame.empty or "error" not in frame.columns:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "summary_by_strategy.csv", index=False)
        return empty
    ok = frame[frame["error"].fillna("") == ""].copy()
    if ok.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "summary_by_strategy.csv", index=False)
        return empty
    ok["label"] = ok.apply(cfg_label, axis=1)
    base = ok[is_base(ok)]
    best = best_rows(ok)
    rows = []
    for (strategy, universe), _ in ok.groupby(["strategy", "universe"]):
        rec: dict = {"strategy": strategy, "universe": universe}
        family = ok.loc[(ok.strategy == strategy) & (ok.universe == universe), "family"]
        rec["family"] = family.iloc[0] if len(family) else ""
        sharpes = []
        for window in ("5y", "3y", "2y", "1y", "fy"):
            b = base[(base.strategy == strategy) & (base.universe == universe) & (base.window == window)]
            w = best[(best.strategy == strategy) & (best.universe == universe) & (best.window == window)]
            if not b.empty:
                rec[f"{window}_base_sharpe"] = b.iloc[0]["sharpe"]
                rec[f"{window}_base_cagr"] = b.iloc[0]["cagr"]
                rec[f"{window}_base_dd"] = b.iloc[0]["max_drawdown"]
                rec[f"{window}_base_exp"] = b.iloc[0]["exposure"]
                rec[f"{window}_base_n"] = b.iloc[0]["n_trades"]
                if window in {"5y", "3y"} and pd.notna(b.iloc[0]["sharpe"]):
                    sharpes.append(float(b.iloc[0]["sharpe"]))
            if not w.empty:
                rec[f"{window}_best_sharpe"] = w.iloc[0]["sharpe"]
                rec[f"{window}_best_cagr"] = w.iloc[0]["cagr"]
                rec[f"{window}_best_cfg"] = cfg_label(w.iloc[0])
        rec["mean_base_sharpe_3y5y"] = sum(sharpes) / len(sharpes) if sharpes else None
        rows.append(rec)
    summary = pd.DataFrame(rows)
    if "mean_base_sharpe_3y5y" in summary.columns:
        summary = summary.sort_values("mean_base_sharpe_3y5y", ascending=False, na_position="last")
    summary.to_csv(out_dir / "summary_by_strategy.csv", index=False)
    return summary


def _fmt(value: object, kind: str = "num") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "pct":
        return f"{number:.1%}"
    if kind == "int":
        return str(int(number))
    return f"{number:.2f}"


def write_html(frame: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    progress = {}
    progress_path = out_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    n_ok = int((frame["error"].fillna("") == "").sum()) if not frame.empty else 0
    n_err = int((frame["error"].fillna("") != "").sum()) if not frame.empty else 0
    rows_html = []
    if not summary.empty:
        for rec in summary.to_dict(orient="records"):
            rows_html.append(
                "<tr>"
                f"<td>{rec.get('strategy','')}</td>"
                f"<td>{rec.get('family','')}</td>"
                f"<td>{rec.get('universe','')}</td>"
                f"<td>{_fmt(rec.get('mean_base_sharpe_3y5y'))}</td>"
                f"<td>{_fmt(rec.get('5y_base_sharpe'))}</td>"
                f"<td>{_fmt(rec.get('5y_base_cagr'), 'pct')}</td>"
                f"<td>{_fmt(rec.get('5y_base_dd'), 'pct')}</td>"
                f"<td>{_fmt(rec.get('5y_base_exp'), 'pct')}</td>"
                f"<td>{_fmt(rec.get('5y_base_n'), 'int')}</td>"
                f"<td>{_fmt(rec.get('5y_best_sharpe'))}</td>"
                f"<td>{rec.get('5y_best_cfg') or ''}</td>"
                f"<td>{_fmt(rec.get('3y_base_sharpe'))}</td>"
                f"<td>{_fmt(rec.get('fy_base_sharpe'))}</td>"
                "</tr>"
            )
    updated = datetime.now().isoformat(timespec="seconds")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="60">
<title>India PIT full factorial</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; white-space: nowrap; }}
th {{ position: sticky; top: 0; background: #f6f6f6; }}
.meta {{ margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>India PIT full-factorial comparison</h1>
<div class="meta">
<p>Updated {updated}.</p>
<p>Cells ok={n_ok} error={n_err}. Groups {progress.get('done_groups', 0)}/{progress.get('total_groups', 0)} fail={progress.get('fail_groups', 0)}.</p>
<p>End date 2026-08-17. FY window 2026-01-01 to 2026-08-17. PIT on. India costs. 10 bps slip. 10 slots.</p>
<p>Research, not financial advice. Page reloads every 60 seconds.</p>
</div>
<table>
<thead>
<tr>
<th>strategy</th><th>family</th><th>universe</th>
<th>mean 3y+5y base Sharpe</th>
<th>5y base Sharpe</th><th>5y CAGR</th><th>5y DD</th><th>5y exp</th><th>5y n</th>
<th>5y best Sharpe</th><th>5y best cfg</th>
<th>3y base Sharpe</th><th>FY base Sharpe</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""
    (out_dir / "compare.html").write_text(html)
    (out_dir / "index.html").write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_rows(out_dir)
    frame.to_csv(out_dir / "results.csv", index=False)
    summary = write_summary(frame, out_dir)
    write_html(frame, summary, out_dir)
    print(f"rows={len(frame)} strategies={frame['strategy'].nunique() if not frame.empty else 0}")
    print(f"wrote {out_dir / 'results.csv'}")
    print(f"wrote {out_dir / 'compare.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
