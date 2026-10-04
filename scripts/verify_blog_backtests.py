"""Check exported portfolio P&L and actual fill dates against frozen bars."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from screener.symbols import tv_to_yf


def verify_blog_backtests() -> None:
    """Reject zero-volume fills or unreconciled ledgers in any completed cell."""
    root = Path(__file__).resolve().parent.parent / "reports/blog_momentum"
    checks = []
    bar_cache: dict[Path, pd.DataFrame] = {}
    for summary in sorted(root.glob("*/summary.json")):
        if summary.parent.name.endswith("_repeat"):
            continue
        for row in json.loads(summary.read_text()):
            if row["status"] not in ("completed", "provisional"):
                raise ValueError(f"Blog verification incomplete cell: {row}")
            trades = pd.read_csv(summary.parent / row["cell"] / "trades.csv")
            if (
                "rotation_roc" in row["strategy"]
                or row["strategy"] == "alvarez_three_factor"
            ):
                normal = trades.loc[trades.exit_reason.eq("expr")]
                if (
                    not normal.empty
                    and (
                        pd.to_datetime(normal.exit_date)
                        == pd.to_datetime(normal.entry_date)
                    ).any()
                ):
                    raise ValueError(
                        f"Blog verification stale monthly exit: {row['cell']}"
                    )
            for symbol, group in trades.groupby("ticker"):
                path = (
                    summary.parent
                    / f"{row['market']}_{row['asset']}_bars"
                    / f"{tv_to_yf(str(symbol), row['market'])}.parquet"
                )
                resolved = path.resolve()
                if resolved not in bar_cache:
                    bar_cache[resolved] = pd.read_parquet(resolved)
                bars = bar_cache[resolved]
                for column in ("entry_date", "exit_date"):
                    if (
                        not bars.loc[pd.to_datetime(group[column]), "volume"]
                        .gt(0)
                        .all()
                    ):
                        raise ValueError(
                            f"Blog verification non-trading fill: {row['cell']} {symbol} {column}"
                        )
            delta = float(
                trades.pnl.sum() - (row["final_equity"] - row["starting_equity"])
            )
            if abs(delta) >= 1e-5:
                raise ValueError(
                    f"Blog verification ledger mismatch: {row['cell']} {delta}"
                )
            checks.append(
                {
                    "cell": row["cell"],
                    "batch": summary.parent.name,
                    "trades": len(trades),
                    "zero_volume_fills": 0,
                    "pnl_reconciliation_error": delta,
                }
            )
    (root / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(f"Verified {len(checks)} cells: P&L reconciles; no zero-volume fills")


if __name__ == "__main__":
    verify_blog_backtests()
