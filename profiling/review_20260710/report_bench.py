"""Measure post-backtest report and browser-payload construction."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import profiling.harness as harness  # noqa: E402
from screener.backtester.dashboard import render_dashboard  # noqa: E402
from screener.backtester.lab import _json_default, _result_payload  # noqa: E402
from screener.backtester.tearsheet import render_tearsheet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=300)
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    n_bars = int(args.years * 252) + harness.SLOW * 3 + 400
    data = harness.build_data(args.tickers, n_bars, args.seed)
    tickers = tuple(f"T{i:04d}" for i in range(args.tickers))
    index = data["SPY"].index
    window_bars = int(args.years * 252)
    start_date = index[-window_bars].date()
    end_date = index[-1].date()
    cfg = harness.make_cfg("rolling", tickers, args.top, end_date)
    result = harness.run_rolling_backtest(
        cfg,
        harness.StubFetcher(data),
        start_date=start_date,
        end_date=end_date,
    )

    with tempfile.TemporaryDirectory(prefix="screener-report-bench-") as raw_dir:
        output_dir = Path(raw_dir)
        started = time.perf_counter()
        tear_path = render_tearsheet(result, output_dir / "tearsheet.html")
        tear_seconds = time.perf_counter() - started

        started = time.perf_counter()
        dashboard_path = render_dashboard(result, output_dir)
        dashboard_seconds = time.perf_counter() - started

        started = time.perf_counter()
        payload = _result_payload("benchmark", result)
        payload_json = json.dumps(payload, default=_json_default)
        payload_seconds = time.perf_counter() - started

        output = {
            "trades": len(result.trades),
            "equity_points": len(result.equity_curve),
            "selection_rows": len(result.selection),
            "tearsheet_seconds": round(tear_seconds, 6),
            "tearsheet_bytes": tear_path.stat().st_size,
            "dashboard_seconds": round(dashboard_seconds, 6),
            "dashboard_bytes": dashboard_path.stat().st_size,
            "lab_payload_seconds": round(payload_seconds, 6),
            "lab_payload_bytes": len(payload_json.encode()),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
