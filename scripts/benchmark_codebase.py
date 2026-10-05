"""Compare offline workloads against a git revision without changing the checkout.

Run with ``uv run python scripts/benchmark_codebase.py --reference 2776bb3``.
The reference must be a trusted revision of this repository.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

_WORKLOADS = """
import json, statistics, tempfile, timeit
from pathlib import Path
import numpy as np
import pandas as pd
from screener import history
from screener.backtester.factor_tearsheet import (
    daily_spearman_ic, quantile_mean_returns, top_quantile_turnover,
)
from screener.operator.screen import label
rng = np.random.default_rng(42)
scores = pd.DataFrame(rng.normal(size=(1000, 300)))
forward = pd.DataFrame(rng.normal(size=scores.shape))
scores = scores.mask(rng.random(scores.shape) < .05)
forward = forward.mask(rng.random(forward.shape) < .05)
n = 10000
operator = pd.DataFrame({
    '_is_fno': rng.random(n) > .2,
    '%_Change_Price': rng.normal(size=n),
    '%_Change_OI': rng.normal(size=n),
    '%_Change_Delivery': rng.uniform(50, 150, n),
    'Dist_From_52W_High': rng.uniform(0, 30, n),
})
screen = pd.DataFrame({
    'name': [f'T{i}' for i in range(n)], 'close': 10.,
    'volume': 1000., 'description': 'test', 'setup_score': 50.,
})
with tempfile.TemporaryDirectory() as directory:
    history.DB_PATH = Path(directory) / 'history.db'
    workloads = {
        'factor_ic': lambda: daily_spearman_ic(scores, forward),
        'quantile_returns': lambda: quantile_mean_returns(scores, forward, n_quantiles=5),
        'quantile_turnover': lambda: top_quantile_turnover(scores, n_quantiles=5),
        'operator_labels': lambda: label(operator),
        'history_save': lambda: history.save_run('us', 'ema', n, screen),
    }
    print(json.dumps({
        name: statistics.median(timeit.repeat(func, number=1, repeat=3))
        for name, func in workloads.items()
    }))
"""


def benchmark_checkout(checkout: Path) -> dict[str, float]:
    """Measure subprocess matrix runtime and fixed in-process workloads in seconds."""
    env = dict(os.environ, PYTHONPATH=str(checkout))
    output = subprocess.check_output(
        [sys.executable, "-c", _WORKLOADS], cwd=checkout, env=env, text=True
    )
    results: dict[str, float] = json.loads(output)
    with tempfile.TemporaryDirectory() as directory:
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            subprocess.run(
                [
                    sys.executable,
                    "scripts/backtest_delta.py",
                    "--out",
                    str(Path(directory) / "matrix.json"),
                ],
                cwd=checkout,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            samples.append(time.perf_counter() - start)
    results["backtest_matrix"] = statistics.median(samples)
    return results


def main() -> None:
    """Export a trusted baseline and report median timing ratios as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", required=True, help="Trusted baseline git revision"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        baseline = Path(directory)
        archive = subprocess.check_output(["git", "archive", args.reference], cwd=root)
        subprocess.run(["tar", "-x", "-C", str(baseline)], input=archive, check=True)
        before = benchmark_checkout(baseline)
        after = benchmark_checkout(root)
    print(
        json.dumps(
            {
                name: {
                    "before_seconds": before[name],
                    "after_seconds": after[name],
                    "speedup": before[name] / after[name],
                }
                for name in before
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
