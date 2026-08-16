#!/usr/bin/env python3
"""Benchmark harness for `screener screen` shell and workflow paths.

Measures M1-M8 as defined in the screen CLI performance plan.
Run from the repo root with an idle machine and thread limits set:

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    .venv/bin/python profiling/scripts/bench_screen.py

Options:
    --label LABEL     Tag for this run (default: baseline / env SCREENER_BENCH_LABEL)
    --runs N          Timed runs per metric after warmup (default: 3)
    --json PATH       Write full JSON results (default: profiling/_analysis/screen_bench_<label>.json)
    --append-log PATH Append markdown section (default: profiling/_analysis/screen_bench_log.md)
    --skip-m8         Skip no-Turso shell metric (M8)
    --skip-workflow   Skip in-process workflow metrics (M4/M5)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS = REPO_ROOT / "profiling" / "_analysis"
VENV_SCREENER = REPO_ROOT / ".venv" / "bin" / "screener"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

METRIC_ORDER = [
    "M1_cli_warm_csv",
    "M2_cli_warm_table",
    "M3_cli_help",
    "M4_workflow_warm_csv",
    "M5_workflow_warm_full",
    "M6_import_cli",
    "M7_usage_pair",
    "M8_cli_warm_csv_no_turso",
]


def _mean(samples: list[float]) -> float:
    return statistics.fmean(samples) if samples else float("nan")


def _shell_time(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> float:
    """Wall seconds via /usr/bin/time -f %e (process tree wall clock)."""
    time_cmd = ["/usr/bin/time", "-f", "%e", *cmd]
    proc = subprocess.run(
        time_cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    # /usr/bin/time writes to stderr; last non-empty line should be the float.
    lines = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"no timing output for {cmd!r} (exit {proc.returncode})")
    try:
        return float(lines[-1])
    except ValueError as exc:
        raise RuntimeError(
            f"could not parse timing from {lines!r} for {cmd!r} (exit {proc.returncode})"
        ) from exc


def _python_time(code: str, *, env: dict | None = None) -> float:
    """Time a short python -c snippet via /usr/bin/time."""
    return _shell_time([str(VENV_PYTHON), "-c", code], env=env)


def _measure_shell(
    cmd: list[str],
    *,
    runs: int,
    cwd: Path | None = None,
    env: dict | None = None,
    warmup: bool = True,
) -> dict:
    if warmup:
        _shell_time(cmd, cwd=cwd, env=env)
    samples = [_shell_time(cmd, cwd=cwd, env=env) for _ in range(runs)]
    return {"mean_s": _mean(samples), "samples_s": samples, "n": runs}


def _turso_available() -> bool:
    url = os.environ.get("TURSO_DATABASE_URL")
    token = os.environ.get("TURSO_AUTH_TOKEN")
    if url and token:
        return True
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return False
    text = env_path.read_text(encoding="utf-8")
    has_url = any(
        line.strip().startswith("TURSO_DATABASE_URL=") and "=" in line
        for line in text.splitlines()
    )
    has_token = any(
        line.strip().startswith("TURSO_AUTH_TOKEN=") and "=" in line
        for line in text.splitlines()
    )
    return has_url and has_token


def _versions() -> dict[str, str]:
    code = (
        "import numpy, pandas; "
        "print(pandas.__version__); print(numpy.__version__)"
    )
    proc = subprocess.run(
        [str(VENV_PYTHON), "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return {"pandas": lines[0], "numpy": lines[1]}


def _git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _measure_workflow(output_csv: bool, *, runs: int) -> dict:
    """In-process warm workflow; import once, then time run_screen_workflow."""
    # Run as a child process so each bench invocation is isolated, but
    # warmup + N runs share one process (matches "after 1 warmup").
    code = f"""
import time
from pathlib import Path
from screener.screen_workflow import ScreenRequest, run_screen_workflow

req = ScreenRequest(
    market="us",
    criteria_names=("ema",),
    limit=50,
    order_by="setup_score",
    output_csv={output_csv!r},
    detail=False,
    refresh=False,
    cache_ttl="15m",
    report_path=None,
    open_report=False,
)
# report_path=None on HEAD writes the default temp HTML report (C8).
# C4-C7 "final" M5 numbers were history-only and do not apply to this path.

# Warm
run_screen_workflow(req)

samples = []
for _ in range({runs}):
    t0 = time.perf_counter()
    outcome = run_screen_workflow(req)
    samples.append(time.perf_counter() - t0)
    assert len(outcome.df) > 0, "empty screen result"

print(",".join(f"{{s:.6f}}" for s in samples))
"""
    proc = subprocess.run(
        [str(VENV_PYTHON), "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"workflow measure failed (exit {proc.returncode}):\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    line = proc.stdout.strip().splitlines()[-1]
    samples = [float(x) for x in line.split(",")]
    return {"mean_s": _mean(samples), "samples_s": samples, "n": len(samples)}


def _measure_usage_pair(*, runs: int) -> dict:
    code = f"""
import time
import os
# Ensure pytest short-circuit is off
os.environ.pop("PYTEST_CURRENT_TEST", None)
from screener import usage

# Warm once (connect + DDL)
usage.record_feature_usage("bench_screen", command_path="bench", duration_ms=1)
usage.record_feature_invocation(
    "bench_screen", command_path="bench", duration_ms=1, status="success", params={{}}
)

samples = []
for _ in range({runs}):
    t0 = time.perf_counter()
    usage.record_feature_usage("bench_screen", command_path="bench", duration_ms=1)
    usage.record_feature_invocation(
        "bench_screen", command_path="bench", duration_ms=1, status="success", params={{}}
    )
    samples.append(time.perf_counter() - t0)

print(",".join(f"{{s:.6f}}" for s in samples))
"""
    proc = subprocess.run(
        [str(VENV_PYTHON), "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"usage pair measure failed (exit {proc.returncode}):\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    line = proc.stdout.strip().splitlines()[-1]
    samples = [float(x) for x in line.split(",")]
    return {"mean_s": _mean(samples), "samples_s": samples, "n": len(samples)}


def _no_turso_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TURSO_DATABASE_URL", None)
    env.pop("TURSO_AUTH_TOKEN", None)
    # Prevent accidental re-read of repo .env via cwd: caller uses cwd=/tmp
    return env


def run_bench(
    *,
    runs: int,
    skip_m8: bool,
    skip_workflow: bool,
) -> dict:
    if not VENV_SCREENER.exists():
        raise SystemExit(f"missing {VENV_SCREENER}; run uv sync first")

    # Populate TV cache before timed runs.
    warm_cmd = [
        str(VENV_SCREENER),
        "--log-level",
        "ERROR",
        "screen",
        "-m",
        "us",
        "-c",
        "ema",
        "-n",
        "50",
        "--csv",
    ]
    subprocess.run(
        warm_cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    metrics: dict[str, dict] = {}

    csv_cmd = [
        str(VENV_SCREENER),
        "--log-level",
        "ERROR",
        "screen",
        "-m",
        "us",
        "-c",
        "ema",
        "-n",
        "50",
        "--csv",
    ]
    table_cmd = [
        str(VENV_SCREENER),
        "--log-level",
        "ERROR",
        "screen",
        "-m",
        "us",
        "-c",
        "ema",
        "-n",
        "50",
    ]
    help_cmd = [str(VENV_SCREENER), "--help"]

    print("Measuring M1_cli_warm_csv ...", flush=True)
    metrics["M1_cli_warm_csv"] = _measure_shell(csv_cmd, runs=runs)

    print("Measuring M2_cli_warm_table ...", flush=True)
    metrics["M2_cli_warm_table"] = _measure_shell(table_cmd, runs=runs)

    print("Measuring M3_cli_help ...", flush=True)
    metrics["M3_cli_help"] = _measure_shell(help_cmd, runs=runs)

    if not skip_workflow:
        print("Measuring M4_workflow_warm_csv ...", flush=True)
        metrics["M4_workflow_warm_csv"] = _measure_workflow(True, runs=runs)
        print("Measuring M5_workflow_warm_full ...", flush=True)
        metrics["M5_workflow_warm_full"] = _measure_workflow(False, runs=runs)
    else:
        metrics["M4_workflow_warm_csv"] = {"mean_s": None, "samples_s": [], "n": 0}
        metrics["M5_workflow_warm_full"] = {"mean_s": None, "samples_s": [], "n": 0}

    print("Measuring M6_import_cli ...", flush=True)
    import_code = "import screener.cli"
    # Each sample is a fresh process (import tax is per-process).
    samples = []
    _python_time(import_code)  # warmup process once (doesn't help much, but ok)
    for _ in range(runs):
        samples.append(_python_time(import_code))
    metrics["M6_import_cli"] = {
        "mean_s": _mean(samples),
        "samples_s": samples,
        "n": runs,
    }

    print("Measuring M7_usage_pair ...", flush=True)
    metrics["M7_usage_pair"] = _measure_usage_pair(runs=runs)

    if not skip_m8:
        print("Measuring M8_cli_warm_csv_no_turso ...", flush=True)
        # Run from /tmp so .env is not found; also unset TURSO_*.
        metrics["M8_cli_warm_csv_no_turso"] = _measure_shell(
            csv_cmd,
            runs=runs,
            cwd=Path("/tmp"),
            env=_no_turso_env(),
        )
    else:
        metrics["M8_cli_warm_csv_no_turso"] = {
            "mean_s": None,
            "samples_s": [],
            "n": 0,
        }

    return metrics


def _fmt_s(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def append_markdown(
    path: Path,
    *,
    label: str,
    sha: str,
    metrics: dict,
    meta: dict,
    prev_means: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if not path.exists() or path.stat().st_size == 0:
        lines.append("# Screen CLI bench log")
        lines.append("")
        lines.append(
            "Timings are mean wall seconds over N timed runs after one warmup "
            "(see `profiling/scripts/bench_screen.py`)."
        )
        lines.append("")

    ts = meta.get("timestamp_utc", "")
    lines.append(f"## {label}")
    lines.append("")
    lines.append(f"- git: `{sha}`")
    lines.append(f"- when: {ts}")
    lines.append(
        f"- turso_configured: {meta.get('turso_configured')}; "
        f"pandas={meta.get('pandas')}; numpy={meta.get('numpy')}"
    )
    lines.append(f"- runs: {meta.get('runs')}")
    lines.append("")
    lines.append("| metric | mean_s | samples | delta_vs_prev |")
    lines.append("| --- | ---: | --- | ---: |")
    for key in METRIC_ORDER:
        m = metrics.get(key) or {}
        mean = m.get("mean_s")
        samples = m.get("samples_s") or []
        sample_str = ", ".join(f"{s:.3f}" for s in samples) if samples else "-"
        delta = ""
        if prev_means is not None and mean is not None and key in prev_means:
            prev = prev_means[key]
            if prev is not None and prev > 0:
                delta = f"{(mean - prev) / prev * 100:+.1f}%"
            elif prev is not None:
                delta = f"{mean - prev:+.3f}s"
        lines.append(f"| {key} | {_fmt_s(mean)} | {sample_str} | {delta or '-'} |")
    lines.append("")

    path.write_text(
        (path.read_text(encoding="utf-8") if path.exists() else "") + "\n".join(lines),
        encoding="utf-8",
    )


def _load_prev_means(log_json_candidates: list[Path]) -> dict[str, float] | None:
    for path in log_json_candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics = data.get("metrics") or {}
            return {
                k: (metrics[k] or {}).get("mean_s")
                for k in METRIC_ORDER
                if k in metrics
            }
        except (json.JSONDecodeError, OSError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default=os.environ.get("SCREENER_BENCH_LABEL", "baseline"),
        help="Label for this measurement section",
    )
    parser.add_argument("--runs", type=int, default=3, help="Timed runs after warmup")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--append-log",
        type=Path,
        default=DEFAULT_ANALYSIS / "screen_bench_log.md",
        help="Markdown log to append",
    )
    parser.add_argument("--skip-m8", action="store_true")
    parser.add_argument("--skip-workflow", action="store_true")
    parser.add_argument(
        "--prev-json",
        type=Path,
        default=None,
        help="Previous results JSON for delta column",
    )
    args = parser.parse_args()

    # Thread limits as required by the plan.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    label = args.label
    json_path = args.json or (DEFAULT_ANALYSIS / f"screen_bench_{label}.json")
    if label == "baseline":
        # Canonical baseline path from the plan.
        json_path = args.json or (DEFAULT_ANALYSIS / "screen_bench_baseline.json")

    versions = _versions()
    sha = _git_sha()
    turso = _turso_available()

    print(f"label={label} sha={sha[:12]} turso={turso} runs={args.runs}", flush=True)
    metrics = run_bench(
        runs=args.runs,
        skip_m8=args.skip_m8,
        skip_workflow=args.skip_workflow,
    )

    meta = {
        "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "turso_configured": turso,
        "pandas": versions["pandas"],
        "numpy": versions["numpy"],
        "runs": args.runs,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "repo": str(REPO_ROOT),
    }
    payload = {"meta": meta, "metrics": metrics}

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {json_path}", flush=True)

    prev = None
    if args.prev_json:
        prev = _load_prev_means([args.prev_json])
    else:
        # Prefer baseline, then most recent non-matching label file is harder;
        # only auto-load baseline for first delta after baseline.
        prev = _load_prev_means([DEFAULT_ANALYSIS / "screen_bench_baseline.json"])
        if label == "baseline":
            prev = None

    append_markdown(
        args.append_log,
        label=label,
        sha=sha,
        metrics=metrics,
        meta=meta,
        prev_means=prev,
    )
    print(f"appended {args.append_log}", flush=True)

    # Print summary table to stdout.
    print()
    print(f"{'metric':28s} {'mean_s':>8s}")
    for key in METRIC_ORDER:
        mean = (metrics.get(key) or {}).get("mean_s")
        print(f"{key:28s} {_fmt_s(mean):>8s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
