#!/usr/bin/env python3
"""Generate an interactive HTML report from the correctness audit pytest run."""

from __future__ import annotations

import html
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = Path("/tmp/audit_results.json")
HTML_PATH = ROOT / "tests" / "backtester_correctness_report.html"


def _ensure_json_plugin() -> None:
    if importlib.util.find_spec("pytest_jsonreport") is not None:
        return
    subprocess.run(
        ["uv", "add", "--dev", "pytest-json-report"],
        cwd=ROOT,
        check=True,
    )


def _run_pytest() -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_correctness_tier1.py",
        "tests/test_correctness_tier2_3.py",
        "--tb=short",
        "-q",
        "--json-report",
        f"--json-report-file={JSON_PATH}",
    ]
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode not in (0, 1):
        sys.exit(proc.returncode)


def _tier_for_nodeid(nodeid: str) -> str:
    if "test_correctness_tier1" in nodeid:
        return "tier1"
    if "test_tier2_" in nodeid:
        return "tier2"
    if "test_tier3_" in nodeid:
        return "tier3"
    return "other"


def _duration_ms(test: dict) -> float:
    total = 0.0
    for phase in ("setup", "call", "teardown"):
        block = test.get(phase) or {}
        total += float(block.get("duration") or 0.0)
    return total * 1000.0


def _failure_text(test: dict) -> str:
    for phase in ("call", "setup", "teardown"):
        block = test.get(phase) or {}
        if block.get("outcome") in ("failed", "error"):
            lr = block.get("longrepr")
            if lr:
                return str(lr)
            crash = block.get("crash")
            if crash and crash.get("longrepr"):
                return str(crash["longrepr"])
    return ""


def _passed_summary(test: dict) -> str:
    """Best-effort short text for passed tests (docstring first line)."""
    doc = test.get("doc") or test.get("description")
    if isinstance(doc, str) and doc.strip():
        return doc.strip().splitlines()[0]
    node = test.get("nodeid", "")
    return f"Passed: {node.split('::')[-1]}"


def _build_html(data: dict) -> str:
    summary = data.get("summary", {})
    total = int(summary.get("total", 0) or 0)
    passed = int(summary.get("passed", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    skipped = int(summary.get("skipped", 0) or 0)
    rate = (100.0 * passed / total) if total else 0.0

    tests = data.get("tests") or []
    by_tier: dict[str, list[dict]] = {
        "tier1": [],
        "tier2": [],
        "tier3": [],
        "other": [],
    }
    for t in tests:
        by_tier[_tier_for_nodeid(t.get("nodeid", ""))].append(t)

    gen = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    raw_json = html.escape(json.dumps(data, indent=2))

    tier_defs = [
        (
            "tier1",
            "Tier 1 — Shared Features (VBT ↔ Core)",
            "Cross-engine fills, risk controls, ranking, and shared metrics.",
        ),
        (
            "tier2",
            "Tier 2 — Core-only Gap / Partial / Entry",
            "Gap-aware fills, partial exits, and alternate entry order types.",
        ),
        (
            "tier3",
            "Tier 3 — Core-only Slippage / Filters / Metrics",
            "Slippage models, universe filters, dividends, and performance metrics.",
        ),
    ]

    def tier_table(tier_key: str, title: str, blurb: str) -> str:
        rows: list[str] = []
        for test in sorted(
            by_tier.get(tier_key, []), key=lambda x: x.get("nodeid", "")
        ):
            nodeid = test.get("nodeid", "")
            short_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
            outcome = test.get("outcome", "unknown")
            ms = _duration_ms(test)
            row_cls = (
                "pass"
                if outcome == "passed"
                else "fail"
                if outcome == "failed"
                else "skip"
            )
            notes = ""
            detail = ""
            if outcome == "passed":
                notes = html.escape(_passed_summary(test))
                detail = html.escape(_passed_summary(test))
            else:
                ft = _failure_text(test)
                notes = (
                    html.escape(ft[:200] + ("…" if len(ft) > 200 else ""))
                    if ft
                    else outcome
                )
                detail = (
                    html.escape(ft)
                    if ft
                    else html.escape(json.dumps(test, indent=2)[:8000])
                )

            rows.append(
                f"""
                <tr class="{row_cls}">
                  <td class="name">
                    <details>
                      <summary>{html.escape(short_name)}</summary>
                      <div class="detail"><pre>{detail}</pre></div>
                    </details>
                    <div class="nodeid">{html.escape(nodeid)}</div>
                  </td>
                  <td>{html.escape(outcome)}</td>
                  <td class="num">{ms:.2f}</td>
                  <td class="notes">{notes}</td>
                </tr>
                """
            )
        body = "\n".join(rows) if rows else "<tr><td colspan='4'>No tests</td></tr>"
        return f"""
        <section class="accordion">
          <button type="button" class="acc-btn" aria-expanded="false">{html.escape(title)}</button>
          <div class="acc-panel" hidden>
            <p class="blurb">{html.escape(blurb)}</p>
            <table>
              <thead><tr><th>Test</th><th>Status</th><th>Duration (ms)</th><th>Notes</th></tr></thead>
              <tbody>{body}</tbody>
            </table>
          </div>
        </section>
        """

    sections = "".join(tier_table(k, t, b) for k, t, b in tier_defs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Backtester Correctness Audit Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem; background: #f4f6f8; color: #1a1a1a; }}
    header {{ margin-bottom: 1rem; }}
    h1 {{ font-size: 1.35rem; margin: 0 0 0.25rem 0; }}
    .meta {{ color: #555; font-size: 0.9rem; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0; }}
    .card {{ background: #fff; border-radius: 8px; padding: 0.75rem 1rem; min-width: 120px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
    .card strong {{ display: block; font-size: 1.5rem; }}
    .card.pass strong {{ color: #1b5e20; }}
    .card.fail strong {{ color: #b71c1c; }}
    .card.skip strong {{ color: #f57f17; }}
    .bar {{ height: 22px; border-radius: 6px; overflow: hidden; display: flex; margin: 0.5rem 0 1.25rem; border: 1px solid #ccc; }}
    .bar .ok {{ background: #66bb6a; height: 100%; }}
    .bar .bad {{ background: #ef5350; height: 100%; }}
    .accordion {{ background: #fff; border-radius: 8px; margin-bottom: 0.75rem; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
    .acc-btn {{ width: 100%; text-align: left; padding: 0.75rem 1rem; font-size: 1rem; border: none; background: #e3e7ec; cursor: pointer; border-radius: 8px 8px 0 0; }}
    .acc-btn:hover {{ background: #d5dbe4; }}
    .acc-panel {{ padding: 0.75rem 1rem 1rem; }}
    .blurb {{ color: #444; font-size: 0.9rem; margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th, td {{ border-bottom: 1px solid #e0e0e0; padding: 0.45rem 0.35rem; vertical-align: top; }}
    th {{ text-align: left; background: #fafafa; }}
    tr.pass td:first-child {{ background: #e8f5e9; }}
    tr.fail td:first-child {{ background: #ffebee; }}
    tr.skip td:first-child {{ background: #fff8e1; }}
    .name summary {{ cursor: pointer; font-weight: 600; }}
    .nodeid {{ font-size: 0.75rem; color: #666; margin-top: 0.2rem; }}
    .detail pre {{ white-space: pre-wrap; word-break: break-word; background: #fafafa; padding: 0.5rem; border-radius: 4px; max-height: 320px; overflow: auto; font-size: 0.82rem; }}
    .num {{ text-align: right; width: 90px; }}
    footer {{ margin-top: 2rem; font-size: 0.85rem; color: #555; }}
    #copyBtn {{ margin: 0.5rem 0 1rem; padding: 0.4rem 0.75rem; cursor: pointer; }}
  </style>
</head>
<body>
  <header>
    <h1>Backtester Correctness Audit Report</h1>
    <div class="meta">Generated: {html.escape(gen)}</div>
  </header>
  <div class="cards">
    <div class="card"><span>Total</span><strong>{total}</strong></div>
    <div class="card pass"><span>Passed</span><strong>{passed}</strong></div>
    <div class="card fail"><span>Failed</span><strong>{failed}</strong></div>
    <div class="card skip"><span>Skipped</span><strong>{skipped}</strong></div>
  </div>
  <div class="bar" title="Pass rate">
    <div class="ok" style="width:{rate:.2f}%"></div>
    <div class="bad" style="width:{100.0 - rate:.2f}%"></div>
  </div>
  <button id="copyBtn" type="button">Copy JSON</button>
  <textarea id="rawJson" hidden>{raw_json}</textarea>
  {sections}
  <footer>Generated by backtester-correctness-audit agent · /workspace</footer>
  <script>
    (function () {{
      document.querySelectorAll(".acc-btn").forEach(function (btn) {{
        btn.addEventListener("click", function () {{
          var p = btn.nextElementSibling;
          var open = btn.getAttribute("aria-expanded") === "true";
          btn.setAttribute("aria-expanded", open ? "false" : "true");
          p.hidden = open;
        }});
      }});
      document.getElementById("copyBtn").addEventListener("click", function () {{
        var t = document.getElementById("rawJson");
        navigator.clipboard.writeText(t.value).then(function () {{
          var b = document.getElementById("copyBtn");
          var o = b.textContent;
          b.textContent = "Copied!";
          setTimeout(function () {{ b.textContent = o; }}, 1500);
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    _ensure_json_plugin()
    _run_pytest()
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    HTML_PATH.write_text(_build_html(data), encoding="utf-8")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
