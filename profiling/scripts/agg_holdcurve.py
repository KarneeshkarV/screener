#!/usr/bin/env python3
"""Aggregate the hold-time curve. 7 finite hold caps {5,10,20,30,40,60,100} from
holdcurve_res/*.csv + the inf (no-cap) point reused from sweep_nohold_all.csv
(equal_slot cells). Emits a JSON blob for the HTML page and prints a text summary."""

import os
import json
import glob

SP = os.environ.get("ANALYSIS_DIR", os.path.join(os.getcwd(), "profiling", "_analysis"))
RES = f"{SP}/holdcurve_res"
STRAT_ORDER = [
    "mark_minervini",
    "mq_us1",
    "mq_us2",
    "mq_us3",
    "mq_in1",
    "mq_in2",
    "mq_in3",
]
TOPS = ["5", "10", "20", "40"]
HOLDS = ["5", "10", "20", "30", "40", "60", "100", "inf"]
HCOLS = [
    "market",
    "strategy",
    "top",
    "hold",
    "sizing",
    "total_return",
    "cagr",
    "sharpe",
    "max_dd",
    "hit_rate",
    "alpha",
    "profit_factor",
    "trades",
    "bench_return",
]
NCOLS = [
    "market",
    "strategy",
    "top",
    "sizing",
    "total_return",
    "cagr",
    "sharpe",
    "max_dd",
    "hit_rate",
    "alpha",
    "profit_factor",
    "trades",
    "bench_return",
]
METRICS = (
    "total_return",
    "cagr",
    "sharpe",
    "max_dd",
    "hit_rate",
    "alpha",
    "profit_factor",
    "bench_return",
)


def num(s):
    if s is None:
        return None
    s = s.strip().replace("%", "").replace("+", "")
    if s in ("", "ERROR", "NO_TRADES"):
        return None
    try:
        return float(s)
    except:
        return None


# key: (market,strategy,top,hold) -> row dict
data = {}
for f in glob.glob(f"{RES}/*.csv"):
    p = open(f).read().rstrip("\n").split(",")
    if len(p) != len(HCOLS):
        continue
    r = dict(zip(HCOLS, p))
    for k in METRICS:
        r[k + "_n"] = num(r[k])
    r["trades_n"] = num(r["trades"])
    data[(r["market"], r["strategy"], r["top"], r["hold"])] = r

# inf point from no-hold equal_slot sweep
for line in open(f"{SP}/sweep_nohold_all.csv"):
    p = line.rstrip("\n").split(",")
    if len(p) != len(NCOLS):
        continue
    r = dict(zip(NCOLS, p))
    if r["sizing"] != "equal_slot":
        continue
    r["hold"] = "inf"
    for k in METRICS:
        r[k + "_n"] = num(r[k])
    r["trades_n"] = num(r["trades"])
    data[(r["market"], r["strategy"], r["top"], "inf")] = r

bench = {m: None for m in ("us", "india")}
for (m, s, t, h), r in data.items():
    if r["bench_return_n"] is not None:
        bench[m] = r["bench_return_n"]


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------- market-level curve: avg over strat×top at each hold ----------
market_curve = {}
for m in ("us", "india"):
    market_curve[m] = {}
    for h in HOLDS:
        rows = [
            data[(m, s, t, h)]
            for s in STRAT_ORDER
            for t in TOPS
            if (m, s, t, h) in data
        ]
        market_curve[m][h] = {
            "sharpe": avg([r["sharpe_n"] for r in rows]),
            "total_return": avg([r["total_return_n"] for r in rows]),
            "cagr": avg([r["cagr_n"] for r in rows]),
            "max_dd": avg([r["max_dd_n"] for r in rows]),
            "trades": avg([r["trades_n"] for r in rows]),
            "n": len(rows),
        }

# ---------- per-strategy curve (avg over top) ----------
strat_curve = {}
for m in ("us", "india"):
    strat_curve[m] = {}
    for s in STRAT_ORDER:
        strat_curve[m][s] = {}
        for h in HOLDS:
            rows = [data[(m, s, t, h)] for t in TOPS if (m, s, t, h) in data]
            strat_curve[m][s][h] = {
                "sharpe": avg([r["sharpe_n"] for r in rows]),
                "total_return": avg([r["total_return_n"] for r in rows]),
                "max_dd": avg([r["max_dd_n"] for r in rows]),
            }


# ---------- best hold per strategy (by avg Sharpe over top) ----------
def best_hold(m, s, metric="sharpe"):
    best = None
    bh = None
    for h in HOLDS:
        v = strat_curve[m][s][h][metric]
        if v is None:
            continue
        if best is None or v > best:
            best = v
            bh = h
    return bh, best


json.dump(
    {
        "market_curve": market_curve,
        "strat_curve": strat_curve,
        "bench": bench,
        "holds": HOLDS,
        "strat_order": STRAT_ORDER,
    },
    open(f"{SP}/holdcurve.json", "w"),
    indent=2,
)

# ================= TEXT REPORT =================
HL = lambda h: "∞" if h == "inf" else h
print(f"benchmark 5yr:  US {bench['us']:+.2f}%   India {bench['india']:+.2f}%\n")

for m in ("us", "india"):
    print(
        f"\n{'=' * 88}\n{m.upper()}  — market-level hold-time curve (avg over 7 strat × 4 top = 28 cells/point)\n{'=' * 88}"
    )
    print(
        f"  {'hold':>5} | {'avgSharpe':>9} | {'avgRet%':>8} | {'avgCAGR%':>8} | {'avgMaxDD%':>9} | {'avgTrades':>9}"
    )
    for h in HOLDS:
        c = market_curve[m][h]
        print(
            f"  {HL(h):>5} | {c['sharpe']:>9.3f} | {c['total_return']:>8.1f} | {c['cagr']:>8.2f} | {c['max_dd']:>9.1f} | {c['trades']:>9.0f}"
        )
    # best hold by market avg sharpe
    bh = max(HOLDS, key=lambda h: market_curve[m][h]["sharpe"])
    br = max(HOLDS, key=lambda h: market_curve[m][h]["total_return"])
    print(
        f"  --> best avg Sharpe at hold={HL(bh)} ({market_curve[m][bh]['sharpe']:.3f});  best avg Return at hold={HL(br)} ({market_curve[m][br]['total_return']:.1f}%)"
    )

print(
    f"\n\n{'#' * 88}\nBEST HOLD PER STRATEGY (by avg Sharpe over top{{5,10,20,40}})\n{'#' * 88}"
)
for m in ("us", "india"):
    print(f"\n  {m.upper()}")
    for s in STRAT_ORDER:
        bh, bv = best_hold(m, s, "sharpe")
        bhr, bvr = best_hold(m, s, "total_return")
        if bh is None:
            continue
        print(
            f"    {s:<14} Sharpe-opt hold={HL(bh):>4} (Sh {bv:.3f})   Return-opt hold={HL(bhr):>4} (Ret {bvr:.1f}%)"
        )
