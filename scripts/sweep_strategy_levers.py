"""Sweep execution levers (stop loss, take profit, trailing stop, regime filter)
over every research strategy on a 3-year window for both markets, find the best
config per strategy x market, and validate it on 5/2/1-year windows.

Phase 1 (sweep): for each strategy x market x regime, prepare the price+signal
panels once and re-simulate the book across the stop/TP/trailing grid.
Phase 2 (validate): run the best config per strategy x market on 5/2/1y.

Run:  uv run python scripts/sweep_strategy_levers.py 1   # sweep only
      uv run python scripts/sweep_strategy_levers.py 2   # validate only
      uv run python scripts/sweep_strategy_levers.py     # both

Outputs under findings/research_study/:
  sweep_results.csv      every (strategy, market, regime, stop, tp, trail) row
  sweep_best.json        best config per strategy x market (by Sharpe, min trades)
  sweep_validate.csv     tuned-vs-baseline across 5/2/1y windows
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

# Force FMP as the price provider for every fetch in this study.
os.environ["SCREENER_PRICE_PROVIDER"] = "fmp"

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import (
    prepare_rolling_backtest,
    run_prepared_rolling_backtest,
    run_rolling_backtest,
)
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "findings" / "research_study"
OUT.mkdir(parents=True, exist_ok=True)

# market -> (universe, cost flags)
MARKETS = {
    "india": {"universe": "nifty500", "cost_model": "india", "slippage_bps": 10.0, "commission_bps": 0.0},
    "us": {"universe": "sp500", "cost_model": "flat", "slippage_bps": 5.0, "commission_bps": 1.0},
}

# strategy -> (top, hold) — same sizing as the main study
STRATEGIES = {
    "golden_cross_50_200": (10, 500),
    "fifty_two_week_high": (10, 500),
    "bll_trading_range_break": (10, 500),
    "keltner_breakout": (10, 100),
    "adx_trend": (10, 250),
    "long_term_reversal": (10, 250),
    "macd_signal_cross": (15, 60),
    "stochastic_cross": (15, 20),
    "connors_rsi2": (20, 20),
    "connors_rsi2_bull": (20, 20),
    "bollinger_mean_reversion": (20, 20),
    "williams_percent_r": (20, 20),
    "cci_reversion": (20, 20),
    "short_term_reversal": (20, 20),
    "turn_of_month": (20, 20),
}

REGIMES = [(), ("bull",), ("bull", "pullback")]

# Execution lever grid (book fields -> fast re-simulation via prepared panels).
GRID = [
    {"stop_loss": sl, "take_profit": tp, "trailing_stop": ts}
    for sl in (None, 0.08, 0.15, 0.25)
    for tp in (None, 0.25)
    for ts in (None, 0.15, 0.25)
]

SIZING_RULES = ("atr_risk", "fixed_fraction", "inverse_vol")

METRIC_FIELDS = [
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "total_return",
    "trades",
    "exposure",
    "benchmark_return",
]


def _metrics_of(result) -> dict[str, float]:
    m = result.metrics
    # Return-like fields stored as fractions by the engine -> percent for CSV.
    return {
        "cagr": None if m.get("cagr") is None else m["cagr"] * 100.0,
        "sharpe": m.get("sharpe"),
        "sortino": m.get("sortino"),
        "calmar": m.get("calmar"),
        "max_drawdown": None if m.get("max_drawdown") is None else m["max_drawdown"] * 100.0,
        "total_return": None if m.get("total_return") is None else m["total_return"] * 100.0,
        "trades": float(m.get("trade_count", len(result.trades)) or 0.0),
        "exposure": float((m.get("exposure") or 0.0) * 100.0),
        "benchmark_return": float((m.get("benchmark_return") or 0.0) * 100.0),
    }


def _sizing_check(prepared, regime_cfg: BacktestConfig, best: dict) -> dict | None:
    """Try alternative sizing rules on the grid-best config; return the winner."""
    updates = {
        "stop_loss": None if best["sl"] == "none" else float(best["sl"]),
        "take_profit": None if best["tp"] == "none" else float(best["tp"]),
        "trailing_stop": None if best["trail"] == "none" else float(best["trail"]),
    }
    winner = best
    for rule in SIZING_RULES:
        cfg_r = regime_cfg.model_copy(update={**updates, "sizing_rule": rule})
        result = run_prepared_rolling_backtest(prepared, cfg_r)
        row = {**best, "sizing": rule, **_metrics_of(result)}
        if (
            row["trades"] >= 8
            and row["sharpe"] is not None
            and (winner["sharpe"] is None or row["sharpe"] > winner["sharpe"])
        ):
            winner = row
    return winner


def _resolve(market: str, strategy: str, years: int) -> tuple[BacktestConfig, object, date, date]:
    top, hold = STRATEGIES[strategy]
    cfg_m = MARKETS[market]
    request = BacktestRequest(
        mode="rolling",
        context_obj=None,
        market=market,
        hold=hold,
        top=top,
        entry_expr=None,
        exit_expr=None,
        strategy_name=strategy,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=cfg_m["slippage_bps"],
        commission_bps=cfg_m["commission_bps"],
        cost_model=cfg_m["cost_model"],
        initial_capital=100_000.0,
        benchmark=None,
        tickers=None,
        universe_file=None,
        max_universe=0,
        min_price=None,
        min_avg_dollar_volume=None,
        adv_window=60,
        slippage_model="fixed",
        half_spread_bps=0.0,
        vol_impact_k=0.0,
        no_gap_fills=False,
        entry_order="moo",
        entry_limit_bps=None,
        partial_exit_args=(),
        price_adjustment="full",
        interval="1d",
        output_csv=False,
        report_path=None,
        open_report=False,
        sizing_rule="equal_slot",
        sizing_risk_pct=0.01,
        sizing_position_pct=0.10,
        sizing_atr_window=14,
        sizing_atr_multiple=2.0,
        sizing_vol_window=20,
        intraday_only=False,
        years=years,
        universe=cfg_m["universe"],
        regime_filter_args=(),
    )
    run = resolve_backtest_run(request)
    assert run.start_date is not None and run.end_date is not None
    return run.config, run.price_fetcher, run.start_date, run.end_date


def _combo_dict(cfg: BacktestConfig) -> dict[str, str]:
    def f(v: float | None) -> str:
        return "none" if v is None else f"{v:.2f}"

    return {
        "sl": f(cfg.stop_loss),
        "tp": f(cfg.take_profit),
        "trail": f(cfg.trailing_stop),
    }


def sweep_one(market: str, strategy: str) -> list[dict]:
    top, hold = STRATEGIES[strategy]
    cfg, fetcher, start, end = _resolve(market, strategy, 3)
    rows: list[dict] = []
    for regime in REGIMES:
        regime_cfg = cfg.model_copy(update={"regime_filter": regime})
        prepared = prepare_rolling_backtest(regime_cfg, fetcher, start_date=start, end_date=end)
        regime_rows: list[dict] = []
        for combo in GRID:
            combo_cfg = regime_cfg.model_copy(update=combo)
            result = run_prepared_rolling_backtest(prepared, combo_cfg)
            row = {
                "market": market,
                "strategy": strategy,
                "regime": ",".join(regime) or "none",
                "sizing": "equal_slot",
                **_combo_dict(combo_cfg),
                **_metrics_of(result),
            }
            regime_rows.append(row)
            rows.append(row)
            print(
                f"  [{market}/{strategy}] regime={row['regime']:12s} "
                f"sl={row['sl']:5s} tp={row['tp']:5s} trail={row['trail']:5s} -> "
                f"CAGR {row['cagr']:+.1f}% Sharpe {row['sharpe']:+.2f} "
                f"MDD {row['max_drawdown']:+.1f}% n={int(row['trades'])}",
                flush=True,
            )
        # Try alternative sizing rules on this regime's grid-best config.
        best_grid = pick_best(regime_rows)
        if best_grid is not None:
            winner = _sizing_check(prepared, regime_cfg, best_grid)
            if winner is not None and winner["sizing"] != "equal_slot":
                rows.append(winner)
                print(
                    f"  [{market}/{strategy}] regime={winner['regime']:12s} "
                    f"sizing={winner['sizing']:12s} -> "
                    f"CAGR {winner['cagr']:+.1f}% Sharpe {winner['sharpe']:+.2f} "
                    f"MDD {winner['max_drawdown']:+.1f}% n={int(winner['trades'])}",
                    flush=True,
                )
    return rows


def pick_best(rows: list[dict], min_trades: int = 8) -> dict | None:
    eligible = [r for r in rows if r["trades"] >= min_trades and r["sharpe"] is not None]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r["sharpe"])


def validate_best(best_by_key: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for market in MARKETS:
        for strategy in STRATEGIES:
            best = best_by_key.get(f"{market}/{strategy}")
            if best is None:
                continue
            cfg, _, _, _ = _resolve(market, strategy, 5)
            regime = () if best["regime"] == "none" else tuple(best["regime"].split(","))
            for years in (5, 2, 1):
                _, fetcher_y, start, end = _resolve(market, strategy, years)
                tuned = cfg.model_copy(
                    update={
                        "regime_filter": regime,
                        "stop_loss": None if best["sl"] == "none" else float(best["sl"]),
                        "take_profit": None if best["tp"] == "none" else float(best["tp"]),
                        "trailing_stop": None if best["trail"] == "none" else float(best["trail"]),
                    }
                )
                # Run the tuned config and the no-lever baseline on the same window.
                for label, run_cfg in (("tuned", tuned), ("baseline", cfg)):
                    result = run_rolling_backtest(run_cfg, fetcher_y, start_date=start, end_date=end)

                    def _lv(v: float | None) -> str:
                        return "none" if v is None else f"{v:.2f}"

                    row = {
                        "market": market,
                        "strategy": strategy,
                        "years": years,
                        "kind": label,
                        "regime": ",".join(regime) or "none",
                        "sl": _lv(run_cfg.stop_loss),
                        "tp": _lv(run_cfg.take_profit),
                        "trail": _lv(run_cfg.trailing_stop),
                        **_metrics_of(result),
                    }
                    rows.append(row)
                    print(
                        f"  validate [{market}/{strategy}] {years}y {label:8s} "
                        f"regime={row['regime']:12s} {row['sl']} {row['tp']} {row['trail']} -> "
                        f"CAGR {row['cagr']:+.1f}% Sharpe {row['sharpe']:+.2f} MDD {row['max_drawdown']:+.1f}%",
                        flush=True,
                    )
    return rows


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "12"
    sweep_csv = OUT / "sweep_results_fmp.csv"
    best_json = OUT / "sweep_best_fmp.json"
    validate_csv = OUT / "sweep_validate_fmp.csv"

    if "1" in phase:
        all_rows: list[dict] = []
        for market in MARKETS:
            for strategy in STRATEGIES:
                all_rows.extend(sweep_one(market, strategy))
        with sweep_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        best: dict[str, dict] = {}
        for market in MARKETS:
            for strategy in STRATEGIES:
                key = f"{market}/{strategy}"
                best[key] = pick_best([r for r in all_rows if r["strategy"] == strategy and r["market"] == market])
        best_json.write_text(json.dumps(best, indent=2, default=str))
        print(f"sweep: {len(all_rows)} rows -> {sweep_csv}")

    if "2" in phase:
        best = json.loads(best_json.read_text())
        rows = validate_best(best)
        with validate_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"validate: {len(rows)} rows -> {validate_csv}")


if __name__ == "__main__":
    main()
