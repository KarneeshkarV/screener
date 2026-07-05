# Paper-Backed Factor Strategy Report

Deterministic, fully-offline verification of three paper-backed equity factor strategies implemented in this screener. **Research, not financial advice.**

## Reproduce

```bash
uv run python scripts/run_strategy_report.py
```

Every metric below is computed from `screener.research.factor_demo`, a closed-form synthetic price panel (8 names + `SPY` benchmark), so the numbers are bit-stable. `tests/test_strategy_report.py` pins them with hardcoded expected values.

- Window: **2023-01-02 → 2023-10-20** (daily rolling backtest, >252 trading-day warmup before the window)
- Portfolio: **top 3** equal-weight slots, **21-day** hold, benchmark `SPY`, no commission/slippage
- Selection: cross-sectional **factor ranking** via the `rank_score` hook (not signal-day dollar volume)

## Results vs benchmark

| Strategy | Sharpe | Total Return | Benchmark | Sortino | Max DD | Trades | Held |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12-1 Momentum | 2.56 | 32.81% | 10.47% | 4.25 | -2.64% | 30 | MOMA, MOMB, WILD |
| Low Volatility | 7.53 | 19.34% | 10.47% | 17.95 | -0.35% | 30 | CALM, MOMB, STDY |
| Momentum + Low-Vol Combo | 5.32 | 28.88% | 10.47% | 10.65 | -0.91% | 30 | MOMA, MOMB, STDY |

## Per-strategy detail

### 12-1 Momentum

- **Paper:** Jegadeesh & Titman (1993), J. Finance 48(1)
- **Signal:** rank by close[t-21]/close[t-252]-1; long positive-momentum winners
- **Sharpe:** 2.5642 | **Total return:** 32.81% | **Benchmark:** 10.47% | **Trades:** 30
- **Reproduction (this report):** `uv run python scripts/run_strategy_report.py`
- **CLI form (live universes):** `uv run screener backtest-rolling -m us --strategy momentum_12_1 --tickers MOMA,MOMB,STDY,CALM,CHOP,WILD,DOWNA,DOWNB --top 3 --hold 21 --start 2023-01-02 --end 2023-10-20 --min-price 0 --min-avg-dollar-volume 0`

### Low Volatility

- **Paper:** Ang, Hodrick, Xing & Zhang (2006), J. Finance 61(1)
- **Signal:** rank by -stdev(daily returns, 252); long the calmest names
- **Sharpe:** 7.5282 | **Total return:** 19.34% | **Benchmark:** 10.47% | **Trades:** 30
- **Reproduction (this report):** `uv run python scripts/run_strategy_report.py`
- **CLI form (live universes):** `uv run screener backtest-rolling -m us --strategy low_volatility --tickers MOMA,MOMB,STDY,CALM,CHOP,WILD,DOWNA,DOWNB --top 3 --hold 21 --start 2023-01-02 --end 2023-10-20 --min-price 0 --min-avg-dollar-volume 0`

### Momentum + Low-Vol Combo

- **Paper:** Blitz & van Vliet (2007) defensive-factor blend; JT (1993) + AHXZ (2006)
- **Signal:** rank by 0.5*pct_rank(mom) + 0.5*pct_rank(-vol); long calm winners
- **Sharpe:** 5.3226 | **Total return:** 28.88% | **Benchmark:** 10.47% | **Trades:** 30
- **Reproduction (this report):** `uv run python scripts/run_strategy_report.py`
- **CLI form (live universes):** `uv run screener backtest-rolling -m us --strategy mom_lowvol_combo --tickers MOMA,MOMB,STDY,CALM,CHOP,WILD,DOWNA,DOWNB --top 3 --hold 21 --start 2023-01-02 --end 2023-10-20 --min-price 0 --min-avg-dollar-volume 0`

## Live illustrative run

The headline metrics above are deterministic/synthetic by design. For a real-market sanity check on the same strategies, see [`results/live_illustrative.md`](live_illustrative.md): a 3-year rolling backtest on a curated liquid US set where all three strategies beat the `SPY` benchmark. Those numbers are **network-dependent and survivorship-biased** (curated current names, not full point-in-time membership), so they are NOT pinned by tests and will drift with live data.

## Skipped (documented)

- **Basu 1/PE value (1977):** requires point-in-time fundamentals (trailing P/E history), which this codebase does not provide for backtests (data is daily OHLCV only). The `value` criterion (`screener/criteria/plugins/value.py`) is available for **live screening** only — `uv run screener screen -m us -c value`. Backtest intentionally skipped to avoid look-ahead/survivorship bias.
- **Magic Formula / Piotroski / Sloan / Fama-French:** same reason — no PIT fundamentals; skipped.
