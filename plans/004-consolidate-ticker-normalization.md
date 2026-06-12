# Plan 004 — Consolidate duplicate ticker normalization (fixes latent BSE bug)

- **Status:** TODO
- **Written against commit:** `9547d4d` (re-verify excerpts if HEAD moved; on mismatch STOP and report drift)
- **Category:** tech-debt (with one latent correctness bug)
- **Effort:** S-M · **Risk of fix:** low-medium
- **Depends on:** plan 003 recommended first (regression net), but not strictly required.

## Why this matters

TradingView→yfinance symbol translation exists twice. The canonical version is `tv_to_yf` in `screener/backtester/data.py`, already used by `conviction.py:43`, `rs_breakout.py:25`, and `earnings_backtest/data.py:23`. `screener/insiders.py` has a private reimplementation `_tv_to_yf` that **disagrees on BSE symbols**: for `"BSE:TCS"` in india mode the canonical function returns `TCS.BO` (correct exchange), while the insiders copy strips the prefix and appends `.NS`, yielding `TCS.NS`. For BSE-only listings that yfinance symbol is wrong or missing — insider/promoter enrichment silently degrades for those tickers. Any future suffix-rule change must currently be made twice.

## Current state (verified excerpts)

Canonical — `screener/backtester/data.py:119-139`:

```python
def tv_to_yf(symbol: str, market: str) -> str:
    """Translate a TradingView-style symbol to a yfinance symbol.

    Examples:
      'NSE:RELIANCE' + india → 'RELIANCE.NS'
      'BSE:TCS'     + india → 'TCS.BO'
      'NASDAQ:AAPL' + us    → 'AAPL'
      ...
    """
    sym = symbol.strip().upper()
    if ":" in sym:
        exch, rest = sym.split(":", 1)
        if exch == "NSE":
            return f"{rest}.NS"
        if exch == "BSE":
            return f"{rest}.BO"
        return rest
    if market == "india" and "." not in sym:
        return f"{sym}.NS"
    return sym
```

Duplicate — `screener/insiders.py:67-71` (`_INDIA_SUFFIXES = (".NS", ".BO")` at line 38):

```python
def _tv_to_yf(ticker: str, market: str) -> str:
    symbol = ticker.split(":", 1)[1] if ":" in ticker else ticker
    if market == "india" and not symbol.endswith(_INDIA_SUFFIXES):
        return f"{symbol}.NS"
    return symbol
```

Behavioral diffs to be aware of (canonical vs insiders copy):

| Input | canonical `tv_to_yf` | insiders `_tv_to_yf` |
|---|---|---|
| `BSE:TCS`, india | `TCS.BO` | `TCS.NS` ← bug |
| `nse:reliance`, india | `RELIANCE.NS` (upper-cases) | `reliance.NS` (preserves case) |
| `RELIANCE.NS`, india | `RELIANCE.NS` (has `.`) | `RELIANCE.NS` (has suffix) |
| `BRK.B`, us | `BRK.B` | `BRK.B` |

The case-handling diff matters only if insiders callers pass lowercase symbols — check the call sites (grep `_tv_to_yf(` in `screener/insiders.py`) and confirm inputs are TradingView-style upper-case tickers from the screen pipeline. They should be; if you find a lowercase-producing caller, normalize at that caller.

## Steps

### Step 1 — switch insiders.py to the canonical function

In `screener/insiders.py`:

- Add `from screener.backtester.data import tv_to_yf` (match the module's existing import style/ordering).
- Delete `_tv_to_yf` (lines 67-71) and replace its call sites with `tv_to_yf`.
- Delete `_INDIA_SUFFIXES` (line 38) **only if** nothing else in the module uses it (grep first).

Check for import cycles: `screener/backtester/data.py` must not import `screener.insiders` (verify with grep). As of `9547d4d` it does not.

### Step 2 — tests

`tests/test_fmp_insiders.py` exists — read it for style. Add a small test (there or in a more fitting existing insiders test module) asserting the promoter/insider path now maps `BSE:TCS` + india → `TCS.BO`. Also add direct unit tests for `tv_to_yf` covering the table above if none exist (grep `tv_to_yf` under `tests/`).

### Step 3 — sweep for further stragglers

```bash
grep -rn "\.NS\"\|'\.NS'\|f\"{.*}.NS\"" screener/ | grep -v backtester/data.py
```

Report (do not fix) any other module that hand-rolls suffix logic instead of calling `tv_to_yf` — candidates become follow-up work, not scope creep here.

## Verification gates

```bash
uv run pytest tests/test_fmp_insiders.py -q
uv run pytest -q
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
uv run mypy
```

## Boundaries

- **In scope:** `screener/insiders.py`, insider-related test files.
- **Out of scope:** the three duplicate *batch price-fetch* implementations (`earnings_backtest/data.py:747`, `rs_breakout.py:333`, the backtester fetchers) — they have genuinely different concurrency/error semantics and consolidating them is a separate, larger task (see plans/README "deferred" notes); any change to `tv_to_yf` itself; promoter-data scraping logic.

## Escape hatches

- If switching to `tv_to_yf` changes behavior for a symbol class the insiders tests actually exercise with lowercase or pre-suffixed input in a way that breaks fixtures, STOP and report the exact inputs rather than adding compatibility shims.
- If an import cycle appears, STOP and report — do not move `tv_to_yf` to a new module unilaterally.

## Maintenance note

After this lands, `tv_to_yf` in `backtester/data.py` is the single symbol-translation authority. Reviewers should reject new private `_tv_to_yf`-style helpers.
