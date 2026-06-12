# Plan 002 — Stop caching failed TradingView scans; log silent fallbacks

- **Status:** DONE
- **Written against commit:** `9547d4d` (if HEAD has moved, re-verify every excerpt before editing; on mismatch STOP and report drift)
- **Category:** correctness / DX
- **Effort:** M · **Risk of fix:** medium
- **Depends on:** nothing

## Why this matters

When TradingView is down or rate-limits, the screen pipeline returns an **empty result that is indistinguishable from "no stocks matched"** — and, worse, the empty fallback frame is **written to the scanner cache**, so the outage is served from cache for the full TTL (default 15 minutes) even after TradingView recovers. Two smaller best-effort code paths also swallow exceptions with no log line, making field debugging guesswork.

## Current state (verified excerpts)

### Bug A — cache poisoning on provider failure

`screener/scanner.py:59-76`:

```python
    key = stable_key(key_parts)
    frame_path = cache_path("tradingview_scanner", key, "parquet")
    meta_path = cache_path("tradingview_scanner", key, "json")
    if not refresh and all_fresh((frame_path, meta_path), cache_ttl):
        cached = read_frame(frame_path)
        meta = read_json(meta_path, default={}) or {}
        if cached is not None:
            return int(meta.get("count", 0)), cached

    count, df = call_with_resilience(
        "tradingview",
        operation,
        query.get_scanner_data,
        fallback=(0, pd.DataFrame(columns=columns)),
    )
    write_frame(frame_path, df)
    write_json(meta_path, {"count": int(count)})
    return count, df
```

`call_with_resilience` (`screener/resilience.py:88-126`) returns the `fallback` value after exhausting retries, so a failure produces `(0, empty_df)` — which lines 74-75 then persist to cache unconditionally.

### Nit B — silent excepts with zero log output

`screener/unusual_volume/nse_client.py:107-113` (best-effort cookie priming — intentionally non-fatal, but silent):

```python
    try:
        resp = session.get(page_url, timeout=10)
        if resp.status_code < 400:
            primed_pages.setdefault(session_id, set()).add(page_url)
            _tls.primed_pages = primed_pages
    except Exception:
        pass
```

`screener/universes.py:197-205` (corrupt membership cache falls back to refetch, silently):

```python
    if use_cache and path.exists():
        try:
            payload = json.loads(path.read_text())
            return { ... }
        except (ValueError, OSError):
            pass
```

Note: `screener/unusual_volume/output.py:155-159` (`except (TypeError, ValueError): pass` around `pd.isna`) was audited and is the **standard pandas idiom** for list-like values — do NOT touch it.

## The fix

### Step 1 — sentinel fallback in `scanner.py` so failure ≠ empty

In `screener/scanner.py`, change the `call_with_resilience` call to use `fallback=None`, then branch:

```python
    result = call_with_resilience(
        "tradingview",
        operation,
        query.get_scanner_data,
        fallback=None,
    )
    if result is None:
        LOG.warning(
            "tradingview scan failed for %s; returning empty results "
            "(not cached) — rerun with --refresh once connectivity is back",
            operation,
        )
        return 0, pd.DataFrame(columns=columns)
    count, df = result
    write_frame(frame_path, df)
    write_json(meta_path, {"count": int(count)})
    return count, df
```

Details to get right:

- Read the whole `screener/scanner.py` first. Reuse the module's existing logger if one exists; otherwise create one the same way `screener/resilience.py` does (look at its `LOG = ...` line and match).
- `call_with_resilience` is generic over `T` — `fallback=None` is valid. mypy runs in CI; make sure the inferred type is `tuple[int, pd.DataFrame] | None` (annotate `result` explicitly if needed).
- A genuinely-empty successful scan (TradingView answered, zero matches) must still be cached — only the `None` failure path skips the cache write. This preserves the "empty screens are cheap to repeat" behavior.

### Step 2 — one debug log line at each silent-except site

- `screener/unusual_volume/nse_client.py:112-113`: replace `pass` with a `logger.debug("NSE page priming failed for %s; will retry on next call", page_url)` (match the module's existing logger; if the module has none, add one in its established style). Keep `except Exception` — priming is best-effort by design (docstring at lines 96-102 explains why).
- `screener/universes.py:204-205`: replace `pass` with a debug log noting the membership cache at `path` was unreadable and is being refetched.

These are DEBUG, not WARNING — they fire in normal degraded operation and must not spam default output.

### Step 3 — tests

Pattern file: `tests/test_tradingview_cache.py` already tests the scanner cache — read it first and extend it (or add alongside in the same style):

1. **Failure is not cached:** stub `query.get_scanner_data` to raise; call the scan function with a tmp cache dir (see how the existing test redirects `cache_path`, likely via monkeypatch/fixture); assert the returned frame is empty AND `frame_path`/`meta_path` do not exist afterwards.
2. **Success-empty IS cached:** stub returns `(0, empty_df)` without raising; assert cache files are written.
3. **Recovery:** after a failed call, a subsequent successful call returns real data (no stale empty cache shadowing it).
4. `caplog` assertion that the failure path logs a WARNING containing "not cached".

## Verification gates

```bash
uv run pytest tests/test_tradingview_cache.py -q
uv run pytest -q
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
uv run mypy
```

## Boundaries

- **In scope:** `screener/scanner.py`, `screener/unusual_volume/nse_client.py` (the one `pass`), `screener/universes.py` (the one `pass`), tests.
- **Out of scope:** `screener/resilience.py` (its retry/breaker/warning behavior is correct and tested); `screener/unusual_volume/output.py`; any change to cache TTLs or the cache module; any change to how empty-but-successful results are displayed.

## Escape hatches

- If `call_with_resilience`'s signature rejects `fallback=None` under mypy in a way you can't fix with a local annotation, STOP and report rather than changing `resilience.py`.
- If the existing tests redirect the cache dir via a mechanism you can't find within `tests/test_tradingview_cache.py` and `screener/cache.py`, STOP and report.

## Maintenance note

The rule this establishes: **fallback values from `call_with_resilience` must never be persisted to caches.** Reviewers should check any new `call_with_resilience` call site whose result is written to disk.
