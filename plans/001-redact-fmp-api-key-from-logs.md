# Plan 001 — Redact FMP API key from warning logs

- **Status:** DONE
- **Written against commit:** `9547d4d` (run `git log --oneline -1` — if HEAD has moved, re-verify every excerpt below before editing; if an excerpt no longer matches, STOP and report drift)
- **Category:** security
- **Effort:** S · **Risk of fix:** low
- **Depends on:** nothing

## Why this matters

The FMP price fetcher sends the API key as a URL query parameter (this is FMP's required auth scheme — it cannot move to a header). When FMP returns an HTTP error (429 rate limit, 403, 5xx), `response.raise_for_status()` raises a `requests.HTTPError` whose message contains the **full request URL including `apikey=<secret>`**. The resilience wrapper then logs that exception at WARNING level, which is visible at the default INFO log level. Net effect: any FMP outage or rate-limit prints the user's API key to the terminal and any captured logs.

## Current state (verified excerpts)

`screener/backtester/data.py:514-532` — key in query params, `raise_for_status` raises with full URL:

```python
            def request_payload() -> object:
                response = self.session.get(
                    f"{self.base_url}/{ticker}",
                    params={
                        "from": start_ts.date().isoformat(),
                        "to": end_ts.date().isoformat(),
                        "apikey": self.api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()

            payload = call_with_resilience(
                "fmp",
                f"historical prices {ticker}",
                request_payload,
                fallback={},
            )
```

`screener/resilience.py:118-126` — the exception (with URL) is interpolated into a WARNING log:

```python
    breaker.record_failure()
    LOG.warning(
        "%s failed for %s after %d attempt(s): %s",
        provider,
        operation,
        max(1, config.attempts),
        last_exc,
    )
    return fallback
```

`screener/insiders.py:246-253` uses `urllib` with `apikey` in the query string. `urllib.error.HTTPError`'s `str()` does **not** include the URL, so it does not leak through this same path today — but any future logging of `req.full_url` would. The redaction helper below protects both paths generically.

Default logging is INFO (`screener/logging_config.py:22-37`), so WARNING messages reach the user.

## The fix

Add a small redaction helper in `screener/resilience.py` and pass the exception text through it in the one place it is logged.

### Step 1 — add the helper

In `screener/resilience.py`, near the top (after the existing imports; `re` will need to be imported):

```python
_SECRET_PARAM_RE = re.compile(r"(?i)\b(apikey|api_key|token|auth)=([^&\s\"']+)")


def redact_secrets(text: str) -> str:
    """Mask credential-bearing query parameters in log/error text."""
    return _SECRET_PARAM_RE.sub(r"\1=***", text)
```

### Step 2 — use it at the logging site

In `call_with_resilience` (`screener/resilience.py:119-125`), change the final argument of the `LOG.warning` call from `last_exc` to `redact_secrets(str(last_exc))`. Do not change the message format otherwise.

### Step 3 — tests

Extend `tests/test_resilience.py` (it already exists — read it first and match its style). Add:

1. A unit test for `redact_secrets`: input `"https://x/api?from=a&apikey=SECRET123&to=b"` → output contains `apikey=***` and does not contain `SECRET123`. Also cover `TOKEN=abc` (case-insensitive) and text with no secrets (returned unchanged).
2. A test that calls `call_with_resilience` with a `func` that always raises `requests.HTTPError("401 Client Error: ... ?apikey=SECRET123")`, using `caplog` at WARNING level, and asserts the captured log text does not contain `SECRET123` but does contain `apikey=***`. Pass `retry=RetryConfig(attempts=1, ...)` (read `RetryConfig` in `screener/resilience.py` for exact fields) and a no-op `sleep` so the test is instant. Note: the circuit breaker is module-global per provider name — use a unique provider name (e.g. `"test-redact"`) so other tests' breaker state can't interfere.

## Verification gates (run from repo root)

```bash
uv run pytest tests/test_resilience.py -q        # new tests pass
uv run pytest -q                                  # full suite passes (177+ tests)
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
uv run mypy
```

Expected: all exit 0.

## Boundaries

- **In scope:** `screener/resilience.py`, `tests/test_resilience.py`.
- **Out of scope:** do NOT move the apikey out of query params (FMP requires it there); do NOT touch `screener/backtester/data.py`, `screener/insiders.py`, or `screener/logging_config.py`; do NOT add a logging Filter class (overkill — single choke point exists).

## Escape hatches

- If `tests/test_resilience.py` does not exist or `RetryConfig` has no `attempts` field, STOP and report — the codebase has drifted from this plan.
- If you find other call sites that log raw provider exceptions (grep `last_exc\|str(exc)` in `screener/`), report them in your summary but do not fix them here.

## Maintenance note

Any future provider that authenticates via query params is automatically covered as long as errors are logged through `call_with_resilience`. Reviewers should reject new `LOG.*` calls that interpolate raw exception text from HTTP clients without `redact_secrets`.
